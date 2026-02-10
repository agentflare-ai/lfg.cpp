// GBNF generation, JSON serialization, and schema support for PEG parser.
// Split from peg-parser.cpp to keep the core parser self-contained
// (peg-parser.cpp only depends on peg-parser.h + unicode.h).
//
// This file depends on json-schema-to-grammar.h and nlohmann/json.hpp.
// It provides: dump(), schema() builder, build_grammar(), to_json/from_json/save/load.

#include "peg-parser.h"
#include "peg-trie.h"
#include "json-schema-to-grammar.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Trick to catch missing branches
template <typename T>
inline constexpr bool is_always_false_v = false;

// Local helper (avoids pulling in common.h for one trivial function)
static std::string string_join(const std::vector<std::string> & vec, const std::string & sep) {
    std::string result;
    for (size_t i = 0; i < vec.size(); i++) {
        if (i > 0) result += sep;
        result += vec[i];
    }
    return result;
}

// ---------------------------------------------------------------------------
// dump() — debug representation of a parser
// ---------------------------------------------------------------------------

std::string common_peg_arena::dump(common_peg_parser_id id) const {
    const auto & parser = parsers_.at(id);

    return std::visit([this](const auto & p) -> std::string {
        using T = std::decay_t<decltype(p)>;

        if constexpr (std::is_same_v<T, common_peg_epsilon_parser>) {
            return "Epsilon";
        } else if constexpr (std::is_same_v<T, common_peg_start_parser>) {
            return "Start";
        } else if constexpr (std::is_same_v<T, common_peg_end_parser>) {
            return "End";
        } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
            return "Literal(" + p.literal + ")";
        } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
            std::vector<std::string> parts;
            for (const auto & child : p.children) {
                parts.push_back(dump(child));
            }
            return "Sequence(" + string_join(parts, ", ") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
            std::vector<std::string> parts;
            for (const auto & child : p.children) {
                parts.push_back(dump(child));
            }
            return "Choice(" + string_join(parts, ", ") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
            if (p.max_count == -1) {
                return "Repetition(" + dump(p.child) + ", " + std::to_string(p.min_count) + ", unbounded)";
            }
            return "Repetition(" + dump(p.child) + ", " + std::to_string(p.min_count) + ", " + std::to_string(p.max_count) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_and_parser>) {
            return "And(" + dump(p.child) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_not_parser>) {
            return "Not(" + dump(p.child) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
            return "Any";
        } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
            return "Space";
        } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
            if (p.max_count == -1) {
                return "CharRepeat(" + p.pattern + ", " + std::to_string(p.min_count) + ", unbounded)";
            }
            return "CharRepeat(" + p.pattern + ", " + std::to_string(p.min_count) + ", " + std::to_string(p.max_count) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_json_string_parser>) {
            return "JsonString()";
        } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
            return "Until(" + string_join(p.delimiters, " | ") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
            return "Schema(" + dump(p.child) + ", " + (p.schema ? p.schema->dump() : "null") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
            return "Rule(" + p.name + ", " + dump(p.child) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
            return "Ref(" + p.name + ")";
        } else {
            return "Unknown";
        }
    }, parser);
}

// ---------------------------------------------------------------------------
// schema() builder method — wraps a parser with JSON schema metadata
// ---------------------------------------------------------------------------

common_peg_parser common_peg_parser_builder::schema(const common_peg_parser & p, const std::string & name, const nlohmann::ordered_json & schema, bool raw) {
    return wrap(arena_.add_parser(common_peg_schema_parser{p.id(), name, std::make_shared<nlohmann::ordered_json>(schema), raw}));
}

// ---------------------------------------------------------------------------
// GBNF generation
// ---------------------------------------------------------------------------

static std::string gbnf_escape_char_class(char c) {
    switch (c) {
        case '\n': return "\\n";
        case '\t': return "\\t";
        case '\r': return "\\r";
        case '\\': return "\\\\";
        case ']':  return "\\]";
        case '[':  return "\\[";
        default:   return std::string(1, c);
    }
}

static std::string gbnf_excluding_pattern(const std::vector<std::string> & strings) {
    trie matcher(strings);
    auto pieces = matcher.collect_prefix_and_next();

    std::string pattern;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (i > 0) {
            pattern += " | ";
        }

        const auto & pre = pieces[i].prefix;
        const auto & chars = pieces[i].next_chars;

        std::string cls;
        cls.reserve(chars.size());
        for (const auto & ch : chars) {
            cls += gbnf_escape_char_class(ch);
        }

        if (!pre.empty()) {
            pattern += gbnf_format_literal(pre) + " [^" + cls + "]";
        } else {
            pattern += "[^" + cls + "]";
        }
    }

    return "(" + pattern + ")*";
}

static std::unordered_set<std::string> collect_reachable_rules(
    const common_peg_arena & arena,
    const common_peg_parser_id & rule
) {
    std::unordered_set<std::string> reachable;
    std::unordered_set<std::string> visited;

    std::function<void(common_peg_parser_id)> visit = [&](common_peg_parser_id id) {
        const auto & parser = arena.get(id);

        std::visit([&](const auto & p) {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, common_peg_epsilon_parser> ||
                          std::is_same_v<T, common_peg_start_parser> ||
                          std::is_same_v<T, common_peg_end_parser> ||
                          std::is_same_v<T, common_peg_until_parser> ||
                          std::is_same_v<T, common_peg_literal_parser> ||
                          std::is_same_v<T, common_peg_chars_parser> ||
                          std::is_same_v<T, common_peg_space_parser> ||
                          std::is_same_v<T, common_peg_any_parser> ||
                          std::is_same_v<T, common_peg_json_string_parser>) {
                // These parsers do not have any children
            } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
                for (auto child : p.children) {
                    visit(child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
                for (auto child : p.children) {
                    visit(child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_repetition_parser> ||
                                 std::is_same_v<T, common_peg_and_parser> ||
                                 std::is_same_v<T, common_peg_not_parser> ||
                                 std::is_same_v<T, common_peg_tag_parser> ||
                                 std::is_same_v<T, common_peg_atomic_parser> ||
                                 std::is_same_v<T, common_peg_schema_parser>) {
                visit(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
                if (visited.find(p.name) == visited.end()) {
                    visited.insert(p.name);
                    reachable.insert(p.name);
                    visit(p.child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
                // Traverse rules so we pick up everything
                auto referenced_rule = arena.get_rule(p.name);
                visit(referenced_rule);
            } else {
                static_assert(is_always_false_v<T>);
            }
        }, parser);
    };

    visit(rule);
    return reachable;
}

// GBNF generation implementation
void common_peg_arena::build_grammar(const common_grammar_builder & builder, bool lazy) const {
    // Generate GBNF for a parser
    std::function<std::string(common_peg_parser_id)> to_gbnf = [&](common_peg_parser_id id) -> std::string {
        const auto & parser = parsers_.at(id);

        return std::visit([&](const auto & p) -> std::string {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, common_peg_epsilon_parser> ||
                          std::is_same_v<T, common_peg_start_parser> ||
                          std::is_same_v<T, common_peg_end_parser>) {
                return "";
            } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
                return gbnf_format_literal(p.literal);
            } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
                std::string s;
                for (const auto & child : p.children) {
                    if (!s.empty()) {
                        s += " ";
                    }
                    auto child_gbnf = to_gbnf(child);
                    const auto & child_parser = parsers_.at(child);
                    if (std::holds_alternative<common_peg_choice_parser>(child_parser) ||
                        std::holds_alternative<common_peg_sequence_parser>(child_parser)) {
                        s += "(" + child_gbnf + ")";
                    } else {
                        s += child_gbnf;
                    }
                }
                return s;
            } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
                std::string s;
                for (const auto & child : p.children) {
                    if (!s.empty()) {
                        s += " | ";
                    }
                    auto child_gbnf = to_gbnf(child);
                    const auto & child_parser = parsers_.at(child);
                    if (std::holds_alternative<common_peg_choice_parser>(child_parser)) {
                        s += "(" + child_gbnf + ")";
                    } else {
                        s += child_gbnf;
                    }
                }
                return s;
            } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
                auto child_gbnf = to_gbnf(p.child);
                const auto & child_parser = parsers_.at(p.child);
                if (std::holds_alternative<common_peg_choice_parser>(child_parser) ||
                    std::holds_alternative<common_peg_sequence_parser>(child_parser)) {
                    child_gbnf = "(" + child_gbnf + ")";
                }
                if (p.min_count == 0 && p.max_count == 1) {
                    return child_gbnf + "?";
                }
                if (p.min_count == 0 && p.max_count == -1) {
                    return child_gbnf + "*";
                }
                if (p.min_count == 1 && p.max_count == -1) {
                    return child_gbnf + "+";
                }
                if (p.max_count == -1) {
                    return child_gbnf + "{" + std::to_string(p.min_count) + ",}";
                }
                if (p.min_count == p.max_count) {
                    if (p.min_count == 1) {
                        return child_gbnf;
                    }
                    return child_gbnf + "{" + std::to_string(p.min_count) + "}";
                }
                return child_gbnf + "{" + std::to_string(p.min_count) + "," + std::to_string(p.max_count) + "}";
            } else if constexpr (std::is_same_v<T, common_peg_and_parser> || std::is_same_v<T, common_peg_not_parser>) {
                return "";  // Lookahead not supported in GBNF
            } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
                return ".";
            } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
                return "space";
            } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
                std::string result = p.pattern;
                if (p.min_count == 0 && p.max_count == 1) {
                    return result + "?";
                }
                if (p.min_count == 0 && p.max_count == -1) {
                    return result + "*";
                }
                if (p.min_count == 1 && p.max_count == -1) {
                    return result + "+";
                }
                if (p.max_count == -1) {
                    return result + "{" + std::to_string(p.min_count) + ",}";
                }
                if (p.min_count == p.max_count) {
                    if (p.min_count == 1) {
                        return result;
                    }
                    return result + "{" + std::to_string(p.min_count) + "}";
                }
                return result + "{" + std::to_string(p.min_count) + "," + std::to_string(p.max_count) + "}";
            } else if constexpr (std::is_same_v<T, common_peg_json_string_parser>) {
                return R"(( [^"\\] | "\\" ( ["\\/ bfnrt] | "u" [0-9a-fA-F]{4} ) )*)";
            } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
                if (p.delimiters.empty()) {
                    return ".*";
                }
                return gbnf_excluding_pattern(p.delimiters);
            } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
                if (p.schema) {
                    if (p.raw && p.schema->contains("type") && p.schema->at("type").is_string() && p.schema->at("type") == "string") {
                        // TODO: Implement more comprehensive grammar generation for raw strings.
                        // For now, use the grammar emitted from the underlying parser.
                        return to_gbnf(p.child);
                    }
                    return builder.add_schema(p.name, *p.schema);
                }
                return to_gbnf(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
                return p.name;
            } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
                // Refs should not exist after flattening, but kept just in case
                return p.name;
            } else if constexpr (std::is_same_v<T, common_peg_tag_parser>) {
                return to_gbnf(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
                return to_gbnf(p.child);
            } else {
                static_assert(is_always_false_v<T>);
            }
        }, parser);
    };

    // Collect reachable rules
    std::unordered_set<std::string> reachable_rules;

    if (lazy) {
        // Collect rules reachable from trigger rules
        for (const auto & [name, id] : rules_) {
            const auto & parser = parsers_.at(id);
            if (auto rule = std::get_if<common_peg_rule_parser>(&parser)) {
                if (rule->trigger) {
                    // Mark trigger as reachable and visit it
                    reachable_rules.insert(name);
                    auto add_rules = collect_reachable_rules(*this, id);
                    reachable_rules.insert(add_rules.begin(), add_rules.end());
                }
            }
        }
    } else {
        // Collect rules reachable from root
        reachable_rules = collect_reachable_rules(*this, root_);
    }

    // Create GBNF rules for all reachable rules
    for (const auto & [name, rule_id] : rules_) {
        if (reachable_rules.find(name) == reachable_rules.end()) {
            continue;
        }

        const auto & parser = parsers_.at(rule_id);
        if (auto rule = std::get_if<common_peg_rule_parser>(&parser)) {
            builder.add_rule(rule->name, to_gbnf(rule->child));
        }
    }

    if (lazy) {
        // Generate root rule from trigger rules only
        std::vector<std::string> trigger_names;
        for (const auto & [name, rule_id] : rules_) {
            const auto & parser = parsers_.at(rule_id);
            if (auto rule = std::get_if<common_peg_rule_parser>(&parser)) {
                if (rule->trigger) {
                    trigger_names.push_back(rule->name);
                }
            }
        }

        // Sort for predictable order
        std::sort(trigger_names.begin(), trigger_names.end());
        builder.add_rule("root", string_join(trigger_names, " | "));
    } else if (root_ != COMMON_PEG_INVALID_PARSER_ID) {
        builder.add_rule("root", to_gbnf(root_));
    }
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

static nlohmann::json serialize_parser_variant(const common_peg_parser_variant & variant) {
    using json = nlohmann::json;

    return std::visit([](const auto & p) -> json {
        using T = std::decay_t<decltype(p)>;

        if constexpr (std::is_same_v<T, common_peg_epsilon_parser>) {
            return json{{"type", "epsilon"}};
        } else if constexpr (std::is_same_v<T, common_peg_start_parser>) {
            return json{{"type", "start"}};
        } else if constexpr (std::is_same_v<T, common_peg_end_parser>) {
            return json{{"type", "end"}};
        } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
            return json{{"type", "literal"}, {"literal", p.literal}};
        } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
            return json{{"type", "sequence"}, {"children", p.children}};
        } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
            return json{{"type", "choice"}, {"children", p.children}};
        } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
            return json{
                {"type", "repetition"},
                {"child", p.child},
                {"min_count", p.min_count},
                {"max_count", p.max_count}
            };
        } else if constexpr (std::is_same_v<T, common_peg_and_parser>) {
            return json{{"type", "and"}, {"child", p.child}};
        } else if constexpr (std::is_same_v<T, common_peg_not_parser>) {
            return json{{"type", "not"}, {"child", p.child}};
        } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
            return json{{"type", "any"}};
        } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
            return json{{"type", "space"}};
        } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
            json ranges = json::array();
            for (const auto & range : p.ranges) {
                ranges.push_back({{"start", range.start}, {"end", range.end}});
            }
            return json{
                {"type", "chars"},
                {"pattern", p.pattern},
                {"ranges", ranges},
                {"negated", p.negated},
                {"min_count", p.min_count},
                {"max_count", p.max_count}
            };
        } else if constexpr (std::is_same_v<T, common_peg_json_string_parser>) {
            return json{{"type", "json_string"}};
        } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
            return json{{"type", "until"}, {"delimiters", p.delimiters}};
        } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
            return json{
                {"type", "schema"},
                {"child", p.child},
                {"name", p.name},
                {"schema", p.schema ? *p.schema : nullptr},
                {"raw", p.raw}
            };
        } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
            return json{
                {"type", "rule"},
                {"name", p.name},
                {"child", p.child},
                {"trigger", p.trigger}
            };
        } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
            return json{{"type", "ref"}, {"name", p.name}};
        } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
            return json{{"type", "atomic"}, {"child", p.child}};
        } else if constexpr (std::is_same_v<T, common_peg_tag_parser>) {
            return json{
                {"type", "tag"},
                {"child", p.child},
                {"tag", p.tag}
            };
        }
    }, variant);
}

nlohmann::json common_peg_arena::to_json() const {
    auto parsers = nlohmann::json::array();
    for (const auto & parser : parsers_) {
        parsers.push_back(serialize_parser_variant(parser));
    }
    return nlohmann::json{
        {"parsers", parsers},
        {"rules", rules_},
        {"root", root_}
    };
}

static common_peg_parser_variant deserialize_parser_variant(const nlohmann::json & j) {
    if (!j.contains("type") || !j["type"].is_string()) {
        throw std::runtime_error("Parser variant JSON missing or invalid 'type' field");
    }

    std::string type = j["type"];

    if (type == "epsilon") {
        return common_peg_epsilon_parser{};
    }
    if (type == "start") {
        return common_peg_start_parser{};
    }
    if (type == "end") {
        return common_peg_end_parser{};
    }
    if (type == "literal") {
        if (!j.contains("literal") || !j["literal"].is_string()) {
            throw std::runtime_error("literal parser missing or invalid 'literal' field");
        }
        return common_peg_literal_parser{j["literal"]};
    }
    if (type == "sequence") {
        if (!j.contains("children") || !j["children"].is_array()) {
            throw std::runtime_error("sequence parser missing or invalid 'children' field");
        }
        return common_peg_sequence_parser{j["children"].get<std::vector<common_peg_parser_id>>()};
    }
    if (type == "choice") {
        if (!j.contains("children") || !j["children"].is_array()) {
            throw std::runtime_error("choice parser missing or invalid 'children' field");
        }
        return common_peg_choice_parser{j["children"].get<std::vector<common_peg_parser_id>>()};
    }
    if (type == "repetition") {
        if (!j.contains("child") || !j.contains("min_count") || !j.contains("max_count")) {
            throw std::runtime_error("repetition parser missing required fields");
        }
        return common_peg_repetition_parser{
            j["child"].get<common_peg_parser_id>(),
            j["min_count"].get<int>(),
            j["max_count"].get<int>()
        };
    }
    if (type == "and") {
        if (!j.contains("child")) {
            throw std::runtime_error("and parser missing 'child' field");
        }
        return common_peg_and_parser{j["child"].get<common_peg_parser_id>()};
    }
    if (type == "not") {
        if (!j.contains("child")) {
            throw std::runtime_error("not parser missing 'child' field");
        }
        return common_peg_not_parser{j["child"].get<common_peg_parser_id>()};
    }
    if (type == "any") {
        return common_peg_any_parser{};
    }
    if (type == "space") {
        return common_peg_space_parser{};
    }
    if (type == "chars") {
        if (!j.contains("pattern") || !j.contains("ranges") || !j.contains("negated") ||
            !j.contains("min_count") || !j.contains("max_count")) {
            throw std::runtime_error("chars parser missing required fields");
        }
        common_peg_chars_parser parser;
        parser.pattern = j["pattern"];
        parser.negated = j["negated"];
        parser.min_count = j["min_count"];
        parser.max_count = j["max_count"];
        for (const auto & range_json : j["ranges"]) {
            if (!range_json.contains("start") || !range_json.contains("end")) {
                throw std::runtime_error("char_range missing 'start' or 'end' field");
            }
            parser.ranges.push_back({
                range_json["start"].get<uint32_t>(),
                range_json["end"].get<uint32_t>()
            });
        }
        return parser;
    }
    if (type == "json_string") {
        return common_peg_json_string_parser{};
    }
    if (type == "until") {
        if (!j.contains("delimiters") || !j["delimiters"].is_array()) {
            throw std::runtime_error("until parser missing or invalid 'delimiters' field");
        }
        return common_peg_until_parser{j["delimiters"].get<std::vector<std::string>>()};
    }
    if (type == "schema") {
        if (!j.contains("child") || !j.contains("name") || !j.contains("schema") || !j.contains("raw")) {
            throw std::runtime_error("schema parser missing required fields");
        }
        common_peg_schema_parser parser;
        parser.child = j["child"].get<common_peg_parser_id>();
        parser.name = j["name"];
        if (!j["schema"].is_null()) {
            parser.schema = std::make_shared<nlohmann::ordered_json>(j["schema"]);
        }
        parser.raw = j["raw"].get<bool>();
        return parser;
    }
    if (type == "rule") {
        if (!j.contains("name") || !j.contains("child") || !j.contains("trigger")) {
            throw std::runtime_error("rule parser missing required fields");
        }
        return common_peg_rule_parser{
            j["name"].get<std::string>(),
            j["child"].get<common_peg_parser_id>(),
            j["trigger"].get<bool>()
        };
    }
    if (type == "ref") {
        if (!j.contains("name") || !j["name"].is_string()) {
            throw std::runtime_error("ref parser missing or invalid 'name' field");
        }
        return common_peg_ref_parser{j["name"]};
    }
    if (type == "atomic") {
        if (!j.contains("child")) {
            throw std::runtime_error("tag parser missing required fields");
        }
        return common_peg_atomic_parser{
            j["child"].get<common_peg_parser_id>(),
        };
    }
    if (type == "tag") {
        if (!j.contains("child") || !j.contains("tag")) {
            throw std::runtime_error("tag parser missing required fields");
        }
        return common_peg_tag_parser{
            j["child"].get<common_peg_parser_id>(),
            j["tag"].get<std::string>(),
        };
    }

    throw std::runtime_error("Unknown parser type: " + type);
}

common_peg_arena common_peg_arena::from_json(const nlohmann::json & j) {
    if (!j.contains("parsers") || !j["parsers"].is_array()) {
        throw std::runtime_error("JSON missing or invalid 'parsers' array");
    }
    if (!j.contains("rules") || !j["rules"].is_object()) {
        throw std::runtime_error("JSON missing or invalid 'rules' object");
    }
    if (!j.contains("root")) {
        throw std::runtime_error("JSON missing 'root' field");
    }

    common_peg_arena arena;

    const auto & parsers_json = j["parsers"];
    arena.parsers_.reserve(parsers_json.size());
    for (const auto & parser_json : parsers_json) {
        arena.parsers_.push_back(deserialize_parser_variant(parser_json));
    }

    arena.rules_ = j["rules"].get<std::unordered_map<std::string, common_peg_parser_id>>();

    for (const auto & [name, id] : arena.rules_) {
        if (id >= arena.parsers_.size()) {
            throw std::runtime_error("Rule '" + name + "' references invalid parser ID: " + std::to_string(id));
        }
    }

    arena.root_ = j["root"].get<common_peg_parser_id>();
    if (arena.root_ != COMMON_PEG_INVALID_PARSER_ID && arena.root_ >= arena.parsers_.size()) {
        throw std::runtime_error("Root references invalid parser ID: " + std::to_string(arena.root_));
    }

    return arena;
}

std::string common_peg_arena::save() const {
    return to_json().dump();
}

void common_peg_arena::load(const std::string & data) {
    *this = from_json(nlohmann::json::parse(data));
}
