#include "peg-parser.h"
#include "peg-trie.h"
#include "unicode.h"

#include <algorithm>
#include <initializer_list>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <unordered_set>

// Trick to catch missing branches
template <typename T>
inline constexpr bool is_always_false_v = false;

const char * common_peg_parse_result_type_name(common_peg_parse_result_type type) {
    switch (type) {
        case COMMON_PEG_PARSE_RESULT_FAIL:            return "fail";
        case COMMON_PEG_PARSE_RESULT_SUCCESS:         return "success";
        case COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT: return "need_more_input";
        default:                                      return "unknown";
    }
}

static bool is_hex_digit(const char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

// Trie is now in peg-trie.h (shared with peg-parser-gbnf.cpp)

static std::pair<uint32_t, size_t> parse_hex_escape(const std::string & str, size_t pos, int hex_count) {
    if (pos + hex_count > str.length()) {
        return {0, 0};
    }

    uint32_t value = 0;
    for (int i = 0; i < hex_count; i++) {
        char c = str[pos + i];
        if (!is_hex_digit(c)) {
            return {0, 0};
        }
        value <<= 4;
        if ('a' <= c && c <= 'f') {
            value += c - 'a' + 10;
        } else if ('A' <= c && c <= 'F') {
            value += c - 'A' + 10;
        } else if ('0' <= c && c <= '9') {
            value += c - '0';
        } else {
            break;
        }
    }
    return {value, static_cast<size_t>(hex_count)};
}

static std::pair<uint32_t, size_t> parse_char_class_char(const std::string & content, size_t pos) {
    if (content[pos] == '\\' && pos + 1 < content.length()) {
        switch (content[pos + 1]) {
            case 'x': {
                auto result = parse_hex_escape(content, pos + 2, 2);
                if (result.second > 0) {
                    return {result.first, 2 + result.second};
                }
                // Invalid escape, treat as literal 'x'
                return {static_cast<uint32_t>('x'), 2};
            }
            case 'u': {
                auto result = parse_hex_escape(content, pos + 2, 4);
                if (result.second > 0) {
                    return {result.first, 2 + result.second};
                }
                // Invalid escape, treat as literal 'u'
                return {static_cast<uint32_t>('u'), 2};
            }
            case 'U': {
                auto result = parse_hex_escape(content, pos + 2, 8);
                if (result.second > 0) {
                    return {result.first, 2 + result.second};
                }
                // Invalid escape, treat as literal 'U'
                return {static_cast<uint32_t>('U'), 2};
            }
            case 'n':  return {'\n', 2};
            case 't':  return {'\t', 2};
            case 'r':  return {'\r', 2};
            case '\\': return {'\\', 2};
            case ']':  return {']', 2};
            case '[':  return {'[', 2};
            default:   return {static_cast<uint32_t>(content[pos + 1]), 2};
        }
    }

    // Regular character - return as codepoint
    return {static_cast<uint32_t>(static_cast<unsigned char>(content[pos])), 1};
}

static std::pair<std::vector<common_peg_chars_parser::char_range>, bool> parse_char_classes(const std::string & classes) {
    std::vector<common_peg_chars_parser::char_range> ranges;
    bool negated = false;

    std::string content = classes;
    if (content.front() == '[') {
        content = content.substr(1);
    }

    if (content.back() == ']') {
        content.pop_back();
    }

    // Check for negation
    if (!content.empty() && content.front() == '^') {
        negated = true;
        content = content.substr(1);
    }

    size_t i = 0;
    while (i < content.length()) {
        auto [start, start_len] = parse_char_class_char(content, i);
        i += start_len;

        if (i + 1 < content.length() && content[i] == '-') {
            // Range detected
            auto [end, end_len] = parse_char_class_char(content, i + 1);
            ranges.push_back(common_peg_chars_parser::char_range{start, end});
            i += 1 + end_len;
        } else {
            ranges.push_back(common_peg_chars_parser::char_range{start, start});
        }
    }

    return {ranges, negated};
}

void common_peg_ast_arena::visit(common_peg_ast_id id, const common_peg_ast_visitor & visitor) const {
    if (id == COMMON_PEG_INVALID_AST_ID) {
        return;
    }
    const auto & node = get(id);
    visitor(node);
    for (const auto & child : node.children) {
        visit(child, visitor);
    }
}

void common_peg_ast_arena::visit(const common_peg_parse_result & result, const common_peg_ast_visitor & visitor) const {
    for (const auto & node : result.nodes) {
        visit(node, visitor);
    }
}

struct parser_executor;

common_peg_parser_id common_peg_arena::add_parser(common_peg_parser_variant parser) {
    common_peg_parser_id id = parsers_.size();
    parsers_.push_back(std::move(parser));
    return id;
}

void common_peg_arena::add_rule(const std::string & name, common_peg_parser_id id) {
    rules_[name] = id;
}

common_peg_parser_id common_peg_arena::get_rule(const std::string & name) const {
    auto it = rules_.find(name);
    if (it == rules_.end()) {
        throw std::runtime_error("Rule not found: " + name);
    }
    return it->second;
}

struct parser_executor {
    const common_peg_arena & arena;
    common_peg_parse_context & ctx;
    size_t start_pos;

    parser_executor(const common_peg_arena & arena, common_peg_parse_context & ctx, size_t start)
        : arena(arena), ctx(ctx), start_pos(start) {}

    common_peg_parse_result operator()(const common_peg_epsilon_parser & /* p */) const {
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_start_parser & /* p */) const {
        return common_peg_parse_result(
            start_pos == 0 ? COMMON_PEG_PARSE_RESULT_SUCCESS : COMMON_PEG_PARSE_RESULT_FAIL,
            start_pos
        );
    }

    common_peg_parse_result operator()(const common_peg_end_parser & /* p */) const {
        return common_peg_parse_result(
            start_pos >= ctx.input.size() ? COMMON_PEG_PARSE_RESULT_SUCCESS : COMMON_PEG_PARSE_RESULT_FAIL,
            start_pos
        );
    }

    common_peg_parse_result operator()(const common_peg_literal_parser & p) {
        auto pos = start_pos;
        for (auto i = 0u; i < p.literal.size(); ++i) {
            if (pos >= ctx.input.size()) {
                if (!ctx.is_partial) {
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
            }
            if (ctx.input[pos] != p.literal[i]) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
            }
            ++pos;
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
    }

    common_peg_parse_result operator()(const common_peg_sequence_parser & p) {
        auto pos = start_pos;
        std::vector<common_peg_ast_id> nodes;

        for (const auto & child_id : p.children) {
            auto result = arena.parse(child_id, ctx, pos);
            if (result.fail()) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos, result.end);
            }

            if (!result.nodes.empty()) {
                nodes.insert(nodes.end(), result.nodes.begin(), result.nodes.end());
            }

            if (result.need_more_input()) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, result.end, std::move(nodes));
            }

            pos = result.end;
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos, std::move(nodes));
    }

    common_peg_parse_result operator()(const common_peg_choice_parser & p) {
        auto pos = start_pos;
        for (const auto & child_id : p.children) {
            auto result = arena.parse(child_id, ctx, pos);
            if (!result.fail()) {
                return result;
            }
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_repetition_parser & p) {
        auto pos = start_pos;
        int match_count = 0;
        std::vector<common_peg_ast_id> nodes;

        // Try to match up to max_count times (or unlimited if max_count is -1)
        while (p.max_count == -1 || match_count < p.max_count) {
            if (pos >= ctx.input.size()) {
                break;
            }

            auto result = arena.parse(p.child, ctx, pos);

            if (result.success()) {
                // Prevent infinite loop on empty matches
                if (result.end == pos) {
                    break;
                }

                if (!result.nodes.empty()) {
                    nodes.insert(nodes.end(), result.nodes.begin(), result.nodes.end());
                }

                pos = result.end;
                match_count++;
                continue;
            }

            if (result.need_more_input()) {
                if (!result.nodes.empty()) {
                    nodes.insert(nodes.end(), result.nodes.begin(), result.nodes.end());
                }

                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, result.end, std::move(nodes));
            }

            // Child failed - stop trying
            break;
        }

        // Check if we got enough matches
        if (p.min_count > 0 && match_count < p.min_count) {
            if (pos >= ctx.input.size() && ctx.is_partial) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos, std::move(nodes));
            }
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos, pos);
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos, std::move(nodes));
    }

    common_peg_parse_result operator()(const common_peg_and_parser & p) {
        auto result = arena.parse(p.child, ctx, start_pos);
        // Pass result but don't consume input
        return common_peg_parse_result(result.type, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_not_parser & p) {
        auto result = arena.parse(p.child, ctx, start_pos);

        if (result.success()) {
            // Fail if the underlying parser matches
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
        }

        if (result.need_more_input()) {
            // Propagate - need to know what child would match before negating
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos);
        }

        // Child failed, so negation succeeds
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_any_parser & /* p */) const {
        // Parse a single UTF-8 codepoint (not just a single byte)
        auto result = parse_utf8_codepoint(ctx.input, start_pos);

        if (result.status == utf8_parse_result::INCOMPLETE) {
            if (!ctx.is_partial) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
            }
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos);
        }
        if (result.status == utf8_parse_result::INVALID) {
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, start_pos + result.bytes_consumed);
    }

    common_peg_parse_result operator()(const common_peg_space_parser & /* p */) {
        auto pos = start_pos;
        while (pos < ctx.input.size()) {
            auto c = static_cast<unsigned char>(ctx.input[pos]);
            if (std::isspace(c)) {
                ++pos;
            } else {
                break;
            }
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
    }

    common_peg_parse_result operator()(const common_peg_chars_parser & p) const {
        auto pos = start_pos;
        int match_count = 0;

        // Try to match up to max_count times (or unlimited if max_count is -1)
        while (p.max_count == -1 || match_count < p.max_count) {
            auto result = parse_utf8_codepoint(ctx.input, pos);

            if (result.status == utf8_parse_result::INCOMPLETE) {
                if (match_count >= p.min_count) {
                    // We have enough matches, succeed with what we have
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
                }
                // Not enough matches yet
                if (!ctx.is_partial) {
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
            }

            if (result.status == utf8_parse_result::INVALID) {
                // Malformed UTF-8 in input
                if (match_count >= p.min_count) {
                    // We have enough matches, succeed up to here
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
                }
                // Not enough matches, fail
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
            }

            // Check if this codepoint matches our character class
            bool matches = false;
            for (const auto & range : p.ranges) {
                if (range.contains(result.codepoint)) {
                    matches = true;
                    break;
                }
            }

            // If negated, invert the match result
            if (p.negated) {
                matches = !matches;
            }

            if (matches) {
                pos += result.bytes_consumed;
                ++match_count;
            } else {
                // Character doesn't match, stop matching
                break;
            }
        }

        // Check if we got enough matches
        if (match_count < p.min_count) {
            if (pos >= ctx.input.size() && ctx.is_partial) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
            }
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos, pos);
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
    }

    static common_peg_parse_result handle_escape_sequence(common_peg_parse_context & ctx, size_t start, size_t & pos) {
        ++pos; // consume '\'
        if (pos >= ctx.input.size()) {
            if (!ctx.is_partial) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start);
            }
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
        }

        switch (ctx.input[pos]) {
            case '"':
            case '\\':
            case '/':
            case 'b':
            case 'f':
            case 'n':
            case 'r':
            case 't':
                ++pos;
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start, pos);
            case 'u':
                return handle_unicode_escape(ctx, start, pos);
            default:
                // Invalid escape sequence
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start);
        }
    }

    static common_peg_parse_result handle_unicode_escape(common_peg_parse_context & ctx, size_t start, size_t & pos) {
        ++pos; // consume 'u'
        for (int i = 0; i < 4; ++i) {
            if (pos >= ctx.input.size()) {
                if (!ctx.is_partial) {
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start);
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
            }
            if (!is_hex_digit(ctx.input[pos])) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start);
            }
            ++pos;
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start, pos);
    }

    common_peg_parse_result operator()(const common_peg_json_string_parser & /* p */) {
        auto pos = start_pos;

        // Parse string content (without quotes)
        while (pos < ctx.input.size()) {
            char c = ctx.input[pos];

            if (c == '"') {
                // Found closing quote - success (don't consume it)
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
            }

            if (c == '\\') {
                auto result = handle_escape_sequence(ctx, start_pos, pos);
                if (!result.success()) {
                    return result;
                }
            } else {
                auto utf8_result = parse_utf8_codepoint(ctx.input, pos);

                if (utf8_result.status == utf8_parse_result::INCOMPLETE) {
                    if (!ctx.is_partial) {
                        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                    }
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
                }

                if (utf8_result.status == utf8_parse_result::INVALID) {
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                }

                pos += utf8_result.bytes_consumed;
            }
        }

        // Reached end without finding closing quote
        if (!ctx.is_partial) {
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos, pos);
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
    }

    common_peg_parse_result operator()(const common_peg_until_parser & p) const {
        trie matcher(p.delimiters);

        // Scan input and check for delimiters
        size_t pos = start_pos;
        size_t last_valid_pos = start_pos;

        while (pos < ctx.input.size()) {
            auto utf8_result = parse_utf8_codepoint(ctx.input, pos);

            if (utf8_result.status == utf8_parse_result::INCOMPLETE) {
                // Incomplete UTF-8 sequence
                if (!ctx.is_partial) {
                    // Input is complete but UTF-8 is incomplete = malformed
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                }
                // Return what we have so far (before incomplete sequence)
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, last_valid_pos);
            }

            if (utf8_result.status == utf8_parse_result::INVALID) {
                // Malformed UTF-8
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
            }

            // Check if a delimiter starts at this position
            auto match = matcher.check_at(ctx.input, pos);

            if (match == trie::COMPLETE_MATCH) {
                // Found a complete delimiter, return everything before it
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
            }

            if (match == trie::PARTIAL_MATCH) {
                // Found a partial match extending to end of input, return everything before it
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
            }

            pos += utf8_result.bytes_consumed;
            last_valid_pos = pos;
        }

        if (last_valid_pos == ctx.input.size() && ctx.is_partial) {
            // Reached the end of a partial stream, there might still be more input that we need to consume.
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, last_valid_pos);
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, last_valid_pos);
    }

    common_peg_parse_result operator()(const common_peg_schema_parser & p) {
        return arena.parse(p.child, ctx, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_rule_parser & p) {
        // Parse the child
        auto result = arena.parse(p.child, ctx, start_pos);

        if (!result.fail()) {
            std::string_view text;
            if (result.start < ctx.input.size()) {
                text = std::string_view(ctx.input).substr(result.start, result.end - result.start);
            }

            auto node_id = ctx.ast.add_node(
                p.name,
                "",
                result.start,
                result.end,
                text,
                std::move(result.nodes),
                result.need_more_input()
            );

            return common_peg_parse_result(result.type, result.start, result.end, { node_id });
        }

        return result;
    }

    common_peg_parse_result operator()(const common_peg_tag_parser & p) {
        // Parse the child
        auto result = arena.parse(p.child, ctx, start_pos);

        if (!result.fail()) {
            std::string_view text;
            if (result.start < ctx.input.size()) {
                text = std::string_view(ctx.input).substr(result.start, result.end - result.start);
            }

            auto node_id = ctx.ast.add_node(
                "",
                p.tag,
                result.start,
                result.end,
                text,
                std::move(result.nodes),
                result.need_more_input()
            );

            return common_peg_parse_result(result.type, result.start, result.end, { node_id });
        }

        return result;
    }

    common_peg_parse_result operator()(const common_peg_ref_parser & p) {
        auto rule_id = arena.get_rule(p.name);
        return arena.parse(rule_id, ctx, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_atomic_parser & p) {
        auto result = arena.parse(p.child, ctx, start_pos);
        if (result.need_more_input()) {
            // Clear nodes so they don't propagate up.
            result.nodes.clear();
        }
        return result;
    }
};

common_peg_parse_result common_peg_arena::parse(common_peg_parse_context & ctx, size_t start) const {
    if (root_ == COMMON_PEG_INVALID_PARSER_ID) {
        throw std::runtime_error("No root parser set");
    }
    return parse(root_, ctx, start);
}

common_peg_parse_result common_peg_arena::parse(common_peg_parser_id id, common_peg_parse_context & ctx, size_t start) const {
    // Execute parser
    const auto & parser = parsers_.at(id);
    parser_executor exec(*this, ctx, start);
    return std::visit(exec, parser);
}

common_peg_parser_id common_peg_arena::resolve_ref(common_peg_parser_id id) {
    const auto & parser = parsers_.at(id);
    if (auto ref = std::get_if<common_peg_ref_parser>(&parser)) {
        return get_rule(ref->name);
    }
    return id;
}

void common_peg_arena::resolve_refs() {
    // Walk through all parsers and replace refs with their corresponding rule IDs
    for (auto & parser : parsers_) {
        std::visit([this](auto & p) {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
                for (auto & child : p.children) {
                    child = resolve_ref(child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
                for (auto & child : p.children) {
                    child = resolve_ref(child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_repetition_parser> ||
                                 std::is_same_v<T, common_peg_and_parser> ||
                                 std::is_same_v<T, common_peg_not_parser> ||
                                 std::is_same_v<T, common_peg_tag_parser> ||
                                 std::is_same_v<T, common_peg_atomic_parser>) {
                p.child = resolve_ref(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
                p.child = resolve_ref(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
                p.child = resolve_ref(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_epsilon_parser> ||
                                 std::is_same_v<T, common_peg_start_parser> ||
                                 std::is_same_v<T, common_peg_end_parser> ||
                                 std::is_same_v<T, common_peg_ref_parser> ||
                                 std::is_same_v<T, common_peg_until_parser> ||
                                 std::is_same_v<T, common_peg_literal_parser> ||
                                 std::is_same_v<T, common_peg_json_string_parser> ||
                                 std::is_same_v<T, common_peg_chars_parser> ||
                                 std::is_same_v<T, common_peg_any_parser> ||
                                 std::is_same_v<T, common_peg_space_parser>) {
                // These rules do not have children
            } else {
                static_assert(is_always_false_v<T>);
            }
        }, parser);
    }

    // Also flatten root if it's a ref
    if (root_ != COMMON_PEG_INVALID_PARSER_ID) {
        root_ = resolve_ref(root_);
    }
}

// dump() is now in peg-parser-gbnf.cpp

common_peg_parser & common_peg_parser::operator=(const common_peg_parser & other) {
    id_ = other.id_;
    return *this;
}

common_peg_parser & common_peg_parser::operator+=(const common_peg_parser & other) {
    id_ = builder_.sequence({id_, other.id_});
    return *this;
}

common_peg_parser & common_peg_parser::operator|=(const common_peg_parser & other) {
    id_ = builder_.choice({id_, other.id_});
    return *this;
}

common_peg_parser common_peg_parser::operator+(const common_peg_parser & other) const {
    return builder_.sequence({id_, other.id_});
}

common_peg_parser common_peg_parser::operator|(const common_peg_parser & other) const {
    return builder_.choice({id_, other.id_});
}

common_peg_parser common_peg_parser::operator<<(const common_peg_parser & other) const {
    return builder_.sequence({id_, builder_.space(), other.id_});
}

common_peg_parser common_peg_parser::operator+(const char * str) const {
    return *this + builder_.literal(str);
}

common_peg_parser common_peg_parser::operator+(const std::string & str) const {
    return *this + builder_.literal(str);
}

common_peg_parser common_peg_parser::operator<<(const char * str) const {
    return *this << builder_.literal(str);
}

common_peg_parser common_peg_parser::operator<<(const std::string & str) const {
    return *this << builder_.literal(str);
}

common_peg_parser common_peg_parser::operator|(const char * str) const {
    return *this | builder_.literal(str);
}

common_peg_parser common_peg_parser::operator|(const std::string & str) const {
    return *this | builder_.literal(str);
}

common_peg_parser operator+(const char * str, const common_peg_parser & p) {
    return p.builder().literal(str) + p;
}

common_peg_parser operator+(const std::string & str, const common_peg_parser & p) {
    return operator+(str.c_str(), p);
}

common_peg_parser operator<<(const char * str, const common_peg_parser & p) {
    return p.builder().literal(str) << p;
}

common_peg_parser operator<<(const std::string & str, const common_peg_parser & p) {
    return operator<<(str.c_str(), p);
}

common_peg_parser operator|(const char * str, const common_peg_parser & p) {
    return p.builder().literal(str) | p;
}

common_peg_parser operator|(const std::string & str, const common_peg_parser & p) {
    return operator|(str.c_str(), p);
}

static std::string rule_name(const std::string & name) {
    static const std::regex invalid_rule_chars_re("[^a-zA-Z0-9-]+");
    return std::regex_replace(name, invalid_rule_chars_re, "-");
}

common_peg_parser_builder::common_peg_parser_builder() {}

common_peg_parser common_peg_parser_builder::sequence(const std::vector<common_peg_parser_id> & parsers) {
    // Flatten nested sequences
    std::vector<common_peg_parser_id> flattened;
    for (const auto & p : parsers) {
        const auto & parser = arena_.get(p);
        if (auto seq = std::get_if<common_peg_sequence_parser>(&parser)) {
            flattened.insert(flattened.end(), seq->children.begin(), seq->children.end());
        } else {
            flattened.push_back(p);
        }
    }
    return wrap(arena_.add_parser(common_peg_sequence_parser{flattened}));
}

common_peg_parser common_peg_parser_builder::sequence(const std::vector<common_peg_parser> & parsers) {
    std::vector<common_peg_parser_id> ids;
    ids.reserve(parsers.size());
    for (const auto & p : parsers) {
        ids.push_back(p.id());
    }
    return sequence(ids);
}

common_peg_parser common_peg_parser_builder::sequence(std::initializer_list<common_peg_parser> parsers) {
    std::vector<common_peg_parser_id> ids;
    ids.reserve(parsers.size());
    for (const auto & p : parsers) {
        ids.push_back(p.id());
    }
    return sequence(ids);
}

common_peg_parser common_peg_parser_builder::choice(const std::vector<common_peg_parser_id> & parsers) {
    // Flatten nested choices
    std::vector<common_peg_parser_id> flattened;
    for (const auto & p : parsers) {
        const auto & parser = arena_.get(p);
        if (auto choice = std::get_if<common_peg_choice_parser>(&parser)) {
            flattened.insert(flattened.end(), choice->children.begin(), choice->children.end());
        } else {
            flattened.push_back(p);
        }
    }
    return wrap(arena_.add_parser(common_peg_choice_parser{flattened}));
}

common_peg_parser common_peg_parser_builder::choice(const std::vector<common_peg_parser> & parsers) {
    std::vector<common_peg_parser_id> ids;
    ids.reserve(parsers.size());
    for (const auto & p : parsers) {
        ids.push_back(p.id());
    }
    return choice(ids);
}

common_peg_parser common_peg_parser_builder::choice(std::initializer_list<common_peg_parser> parsers) {
    std::vector<common_peg_parser_id> ids;
    ids.reserve(parsers.size());
    for (const auto & p : parsers) {
        ids.push_back(p.id());
    }
    return choice(ids);
}

common_peg_parser common_peg_parser_builder::chars(const std::string & classes, int min, int max) {
    auto [ranges, negated] = parse_char_classes(classes);
    return wrap(arena_.add_parser(common_peg_chars_parser{classes, ranges, negated, min, max}));
}

// schema() is now in peg-parser-gbnf.cpp

common_peg_parser common_peg_parser_builder::rule(const std::string & name, const common_peg_parser & p, bool trigger) {
    auto clean_name = rule_name(name);
    auto rule_id = arena_.add_parser(common_peg_rule_parser{clean_name, p.id(), trigger});
    arena_.add_rule(clean_name, rule_id);
    return ref(clean_name);
}

common_peg_parser common_peg_parser_builder::rule(const std::string & name, const std::function<common_peg_parser()> & builder_fn, bool trigger) {
    auto clean_name = rule_name(name);
    if (arena_.has_rule(clean_name)) {
        return ref(clean_name);
    }

    // Create placeholder rule to allow recursive references
    auto placeholder = any();  // Temporary placeholder
    auto placeholder_rule_id = arena_.add_parser(common_peg_rule_parser{clean_name, placeholder.id(), trigger});
    arena_.add_rule(clean_name, placeholder_rule_id);

    // Build the actual parser
    auto parser = builder_fn();

    // Replace placeholder with actual rule
    auto rule_id = arena_.add_parser(common_peg_rule_parser{clean_name, parser.id(), trigger});
    arena_.rules_[clean_name] = rule_id;

    return ref(clean_name);
}

void common_peg_parser_builder::set_root(const common_peg_parser & p) {
    arena_.set_root(p.id());
}

common_peg_arena common_peg_parser_builder::build() {
    arena_.resolve_refs();
    return std::move(arena_);
}

// JSON parsers
common_peg_parser common_peg_parser_builder::json_number() {
   return rule("json-number", [this]() {
        auto digit1_9 = chars("[1-9]", 1, 1);
        auto digits = chars("[0-9]");
        auto int_part = choice({literal("0"), sequence({digit1_9, chars("[0-9]", 0, -1)})});
        auto frac = sequence({literal("."), digits});
        auto exp = sequence({choice({literal("e"), literal("E")}), optional(chars("[+-]", 1, 1)), digits});
        return sequence({optional(literal("-")), int_part, optional(frac), optional(exp), space()});
    });
}

common_peg_parser common_peg_parser_builder::json_string() {
    return rule("json-string", [this]() {
        return sequence({literal("\""), json_string_content(), literal("\""), space()});
    });
}

common_peg_parser common_peg_parser_builder::json_bool() {
    return rule("json-bool", [this]() {
        return sequence({choice({literal("true"), literal("false")}), space()});
    });
}

common_peg_parser common_peg_parser_builder::json_null() {
    return rule("json-null", [this]() {
        return sequence({literal("null"), space()});
    });
}

common_peg_parser common_peg_parser_builder::json_object() {
    return rule("json-object", [this]() {
        auto ws = space();
        auto member = sequence({json_string(), ws, literal(":"), ws, json()});
        auto members = sequence({member, zero_or_more(sequence({ws, literal(","), ws, member}))});
        return sequence({
            literal("{"),
            ws,
            choice({
                literal("}"),
                sequence({members, ws, literal("}")})
            }),
            ws
        });
    });
}

common_peg_parser common_peg_parser_builder::json_array() {
    return rule("json-array", [this]() {
        auto ws = space();
        auto elements = sequence({json(), zero_or_more(sequence({literal(","), ws, json()}))});
        return sequence({
            literal("["),
            ws,
            choice({
                literal("]"),
                sequence({elements, ws, literal("]")})
            }),
            ws
        });
    });
}

common_peg_parser common_peg_parser_builder::json() {
    return rule("json-value", [this]() {
        return choice({
            json_object(),
            json_array(),
            json_string(),
            json_number(),
            json_bool(),
            json_null()
        });
    });
}

common_peg_parser common_peg_parser_builder::json_string_content() {
    return wrap(arena_.add_parser(common_peg_json_string_parser{}));
}

common_peg_parser common_peg_parser_builder::json_member(const std::string & key, const common_peg_parser & p) {
    auto ws = space();
    return sequence({
        literal("\"" + key + "\""),
        ws,
        literal(":"),
        ws,
        p,
    });
}


// GBNF generation, serialization, dump(), and schema() are now in peg-parser-gbnf.cpp

common_peg_arena build_peg_parser(const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn) {
    common_peg_parser_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}
