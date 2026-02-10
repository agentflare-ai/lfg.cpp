#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

// Trie for matching multiple literals.
// This is used in common_peg_until_parser and to build a GBNF exclusion grammar
struct trie {
    struct node {
        size_t depth = 0;
        std::map<unsigned char, size_t> children;
        bool is_word;
    };

    std::vector<node> nodes;

    trie(const std::vector<std::string> & words) {
      create_node(); // root node
      for (const auto & w : words) {
          insert(w);
      }
    }

    enum match_result { NO_MATCH, PARTIAL_MATCH, COMPLETE_MATCH };

    // Check if a delimiter starts at the given position
    match_result check_at(std::string_view sv, size_t start_pos) const {
        size_t current = 0; // Start at root
        size_t pos = start_pos;

        while (pos < sv.size()) {
            auto it = nodes[current].children.find(sv[pos]);
            if (it == nodes[current].children.end()) {
                // Can't continue matching
                return match_result{match_result::NO_MATCH};
            }

            current = it->second;
            pos++;

            // Check if we've matched a complete word
            if (nodes[current].is_word) {
                return match_result{match_result::COMPLETE_MATCH};
            }
        }

        // Reached end of input while still in the trie (not at root)
        if (current != 0) {
            // We're in the middle of a potential match
            return match_result{match_result::PARTIAL_MATCH};
        }

        // Reached end at root (no match)
        return match_result{match_result::NO_MATCH};
    }

    struct prefix_and_next {
        std::string prefix;
        std::string next_chars;
    };

    std::vector<prefix_and_next> collect_prefix_and_next() {
        std::string prefix;
        std::vector<prefix_and_next> result;
        collect_prefix_and_next(0, prefix, result);
        return result;
    }

  private:
    void collect_prefix_and_next(size_t index, std::string & prefix, std::vector<prefix_and_next> & out) {
        if (!nodes[index].is_word) {
            if (!nodes[index].children.empty()) {
                std::string chars;
                chars.reserve(nodes[index].children.size());
                for (const auto & p : nodes[index].children) {
                    chars.push_back(p.first);
                }
                out.emplace_back(prefix_and_next{prefix, chars});
            }
        }

        for (const auto & p : nodes[index].children) {
            unsigned char ch = p.first;
            auto child = p.second;
            prefix.push_back(ch);
            collect_prefix_and_next(child, prefix, out);
            prefix.pop_back();
        }
    }

    size_t create_node() {
        size_t index = nodes.size();
        nodes.emplace_back();
        return index;
    }

    void insert(const std::string & word) {
        size_t current = 0;
        for (unsigned char ch : word) {
            auto it = nodes[current].children.find(ch);
            if (it == nodes[current].children.end()) {
                size_t child = create_node();
                nodes[child].depth = nodes[current].depth + 1;
                nodes[current].children[ch] = child;
                current = child;
            } else {
                current = it->second;
            }
        }
        nodes[current].is_word = true;
    }
};
