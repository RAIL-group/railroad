#pragma once
#include <string>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <sstream>
#include <functional>
#include <unordered_set>
#include <algorithm>

namespace mrppddl {

class Fluent {
public:
    Fluent(std::string name, std::vector<std::string> args = {}, bool negated = false)
        : name_(std::move(name)), args_(std::move(args)), negated_(negated)
    {
        if (!args_.empty()) {
            if (name_ == "not") {
                throw std::invalid_argument("Use the 'negated' argument or ~Fluent to negate.");
            }
        } else {
            // Parse from a flat string
            std::istringstream iss(name_);
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token) tokens.push_back(token);

            if (tokens.empty()) {
                throw std::invalid_argument("Empty Fluent string.");
            }

            if (tokens[0] == "not") {
                negated_ = true;
                tokens.erase(tokens.begin());
            } else {
                negated_ = false;
            }

            if (tokens.empty()) {
                throw std::invalid_argument("Missing Fluent name after 'not'.");
            }

            name_ = tokens[0];
            args_.assign(tokens.begin() + 1, tokens.end());
        }

        cached_hash_ = compute_hash();
    }

   Fluent invert() const {
    Fluent flipped = *this;
    flipped.negated_ = !negated_;
    flipped.cached_hash_ = flipped.compute_hash();
    return flipped;
  }


    std::string name() const { return name_; }
    const std::vector<std::string>& args() const { return args_; }
    bool is_negated() const { return negated_; }

    bool operator==(const Fluent& other) const {
        return cached_hash_ == other.cached_hash_;
    }

    std::size_t hash() const {
        return cached_hash_;
    }

private:
    std::string name_;
    std::vector<std::string> args_;
    bool negated_;
    std::size_t cached_hash_;

  std::size_t compute_hash() const {
    std::size_t h = std::hash<std::string>{}(name_);
    for (const auto& arg : args_) {
      h ^= std::hash<std::string>{}(arg);
    }
    if (negated_) {
      h = ~h;  // flip all bits if negated
    }
    return h;
  }
};

}


namespace std {
template <>
struct hash<mrppddl::Fluent> {
    std::size_t operator()(const mrppddl::Fluent& f) const noexcept {
        return f.hash();
    }
};
}


namespace mrppddl {

class ActiveFluents {
public:
    ActiveFluents(std::unordered_set<Fluent> fluents = {})
        : fluents_(std::move(fluents)) {
        cached_hash_ = compute_hash();
    }

    ActiveFluents update(const std::unordered_set<Fluent>& fluents, bool relax = false) const {
        std::unordered_set<Fluent> updated = fluents_;

        // Add positive fluents from the input
        for (const auto& f : fluents) {
            if (!f.is_negated()) {
                updated.insert(f);
            }
        }

        // Remove inverted negated fluents if not in relax mode
        if (!relax) {
            for (const auto& f : fluents) {
                if (f.is_negated()) {
                    updated.erase(f.invert());
                }
            }
        }

        return ActiveFluents(std::move(updated));
    }

    bool contains(const Fluent& f) const {
        return fluents_.find(f) != fluents_.end();
    }

    std::unordered_set<Fluent>::const_iterator begin() const { return fluents_.begin(); }
    std::unordered_set<Fluent>::const_iterator end() const { return fluents_.end(); }

    bool operator==(const ActiveFluents& other) const {
        return cached_hash_ == other.cached_hash_;
    }

    std::size_t hash() const { return cached_hash_; }

    std::string repr() const {
        std::vector<std::string> strs;
        for (const auto& f : fluents_) {
            std::ostringstream oss;
            if (f.is_negated()) oss << "not ";
            oss << f.name();
            for (const auto& arg : f.args()) {
                oss << " " << arg;
            }
            strs.push_back(oss.str());
        }
        std::sort(strs.begin(), strs.end());
        std::ostringstream out;
        out << "{";
        for (size_t i = 0; i < strs.size(); ++i) {
            out << strs[i];
            if (i + 1 < strs.size()) out << ", ";
        }
        out << "}";
        return out.str();
    }

    const std::unordered_set<Fluent>& fluents() const { return fluents_; }

private:
    std::unordered_set<Fluent> fluents_;
    std::size_t cached_hash_;

    std::size_t compute_hash() const {
        std::size_t h = 0;
        for (const auto& f : fluents_) {
            h ^= f.hash();  // fast XOR-based hash combiner
        }
        return h;
    }
};

}  // namespace mrppddl
