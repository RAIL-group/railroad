#pragma once
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory> // for std::shared_ptr
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace mrppddl {

inline void hash_combine(std::size_t &seed, std::size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

class Fluent {
public:
  Fluent(std::string name, std::vector<std::string> args = {},
         bool negated = false)
      : name_(std::move(name)), args_(std::move(args)), negated_(negated) {
    if (!args_.empty()) {
      if (name_ == "not") {
        throw std::invalid_argument(
            "Use the 'negated' argument or ~Fluent to negate.");
      }
    } else {
      // Parse from a flat string
      std::istringstream iss(name_);
      std::vector<std::string> tokens;
      std::string token;
      while (iss >> token)
        tokens.push_back(token);

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

    is_free_ = (name_ == "free");
    is_free_ = (name_ == "waiting");
    cached_hash_ = compute_hash();
  }

  const Fluent invert() const {
    Fluent flipped = *this;
    flipped.negated_ = !negated_;
    flipped.cached_hash_ = ~cached_hash_;
    return flipped;
  }

  std::string name() const { return name_; }
  bool is_free() const { return is_free_; }
  bool is_waiting() const { return is_waiting_; }
  const std::vector<std::string> &args() const { return args_; }
  bool is_negated() const { return negated_; }

  bool operator==(const Fluent &other) const { return hash() == other.hash(); }

  std::size_t hash() const { return cached_hash_; }

private:
  std::string name_;
  std::vector<std::string> args_;
  bool negated_;
  std::size_t cached_hash_;
  bool is_free_;

  std::size_t compute_hash() const {
    std::size_t h = std::hash<std::string>{}(name_);
    for (const auto &arg : args_) {
      hash_combine(h, std::hash<std::string>{}(arg));
    }
    if (negated_) {
      h = ~h; // flip all bits if negated
    }
    return h;
  }
};

} // namespace mrppddl

namespace std {
template <> struct hash<mrppddl::Fluent> {
  std::size_t operator()(const mrppddl::Fluent &f) const noexcept {
    return f.hash();
  }
};
} // namespace std

namespace mrppddl {

class GroundedEffect; // Forward Declaration

class ProbBranchWrapper {
public:
  ProbBranchWrapper(double prob,
                    std::vector<std::shared_ptr<const GroundedEffect>> effects)
      : prob_(prob), effects_(std::move(effects)) {}

  double prob() const { return prob_; }
  const std::vector<std::shared_ptr<const GroundedEffect>> &effects() const {
    return effects_;
  }

  std::size_t hash() const;

private:
  double prob_;
  std::vector<std::shared_ptr<const GroundedEffect>> effects_;
  mutable std::optional<std::size_t> cached_hash_;
};

class GroundedEffect {
public:
  GroundedEffect(
      double time, std::unordered_set<Fluent> resulting_fluents,
      std::vector<
          std::pair<double, std::vector<std::shared_ptr<const GroundedEffect>>>>
          prob_pairs)
      : time_(time), resulting_fluents_(std::move(resulting_fluents)),
        cached_hash_(std::nullopt) {
    for (auto &[p, effects] : prob_pairs) {
      prob_effects_.emplace_back(p, std::move(effects));
    }
    for (const auto &f : resulting_fluents_) {
      if (f.is_negated()) {
        flipped_neg_fluents_.insert(f.invert());
      } else {
        pos_fluents_.insert(f);
      }
    }

    hash();
  }
  GroundedEffect(double time, std::unordered_set<Fluent> resulting_fluents)
      : time_(time), resulting_fluents_(std::move(resulting_fluents)),
        prob_effects_{} {}

  double time() const { return time_; }
  const std::unordered_set<Fluent> &resulting_fluents() const {
    return resulting_fluents_;
  }
  const std::unordered_set<Fluent> &pos_fluents() const { return pos_fluents_; }
  const std::unordered_set<Fluent> &flipped_neg_fluents() const {
    return flipped_neg_fluents_;
  }
  const std::vector<ProbBranchWrapper> &prob_effects() const {
    return prob_effects_;
  }

  bool is_probabilistic() const { return !prob_effects_.empty(); }

  bool operator<(const GroundedEffect &other) const {
    return time_ < other.time_;
  }

  bool operator==(const GroundedEffect &other) const {
    return hash() == other.hash();
  }

  std::size_t hash() const {
    if (cached_hash_) {
      return *cached_hash_;
    }

    std::size_t h_time = std::hash<double>{}(time_);

    // Hash fluents
    std::size_t h_fluents = 0;
    for (const auto &f : resulting_fluents_) {
      std::size_t h = f.hash();
      hash_combine(h, 0);
      h_fluents ^= h;
    }

    // Hash branches
    std::size_t h_branches = 0;
    for (const auto &branch : prob_effects_) {
      std::size_t h = branch.hash();
      hash_combine(h, 0);
      h_branches ^= h;
    }

    // Final combination: time, fluents, branches (ordered)
    hash_combine(h_time, h_fluents);
    hash_combine(h_time, h_branches);
    cached_hash_ = h_time;

    return h_time;
  }

  std::string str() const {
    std::ostringstream out;
    if (is_probabilistic()) {
      out << "probabilistic after " << time_ << ": { ";
      for (const auto &branch : prob_effects_) {
        out << branch.prob() << ": [";
        for (size_t i = 0; i < branch.effects().size(); ++i) {
          out << branch.effects()[i]->str();
          if (i + 1 < branch.effects().size())
            out << "; ";
        }
        out << "], ";
      }
      out << "}";
    } else {
      out << "after " << time_ << ": ";
      bool first = true;
      for (const auto &f : resulting_fluents_) {
        if (!first)
          out << ", ";
        out << (f.is_negated() ? "not " : "") << f.name();
        for (const auto &arg : f.args())
          out << " " << arg;
        first = false;
      }
    }
    return out.str();
  }

private:
  double time_;
  std::unordered_set<Fluent> resulting_fluents_;
  std::unordered_set<Fluent> pos_fluents_;
  std::unordered_set<Fluent> flipped_neg_fluents_;
  std::vector<ProbBranchWrapper> prob_effects_;
  mutable std::optional<std::size_t> cached_hash_;
};

inline bool operator==(const ProbBranchWrapper &a, const ProbBranchWrapper &b) {
  return a.prob() == b.prob() && a.effects() == b.effects();
}

} // namespace mrppddl

namespace std {
template <> struct hash<mrppddl::GroundedEffect> {
  std::size_t operator()(const mrppddl::GroundedEffect &eff) const noexcept {
    return eff.hash();
  }
};
} // namespace std

namespace mrppddl {

class Action {
public:
  Action(std::unordered_set<Fluent> preconditions,
         std::vector<std::shared_ptr<const GroundedEffect>> effects,
         std::string name = "anonymous")
      : preconditions_(std::move(preconditions)), effects_(std::move(effects)),
        name_(std::move(name)) {
    for (const auto &f : preconditions_) {
      if (f.is_negated()) {
        neg_precond_flipped_.insert(f.invert());
      } else {
        pos_precond_.insert(f);
      }
    }
  }

  // Keep defaults
  Action() = default;
  Action(const Action &) = default;
  Action &operator=(const Action &) = default;

  const std::unordered_set<Fluent> &preconditions() const {
    return preconditions_;
  }
  const std::vector<std::shared_ptr<const GroundedEffect>> &effects() const {
    return effects_;
  }
  const std::string &name() const { return name_; }
  const std::unordered_set<Fluent> &pos_preconditions() const {
    return pos_precond_;
  }
  const std::unordered_set<Fluent> &neg_precond_flipped() const {
    return neg_precond_flipped_;
  }

  std::string str() const {
    std::ostringstream out;
    out << "Action('" << name_ << "'\n  Preconditions: [";

    bool first = true;
    for (const auto &p : preconditions_) {
      if (!first)
        out << ", ";
      std::ostringstream p_str;
      if (p.is_negated())
        p_str << "not ";
      p_str << p.name();
      for (const auto &arg : p.args()) {
        p_str << " " << arg;
      }
      out << p_str.str();
      first = false;
    }

    out << "]\n  Effects:\n";
    for (const auto &eff : effects_) {
      out << "    after " << eff->time() << "...\n";
    }
    out << ")";
    return out.str();
  }

  bool operator==(const Action &other) const { return hash() == other.hash(); }

  std::size_t hash() const {
    std::size_t h_name = std::hash<std::string>{}(name_);

    // Hash preconditions
    std::size_t h_preconds = 0;
    for (const auto &f : preconditions_) {
      std::size_t h = f.hash();
      hash_combine(h, 0);
      h_preconds ^= h;
    }

    // Hash effects
    std::size_t h_effects = 0;
    for (const auto &eff : effects_) {
      std::size_t h = eff->hash();
      hash_combine(h, 0);
      h_effects ^= h;
    }

    // Combine components
    std::size_t h = h_name;
    hash_combine(h, h_preconds);
    hash_combine(h, h_effects);

    return h;
  }

private:
  std::unordered_set<Fluent> preconditions_;
  std::vector<std::shared_ptr<const GroundedEffect>> effects_;
  std::string name_;
  std::unordered_set<Fluent> pos_precond_;
  std::unordered_set<Fluent> neg_precond_flipped_;
};

} // namespace mrppddl

namespace std {
template <> struct hash<mrppddl::Action> {
  std::size_t operator()(const mrppddl::Action &action) const noexcept {
    return action.hash();
  }
};
} // namespace std

namespace mrppddl {

std::size_t ProbBranchWrapper::hash() const {
  if (cached_hash_)
    return *cached_hash_;

  std::size_t h_branch = 0;
  for (const auto &inner_eff : effects_) {
    std::size_t h = inner_eff->hash();
    hash_combine(h, 0);
    h_branch ^= h;
  }
  hash_combine(h_branch, std::hash<double>{}(prob_));
  cached_hash_ = h_branch;
  return h_branch;
}

} // namespace mrppddl
