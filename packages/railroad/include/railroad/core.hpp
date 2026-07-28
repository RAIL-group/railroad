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
#include <optional>

namespace railroad {

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
        negated_ = negated;
      }

      if (tokens.empty()) {
        throw std::invalid_argument("Missing Fluent name after 'not'.");
      }

      name_ = tokens[0];
      args_.assign(tokens.begin() + 1, tokens.end());
    }

    is_free_ = (name_ == "free");
    is_waiting_ = (name_ == "waiting");
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
  bool is_waiting_;

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

} // namespace railroad

namespace std {
template <> struct hash<railroad::Fluent> {
  std::size_t operator()(const railroad::Fluent &f) const noexcept {
    return f.hash();
  }
};
} // namespace std

namespace railroad {

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

// A conditional effect branch (PDDL `when`): sub-effects applied only if the
// conditions hold in the state at the moment the parent effect fires.
// The state-selected analogue of ProbBranchWrapper. Negated conditions use
// negation-as-absence, mirroring negative preconditions.
class CondBranchWrapper {
public:
  CondBranchWrapper(std::unordered_set<Fluent> conditions,
                    std::vector<std::shared_ptr<const GroundedEffect>> effects)
      : conditions_(std::move(conditions)), effects_(std::move(effects)) {
    for (const auto &f : conditions_) {
      if (f.is_negated()) {
        neg_conditions_.insert(f.invert());
      } else {
        pos_conditions_.insert(f);
      }
    }
  }

  const std::unordered_set<Fluent> &conditions() const { return conditions_; }
  const std::vector<std::shared_ptr<const GroundedEffect>> &effects() const {
    return effects_;
  }

  bool holds(const std::unordered_set<Fluent> &fluents) const {
    for (const auto &f : pos_conditions_) {
      if (!fluents.count(f)) {
        return false;
      }
    }
    for (const auto &f : neg_conditions_) {
      if (fluents.count(f)) {
        return false;
      }
    }
    return true;
  }

  std::size_t hash() const;

private:
  std::unordered_set<Fluent> conditions_;
  std::unordered_set<Fluent> pos_conditions_;
  std::unordered_set<Fluent> neg_conditions_;
  std::vector<std::shared_ptr<const GroundedEffect>> effects_;
  mutable std::optional<std::size_t> cached_hash_;
};

class GroundedEffect {
public:
  GroundedEffect(
      double time, std::unordered_set<Fluent> resulting_fluents,
      std::vector<
          std::pair<double, std::vector<std::shared_ptr<const GroundedEffect>>>>
          prob_pairs,
      std::vector<std::pair<std::unordered_set<Fluent>,
                            std::vector<std::shared_ptr<const GroundedEffect>>>>
          cond_pairs = {})
      : time_(time), resulting_fluents_(std::move(resulting_fluents)),
        cached_hash_(std::nullopt) {
    for (auto &[p, effects] : prob_pairs) {
      prob_effects_.emplace_back(p, std::move(effects));
    }
    for (auto &[conditions, effects] : cond_pairs) {
      cond_effects_.emplace_back(std::move(conditions), std::move(effects));
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
  const std::vector<CondBranchWrapper> &cond_effects() const {
    return cond_effects_;
  }

  bool is_probabilistic() const { return !prob_effects_.empty(); }
  bool is_conditional() const { return !cond_effects_.empty(); }

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

    // Hash conditional branches
    std::size_t h_cond_branches = 0;
    for (const auto &branch : cond_effects_) {
      std::size_t h = branch.hash();
      hash_combine(h, 1);
      h_cond_branches ^= h;
    }

    // Final combination: time, fluents, branches (ordered)
    hash_combine(h_time, h_fluents);
    hash_combine(h_time, h_branches);
    hash_combine(h_time, h_cond_branches);
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
    for (const auto &branch : cond_effects_) {
      out << " when {";
      bool first = true;
      for (const auto &f : branch.conditions()) {
        if (!first)
          out << ", ";
        out << (f.is_negated() ? "not " : "") << f.name();
        for (const auto &arg : f.args())
          out << " " << arg;
        first = false;
      }
      out << "}: [";
      for (size_t i = 0; i < branch.effects().size(); ++i) {
        out << branch.effects()[i]->str();
        if (i + 1 < branch.effects().size())
          out << "; ";
      }
      out << "]";
    }
    return out.str();
  }

private:
  double time_;
  std::unordered_set<Fluent> resulting_fluents_;
  std::unordered_set<Fluent> pos_fluents_;
  std::unordered_set<Fluent> flipped_neg_fluents_;
  std::vector<ProbBranchWrapper> prob_effects_;
  std::vector<CondBranchWrapper> cond_effects_;
  mutable std::optional<std::size_t> cached_hash_;
};

// The single implementation of effect firing, shared by the planner's
// transition (advance_to_terminal) and environment execution. Conditional
// branches read `fluents` as they are when the effect fires, BEFORE the
// effect's own fluents apply, so collect_triggered_effects must be called
// before apply_effect_fluents.
inline std::vector<std::shared_ptr<const GroundedEffect>>
collect_triggered_effects(const GroundedEffect &effect,
                          const std::unordered_set<Fluent> &fluents,
                          bool relax = false) {
  std::vector<std::shared_ptr<const GroundedEffect>> triggered;
  if (effect.is_conditional()) {
    for (const auto &branch : effect.cond_effects()) {
      if (relax || branch.holds(fluents)) {
        triggered.insert(triggered.end(), branch.effects().begin(),
                         branch.effects().end());
      }
    }
  }
  return triggered;
}

// Deletes are applied before adds (PDDL semantics), so an effect that both
// deletes and adds the same fluent leaves it present. Returns whether the
// effect freed a robot.
inline bool apply_effect_fluents(std::unordered_set<Fluent> &fluents,
                                 const GroundedEffect &effect,
                                 bool relax = false) {
  if (!relax) {
    for (const auto &f : effect.flipped_neg_fluents()) {
      fluents.erase(f);
    }
  }
  bool freed_robot = false;
  for (const auto &f : effect.pos_fluents()) {
    if (f.is_free()) {
      freed_robot = true;
    }
    fluents.insert(f);
  }
  return freed_robot;
}

inline bool operator==(const ProbBranchWrapper &a, const ProbBranchWrapper &b) {
  return a.prob() == b.prob() && a.effects() == b.effects();
}

inline bool operator==(const CondBranchWrapper &a, const CondBranchWrapper &b) {
  return a.conditions() == b.conditions() && a.effects() == b.effects();
}

} // namespace railroad

namespace std {
template <> struct hash<railroad::GroundedEffect> {
  std::size_t operator()(const railroad::GroundedEffect &eff) const noexcept {
    return eff.hash();
  }
};
} // namespace std

namespace railroad {

// Forward declaration for lazy computation of relaxed successors
class State;

class Action {
public:
  Action(std::unordered_set<Fluent> preconditions,
         std::vector<std::shared_ptr<const GroundedEffect>> effects,
         std::string name = "anonymous",
         double extra_cost = 0.0)
      : preconditions_(std::move(preconditions)), effects_(std::move(effects)),
        name_(std::move(name)), extra_cost_(extra_cost) {
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
  double extra_cost() const { return extra_cost_; }
  const std::unordered_set<Fluent> &pos_preconditions() const {
    return pos_precond_;
  }
  const std::unordered_set<Fluent> &neg_precond_flipped() const {
    return neg_precond_flipped_;
  }

  // Get precomputed relaxed successors (computed lazily on first call)
  const std::vector<std::pair<State, double>>& get_relaxed_successors() const;

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
      out << "    after " << eff->time() << " " << eff->str() << "\n";
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
  double extra_cost_;
  std::unordered_set<Fluent> pos_precond_;
  std::unordered_set<Fluent> neg_precond_flipped_;

  // Cache for lazily computed relaxed successors
  mutable std::optional<std::vector<std::pair<State, double>>> relaxed_successors_cache_;
};

} // namespace railroad

namespace std {
template <> struct hash<railroad::Action> {
  std::size_t operator()(const railroad::Action &action) const noexcept {
    return action.hash();
  }
};
} // namespace std

namespace railroad {

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

std::size_t CondBranchWrapper::hash() const {
  if (cached_hash_)
    return *cached_hash_;

  std::size_t h_branch = 0;
  for (const auto &inner_eff : effects_) {
    std::size_t h = inner_eff->hash();
    hash_combine(h, 0);
    h_branch ^= h;
  }
  std::size_t h_conditions = 0;
  for (const auto &f : conditions_) {
    std::size_t h = f.hash();
    hash_combine(h, 0);
    h_conditions ^= h;
  }
  hash_combine(h_branch, h_conditions);
  cached_hash_ = h_branch;
  return h_branch;
}

} // namespace railroad
