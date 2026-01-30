#pragma once

#include "railroad/core.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace railroad {

enum class GoalType { LITERAL, AND, OR, TRUE_GOAL, FALSE_GOAL };

class GoalBase;
using GoalPtr = std::shared_ptr<GoalBase>;
using FluentSet = std::unordered_set<Fluent>;

namespace detail {

// Lexicographic compare for fingerprints (vector<size_t>).
inline bool lex_less(const std::vector<std::size_t>& a,
                     const std::vector<std::size_t>& b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

// Deterministic ordering comparator:
//  1) primary: hash()
//  2) secondary: fingerprint() (only computed on collision)
inline bool goal_less(const GoalPtr& a, const GoalPtr& b);

// Canonicalize a child list:
// - sort deterministically
// - unique structurally (no hash-only dedup)
// Note: requires GoalBase::operator== to be structural.
inline void canonicalize_children(std::vector<GoalPtr>& children);

// Complement check helper for literal fluents.
// If Fluent is "~P", Fluent::invert() should be "P" and vice versa.
inline bool contains_complement(const std::vector<GoalPtr>& children);

}  // namespace detail

class GoalBase {
public:
  virtual ~GoalBase() = default;

  // Check if goal is satisfied by given state fluents.
  virtual bool evaluate(const FluentSet& fluents) const = 0;

  // Node type.
  virtual GoalType get_type() const = 0;

  // Return normalized (canonical) form.
  virtual GoalPtr normalize() const = 0;

  // Leaf literal set (syntactic union of all literals appearing anywhere under this goal).
  virtual FluentSet get_all_literals() const = 0;

  // Children (for AND/OR goals).
  virtual const std::vector<GoalPtr>& children() const {
    static const std::vector<GoalPtr> empty;
    return empty;
  }

  // Progress metric: count of satisfied leaf literals, structure-aware.
  // - Literal: 1 if satisfied else 0 (handles negation correctly)
  // - AND: sum
  // - OR: max (best branch progress)
  // - TRUE/FALSE: 0
  int goal_count(const FluentSet& fluents) const { return satisfied_leaf_count(fluents); }

  // Get DNF (Disjunctive Normal Form) branches.
  // Returns a vector of fluent sets where:
  // - Each FluentSet is a pure conjunction (all fluents must hold)
  // - The vector represents OR of these conjunctions
  // - Empty vector means unsatisfiable (FalseGoal)
  // - Vector with empty set means trivially satisfied (TrueGoal)
  // Result is cached for efficiency.
  const std::vector<FluentSet>& get_dnf_branches() const {
    if (!cached_dnf_branches_) {
      cached_dnf_branches_ = compute_dnf_branches();
    }
    return *cached_dnf_branches_;
  }

  // Structural equality (never hash-only).
  virtual bool equals(const GoalBase& other) const = 0;

  bool operator==(const GoalBase& other) const {
    return get_type() == other.get_type() && equals(other);
  }

  // Deterministic hash with caching. Hash is not used for equality.
  std::size_t hash() const {
    if (!cached_hash_) {
      cached_hash_ = compute_hash();
    }
    return *cached_hash_;
  }

  // A deterministic fingerprint used only as a tie-breaker for sorting
  // when hashes collide. Should encode structure, not pointer addresses.
  virtual void fingerprint(std::vector<std::size_t>& out) const = 0;

protected:
  virtual std::size_t compute_hash() const = 0;
  virtual int satisfied_leaf_count(const FluentSet& fluents) const = 0;
  virtual std::vector<FluentSet> compute_dnf_branches() const = 0;

  // Cache is safe because goals are immutable after construction.
  mutable std::optional<std::size_t> cached_hash_;
  mutable std::optional<std::vector<FluentSet>> cached_dnf_branches_;
};

// Forward declarations for factories used by normalize().
inline GoalPtr make_true_goal();
inline GoalPtr make_false_goal();

// TrueGoal: always satisfied
class TrueGoal final : public GoalBase {
public:
  bool evaluate(const FluentSet&) const override { return true; }
  GoalType get_type() const override { return GoalType::TRUE_GOAL; }
  GoalPtr normalize() const override { return make_true_goal(); }
  FluentSet get_all_literals() const override { return {}; }

  bool equals(const GoalBase& other) const override {
    return other.get_type() == GoalType::TRUE_GOAL;
  }

  void fingerprint(std::vector<std::size_t>& out) const override {
    out.push_back(static_cast<std::size_t>(GoalType::TRUE_GOAL));
  }

protected:
  std::size_t compute_hash() const override {
    // Deterministic tag.
    return static_cast<std::size_t>(0xB9F3'9A47'1D2C'3E11ULL);
  }

  int satisfied_leaf_count(const FluentSet&) const override { return 0; }

  std::vector<FluentSet> compute_dnf_branches() const override {
    // One branch with no requirements (trivially satisfied)
    return {{}};
  }
};

// FalseGoal: never satisfied
class FalseGoal final : public GoalBase {
public:
  bool evaluate(const FluentSet&) const override { return false; }
  GoalType get_type() const override { return GoalType::FALSE_GOAL; }
  GoalPtr normalize() const override { return make_false_goal(); }
  FluentSet get_all_literals() const override { return {}; }

  bool equals(const GoalBase& other) const override {
    return other.get_type() == GoalType::FALSE_GOAL;
  }

  void fingerprint(std::vector<std::size_t>& out) const override {
    out.push_back(static_cast<std::size_t>(GoalType::FALSE_GOAL));
  }

protected:
  std::size_t compute_hash() const override {
    return static_cast<std::size_t>(0x4C17'5D80'AA3B'91F7ULL);
  }

  int satisfied_leaf_count(const FluentSet&) const override { return 0; }

  std::vector<FluentSet> compute_dnf_branches() const override {
    // No valid branches (unsatisfiable)
    return {};
  }
};

// LiteralGoal: leaf fluent that must hold (or must not hold if negated)
class LiteralGoal final : public GoalBase {
public:
  explicit LiteralGoal(Fluent fluent) : fluent_(std::move(fluent)) {}

  bool evaluate(const FluentSet& fluents) const override {
    if (fluent_.is_negated()) {
      // For ~P, require P not in state.
      return fluents.count(fluent_.invert()) == 0;
    }
    // For P, require P in state.
    return fluents.count(fluent_) > 0;
  }

  GoalType get_type() const override { return GoalType::LITERAL; }

  GoalPtr normalize() const override {
    // Already normalized; return a new node (no shared_from_this dependency).
    return std::make_shared<LiteralGoal>(fluent_);
  }

  FluentSet get_all_literals() const override { return {fluent_}; }

  const Fluent& fluent() const { return fluent_; }

  bool equals(const GoalBase& other) const override {
    if (other.get_type() != GoalType::LITERAL) return false;
    const auto& o = static_cast<const LiteralGoal&>(other);
    return fluent_ == o.fluent_;
  }

  void fingerprint(std::vector<std::size_t>& out) const override {
    out.push_back(static_cast<std::size_t>(GoalType::LITERAL));
    out.push_back(fluent_.hash());
  }

protected:
  std::size_t compute_hash() const override {
    std::size_t h = static_cast<std::size_t>(0xA2C1'9E7D'0134'5B6FULL);
    hash_combine(h, static_cast<std::size_t>(GoalType::LITERAL));
    hash_combine(h, fluent_.hash());
    return h;
  }

  int satisfied_leaf_count(const FluentSet& fluents) const override {
    return evaluate(fluents) ? 1 : 0;
  }

  std::vector<FluentSet> compute_dnf_branches() const override {
    // One branch with this single literal
    return {{fluent_}};
  }

private:
  Fluent fluent_;
};

class AndGoal;
class OrGoal;

// AndGoal: conjunction
class AndGoal final : public GoalBase {
public:
  explicit AndGoal(std::vector<GoalPtr> children) : children_(std::move(children)) {}

  bool evaluate(const FluentSet& fluents) const override {
    for (const auto& child : children_) {
      if (!child->evaluate(fluents)) return false;
    }
    return true;
  }

  GoalType get_type() const override { return GoalType::AND; }

  GoalPtr normalize() const override;

  FluentSet get_all_literals() const override {
    FluentSet result;
    for (const auto& child : children_) {
      auto s = child->get_all_literals();
      result.insert(s.begin(), s.end());
    }
    return result;
  }

  const std::vector<GoalPtr>& children() const override { return children_; }

  bool equals(const GoalBase& other) const override {
    if (other.get_type() != GoalType::AND) return false;
    const auto& o = static_cast<const AndGoal&>(other);

    // Equality is defined over canonical ordering; normalize() enforces this.
    if (children_.size() != o.children_.size()) return false;
    for (std::size_t i = 0; i < children_.size(); ++i) {
      if (!(*children_[i] == *o.children_[i])) return false;
    }
    return true;
  }

  void fingerprint(std::vector<std::size_t>& out) const override {
    out.push_back(static_cast<std::size_t>(GoalType::AND));
    out.push_back(children_.size());
    for (const auto& c : children_) c->fingerprint(out);
  }

protected:
  std::size_t compute_hash() const override {
    // Order-sensitive because children are canonicalized in normalize().
    std::size_t h = static_cast<std::size_t>(0x6D9A'B7C3'21EF'441DULL);
    hash_combine(h, static_cast<std::size_t>(GoalType::AND));
    hash_combine(h, children_.size());
    for (const auto& c : children_) hash_combine(h, c->hash());
    return h;
  }

  int satisfied_leaf_count(const FluentSet& fluents) const override {
    int sum = 0;
    for (const auto& child : children_) sum += child->goal_count(fluents);
    return sum;
  }

  std::vector<FluentSet> compute_dnf_branches() const override {
    // AND distributes over OR: AND(A, OR(B,C)) -> [{A,B}, {A,C}]
    // Start with a single empty branch
    std::vector<FluentSet> result = {{}};

    for (const auto& child : children_) {
      const auto& child_branches = child->get_dnf_branches();

      if (child_branches.empty()) {
        // Child is unsatisfiable (FalseGoal) - entire AND is unsatisfiable
        return {};
      }

      if (child_branches.size() == 1) {
        // Single branch from child - add its fluents to all current branches
        for (auto& branch : result) {
          for (const auto& fluent : child_branches[0]) {
            branch.insert(fluent);
          }
        }
      } else {
        // Multiple branches from child (OR) - distribute via Cartesian product
        std::vector<FluentSet> new_result;
        new_result.reserve(result.size() * child_branches.size());

        for (const auto& existing_branch : result) {
          for (const auto& child_branch : child_branches) {
            FluentSet combined = existing_branch;
            for (const auto& fluent : child_branch) {
              combined.insert(fluent);
            }
            new_result.push_back(std::move(combined));
          }
        }
        result = std::move(new_result);
      }
    }

    return result;
  }

private:
  std::vector<GoalPtr> children_;
};

// OrGoal: disjunction
class OrGoal final : public GoalBase {
public:
  explicit OrGoal(std::vector<GoalPtr> children) : children_(std::move(children)) {}

  bool evaluate(const FluentSet& fluents) const override {
    for (const auto& child : children_) {
      if (child->evaluate(fluents)) return true;
    }
    return false;
  }

  GoalType get_type() const override { return GoalType::OR; }

  GoalPtr normalize() const override;

  FluentSet get_all_literals() const override {
    FluentSet result;
    for (const auto& child : children_) {
      auto s = child->get_all_literals();
      result.insert(s.begin(), s.end());
    }
    return result;
  }

  const std::vector<GoalPtr>& children() const override { return children_; }

  bool equals(const GoalBase& other) const override {
    if (other.get_type() != GoalType::OR) return false;
    const auto& o = static_cast<const OrGoal&>(other);

    // Equality is defined over canonical ordering; normalize() enforces this.
    if (children_.size() != o.children_.size()) return false;
    for (std::size_t i = 0; i < children_.size(); ++i) {
      if (!(*children_[i] == *o.children_[i])) return false;
    }
    return true;
  }

  void fingerprint(std::vector<std::size_t>& out) const override {
    out.push_back(static_cast<std::size_t>(GoalType::OR));
    out.push_back(children_.size());
    for (const auto& c : children_) c->fingerprint(out);
  }

protected:
  std::size_t compute_hash() const override {
    // Order-sensitive because children are canonicalized in normalize().
    std::size_t h = static_cast<std::size_t>(0x2F4C'8D11'8B76'CC09ULL);
    hash_combine(h, static_cast<std::size_t>(GoalType::OR));
    hash_combine(h, children_.size());
    for (const auto& c : children_) hash_combine(h, c->hash());
    return h;
  }

  int satisfied_leaf_count(const FluentSet& fluents) const override {
    int best = 0;
    for (const auto& child : children_) best = std::max(best, child->goal_count(fluents));
    return best;
  }

  std::vector<FluentSet> compute_dnf_branches() const override {
    // OR: union of branches from all children
    std::vector<FluentSet> result;

    for (const auto& child : children_) {
      const auto& child_branches = child->get_dnf_branches();
      for (const auto& branch : child_branches) {
        result.push_back(branch);
      }
    }

    return result;
  }

private:
  std::vector<GoalPtr> children_;
};

// ---------------------- detail helpers ----------------------

namespace detail {

inline bool goal_less(const GoalPtr& a, const GoalPtr& b) {
  const std::size_t ha = a->hash();
  const std::size_t hb = b->hash();
  if (ha != hb) return ha < hb;

  // Rare: hash collision. Use deterministic structural fingerprint.
  std::vector<std::size_t> fa;
  std::vector<std::size_t> fb;
  fa.reserve(16);
  fb.reserve(16);
  a->fingerprint(fa);
  b->fingerprint(fb);
  return lex_less(fa, fb);
}

inline void canonicalize_children(std::vector<GoalPtr>& children) {
  std::sort(children.begin(), children.end(), goal_less);

  // Structural unique (NOT hash-only).
  auto new_end = std::unique(children.begin(), children.end(),
                             [](const GoalPtr& a, const GoalPtr& b) {
                               return *a == *b;
                             });
  children.erase(new_end, children.end());
}

inline bool contains_complement(const std::vector<GoalPtr>& children) {
  // Only checks explicit literal complements among direct children.
  // This is sufficient for many practical simplifications and avoids
  // distributing AND over OR.
  FluentSet seen;
  for (const auto& c : children) {
    if (c->get_type() != GoalType::LITERAL) continue;
    const auto& lit = static_cast<const LiteralGoal&>(*c);
    const Fluent& f = lit.fluent();
    if (seen.count(f.invert()) > 0) return true;
    seen.insert(f);
  }
  return false;
}

}  // namespace detail

// ---------------------- Normalization ----------------------

inline GoalPtr AndGoal::normalize() const {
  std::vector<GoalPtr> normalized_children;
  normalized_children.reserve(children_.size());

  // 1) Normalize children, absorb TRUE/FALSE, flatten nested ANDs.
  for (const auto& child : children_) {
    GoalPtr c = child->normalize();

    if (c->get_type() == GoalType::FALSE_GOAL) return make_false_goal();
    if (c->get_type() == GoalType::TRUE_GOAL) continue;

    if (c->get_type() == GoalType::AND) {
      for (const auto& gc : c->children()) normalized_children.push_back(gc);
    } else {
      normalized_children.push_back(std::move(c));
    }
  }

  // 2) Empty AND is TRUE.
  if (normalized_children.empty()) return make_true_goal();

  // 3) Canonicalize + structural dedup.
  detail::canonicalize_children(normalized_children);

  // 4) Check direct literal contradiction: P ∧ ~P => FALSE.
  if (detail::contains_complement(normalized_children)) return make_false_goal();

  // 5) Single child: return it.
  if (normalized_children.size() == 1) return normalized_children[0];

  return std::make_shared<AndGoal>(std::move(normalized_children));
}

inline GoalPtr OrGoal::normalize() const {
  std::vector<GoalPtr> normalized_children;
  normalized_children.reserve(children_.size());

  // 1) Normalize children, absorb TRUE/FALSE, flatten nested ORs.
  for (const auto& child : children_) {
    GoalPtr c = child->normalize();

    if (c->get_type() == GoalType::TRUE_GOAL) return make_true_goal();
    if (c->get_type() == GoalType::FALSE_GOAL) continue;

    if (c->get_type() == GoalType::OR) {
      for (const auto& gc : c->children()) normalized_children.push_back(gc);
    } else {
      normalized_children.push_back(std::move(c));
    }
  }

  // 2) Empty OR is FALSE.
  if (normalized_children.empty()) return make_false_goal();

  // 3) Canonicalize + structural dedup.
  detail::canonicalize_children(normalized_children);

  // 4) Check direct literal tautology: P ∨ ~P => TRUE.
  if (detail::contains_complement(normalized_children)) return make_true_goal();

  // 5) Single child: return it.
  if (normalized_children.size() == 1) return normalized_children[0];

  return std::make_shared<OrGoal>(std::move(normalized_children));
}

// ---------------------- Factory functions ----------------------

inline GoalPtr make_literal_goal(Fluent f) { return std::make_shared<LiteralGoal>(std::move(f)); }

inline GoalPtr make_and_goal(std::vector<GoalPtr> children) {
  return std::make_shared<AndGoal>(std::move(children));
}

inline GoalPtr make_or_goal(std::vector<GoalPtr> children) {
  return std::make_shared<OrGoal>(std::move(children));
}

inline GoalPtr make_true_goal() {
  static GoalPtr g = std::make_shared<TrueGoal>();
  return g;
}

inline GoalPtr make_false_goal() {
  static GoalPtr g = std::make_shared<FalseGoal>();
  return g;
}

inline GoalPtr goal_from_fluent_set(const FluentSet& fluents) {
  if (fluents.empty()) return make_true_goal();
  if (fluents.size() == 1) return make_literal_goal(*fluents.begin());

  std::vector<GoalPtr> children;
  children.reserve(fluents.size());
  for (const auto& f : fluents) children.push_back(make_literal_goal(f));

  return std::make_shared<AndGoal>(std::move(children))->normalize();
}

}  // namespace railroad
