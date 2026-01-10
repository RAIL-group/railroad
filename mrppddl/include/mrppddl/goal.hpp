#pragma once

#include "mrppddl/core.hpp"
#include <algorithm>
#include <memory>
#include <unordered_set>
#include <vector>

namespace mrppddl {

enum class GoalType { LITERAL, AND, OR, TRUE_GOAL, FALSE_GOAL };

class GoalBase;
using GoalPtr = std::shared_ptr<GoalBase>;

class GoalBase {
public:
  virtual ~GoalBase() = default;

  // Check if goal is satisfied by given fluents
  virtual bool evaluate(const std::unordered_set<Fluent> &fluents) const = 0;

  // Get the type of this goal
  virtual GoalType get_type() const = 0;

  // Return normalized form of this goal
  virtual GoalPtr normalize() const = 0;

  // Get all literal fluents in this goal
  virtual std::unordered_set<Fluent> get_all_literals() const = 0;

  // Check if this is a pure conjunction of literals (no ORs)
  virtual bool is_pure_conjunction() const = 0;

  // Get children (for AND/OR goals)
  virtual const std::vector<GoalPtr> &children() const {
    static const std::vector<GoalPtr> empty;
    return empty;
  }

  // Count how many goal literals are achieved
  virtual int goal_count(const std::unordered_set<Fluent> &fluents) const {
    auto literals = get_all_literals();
    int count = 0;
    for (const auto &f : literals) {
      if (fluents.count(f)) {
        count++;
      }
    }
    return count;
  }

  // Hash for deduplication
  virtual std::size_t hash() const = 0;

  bool operator==(const GoalBase &other) const { return hash() == other.hash(); }
};

// TrueGoal: Always satisfied
class TrueGoal : public GoalBase {
public:
  bool evaluate(const std::unordered_set<Fluent> &) const override {
    return true;
  }

  GoalType get_type() const override { return GoalType::TRUE_GOAL; }

  GoalPtr normalize() const override {
    return std::make_shared<TrueGoal>(*this);
  }

  std::unordered_set<Fluent> get_all_literals() const override { return {}; }

  bool is_pure_conjunction() const override { return true; }

  std::size_t hash() const override {
    return std::hash<std::string>{}("TRUE_GOAL");
  }
};

// FalseGoal: Never satisfied
class FalseGoal : public GoalBase {
public:
  bool evaluate(const std::unordered_set<Fluent> &) const override {
    return false;
  }

  GoalType get_type() const override { return GoalType::FALSE_GOAL; }

  GoalPtr normalize() const override {
    return std::make_shared<FalseGoal>(*this);
  }

  std::unordered_set<Fluent> get_all_literals() const override { return {}; }

  bool is_pure_conjunction() const override { return true; }

  std::size_t hash() const override {
    return std::hash<std::string>{}("FALSE_GOAL");
  }
};

// LiteralGoal: A single fluent that must be true (or absent if negated)
class LiteralGoal : public GoalBase {
public:
  explicit LiteralGoal(Fluent fluent) : fluent_(std::move(fluent)) {}

  bool evaluate(const std::unordered_set<Fluent> &fluents) const override {
    if (fluent_.is_negated()) {
      // For negative fluent ~P, check that P is NOT in state
      return fluents.count(fluent_.invert()) == 0;
    } else {
      // For positive fluent P, check that P is in state
      return fluents.count(fluent_) > 0;
    }
  }

  GoalType get_type() const override { return GoalType::LITERAL; }

  GoalPtr normalize() const override {
    return std::make_shared<LiteralGoal>(*this);
  }

  std::unordered_set<Fluent> get_all_literals() const override {
    return {fluent_};
  }

  bool is_pure_conjunction() const override { return true; }

  const Fluent &fluent() const { return fluent_; }

  std::size_t hash() const override {
    std::size_t h = std::hash<std::string>{}("LITERAL");
    hash_combine(h, fluent_.hash());
    return h;
  }

private:
  Fluent fluent_;
};

// Forward declarations for normalization
class AndGoal;
class OrGoal;

// AndGoal: Conjunction - all children must be satisfied
class AndGoal : public GoalBase {
public:
  explicit AndGoal(std::vector<GoalPtr> children)
      : children_(std::move(children)) {}

  bool evaluate(const std::unordered_set<Fluent> &fluents) const override {
    for (const auto &child : children_) {
      if (!child->evaluate(fluents)) {
        return false; // Short-circuit
      }
    }
    return true;
  }

  GoalType get_type() const override { return GoalType::AND; }

  GoalPtr normalize() const override;

  std::unordered_set<Fluent> get_all_literals() const override {
    std::unordered_set<Fluent> result;
    for (const auto &child : children_) {
      auto child_literals = child->get_all_literals();
      result.insert(child_literals.begin(), child_literals.end());
    }
    return result;
  }

  bool is_pure_conjunction() const override {
    for (const auto &child : children_) {
      if (!child->is_pure_conjunction()) {
        return false;
      }
    }
    return true;
  }

  const std::vector<GoalPtr> &children() const override { return children_; }

  std::size_t hash() const override {
    std::size_t h = std::hash<std::string>{}("AND");
    // XOR for order-independence after canonical ordering
    std::size_t children_hash = 0;
    for (const auto &child : children_) {
      children_hash ^= child->hash();
    }
    hash_combine(h, children_hash);
    return h;
  }

private:
  std::vector<GoalPtr> children_;
};

// OrGoal: Disjunction - at least one child must be satisfied
class OrGoal : public GoalBase {
public:
  explicit OrGoal(std::vector<GoalPtr> children)
      : children_(std::move(children)) {}

  bool evaluate(const std::unordered_set<Fluent> &fluents) const override {
    for (const auto &child : children_) {
      if (child->evaluate(fluents)) {
        return true; // Short-circuit
      }
    }
    return false;
  }

  GoalType get_type() const override { return GoalType::OR; }

  GoalPtr normalize() const override;

  std::unordered_set<Fluent> get_all_literals() const override {
    std::unordered_set<Fluent> result;
    for (const auto &child : children_) {
      auto child_literals = child->get_all_literals();
      result.insert(child_literals.begin(), child_literals.end());
    }
    return result;
  }

  bool is_pure_conjunction() const override {
    return false; // OR is never a pure conjunction
  }

  const std::vector<GoalPtr> &children() const override { return children_; }

  std::size_t hash() const override {
    std::size_t h = std::hash<std::string>{}("OR");
    // XOR for order-independence after canonical ordering
    std::size_t children_hash = 0;
    for (const auto &child : children_) {
      children_hash ^= child->hash();
    }
    hash_combine(h, children_hash);
    return h;
  }

private:
  std::vector<GoalPtr> children_;
};

// Normalization implementations

inline GoalPtr AndGoal::normalize() const {
  std::vector<GoalPtr> normalized_children;

  // Step 1: Recursively normalize and flatten nested ANDs
  for (const auto &child : children_) {
    GoalPtr normalized_child = child->normalize();

    // Check for FALSE (short-circuit: AND with FALSE = FALSE)
    if (normalized_child->get_type() == GoalType::FALSE_GOAL) {
      return std::make_shared<FalseGoal>();
    }

    // Skip TRUE (AND with TRUE doesn't change result)
    if (normalized_child->get_type() == GoalType::TRUE_GOAL) {
      continue;
    }

    // Flatten nested ANDs
    if (normalized_child->get_type() == GoalType::AND) {
      for (const auto &grandchild : normalized_child->children()) {
        normalized_children.push_back(grandchild);
      }
    } else {
      normalized_children.push_back(normalized_child);
    }
  }

  // Step 2: Handle empty case
  if (normalized_children.empty()) {
    return std::make_shared<TrueGoal>();
  }

  // Step 3: Single child case
  if (normalized_children.size() == 1) {
    return normalized_children[0];
  }

  // Step 4: Deduplication using hash
  std::unordered_set<std::size_t> seen_hashes;
  std::vector<GoalPtr> deduped;
  for (const auto &child : normalized_children) {
    std::size_t h = child->hash();
    if (seen_hashes.find(h) == seen_hashes.end()) {
      seen_hashes.insert(h);
      deduped.push_back(child);
    }
  }

  if (deduped.size() == 1) {
    return deduped[0];
  }

  // Step 5: Canonical ordering (sort by hash for determinism)
  std::sort(deduped.begin(), deduped.end(),
            [](const GoalPtr &a, const GoalPtr &b) {
              return a->hash() < b->hash();
            });

  return std::make_shared<AndGoal>(std::move(deduped));
}

inline GoalPtr OrGoal::normalize() const {
  std::vector<GoalPtr> normalized_children;

  // Step 1: Recursively normalize and flatten nested ORs
  for (const auto &child : children_) {
    GoalPtr normalized_child = child->normalize();

    // Check for TRUE (short-circuit: OR with TRUE = TRUE)
    if (normalized_child->get_type() == GoalType::TRUE_GOAL) {
      return std::make_shared<TrueGoal>();
    }

    // Skip FALSE (OR with FALSE doesn't change result)
    if (normalized_child->get_type() == GoalType::FALSE_GOAL) {
      continue;
    }

    // Flatten nested ORs
    if (normalized_child->get_type() == GoalType::OR) {
      for (const auto &grandchild : normalized_child->children()) {
        normalized_children.push_back(grandchild);
      }
    } else {
      normalized_children.push_back(normalized_child);
    }
  }

  // Step 2: Handle empty case
  if (normalized_children.empty()) {
    return std::make_shared<FalseGoal>();
  }

  // Step 3: Single child case
  if (normalized_children.size() == 1) {
    return normalized_children[0];
  }

  // Step 4: Deduplication using hash
  std::unordered_set<std::size_t> seen_hashes;
  std::vector<GoalPtr> deduped;
  for (const auto &child : normalized_children) {
    std::size_t h = child->hash();
    if (seen_hashes.find(h) == seen_hashes.end()) {
      seen_hashes.insert(h);
      deduped.push_back(child);
    }
  }

  if (deduped.size() == 1) {
    return deduped[0];
  }

  // Step 5: Canonical ordering (sort by hash for determinism)
  std::sort(deduped.begin(), deduped.end(),
            [](const GoalPtr &a, const GoalPtr &b) {
              return a->hash() < b->hash();
            });

  return std::make_shared<OrGoal>(std::move(deduped));
}

// Factory functions

inline GoalPtr make_literal_goal(const Fluent &f) {
  return std::make_shared<LiteralGoal>(f);
}

inline GoalPtr make_and_goal(const std::vector<GoalPtr> &children) {
  return std::make_shared<AndGoal>(children);
}

inline GoalPtr make_or_goal(const std::vector<GoalPtr> &children) {
  return std::make_shared<OrGoal>(children);
}

inline GoalPtr make_true_goal() { return std::make_shared<TrueGoal>(); }

inline GoalPtr make_false_goal() { return std::make_shared<FalseGoal>(); }

inline GoalPtr goal_from_fluent_set(const std::unordered_set<Fluent> &fluents) {
  if (fluents.empty()) {
    return std::make_shared<TrueGoal>();
  }

  if (fluents.size() == 1) {
    return std::make_shared<LiteralGoal>(*fluents.begin());
  }

  std::vector<GoalPtr> children;
  children.reserve(fluents.size());
  for (const auto &f : fluents) {
    children.push_back(std::make_shared<LiteralGoal>(f));
  }
  return std::make_shared<AndGoal>(children)->normalize();
}

} // namespace mrppddl
