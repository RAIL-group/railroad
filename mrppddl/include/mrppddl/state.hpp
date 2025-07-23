#pragma once
#include "mrppddl/core.hpp"
#include <string>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <sstream>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

namespace mrppddl {

class State {
public:
    using EffectQueue = std::vector<std::pair<double, GroundedEffect>>;

    State(double time = 0,
          std::unordered_set<Fluent> fluents = {},
          EffectQueue upcoming_effects = {})
        : time_(time),
          fluents_(std::move(fluents)),
          upcoming_effects_(std::move(upcoming_effects)),
          cached_hash_(std::nullopt) {}

    double time() const { return time_; }
    const std::unordered_set<Fluent>& fluents() const { return fluents_; }
    const EffectQueue& upcoming_effects() const { return upcoming_effects_; }

    void set_time(double new_time) {
        time_ = new_time;
        cached_hash_ = std::nullopt;
    }

    bool satisfies_precondition(const Action& action, bool relax = false) const {
        const auto& pos = action.pos_preconditions();
        bool has_all_positive = std::all_of(pos.begin(), pos.end(),
                                            [&](const Fluent& f) { return fluents_.count(f); });

	if (!has_all_positive || relax) {
	  return has_all_positive;
	}

        const auto& neg = action.neg_precond_flipped();
        bool disjoint_negative = std::none_of(neg.begin(), neg.end(),
                                             [&](const Fluent& f) { return fluents_.count(f); });
        return has_all_positive && disjoint_negative;
    }

    State copy() const {
        return State(time_, fluents_, upcoming_effects_);
    }

    void update_fluents(const std::unordered_set<Fluent>& new_fluents, bool relax = false) {
        cached_hash_ = std::nullopt;
        for (const auto& f : new_fluents) {
            if (f.is_negated()) {
                if (!relax) {
		  fluents_.erase(f.invert());
                }
            } else {
	      fluents_.insert(f);
            }
        }
    }

    void queue_effect(const GroundedEffect& effect) {
        cached_hash_ = std::nullopt;
        upcoming_effects_.emplace_back(time_ + effect.time(), effect);
        std::push_heap(upcoming_effects_.begin(), upcoming_effects_.end(), effect_cmp);
    }

    void pop_effect() {
        cached_hash_ = std::nullopt;
        std::pop_heap(upcoming_effects_.begin(), upcoming_effects_.end(), effect_cmp);
        upcoming_effects_.pop_back();
    }

    State copy_and_zero_out_time() const {
        EffectQueue new_effects;
        for (const auto& [t, e] : upcoming_effects_) {
            new_effects.emplace_back(t - time_, e);
        }
        return State(0, fluents_, new_effects);
    }

    std::size_t hash() const {
      if (cached_hash_) {
	return *cached_hash_;
      }

      std::size_t h_time = std::hash<double>{}(time_);

      // Hash fluents
      std::size_t h_fluents = 0;
      for (const auto& f : fluents_) {
	h_fluents ^= f.hash();
      }

      std::size_t h_upcoming = 0;
      for (const auto& [t, eff] : upcoming_effects_) {
	std::size_t h_effect = 0;
	hash_combine(h_effect, std::hash<double>{}(t));
	hash_combine(h_effect, eff.hash());
	h_upcoming ^= h_effect;
      }

      std::size_t h = 0;
      hash_combine(h, h_time);
      hash_combine(h, h_fluents);
      hash_combine(h, h_upcoming);
      cached_hash_ = h;

      return h;
    }

    bool operator==(const State& other) const {
        return this->hash() == other.hash();
    }

    bool operator<(const State& other) const {
        return time_ < other.time_;
    }

    std::string str() const {
        std::ostringstream out;
        out << "State<time=" << time_ << ", fluents={";
        bool first = true;
        for (const auto& f : fluents_) {
            if (!first) out << ", ";
            out << (f.is_negated() ? "not " : "") << f.name();
            for (const auto& arg : f.args()) out << " " << arg;
            first = false;
        }
        out << "}, upcoming_effects=[";
        for (size_t i = 0; i < upcoming_effects_.size(); ++i) {
            out << "(" << upcoming_effects_[i].first << ", ...)";
            if (i + 1 < upcoming_effects_.size()) out << ", ";
        }
        out << "]>";
        return out.str();
    }

private:
    double time_;
    std::unordered_set<Fluent> fluents_;
    EffectQueue upcoming_effects_;
    mutable std::optional<std::size_t> cached_hash_;
    static bool effect_cmp(const std::pair<double, GroundedEffect>& a,
                           const std::pair<double, GroundedEffect>& b) {
        return a.first > b.first;
    }
};

}

namespace std {
  template <>
  struct hash<mrppddl::State> {
    std::size_t operator()(const mrppddl::State& s) const noexcept {
      return s.hash();  // assumes State::hash() is a valid const method
    }
  };
}

namespace mrppddl {

inline void advance_to_terminal(
    State state,
    double prob,
    std::unordered_map<State, double>& outcomes,
    bool relax = false
) {
    while (!state.upcoming_effects().empty()) {
        auto [scheduled_time, effect] = state.upcoming_effects().front();

        if (scheduled_time > state.time() && !relax &&
            std::any_of(state.fluents().begin(), state.fluents().end(),
                        [](const Fluent& f) { return f.is_free(); }))
        {
            outcomes[state] += prob;
            return;
        }

        if (scheduled_time > state.time()) {
            state.set_time(scheduled_time);
        }

        state.pop_effect();
        state.update_fluents(effect.resulting_fluents(), relax);

	if (effect.is_probabilistic()) {
	  for (const auto& branch : effect.prob_effects()) {
	    State branched = state;
	    for (const auto& e : branch.effects()) {
	      branched.queue_effect(e);
	    }
	    advance_to_terminal(branched, prob * branch.prob(), outcomes, relax);
	  }
	  return;  // stop after branching
	}
    }

    outcomes[state] += prob;
}

inline std::vector<std::pair<State, double>> transition(
    const State& state,
    const Action* action,
    bool relax = false
) {
    // if (action && !state.satisfies_precondition(*action, relax)) {
    //     throw std::runtime_error("Precondition not satisfied for applying action");
    // }

    State new_state = state;
    for (const auto& effect : action->effects()) {
      new_state.queue_effect(effect);
    }

    // // Base: 631 ms
    // static std::chrono::nanoseconds cumulative_time(0); // Static, initialized once
    // auto start_time = std::chrono::high_resolution_clock::now();
    std::unordered_map<State, double> outcomes;
    advance_to_terminal(new_state, 1.0, outcomes, relax);
    // auto end_time = std::chrono::high_resolution_clock::now();
    // cumulative_time += (end_time - start_time);
    // std::cout << "Cumulative time in myFunction: "
    //           << std::chrono::duration_cast<std::chrono::milliseconds>(cumulative_time).count()
    //           << " ms" << std::endl;

    return {outcomes.begin(), outcomes.end()};
}

}  // namespace mrppddl
