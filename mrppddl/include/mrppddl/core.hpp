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

// class GroundedEffectType {
// public:
//     GroundedEffectType(double time, std::unordered_set<Fluent> resulting_fluents)
//         : time_(time), resulting_fluents_(std::move(resulting_fluents)) {}

//     double time() const { return time_; }
//     const std::unordered_set<Fluent>& resulting_fluents() const { return resulting_fluents_; }

//     bool operator<(const GroundedEffectType& other) const {
//         return time_ < other.time_;
//     }

// protected:
//     double time_;
//     std::unordered_set<Fluent> resulting_fluents_;
// };

// class GroundedEffectType {
// public:
//     using ProbBranch = ProbBranchWrapper;

//     GroundedEffectType(
//         double time,
//         std::unordered_set<Fluent> resulting_fluents = {},
//         std::vector<ProbBranch> prob_effects = {}
//     )
//         : time_(time),
//           resulting_fluents_(std::move(resulting_fluents)),
//           prob_effects_(std::move(prob_effects)) {}

//     double time() const { return time_; }
//     const std::unordered_set<Fluent>& resulting_fluents() const { return resulting_fluents_; }
//     const std::vector<ProbBranch>& prob_effects() const { return prob_effects_; }

//     bool is_probabilistic() const { return !prob_effects_.empty(); }

//     bool operator<(const GroundedEffectType& other) const {
//         return time_ < other.time_;
//     }

// private:
//     double time_;
//     std::unordered_set<Fluent> resulting_fluents_;
//     std::vector<ProbBranch> prob_effects_;
// };


// class GroundedEffect : public GroundedEffectType {
// public:
//     GroundedEffect(double time, std::unordered_set<Fluent> resulting_fluents)
//         : GroundedEffectType(time, std::move(resulting_fluents)) {
//         cached_hash_ = compute_hash();
//     }

//     std::size_t hash() const { return cached_hash_; }

//     std::string str() const {
//         std::vector<std::string> parts;
//         for (const auto& f : resulting_fluents_) {
//             std::ostringstream oss;
//             if (f.is_negated()) oss << "not ";
//             oss << f.name();
//             for (const auto& arg : f.args()) {
//                 oss << " " << arg;
//             }
//             parts.push_back(oss.str());
//         }
//         std::ostringstream out;
//         out << "after " << time_ << ": ";
//         for (size_t i = 0; i < parts.size(); ++i) {
//             out << parts[i];
//             if (i + 1 < parts.size()) out << ", ";
//         }
//         return out.str();
//     }

//     bool operator==(const GroundedEffect& other) const {
//         return cached_hash_ == other.cached_hash_ &&
//                time_ == other.time_ &&
//                resulting_fluents_ == other.resulting_fluents_;
//     }

// private:
//     std::size_t cached_hash_;

//     std::size_t compute_hash() const {
//         std::size_t h = std::hash<double>{}(time_);
//         for (const auto& f : resulting_fluents_) {
//             h ^= f.hash();
//         }
//         return h;
//     }
// };

// class GroundedProbEffect : public GroundedEffectType {
// public:
//     using ProbBranch = std::pair<double, std::vector<GroundedEffectType>>;

//     GroundedProbEffect(
//         double time,
//         std::vector<ProbBranch> prob_effects,
//         std::unordered_set<Fluent> resulting_fluents = {}
//     )
//         : GroundedEffectType(time, std::move(resulting_fluents)),
//           prob_effects_(std::move(prob_effects)) {
//         cached_hash_ = compute_hash();
//     }

//     const std::vector<ProbBranch>& prob_effects() const { return prob_effects_; }

//     std::size_t hash() const { return cached_hash_; }

//     std::string str() const {
//         std::ostringstream out;
//         out << "probabilistic after " << time_ << ": { ";
//         for (size_t i = 0; i < prob_effects_.size(); ++i) {
//             const auto& [prob, effs] = prob_effects_[i];
//             out << prob << ": [";
//             for (size_t j = 0; j < effs.size(); ++j) {
//                 out << "after " << effs[j].time() << "...";  // Placeholder, customize as needed
//                 if (j + 1 < effs.size()) out << "; ";
//             }
//             out << "]";
//             if (i + 1 < prob_effects_.size()) out << ", ";
//         }
//         out << " }";
//         return out.str();
//     }

// private:
//     std::vector<ProbBranch> prob_effects_;
//     std::size_t cached_hash_;

//     std::size_t compute_hash() const {
//         std::size_t h = std::hash<double>{}(time_);
//         for (const auto& f : resulting_fluents_) {
//             h ^= f.hash();
//         }
//         for (const auto& [p, effs] : prob_effects_) {
//             h ^= std::hash<double>{}(p);
//             for (const auto& e : effs) {
//                 h ^= std::hash<double>{}(e.time());  // Simplified
//             }
//         }
//         return h;
//     }
// };

}

namespace mrppddl {

class GroundedEffectType;  // Forward declaration

class ProbBranchWrapper {
public:
    ProbBranchWrapper(double prob, std::vector<GroundedEffectType> effects)
        : prob_(prob), effects_(std::move(effects)) {}

    double prob() const { return prob_; }
    const std::vector<GroundedEffectType>& effects() const { return effects_; }

private:
    double prob_;
    std::vector<GroundedEffectType> effects_;
};

class GroundedEffectType {
public:
  GroundedEffectType(
		     double time,
		     std::unordered_set<Fluent> resulting_fluents,
		     std::vector<std::pair<double, std::vector<GroundedEffectType>>> prob_pairs
		     )
    : time_(time), resulting_fluents_(std::move(resulting_fluents))
  {
    for (auto& [p, effects] : prob_pairs) {
      prob_effects_.emplace_back(p, std::move(effects));
    }
  }
GroundedEffectType(
    double time,
    std::unordered_set<Fluent> resulting_fluents
)
    : time_(time),
      resulting_fluents_(std::move(resulting_fluents)),
      prob_effects_{} {}


    // GroundedEffectType(
    //     double time,
    //     std::unordered_set<Fluent> resulting_fluents = {},
    //     std::vector<ProbBranchWrapper> prob_effects = {}
    // )
    //     : time_(time),
    //       resulting_fluents_(std::move(resulting_fluents)),
    //       prob_effects_(std::move(prob_effects)) {}

    double time() const { return time_; }
    const std::unordered_set<Fluent>& resulting_fluents() const { return resulting_fluents_; }
    const std::vector<ProbBranchWrapper>& prob_effects() const { return prob_effects_; }

    bool is_probabilistic() const { return !prob_effects_.empty(); }

    bool operator<(const GroundedEffectType& other) const {
        return time_ < other.time_;
    }

    bool operator==(const GroundedEffectType& other) const {
        return time_ == other.time_ &&
               resulting_fluents_ == other.resulting_fluents_ &&
               prob_effects_ == other.prob_effects_;  // Optional: you can omit this
    }

    std::size_t hash() const {
        std::size_t h = std::hash<double>{}(time_);
        for (const auto& f : resulting_fluents_) {
            h ^= f.hash();
        }
        for (const auto& branch : prob_effects_) {
            h ^= std::hash<double>{}(branch.prob());
            for (const auto& inner_eff : branch.effects()) {
                h ^= inner_eff.hash();
            }
        }
        return h;
    }

    std::string str() const {
        std::ostringstream out;
        if (is_probabilistic()) {
            out << "probabilistic after " << time_ << ": { ";
            for (const auto& branch : prob_effects_) {
                out << branch.prob() << ": [";
                for (size_t i = 0; i < branch.effects().size(); ++i) {
                    out << branch.effects()[i].str();
                    if (i + 1 < branch.effects().size()) out << "; ";
                }
                out << "], ";
            }
            out << "}";
        } else {
            out << "after " << time_ << ": ";
            bool first = true;
            for (const auto& f : resulting_fluents_) {
                if (!first) out << ", ";
                out << (f.is_negated() ? "not " : "") << f.name();
                for (const auto& arg : f.args()) out << " " << arg;
                first = false;
            }
        }
        return out.str();
    }

private:
    double time_;
    std::unordered_set<Fluent> resulting_fluents_;
    std::vector<ProbBranchWrapper> prob_effects_;
};

inline bool operator==(const ProbBranchWrapper& a, const ProbBranchWrapper& b) {
    return a.prob() == b.prob() && a.effects() == b.effects();
}

}  // namespace mrppddl

namespace std {
  template <>
  struct hash<mrppddl::GroundedEffectType> {
    std::size_t operator()(const mrppddl::GroundedEffectType& eff) const noexcept {
      return eff.hash();
    }
  };
}


namespace mrppddl {

class Action {
public:
    Action(std::unordered_set<Fluent> preconditions,
           std::vector<GroundedEffectType> effects,
           std::string name = "anonymous")
        : preconditions_(std::move(preconditions)),
          effects_(std::move(effects)),
          name_(std::move(name))
    {
        for (const auto& f : preconditions_) {
            if (f.is_negated()) {
                neg_precond_flipped_.insert(f.invert());
            } else {
                pos_precond_.insert(f);
            }
        }
    }

  Action() = default;
  Action(const Action&) = default;
  Action& operator=(const Action&) = default;

    const std::unordered_set<Fluent>& preconditions() const { return preconditions_; }
    const std::vector<GroundedEffectType>& effects() const { return effects_; }
    const std::string& name() const { return name_; }
    const std::unordered_set<Fluent>& pos_preconditions() const { return pos_precond_; }
    const std::unordered_set<Fluent>& neg_precond_flipped() const { return neg_precond_flipped_; }

    std::string str() const {
        std::ostringstream out;
        out << "Action('" << name_ << "'\n  Preconditions: [";

        bool first = true;
        for (const auto& p : preconditions_) {
            if (!first) out << ", ";
            std::ostringstream p_str;
            if (p.is_negated()) p_str << "not ";
            p_str << p.name();
            for (const auto& arg : p.args()) {
                p_str << " " << arg;
            }
            out << p_str.str();
            first = false;
        }

        out << "]\n  Effects:\n";
        for (const auto& eff : effects_) {
            out << "    after " << eff.time() << "...";  // customize if desired
            out << "\n";
        }
        out << ")";
        return out.str();
    }

private:
    std::unordered_set<Fluent> preconditions_;
    std::vector<GroundedEffectType> effects_;
    std::string name_;
    std::unordered_set<Fluent> pos_precond_;
    std::unordered_set<Fluent> neg_precond_flipped_;
};


}  // namespace mrppddl
