#define RAILROAD_USE_PYBIND
#include "railroad/core.hpp"
#include "railroad/heuristic.hpp"
#include "railroad/goal.hpp"
#include "railroad/planner.hpp"
#include "railroad/state.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace railroad;

PYBIND11_MODULE(_bindings, m) {
  py::class_<Fluent>(m, "Fluent")
      .def(py::init([](const std::string &name, py::args args, bool negated) {
             std::vector<std::string> arg_list;
             for (const auto &arg : args) {
               arg_list.push_back(arg.cast<std::string>());
             }
             return Fluent(name, std::move(arg_list), negated);
           }),
           py::arg("name"), py::arg("negated") = false)
      .def("__invert__", &Fluent::invert)
      .def("__eq__", &Fluent::operator==)
      .def("__hash__", &Fluent::hash)
      .def_property_readonly("name", &Fluent::name)
      .def_property_readonly("args", &Fluent::args)
      .def_property_readonly("negated", &Fluent::is_negated)
      .def(py::pickle(
          [](const Fluent &f) {
            std::string full = f.name();
            for (const auto &arg : f.args()) {
              full += " " + arg;
            }
            if (f.is_negated()) {
              full = "not " + full;
            }
            return py::make_tuple(full); // what to serialize
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error("Invalid state for Fluent!");
            return Fluent(t[0].cast<std::string>()); // how to restore
          }))
      .def("__repr__", [](const Fluent &f) {
        std::ostringstream oss;
        oss << "(";
        if (f.is_negated())
          oss << "not ";
        oss << f.name();
        for (const auto &arg : f.args()) {
          oss << " " << arg;
        }
        oss << ")";
        return oss.str();
      })
      // Operators for goal construction: F("a") & F("b") -> AndGoal
      .def("__and__", [](const Fluent &self, const py::object &other) -> GoalPtr {
        auto self_goal = std::make_shared<LiteralGoal>(self);
        if (py::isinstance<Fluent>(other)) {
          auto other_goal = std::make_shared<LiteralGoal>(other.cast<Fluent>());
          return std::make_shared<AndGoal>(std::vector<GoalPtr>{self_goal, other_goal});
        }
        // Try cast to GoalPtr - works for Goal and all subclasses
        try {
          auto other_goal = other.cast<GoalPtr>();
          if (other_goal->get_type() == GoalType::AND) {
            // Flatten: self & AND(a, b) -> AND(self, a, b)
            std::vector<GoalPtr> children = {self_goal};
            for (const auto &child : other_goal->children()) {
              children.push_back(child);
            }
            return std::make_shared<AndGoal>(std::move(children));
          }
          return std::make_shared<AndGoal>(std::vector<GoalPtr>{self_goal, other_goal});
        } catch (const py::cast_error&) {
          throw py::type_error("unsupported operand type for &");
        }
      })
      .def("__or__", [](const Fluent &self, const py::object &other) -> GoalPtr {
        auto self_goal = std::make_shared<LiteralGoal>(self);
        if (py::isinstance<Fluent>(other)) {
          auto other_goal = std::make_shared<LiteralGoal>(other.cast<Fluent>());
          return std::make_shared<OrGoal>(std::vector<GoalPtr>{self_goal, other_goal});
        }
        // Try cast to GoalPtr - works for Goal and all subclasses
        try {
          auto other_goal = other.cast<GoalPtr>();
          if (other_goal->get_type() == GoalType::OR) {
            // Flatten: self | OR(a, b) -> OR(self, a, b)
            std::vector<GoalPtr> children = {self_goal};
            for (const auto &child : other_goal->children()) {
              children.push_back(child);
            }
            return std::make_shared<OrGoal>(std::move(children));
          }
          return std::make_shared<OrGoal>(std::vector<GoalPtr>{self_goal, other_goal});
        } catch (const py::cast_error&) {
          throw py::type_error("unsupported operand type for |");
        }
      })
      .def("__rand__", [](const Fluent &self, const py::object &other) -> GoalPtr {
        // Reverse and: other & self (when other doesn't define __and__)
        if (py::isinstance<Fluent>(other)) {
          auto other_goal = std::make_shared<LiteralGoal>(other.cast<Fluent>());
          auto self_goal = std::make_shared<LiteralGoal>(self);
          return std::make_shared<AndGoal>(std::vector<GoalPtr>{other_goal, self_goal});
        }
        throw py::type_error("unsupported operand type for &");
      })
      .def("__ror__", [](const Fluent &self, const py::object &other) -> GoalPtr {
        // Reverse or: other | self (when other doesn't define __or__)
        if (py::isinstance<Fluent>(other)) {
          auto other_goal = std::make_shared<LiteralGoal>(other.cast<Fluent>());
          auto self_goal = std::make_shared<LiteralGoal>(self);
          return std::make_shared<OrGoal>(std::vector<GoalPtr>{other_goal, self_goal});
        }
        throw py::type_error("unsupported operand type for |");
      })
      // Goal-like methods for Fluent (delegate to LiteralGoal)
      .def("evaluate", [](const Fluent &self, const std::unordered_set<Fluent> &fluents) {
        return LiteralGoal(self).evaluate(fluents);
      }, py::arg("fluents"), "Check if this fluent is satisfied by given fluents")
      .def("get_all_literals", [](const Fluent &self) {
        return LiteralGoal(self).get_all_literals();
      }, "Get all literal fluents (just this fluent)")
      .def("goal_count", [](const Fluent &self, const std::unordered_set<Fluent> &fluents) {
        return LiteralGoal(self).goal_count(fluents);
      }, py::arg("fluents"), "Count how many goal literals are achieved (0 or 1)");

  // GroundedEffect
  py::class_<GroundedEffect, std::shared_ptr<GroundedEffect>>(m,
                                                              "GroundedEffect")
      .def(py::init<double, std::unordered_set<Fluent>,
                    std::vector<std::pair<
                        double,
                        std::vector<std::shared_ptr<const GroundedEffect>>>>,
                    std::vector<std::pair<
                        std::unordered_set<Fluent>,
                        std::vector<std::shared_ptr<const GroundedEffect>>>>>(),
           py::arg("time"),
           py::arg("resulting_fluents") = std::unordered_set<Fluent>{},
           py::arg("prob_effects") =
               std::vector<std::pair<double, std::vector<GroundedEffect>>>{},
           py::arg("cond_effects") = std::vector<std::pair<
               std::unordered_set<Fluent>, std::vector<GroundedEffect>>>{})
      .def_property_readonly("time", &GroundedEffect::time)
      .def_property_readonly("resulting_fluents",
                             &GroundedEffect::resulting_fluents)
      .def_property_readonly("prob_effects", &GroundedEffect::prob_effects)
      .def_property_readonly("cond_effects", &GroundedEffect::cond_effects)
      .def_property_readonly("is_probabilistic",
                             &GroundedEffect::is_probabilistic)
      .def_property_readonly("is_conditional",
                             &GroundedEffect::is_conditional)
      .def(py::pickle(
          [](const GroundedEffect &eff) {
            // Serialize the GroundedEffect into a tuple
            // Flatten prob_effects recursively
            py::list pickled_prob_effects;
            for (const auto &pair : eff.prob_effects()) {
              double prob = pair.prob();
              py::list sub_effects;
              for (const auto &sub_eff : pair.effects()) {
                sub_effects.append(
                    sub_eff); // Assumes recursive GroundedEffect pickling
              }
              pickled_prob_effects.append(py::make_tuple(prob, sub_effects));
            }

            py::list pickled_cond_effects;
            for (const auto &branch : eff.cond_effects()) {
              py::list sub_effects;
              for (const auto &sub_eff : branch.effects()) {
                sub_effects.append(sub_eff);
              }
              pickled_cond_effects.append(
                  py::make_tuple(py::cast(branch.conditions()), sub_effects));
            }

            return py::make_tuple(
                eff.time(),
                py::cast(eff.resulting_fluents()), // std::unordered_set<Fluent>
                pickled_prob_effects, // List of (double, list of GroundedEffect)
                pickled_cond_effects  // List of (set of Fluent, list of GroundedEffect)
            );
          },
          [](py::tuple t) {
            // Size 3 accepted for pickles predating conditional effects.
            if (t.size() != 3 && t.size() != 4)
              throw std::runtime_error("Invalid state for GroundedEffect!");

            double time = t[0].cast<double>();
            auto fluents = t[1].cast<std::unordered_set<Fluent>>();

            auto py_prob_effects = t[2].cast<py::list>();
            std::vector<std::pair<
                double, std::vector<std::shared_ptr<const GroundedEffect>>>>
                prob_effects;

            for (auto item : py_prob_effects) {
              auto tup = item.cast<py::tuple>();
              if (tup.size() != 2)
                throw std::runtime_error("Invalid subeffect structure!");

              double prob = tup[0].cast<double>();
              std::vector<std::shared_ptr<const GroundedEffect>> effects;
              for (auto sub : tup[1].cast<py::list>()) {
                effects.push_back(
                    sub.cast<std::shared_ptr<const GroundedEffect>>());
              }
              prob_effects.emplace_back(prob, std::move(effects));
            }

            std::vector<std::pair<std::unordered_set<Fluent>,
                                  std::vector<std::shared_ptr<const GroundedEffect>>>>
                cond_effects;
            if (t.size() == 4) {
              for (auto item : t[3].cast<py::list>()) {
                auto tup = item.cast<py::tuple>();
                if (tup.size() != 2)
                  throw std::runtime_error("Invalid conditional structure!");

                auto conditions = tup[0].cast<std::unordered_set<Fluent>>();
                std::vector<std::shared_ptr<const GroundedEffect>> effects;
                for (auto sub : tup[1].cast<py::list>()) {
                  effects.push_back(
                      sub.cast<std::shared_ptr<const GroundedEffect>>());
                }
                cond_effects.emplace_back(std::move(conditions),
                                          std::move(effects));
              }
            }

            return std::make_shared<GroundedEffect>(time, std::move(fluents),
                                                    std::move(prob_effects),
                                                    std::move(cond_effects));
          }))
      .def("__str__", &GroundedEffect::str)
      .def("__repr__",
           [](const GroundedEffect &eff) {
             return "GroundedEffect(" + eff.str() + ")";
           })
      .def("__eq__", &GroundedEffect::operator==)
      .def("__hash__", [](const GroundedEffect &eff) {
        return static_cast<std::size_t>(eff.hash());
      });

  py::class_<Action>(m, "Action")
      .def(py::init<std::unordered_set<Fluent>,
                    std::vector<std::shared_ptr<const GroundedEffect>>,
                    std::string,
                    double>(),
           py::arg("preconditions"), py::arg("effects"),
           py::arg("name") = "anonymous", py::arg("extra_cost") = 0.0)
      .def_property_readonly("name", &Action::name)
      .def_property_readonly("extra_cost", &Action::extra_cost)
      .def_property_readonly("preconditions", &Action::preconditions)
      .def_property_readonly("effects", &Action::effects)
      .def_property_readonly("_pos_precond", &Action::pos_preconditions)
      .def_property_readonly("_neg_precond_flipped",
                             &Action::neg_precond_flipped)
      .def(py::pickle(
          [](const Action &a) {
            // Serialization
            return py::make_tuple(
                py::cast(a.preconditions()), // unordered_set<Fluent>
                py::cast(a.effects()), // vector<shared_ptr<GroundedEffect>>
                py::cast(a.name()),     // string
                py::cast(a.extra_cost()) // double
            );
          },
          [](py::tuple t) {
            if (t.size() != 4)
              throw std::runtime_error("Invalid state for Action!");

            auto preconds = t[0].cast<std::unordered_set<Fluent>>();
            auto effects =
                t[1].cast<std::vector<std::shared_ptr<const GroundedEffect>>>();
            auto name = t[2].cast<std::string>();
            auto extra_cost = t[3].cast<double>();

            return Action(std::move(preconds), std::move(effects),
                          std::move(name), extra_cost);
          }))
      .def("__str__", &Action::str)
      .def("__repr__", [](const Action &a) { return a.str(); })
      .def("__eq__", &Action::operator==)
      .def("__hash__", [](const Action &action) {
        return static_cast<std::size_t>(action.hash());
      });

  py::class_<State>(m, "State")
      .def(py::init<double, std::unordered_set<Fluent>,
                    std::vector<std::pair<
                        double, std::shared_ptr<const GroundedEffect>>>>(),
           py::arg("time") = 0.0,
           py::arg("fluents") = std::unordered_set<Fluent>{},
           py::arg("upcoming_effects") =
               std::vector<std::pair<double, GroundedEffect>>{})
      .def_property("time", &State::time, &State::set_time)
      .def_property_readonly("fluents", &State::fluents)
      .def_property_readonly("upcoming_effects", &State::upcoming_effects)
      .def("satisfies_precondition", &State::satisfies_precondition,
           py::arg("action"), py::arg("relax") = false)
      .def("update_fluents", &State::update_fluents, py::arg("new_fluents"),
           py::arg("relax") = false)
      .def("copy", &State::copy)
      .def("copy_and_zero_out_time", &State::copy_and_zero_out_time)
      .def("queue_effect", &State::queue_effect)
      .def("pop_effect", &State::pop_effect)
      .def("set_time", &State::set_time)
      .def(py::pickle(
          [](const State &s) {
            py::list pickled_effects;
            for (const auto &pair : s.upcoming_effects()) {
              double t = pair.first;
              pickled_effects.append(py::make_tuple(t, pair.second));
            }

            return py::make_tuple(
                s.time(),
                py::cast(s.fluents()), // std::unordered_set<Fluent>
                pickled_effects        // List of (double, GroundedEffect)
            );
          },
          [](py::tuple t) {
            if (t.size() != 3)
              throw std::runtime_error("Invalid state for State!");

            double time = t[0].cast<double>();
            auto fluents = t[1].cast<std::unordered_set<Fluent>>();

            auto py_effects = t[2].cast<py::list>();
            std::vector<
                std::pair<double, std::shared_ptr<const GroundedEffect>>>
                effects;

            for (auto item : py_effects) {
              auto tup = item.cast<py::tuple>();
              if (tup.size() != 2)
                throw std::runtime_error("Invalid effect entry!");

              double t_offset = tup[0].cast<double>();
              auto effect =
                  tup[1].cast<std::shared_ptr<const GroundedEffect>>();
              effects.emplace_back(t_offset, std::move(effect));
            }

            return State(time, std::move(fluents), std::move(effects));
          }))

      .def("__hash__", &State::hash)
      .def("__eq__",
           [](const State &self, py::object other) {
             // Check if 'other' is instance of State
             if (!py::isinstance<State>(other))
               return false;
             // Perform the actual comparison
             const auto &other_ref = other.cast<const State &>();
             return self == other_ref;
           })
      .def("__lt__", &State::operator<)
      .def("__str__", &State::str)
      .def("__repr__", [](const State &s) { return s.str(); });
  m.def(
      "transition",
      [](const State &state, const Action *action, bool relax) {
        return transition(state, action, relax);
      },
      py::arg("state"), py::arg("action") = nullptr, py::arg("relax") = false);
  py::class_<ProbBranchWrapper>(m, "ProbBranch")
      .def(py::init<double,
                    std::vector<std::shared_ptr<const GroundedEffect>>>())
      .def_property_readonly("prob", &ProbBranchWrapper::prob)
      .def_property_readonly("effects", &ProbBranchWrapper::effects)
      .def("__getitem__",
           [](const ProbBranchWrapper &b, std::size_t i) -> py::object {
             if (i == 0)
               return py::float_(b.prob());
             if (i == 1)
               return py::cast(b.effects());
             throw py::index_error("ProbBranch index out of range");
           })
      .def("__repr__", [](const ProbBranchWrapper &b) {
        std::ostringstream oss;
        oss << "<ProbBranch p=" << b.prob()
            << ", n_effects=" << b.effects().size() << ">";
        return oss.str();
      });

  py::class_<CondBranchWrapper>(m, "CondBranch")
      .def(py::init<std::unordered_set<Fluent>,
                    std::vector<std::shared_ptr<const GroundedEffect>>>())
      .def_property_readonly("conditions", &CondBranchWrapper::conditions)
      .def_property_readonly("effects", &CondBranchWrapper::effects)
      .def("holds", &CondBranchWrapper::holds, py::arg("fluents"),
           "Whether the conditions hold in the given fluent set")
      .def("__repr__", [](const CondBranchWrapper &b) {
        std::ostringstream oss;
        oss << "<CondBranch n_conditions=" << b.conditions().size()
            << ", n_effects=" << b.effects().size() << ">";
        return oss.str();
      });

  m.def(
      "transition",
      [](const State &state,
         std::optional<std::reference_wrapper<const Action>> action,
         bool relax) {
        return railroad::transition(state, action ? &action->get() : nullptr,
                                   relax);
      },
      py::arg("state"), py::arg("action") = std::nullopt,
      py::arg("relax") = false);

  m.def("get_next_actions",
        [](const State &state, const std::vector<Action> &all_actions) {
          // Return owned copies: get_next_actions yields pointers into
          // `all_actions`, which is a temporary built from the Python list and
          // destroyed when the call returns, so the pointers must not escape.
          std::vector<Action> out;
          for (const Action *a : get_next_actions(state, all_actions)) {
            out.push_back(*a);
          }
          return out;
        },
        py::arg("state"), py::arg("all_actions"),
        "Return list of applicable actions for at least one free robot");
  m.def("get_usable_actions",
        [](const State &input_state, const std::vector<Action> &all_actions) {
          return get_usable_actions(input_state, all_actions);
        },
        py::arg("input_state"), py::arg("all_actions"),
        "Get actions usable from the given state via forward reachability");

  m.def("get_relaxed_optimistic_costs",
        [](const State &input_state, const std::vector<Action> &all_actions) {
          return get_relaxed_optimistic_costs(input_state, all_actions);
        },
        py::arg("input_state"), py::arg("all_actions"),
        "Optimistic relaxed-plan costs for all reachable fluents from the given state");

  m.def("get_relaxed_optimistic_cost",
        [](const State &input_state, const Fluent &fluent, const std::vector<Action> &all_actions) {
          return get_relaxed_optimistic_cost(input_state, fluent, all_actions);
        },
        py::arg("input_state"), py::arg("fluent"), py::arg("all_actions"),
        "Optimistic relaxed-plan cost for a single fluent from the given state");

  m.def("get_probabilistic_path_achievers",
        [](const State &input_state, const GoalPtr &goal,
           const std::vector<Action> &all_actions, bool at_implies_found) {
          return get_probabilistic_path_achievers(input_state, goal.get(),
                                                   all_actions, at_implies_found);
        },
        py::arg("input_state"), py::arg("goal"), py::arg("all_actions"),
        py::arg("at_implies_found") = true,
        "For each probabilistic fluent on the relaxed path to the goal, return "
        "its achievers as (action_name, probability, exec_cost, wait_cost).");

  m.def("get_goal_relevant_action_names",
        [](const State &input_state, const GoalPtr &goal,
           const std::vector<Action> &all_actions, bool at_implies_found) {
          return get_goal_relevant_action_names(input_state, goal.get(),
                                                all_actions, at_implies_found);
        },
        py::arg("input_state"), py::arg("goal"), py::arg("all_actions"),
        py::arg("at_implies_found") = true,
        "Names of all actions that can contribute to the goal under relaxed "
        "reachability (the backward closure following every achiever).");

  // Complex Goal classes
  py::enum_<GoalType>(m, "GoalType")
      .value("LITERAL", GoalType::LITERAL)
      .value("AND", GoalType::AND)
      .value("OR", GoalType::OR)
      .value("TRUE_GOAL", GoalType::TRUE_GOAL)
      .value("FALSE_GOAL", GoalType::FALSE_GOAL);

  py::class_<GoalBase, GoalPtr>(m, "Goal")
      .def("evaluate", &GoalBase::evaluate, py::arg("fluents"),
           "Check if goal is satisfied by given fluents")
      .def("get_type", &GoalBase::get_type,
           "Get the type of this goal")
      .def("normalize", &GoalBase::normalize,
           "Return normalized form of this goal")
      .def("get_all_literals", &GoalBase::get_all_literals,
           "Get all literal fluents in this goal")
      .def("get_dnf_branches", &GoalBase::get_dnf_branches,
           "Get DNF branches: list of fluent sets (OR of ANDs)")
      .def("dnf_branch_count", &GoalBase::dnf_branch_count,
           "How many branches get_dnf_branches() would build, computed "
           "structurally without building any. AND multiplies and OR adds, so "
           "this grows exponentially in the number of conjoined disjunctions "
           "-- check it before walking the DNF of a machine-generated goal. "
           "Saturates rather than overflowing.")
      .def("children", &GoalBase::children,
           "Get children (for AND/OR goals)")
      .def("goal_count", &GoalBase::goal_count, py::arg("fluents"),
           "Count how many goal literals are achieved")
      .def("__eq__", &GoalBase::operator==)
      .def("__hash__", &GoalBase::hash)
      // Operators for goal chaining
      .def("__and__", [](const GoalPtr &self, const py::object &other) -> GoalPtr {
        GoalPtr other_goal;
        if (py::isinstance<Fluent>(other)) {
          other_goal = std::make_shared<LiteralGoal>(other.cast<Fluent>());
        } else {
          // Try cast to GoalPtr - works for Goal and all subclasses (AndGoal, OrGoal, etc.)
          try {
            other_goal = other.cast<GoalPtr>();
          } catch (const py::cast_error&) {
            throw py::type_error("unsupported operand type for &");
          }
        }
        // Flatten if possible
        std::vector<GoalPtr> children;
        if (self->get_type() == GoalType::AND) {
          for (const auto &child : self->children()) {
            children.push_back(child);
          }
        } else {
          children.push_back(self);
        }
        if (other_goal->get_type() == GoalType::AND) {
          for (const auto &child : other_goal->children()) {
            children.push_back(child);
          }
        } else {
          children.push_back(other_goal);
        }
        return std::make_shared<AndGoal>(std::move(children));
      })
      .def("__or__", [](const GoalPtr &self, const py::object &other) -> GoalPtr {
        GoalPtr other_goal;
        if (py::isinstance<Fluent>(other)) {
          other_goal = std::make_shared<LiteralGoal>(other.cast<Fluent>());
        } else {
          // Try cast to GoalPtr - works for Goal and all subclasses (AndGoal, OrGoal, etc.)
          try {
            other_goal = other.cast<GoalPtr>();
          } catch (const py::cast_error&) {
            throw py::type_error("unsupported operand type for |");
          }
        }
        // Flatten if possible
        std::vector<GoalPtr> children;
        if (self->get_type() == GoalType::OR) {
          for (const auto &child : self->children()) {
            children.push_back(child);
          }
        } else {
          children.push_back(self);
        }
        if (other_goal->get_type() == GoalType::OR) {
          for (const auto &child : other_goal->children()) {
            children.push_back(child);
          }
        } else {
          children.push_back(other_goal);
        }
        return std::make_shared<OrGoal>(std::move(children));
      });

  py::class_<TrueGoal, GoalBase, std::shared_ptr<TrueGoal>>(m, "TrueGoal")
      .def(py::init<>())
      .def("__repr__", [](const TrueGoal &) { return "TrueGoal()"; })
      .def(py::pickle(
          [](const TrueGoal &) {
            return py::make_tuple();  // No state needed
          },
          [](py::tuple t) {
            if (t.size() != 0)
              throw std::runtime_error("Invalid state for TrueGoal!");
            return std::make_shared<TrueGoal>();
          }));

  py::class_<FalseGoal, GoalBase, std::shared_ptr<FalseGoal>>(m, "FalseGoal")
      .def(py::init<>())
      .def("__repr__", [](const FalseGoal &) { return "FalseGoal()"; })
      .def(py::pickle(
          [](const FalseGoal &) {
            return py::make_tuple();  // No state needed
          },
          [](py::tuple t) {
            if (t.size() != 0)
              throw std::runtime_error("Invalid state for FalseGoal!");
            return std::make_shared<FalseGoal>();
          }));

  py::class_<LiteralGoal, GoalBase, std::shared_ptr<LiteralGoal>>(m, "LiteralGoal")
      .def(py::init<Fluent>(), py::arg("fluent"))
      .def("fluent", &LiteralGoal::fluent,
           "Get the fluent for this literal goal")
      .def("__repr__", [](const LiteralGoal &g) {
        std::ostringstream oss;
        const Fluent& f = g.fluent();
        oss << "(";
        if (f.is_negated()) oss << "not ";
        oss << f.name();
        for (const auto &arg : f.args()) {
          oss << " " << arg;
        }
        oss << ")";
        return oss.str();
      })
      .def(py::pickle(
          [](const LiteralGoal &g) {
            return py::make_tuple(g.fluent());
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error("Invalid state for LiteralGoal!");
            return std::make_shared<LiteralGoal>(t[0].cast<Fluent>());
          }));

  py::class_<AndGoal, GoalBase, std::shared_ptr<AndGoal>>(m, "AndGoal")
      .def(py::init<std::vector<GoalPtr>>(), py::arg("children"))
      .def("__repr__", [](const AndGoal &g) {
        std::ostringstream oss;
        oss << "(";
        bool first = true;
        for (const auto& child : g.children()) {
          if (!first) oss << " & ";
          first = false;
          oss << py::repr(py::cast(child)).cast<std::string>();
        }
        oss << ")";
        return oss.str();
      })
      .def(py::pickle(
          [](const AndGoal &g) {
            return py::make_tuple(py::cast(g.children()));
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error("Invalid state for AndGoal!");
            auto children = t[0].cast<std::vector<GoalPtr>>();
            return std::make_shared<AndGoal>(std::move(children));
          }));

  py::class_<OrGoal, GoalBase, std::shared_ptr<OrGoal>>(m, "OrGoal")
      .def(py::init<std::vector<GoalPtr>>(), py::arg("children"))
      .def("__repr__", [](const OrGoal &g) {
        std::ostringstream oss;
        oss << "(";
        bool first = true;
        for (const auto& child : g.children()) {
          if (!first) oss << " | ";
          first = false;
          oss << py::repr(py::cast(child)).cast<std::string>();
        }
        oss << ")";
        return oss.str();
      })
      .def(py::pickle(
          [](const OrGoal &g) {
            return py::make_tuple(py::cast(g.children()));
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error("Invalid state for OrGoal!");
            auto children = t[0].cast<std::vector<GoalPtr>>();
            return std::make_shared<OrGoal>(std::move(children));
          }));


  py::class_<MCTSPlanner>(m, "MCTSPlanner")
      .def(py::init<std::vector<Action>, double, double, double,
                    std::optional<double>>(),
           py::arg("all_actions"),
           py::arg("lambda_add") = 0.5,
           py::arg("lambda_max") = 0.0,
           py::arg("lambda_ff")  = 0.5,
           py::arg("dead_end_penalty") = py::none(),
           "Construct an MCTSPlanner. The lambda_* weights mix the additive "
           "(h_add), max (h_max), and relaxed-plan-cost (h_ff) heuristic "
           "components used during search; defaults are an even split between "
           "h_add and h_ff (0.5, 0.0, 0.5).\n\n"
           "dead_end_penalty: reward charged for a branch the relaxation "
           "proves cannot reach the goal (h = inf), as a flat cost -- the "
           "time and extra_cost the branch already spent are not added, so a "
           "slow failure is not ranked below a fast one. None (the default) "
           "keeps the legacy behavior of clamping h to "
           "HEURISTIC_CANNOT_FIND_GOAL_PENALTY, under which dead ends score "
           "*better* than reachable states and the search is drawn to them.")
      .def(
          "__call__",
          [](MCTSPlanner &self, const State &s,
             const GoalPtr &goal, int max_iterations,
             int max_depth, double c, double heuristic_multiplier,
             std::optional<HeuristicFn> heuristic_fn,
             double unreachable_penalty) {
            // Release the GIL for the search, but only when the leaf value is pure C++.
            // A Python heuristic_fn is called from inside mcts(), so the GIL has to be held
            // for the whole run; dropping it here would deadlock on the first leaf.
            return self(s, goal, max_iterations, max_depth, c, heuristic_multiplier,
                        heuristic_fn ? *heuristic_fn : nullptr, unreachable_penalty);
          },
          py::arg("state"), py::arg("goal"),
          py::arg("max_iterations") = 1000, py::arg("max_depth") = 20,
          py::arg("c") = 1.414, py::arg("heuristic_multiplier") = 5.0,
          py::arg("heuristic_fn") = py::none(),
          py::arg("unreachable_penalty") = HEURISTIC_CANNOT_FIND_GOAL_PENALTY,
          "Plan with a Goal object (supports complex AND/OR goals).\n\n"
          "heuristic_fn: optional callable(State) -> float used as the leaf value in place "
          "of the built-in FF mix. When given, the lambda_* weights are unused.\n"
          "unreachable_penalty: leaf value substituted when the heuristic reports the goal "
          "unreachable and no dead_end_penalty was set on the planner.")
      .def_property_readonly("lambda_add", &MCTSPlanner::lambda_add)
      .def_property_readonly("lambda_max", &MCTSPlanner::lambda_max)
      .def_property_readonly("lambda_ff",  &MCTSPlanner::lambda_ff)
      .def_property_readonly("dead_end_penalty", &MCTSPlanner::dead_end_penalty)
      .def("get_trace_from_last_mcts_tree", &MCTSPlanner::get_trace_from_last_mcts_tree,
           "Get the tree trace from the most recent MCTS planning call");

  m.def(
      "apply_effect_deterministic",
      [](std::unordered_set<Fluent> fluents, const GroundedEffect &effect) {
        auto triggered = collect_triggered_effects(effect, fluents);
        apply_effect_fluents(fluents, effect);
        return py::make_tuple(std::move(fluents), std::move(triggered));
      },
      py::arg("fluents"), py::arg("effect"),
      "Apply one effect's deterministic part to a fluent set, sharing the "
      "core's implementation: conditional branches are evaluated against the "
      "pre-effect fluents, then deletes apply before adds. Returns "
      "(new_fluents, triggered_sub_effects); probabilistic branches are left "
      "to the caller.");

  m.def("seed_planner_rng", &seed_mcts_rng, py::arg("seed"),
        "Seed the MCTS outcome-sampling RNG for reproducible planning "
        "(thread-local: applies to planner calls from the calling thread)");

  // ff_heuristic with Goal object
  m.def("ff_heuristic",
        [](const State &state, const GoalPtr &goal,
           const std::vector<Action> &all_actions,
           double lambda_add, double lambda_max, double lambda_ff,
           bool at_implies_found) {
          return ff_heuristic(state, goal.get(), all_actions, nullptr,
                              lambda_add, lambda_max, lambda_ff,
                              at_implies_found);
        },
        "Compute FF heuristic value for a state with a Goal object. "
        "lambda_add/lambda_max/lambda_ff mix the additive (h_add), max (h_max), "
        "and relaxed-plan-cost (h_ff) components; defaults are an even split "
        "between h_add and h_ff (0.5, 0.0, 0.5). When at_implies_found is true "
        "(default), any required `at <entity> <loc>` also requires a reachable "
        "`found <entity>`, so search goals need not be spelled out.",
        py::arg("state"), py::arg("goal"), py::arg("all_actions"),
        py::arg("lambda_add") = 0.5,
        py::arg("lambda_max") = 0.0,
        py::arg("lambda_ff")  = 0.5,
        py::arg("at_implies_found") = true);
}
