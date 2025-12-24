#define MRPPDDL_USE_PYBIND
#include "mrppddl/core.hpp"
#include "mrppddl/ff_heuristic.hpp"
#include "mrppddl/planner.hpp"
#include "mrppddl/state.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace mrppddl;

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
      });

  // GroundedEffect
  py::class_<GroundedEffect, std::shared_ptr<GroundedEffect>>(m,
                                                              "GroundedEffect")
      .def(py::init<double, std::unordered_set<Fluent>,
                    std::vector<std::pair<
                        double,
                        std::vector<std::shared_ptr<const GroundedEffect>>>>>(),
           py::arg("time"),
           py::arg("resulting_fluents") = std::unordered_set<Fluent>{},
           py::arg("prob_effects") =
               std::vector<std::pair<double, std::vector<GroundedEffect>>>{})
      .def_property_readonly("time", &GroundedEffect::time)
      .def_property_readonly("resulting_fluents",
                             &GroundedEffect::resulting_fluents)
      .def_property_readonly("prob_effects", &GroundedEffect::prob_effects)
      .def_property_readonly("is_probabilistic",
                             &GroundedEffect::is_probabilistic)
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

            return py::make_tuple(
                eff.time(),
                py::cast(eff.resulting_fluents()), // std::unordered_set<Fluent>
                pickled_prob_effects // List of (double, list of GroundedEffect)
            );
          },
          [](py::tuple t) {
            if (t.size() != 3)
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

            return std::make_shared<GroundedEffect>(time, std::move(fluents),
                                                    std::move(prob_effects));
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

  m.def(
      "transition",
      [](const State &state,
         std::optional<std::reference_wrapper<const Action>> action,
         bool relax) {
        return mrppddl::transition(state, action ? &action->get() : nullptr,
                                   relax);
      },
      py::arg("state"), py::arg("action") = std::nullopt,
      py::arg("relax") = false);

  m.def("get_next_actions", &get_next_actions, py::arg("state"),
        py::arg("all_actions"),
        "Return list of applicable actions for at least one free robot");
  m.def("get_usable_actions", &get_usable_actions_fluent_list, py::arg("input_state"),
	py::arg("goal_fluents"), py::arg("all_actions"));

  m.def("make_goal_test", &mrppddl::make_goal_test,
        "Construct a goal-checking function", py::arg("goal_fluents"));
  m.def("astar", &astar, py::arg("start_state"), py::arg("all_actions"),
        py::arg("is_goal_state"), py::arg("heuristic_fn") = nullptr,
        "Run A* search and return the action path");

  py::class_<GoalFn>(m, "GoalFn")
      .def(py::init<const std::unordered_set<Fluent> &>(),
           py::arg("goal_fluents"),
           "Create a goal function from a set of goal fluents")
      .def("__call__", &GoalFn::operator(),
           py::arg("fluents"),
           "Check if all goal fluents are present in the given set")
      .def("goal_count", &GoalFn::goal_count,
           py::arg("active_fluents"),
           "Count how many goal fluents are present in the active set");

  py::class_<MCTSPlanner>(m, "MCTSPlanner")
      .def(py::init<std::vector<Action>>(), py::arg("all_actions"))
      .def(
          "__call__",
          [](MCTSPlanner &self, const State &s,
             const std::unordered_set<Fluent> &goal_fluents, int max_iterations,
             int max_depth, double c, double heuristic_multiplier) {
            return self(s, goal_fluents, max_iterations, max_depth, c, heuristic_multiplier);
          },
          py::arg("state"), py::arg("goal_fluents"),
          py::arg("max_iterations") = 1000, py::arg("max_depth") = 20,
          py::arg("c") = 1.414, py::arg("heuristic_multiplier") = 5.0)
      .def("get_trace_from_last_mcts_tree", &MCTSPlanner::get_trace_from_last_mcts_tree,
           "Get the tree trace from the most recent MCTS planning call");
  m.def("ff_heuristic",
        [](const State &state, const std::unordered_set<Fluent> &goal_fluents,
           const std::vector<Action> &all_actions) {
	  auto goal_fn = GoalFn(goal_fluents);
          return ff_heuristic(state, &goal_fn, all_actions);
        },
        "Compute FF heuristic value for a state",
        py::arg("state"), py::arg("goal_fluents"), py::arg("all_actions"));
}
