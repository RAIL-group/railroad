#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include "mrppddl/core.hpp"
#include "mrppddl/state.hpp"
#include "mrppddl/planner.hpp"


namespace py = pybind11;
using namespace mrppddl;

PYBIND11_MODULE(_bindings, m) {
    py::class_<Fluent>(m, "Fluent")
      .def(py::init([](const std::string& name, py::args args, bool negated) {
	std::vector<std::string> arg_list;
	for (const auto& arg : args) {
	  arg_list.push_back(arg.cast<std::string>());
	}
	return Fluent(name, std::move(arg_list), negated);
      }), py::arg("name"), py::arg("negated") = false)
      .def("__invert__", &Fluent::invert)
      .def("__eq__", &Fluent::operator==)
      .def("__hash__", &Fluent::hash)
      .def_property_readonly("name", &Fluent::name)
      .def_property_readonly("args", &Fluent::args)
      .def_property_readonly("negated", &Fluent::is_negated)
      .def("__repr__", [](const Fluent& f) {
	std::ostringstream oss;
	oss << "<Fluent: ";
	if (f.is_negated()) oss << "not ";
	oss << f.name();
	for (const auto& arg : f.args()) {
	  oss << " " << arg;
	}
	oss << ">";
	return oss.str();
      });

    // py::class_<GroundedEffectType>(m, "GroundedEffect")
    //   .def(py::init<double, std::unordered_set<Fluent>>(),
    // 	   py::arg("time"), py::arg("resulting_fluents"))
    //   .def_property_readonly("time", &GroundedEffectType::time)
    //   .def_property_readonly("resulting_fluents", &GroundedEffectType::resulting_fluents)
    //   .def("__lt__", &GroundedEffectType::operator<)
    //   .def("__repr__", [](const GroundedEffectType& ge) {
    // 	std::ostringstream oss;
    // 	oss << "<GroundedEffect t=" << ge.time() << ", size=" << ge.resulting_fluents().size() << ">";
    // 	return oss.str();
    //   });


// // ProbBranchWrapper
// py::class_<ProbBranchWrapper>(m, "ProbBranch")
//     .def(py::init<double, std::vector<GroundedEffectType>>(),
//          py::arg("prob"), py::arg("effects"))
//     .def_property_readonly("prob", &ProbBranchWrapper::prob)
//     .def_property_readonly("effects", &ProbBranchWrapper::effects)
//     .def("__repr__", [](const ProbBranchWrapper& b) {
//         std::ostringstream oss;
//         oss << "<ProbBranch p=" << b.prob()
//             << ", n_effects=" << b.effects().size() << ">";
//         return oss.str();
//     });

// GroundedEffectType
py::class_<GroundedEffectType>(m, "GroundedEffectType")
    .def(py::init<double,
	 std::unordered_set<Fluent>,
	 std::vector<std::pair<double, std::vector<GroundedEffectType>>>>(),
         py::arg("time"),
         py::arg("resulting_fluents") = std::unordered_set<Fluent>{},
	 py::arg("prob_effects") = std::vector<std::pair<double, std::vector<GroundedEffectType>>>{})
    .def_property_readonly("time", &GroundedEffectType::time)
    .def_property_readonly("resulting_fluents", &GroundedEffectType::resulting_fluents)
    .def_property_readonly("prob_effects", &GroundedEffectType::prob_effects)
    .def("is_probabilistic", &GroundedEffectType::is_probabilistic)
    .def("__str__", &GroundedEffectType::str)
    .def("__repr__", [](const GroundedEffectType& eff) {
        return "GroundedEffect(" + eff.str() + ")";
    })
    .def("__eq__", &GroundedEffectType::operator==)
    .def("__hash__", [](const GroundedEffectType& eff) {
        return static_cast<std::size_t>(eff.hash());
    });


    // py::class_<GroundedEffect, GroundedEffectType>(m, "GroundedEffect")
    //   .def(py::init<double, std::unordered_set<Fluent>>(),
    // 	   py::arg("time"), py::arg("resulting_fluents"))
    //   .def("__hash__", &GroundedEffect::hash)
    //   .def("__eq__", &GroundedEffect::operator==)
    //   .def("__str__", &GroundedEffect::str)
    //   .def("__repr__", [](const GroundedEffect& ge) {
    //     return "GroundedEffect(" + ge.str() + ")";
    //   });

    // py::class_<GroundedProbEffect, GroundedEffectType>(m, "GroundedProbEffect")
    //   .def(py::init<double, std::vector<ProbBranch>, std::unordered_set<Fluent>>(),
    // 	   py::arg("time"),
    // 	   py::arg("prob_effects"),
    // 	   py::arg("resulting_fluents") = std::unordered_set<Fluent>{})
    //   .def_property_readonly("prob_effects", &GroundedProbEffect::prob_effects)
    //   .def("__hash__", &GroundedProbEffect::hash)
    //   .def("__str__", &GroundedProbEffect::str)
    //   .def("__repr__", [](const GroundedProbEffect& gpe) {
    //     return "ProbEffects(" + gpe.str() + ")";
    //   });

    py::class_<Action>(m, "Action")
	.def(py::init<std::unordered_set<Fluent>, std::vector<GroundedEffectType>, std::string>(),
	    py::arg("preconditions"), py::arg("effects"), py::arg("name") = "anonymous")
	.def_property_readonly("name", &Action::name)
	.def_property_readonly("preconditions", &Action::preconditions)
	.def_property_readonly("effects", &Action::effects)
	.def_property_readonly("_pos_precond", &Action::pos_preconditions)
	.def_property_readonly("_neg_precond_flipped", &Action::neg_precond_flipped)
	.def("__str__", &Action::str)
	.def("__repr__", [](const Action& a) { return a.str(); });

    py::class_<State>(m, "State")
      .def(py::init<double,
	   std::unordered_set<Fluent>,
	   std::vector<std::pair<double, GroundedEffectType>>>(),
	   py::arg("time") = 0.0,
	   py::arg("fluents") = std::unordered_set<Fluent>{},
	   py::arg("upcoming_effects") = std::vector<std::pair<double, GroundedEffectType>>{})
      .def_property("time", &State::time, &State::set_time)
      .def_property_readonly("fluents", &State::fluents)
      .def_property_readonly("upcoming_effects", &State::upcoming_effects)
      .def("satisfies_precondition", &State::satisfies_precondition,
	   py::arg("action"), py::arg("relax") = false)
      .def("update_fluents", &State::update_fluents, py::arg("new_fluents"), py::arg("relax") = false)
      .def("copy", &State::copy)
      .def("copy_and_zero_out_time", &State::copy_and_zero_out_time)
      .def("queue_effect", &State::queue_effect)
      .def("pop_effect", &State::pop_effect)
      .def("__hash__", &State::hash)
      .def("__eq__", &State::operator==)
      .def("__lt__", &State::operator<)
      .def("__str__", &State::str)
      .def("__repr__", [](const State& s) { return s.str(); });
    py::class_<ProbBranchWrapper>(m, "ProbBranch")
      .def(py::init<double, std::vector<GroundedEffectType>>())
      .def_property_readonly("prob", &ProbBranchWrapper::prob)
      .def_property_readonly("effects", &ProbBranchWrapper::effects)
      .def("__getitem__", [](const ProbBranchWrapper& b, std::size_t i) -> py::object {
	if (i == 0) return py::float_(b.prob());
	if (i == 1) return py::cast(b.effects());
	throw py::index_error("ProbBranch index out of range");
      })
      .def("__repr__", [](const ProbBranchWrapper& b) {
	std::ostringstream oss;
	oss << "<ProbBranch p=" << b.prob()
	    << ", n_effects=" << b.effects().size() << ">";
	return oss.str();
      });


    m.def("transition",
	  [](const State& state, std::optional<std::reference_wrapper<const Action>> action, bool relax) {
	    return mrppddl::transition(state, action ? &action->get() : nullptr, relax);
	  },
	  py::arg("state"), py::arg("action") = std::nullopt, py::arg("relax") = false);

    m.def("get_next_actions", &get_next_actions,
	  py::arg("state"), py::arg("all_actions"),
	  "Return list of applicable actions for at least one free robot");

    m.def("make_goal_test", &mrppddl::make_goal_test, "Construct a goal-checking function",
	  py::arg("goal_fluents"));
    m.def("astar", &astar,
          py::arg("start_state"),
          py::arg("all_actions"),
          py::arg("is_goal_state"),
          py::arg("heuristic_fn") = nullptr,
          "Run A* search and return the action path");



    // py::bind_vector<std::vector<GroundedEffectType>>(m, "GroundedEffectTypeVec");
    // py::bind_vector<std::vector<ProbBranch>>(m, "GroundedProbBranchVec");
    // py::bind_tuple<ProbBranch>(m, "ProbBranch");
}
