#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mrppddl/core.hpp"

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
    py::class_<ActiveFluents>(m, "ActiveFluents")
      .def(py::init<std::unordered_set<Fluent>>(), py::arg("fluents") = std::unordered_set<Fluent>{})
      .def("update", &ActiveFluents::update, py::arg("fluents"), py::arg("relax") = false)
      .def("__contains__", &ActiveFluents::contains)
      .def("__iter__", [](const ActiveFluents& af) {
	return py::make_iterator(af.begin(), af.end());
      }, py::keep_alive<0, 1>())
      .def("__eq__", &ActiveFluents::operator==)
      .def("__hash__", &ActiveFluents::hash)
      .def_property_readonly("fluents", &ActiveFluents::fluents)
      .def("__repr__", [](const ActiveFluents& af) {
	return af.repr();
      });
}
