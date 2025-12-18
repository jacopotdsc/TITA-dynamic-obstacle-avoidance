#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "MPC.hpp"
#include "RobotState.hpp"


namespace py = pybind11;

PYBIND11_MODULE(mpc, m) {
  m.doc() = "pybind11 bindings for labrob::MPC";

  py::class_<labrob::MPC>(m, "MPC")
    .def(py::init<>())
    .def("solve", [](labrob::MPC &self, const Eigen::Matrix<double,10,1> &x0){ self.solve(x0); }, py::arg("x0"))
    .def("get_solution", [](labrob::MPC &self){
      auto s = self.get_solution();
      py::dict out;
      out["com_pos"] = s.com.pos;
      out["com_vel"] = s.com.vel;
      out["com_acc"] = s.com.acc;
      out["zmp_pos"] = s.pc.pos;
      out["zmp_vel"] = s.pc.vel;
      out["zmp_acc"] = s.pc.acc;
      return out;
    });
}