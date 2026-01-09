#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "WalkingManager.hpp"
#include "MPC.hpp"
#include "JointState.hpp"
#include "RobotState.hpp"
#include "JointCommand.hpp"

namespace py = pybind11;
using namespace labrob;

struct WalkingManagerResult {
    JointCommand cmd;
    SolutionMPC solution;
};

uintptr_t get_mujoco_ptr(py::object obj) {
    // 1. Se è già un intero (l'utente ha passato l'indirizzo a mano)
    if (py::isinstance<py::int_>(obj)) {
        return obj.cast<uintptr_t>();
    }
    
    // 2. Prova attributi noti (mujoco-py usa .ptr, altri usano ._address o .address)
    if (py::hasattr(obj, "ptr")) {
        return obj.attr("ptr").cast<uintptr_t>();
    }
    if (py::hasattr(obj, "_address")) { // Spesso usato nei binding interni
        return obj.attr("_address").cast<uintptr_t>();
    }
    if (py::hasattr(obj, "address")) {
        return obj.attr("address").cast<uintptr_t>();
    }

    throw std::runtime_error("Impossibile estrarre l'indirizzo di memoria dall'oggetto MuJoCo. "
                             "L'oggetto non ha attributi .ptr, ._address o .address e non è un int.");
}

PYBIND11_MODULE(wm, m) {
    m.doc() = "Python bindings for WalkingManager";

    m.def("robot_state_from_mujoco", [](py::object model_obj, py::object data_obj) {
        uintptr_t model_ptr = get_mujoco_ptr(model_obj);
        uintptr_t data_ptr = get_mujoco_ptr(data_obj);

        mjModel* model = reinterpret_cast<mjModel*>(model_ptr);
        mjData* data = reinterpret_cast<mjData*>(data_ptr);
        return robot_state_from_mujoco(model, data);
    }, "Converte lo stato di MuJoCo in RobotState", 
       py::arg("model_address"), py::arg("data_address"));

    py::class_<JointData>(m, "JointData")
        .def(py::init<>())
        .def_readwrite("pos", &JointData::pos)
        .def_readwrite("vel", &JointData::vel)
        .def_readwrite("acc", &JointData::acc)
        .def_readwrite("eff", &JointData::eff);
    
    py::class_<JointState>(m, "JointState")
        .def(py::init<>())
        .def("__getitem__", [](JointState &js, const std::string &key) {
            return js[key]; 
        })
        .def("__setitem__", [](JointState &js, const std::string &key, const JointData &val) {
            js[key] = val;
        })
        .def("__iter__", [](JointState &js) {
            return py::make_iterator(js.begin(), js.end());
        }, py::keep_alive<0, 1>())

        //.def("__len__", [](JointState &js) {
        //    return js.size();
        //})

        .def("__repr__", [](JointCommand &jc) {
            std::stringstream ss;
            ss << "JointCommand(\n";
            for (const auto &pair : jc) {
                ss << "  " << pair.first << ": " << std::fixed << std::setprecision(4) << pair.second << "\n";
            }
            ss << ")";
            return ss.str();
        });

    py::class_<RobotState>(m, "RobotState")
        .def(py::init<>())
        .def_readwrite("position", &RobotState::position)
        .def_property("orientation",
            [](const RobotState &r) { 
                return r.orientation.coeffs(); 
            },
            [](RobotState &r, const Eigen::Vector4d &v) {
                r.orientation.coeffs() = v;   
            }
        )
        .def_readwrite("linear_velocity", &RobotState::linear_velocity)
        .def_readwrite("angular_velocity", &RobotState::angular_velocity)
        .def_readwrite("joint_state", &RobotState::joint_state)
        .def_readwrite("total_force", &RobotState::total_force)
        .def_readwrite("contact_points", &RobotState::contact_points)
        .def_readwrite("contact_forces", &RobotState::contact_forces)
        
        .def("__repr__", [](const RobotState &r) {
            std::stringstream ss;
            ss << "  Pos: [" << r.position.transpose() << "]\n";
            ss << "  Ori: [" << r.orientation.coeffs().transpose() << "]\n";
            ss << "  LinVel: [" << r.linear_velocity.transpose() << "]\n";
            ss << "  AngVel: [" << r.angular_velocity.transpose() << "]\n";
            
            //ss << "  Joints: " << r.joint_state.size() << " active\n";
            ss << "  Contacts: " << r.contact_points.size() << " active\n";
            ss << "  Total Force: [" << r.total_force.transpose() << "]";
            
            return ss.str();
        });
    
    py::class_<SolutionMPC::Com>(m, "Com")
        .def_readwrite("pos", &SolutionMPC::Com::pos)
        .def_readwrite("vel", &SolutionMPC::Com::vel)
        .def_readwrite("acc", &SolutionMPC::Com::acc)

        .def("__repr__", [](const SolutionMPC::Com &c) {
            std::stringstream ss;
            ss << "  pos: [" << c.pos.transpose() << "]\n";
            ss << "  vel: [" << c.vel.transpose() << "]\n";
            ss << "  acc: [" << c.acc.transpose() << "]\n";
            return ss.str();
        });

    py::class_<SolutionMPC::Pc>(m, "Pc")
        .def_readwrite("pos", &SolutionMPC::Pc::pos)
        .def_readwrite("vel", &SolutionMPC::Pc::vel)
        .def_readwrite("acc", &SolutionMPC::Pc::acc)

        .def("__repr__", [](const SolutionMPC::Pc &p) {
            std::stringstream ss;
            ss << "  pos: [" << p.pos.transpose() << "]\n";
            ss << "  vel: [" << p.vel.transpose() << "]\n";
            ss << "  acc: [" << p.acc.transpose() << "]\n";
            return ss.str();
        });

    py::class_<SolutionMPC>(m, "SolutionMPC")
        .def(py::init<>())
        .def_readwrite("com", &SolutionMPC::com)
        .def_readwrite("pc", &SolutionMPC::pc)

        .def("__repr__", [](const SolutionMPC &r) {
            std::stringstream ss;
            ss << py::repr(py::cast(r.com)).cast<std::string>() << "\n";
            ss << py::repr(py::cast(r.pc)).cast<std::string>() << "\n";            
            return ss.str();
        });

    py::class_<JointCommand>(m, "JointCommand")
        .def(py::init<>())
        .def("__getitem__", [](const JointCommand &jc, const std::string &key) {
            return jc[key];
        })
        .def("__setitem__", [](JointCommand &jc, const std::string &key, double value) {
            jc[key] = value;
        })
        .def("__iter__", [](JointCommand &jc) {
            return py::make_iterator(jc.begin(), jc.end());
        }, py::keep_alive<0, 1>())

        .def("__repr__", [](JointCommand &jc) {
            std::stringstream ss;
            ss << "[ ";
            bool first = true;
            for (const auto &pair : jc) {
                if (!first) ss << ", ";
                ss << "(" << pair.first << ":" << std::fixed << std::setprecision(4) << pair.second << ")";
                first = false;
            }
            ss << " ]";
            return ss.str();
        });

    py::class_<WalkingManagerResult>(m, "WalkingManagerResult")
        .def_readwrite("cmd", &WalkingManagerResult::cmd)
        .def_readwrite("solution", &WalkingManagerResult::solution);

    py::class_<WalkingManager>(m, "WalkingManager")
        .def(py::init<>())
        .def("init", [](WalkingManager &wm, const RobotState &state, py::dict &armatures) {
            
            std::map<std::string, double> armatures_map;
            for (auto item : armatures) {
                armatures_map[item.first.cast<std::string>()] = item.second.cast<double>();
            }
            return wm.init(state, armatures_map);
        })

        .def("update", [](WalkingManager &wm, const RobotState &state, Eigen::Vector3d &position_desired) {
            JointCommand cmd;
            SolutionMPC solution;
            wm.update(state, position_desired, cmd, solution);

            WalkingManagerResult result;
            result.cmd = cmd;
            result.solution = solution;
            return result;
        });
}
