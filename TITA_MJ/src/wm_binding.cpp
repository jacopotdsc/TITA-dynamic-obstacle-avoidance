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
            ss << "<RobotState>\n";
            ss << "  Pos: [" << r.position.transpose() << "]\n";
            ss << "  Ori: [" << r.orientation.coeffs().transpose() << "]\n";
            ss << "  LinVel: [" << r.linear_velocity.transpose() << "]\n";
            ss << "  AngVel: [" << r.angular_velocity.transpose() << "]\n";
            
            //ss << "  Joints: " << r.joint_state.size() << " active\n";
            ss << "  Contacts: " << r.contact_points.size() << " active\n";
            ss << "  Total Force: [" << r.total_force.transpose() << "]";
            
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
        }, py::keep_alive<0, 1>());

    py::class_<WalkingManager>(m, "WalkingManager")
        .def(py::init<>())
        .def("init", [](WalkingManager &wm, const RobotState &state, py::dict &armatures) {
            
            std::map<std::string, double> armatures_map;
            for (auto item : armatures) {
                armatures_map[item.first.cast<std::string>()] = item.second.cast<double>();
            }
            return wm.init(state, armatures_map);
        })
        .def("update", [](WalkingManager &wm, const RobotState &state) {
            JointCommand cmd;
            wm.update(state, cmd);
            return cmd;
        });
}
