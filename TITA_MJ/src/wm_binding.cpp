#include <Python.h>
#include <frameobject.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <mujoco/mujoco.h>

#include "WalkingManager.hpp"
#include "MPC.hpp"
#include "JointState.hpp"
#include "RobotState.hpp"
#include "JointCommand.hpp"
#include "walkingPlanner.hpp"

namespace py = pybind11;
using namespace labrob;

struct WalkingManagerResult {
    JointCommand cmd;
    SolutionMPC solution;
    infoPinocchio pinocchio_info;
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

    m.def("robot_state_from_mujoco", [](uintptr_t model_ptr, uintptr_t data_ptr) {
        mjModel* model = reinterpret_cast<mjModel*>(model_ptr);
        mjData* data   = reinterpret_cast<mjData*>(data_ptr);

        if (!model || !data)
            throw std::runtime_error("Errore: non è stato possibile ottenere i puntatori MuJoCo");


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

    py::class_<SolutionMPC::Com>(m, "Com")
        .def_readwrite("pos", &SolutionMPC::Com::pos)
        .def_readwrite("vel", &SolutionMPC::Com::vel)
        .def_readwrite("acc", &SolutionMPC::Com::acc)
        .def("__repr__", [](const SolutionMPC::Com &c) {
            std::stringstream ss;
            ss << "Com(pos=[" << c.pos.transpose() 
            << "], vel=[" << c.vel.transpose() 
            << "], acc=[" << c.acc.transpose() << "])";
            return ss.str();
        });

    py::class_<SolutionMPC::Pl>(m, "Pl")
        .def_readwrite("pos", &SolutionMPC::Pl::pos)
        .def_readwrite("vel", &SolutionMPC::Pl::vel)
        .def_readwrite("acc", &SolutionMPC::Pl::acc)
        .def("__repr__", [](const SolutionMPC::Pl &p) {
            std::stringstream ss;
            ss << "Pl(pos=[" << p.pos.transpose() 
            << "], vel=[" << p.vel.transpose() 
            << "], acc=[" << p.acc.transpose() << "])";
            return ss.str();
        });

    py::class_<SolutionMPC::Pr>(m, "Pr")
        .def_readwrite("pos", &SolutionMPC::Pr::pos)
        .def_readwrite("vel", &SolutionMPC::Pr::vel)
        .def_readwrite("acc", &SolutionMPC::Pr::acc)
        .def("__repr__", [](const SolutionMPC::Pr &p) {
            std::stringstream ss;
            ss << "Pr(pos=[" << p.pos.transpose() 
            << "], vel=[" << p.vel.transpose() 
            << "], acc=[" << p.acc.transpose() << "])";
            return ss.str();
        });
    
    py::class_<infoPinocchio>(m, "InfoPinocchio")
        .def(py::init<>())  
        .def_readwrite("p_com", &infoPinocchio::p_CoM)
        .def_readwrite("v_com", &infoPinocchio::v_CoM)
        .def_readwrite("a_com", &infoPinocchio::a_CoM)
        .def_readwrite("right_rcp", &infoPinocchio::right_rCP)
        .def_readwrite("left_rcp", &infoPinocchio::left_rCP)
        .def_readwrite("right_contact", &infoPinocchio::right_contact)
        .def_readwrite("left_contact", &infoPinocchio::left_contact)
        .def("__repr__", [](const infoPinocchio &self) {
            std::ostringstream oss;
            oss << "<InfoPinocchio "
                << "p_CoM=" << self.p_CoM.transpose() << ", "
                << "v_CoM=" << self.v_CoM.transpose() << ", "
                << "a_CoM=" << self.a_CoM.transpose() << ", "
                << "right_rCP=" << self.right_rCP.transpose() << ", "
                << "left_rCP=" << self.left_rCP.transpose() << ", "
                << "right_contact=" << self.right_contact.transpose() << ", "
                << "left_contact=" << self.left_contact.transpose()
                << ">";
            return oss.str();
        });

    py::class_<SolutionMPC>(m, "SolutionMPC")
        .def_readwrite("com", &SolutionMPC::com)
        .def_readwrite("pl", &SolutionMPC::pl)
        .def_readwrite("pr", &SolutionMPC::pr)
        .def_readwrite("theta", &SolutionMPC::theta)
        .def_readwrite("omega", &SolutionMPC::omega)
        .def_readwrite("alpha", &SolutionMPC::alpha)
        .def_readwrite("contact_force_left", &SolutionMPC::contact_force_left)
        .def_readwrite("contact_force_right", &SolutionMPC::contact_force_right)
        .def("__repr__", [](const SolutionMPC &s) {
            std::stringstream ss;
            ss << "SolutionMPC(\n"
                << "  com=" << py::repr(py::cast(s.com)) << ",\n"
                << "  pl=" << py::repr(py::cast(s.pl)) << ",\n"
                << "  pr=" << py::repr(py::cast(s.pr)) << ",\n"
                << "  contact_force_left=[" << s.contact_force_left.transpose() << "],\n"
                << "  contact_force_right=[" << s.contact_force_right.transpose() << "],\n"
                << "  theta=" << s.theta 
                << ", omega=" << s.omega 
                << ", alpha=" << s.alpha << ")";
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
        .def_readwrite("solution", &WalkingManagerResult::solution)
        .def_readwrite("pinocchio_info", &WalkingManagerResult::pinocchio_info);

    py::class_<walkingPlanner>(m, "WalkingPlanner")
        .def(py::init<double, double, double, double, double>(),
            py::arg("vel_lin"),
            py::arg("vel_ang"),
            py::arg("vel_z"),
            py::arg("z_min"),
            py::arg("z_max")
            )
        .def("get_variables", &walkingPlanner::getVariables)
        .def("get_xref_at_time_ms", &walkingPlanner::get_xref_at_time_ms)
        .def("get_uref_at_time_ms", &walkingPlanner::get_uref_at_time_ms)
        .def("get_x_ref", &walkingPlanner::get_x_ref)  
        .def("get_u_ref", &walkingPlanner::get_u_ref); 
    
    py::class_<WalkingManager>(m, "WalkingManager")
        .def(py::init<>())
        .def("init", [](WalkingManager &wm, const RobotState &state, py::dict &armatures, walkingPlanner& walking_planner) {
            
            std::map<std::string, double> armatures_map;
            for (auto item : armatures) {
                armatures_map[item.first.cast<std::string>()] = item.second.cast<double>();
            }
            WalkingManagerResult result;
            infoPinocchio pinocchio_info;
            bool ret = wm.init(state, armatures_map, walking_planner, pinocchio_info);
            result.pinocchio_info = pinocchio_info;
            return result;
        })

        .def("get_walking_planner",
            py::overload_cast<>(&WalkingManager::get_walking_planner, py::const_),
            py::return_value_policy::reference_internal)

        .def("update", [](WalkingManager &wm, const RobotState &state, const Eigen::Vector3d position_desired) {
            JointCommand cmd;
            SolutionMPC solution;
            infoPinocchio pinocchio_info;
            wm.update(state, position_desired, cmd, solution, pinocchio_info);

            WalkingManagerResult result;
            result.cmd = cmd;
            result.solution = solution;
            result.pinocchio_info = pinocchio_info;
            return result;
        });
}
