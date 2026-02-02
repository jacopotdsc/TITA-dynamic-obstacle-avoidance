#pragma once

#include <WholeBodyController.hpp>
#include <DesiredConfiguration.hpp>
#include <MPC.hpp>
#include <LQR.hpp>

#include <labrob_qpsolvers/qpsolvers.hpp>


namespace labrob {

struct infoPinocchio {
    Eigen::Vector3d p_CoM;
    Eigen::Vector3d v_CoM;
    Eigen::Vector3d a_CoM;

    Eigen::Vector3d right_rCP;
    Eigen::Vector3d left_rCP;
    Eigen::Vector3d right_contact;
    Eigen::Vector3d left_contact;
};

class WalkingManager {
 public:

  bool init(const labrob::RobotState& initial_robot_state, std::map<std::string, double> &armatures, const labrob::walkingPlanner& walkingPlanner, labrob::infoPinocchio& pinocchio_info);

  void update(
      const labrob::RobotState& robot_state,
      const Eigen::Vector3d position_desired,
      labrob::JointCommand& joint_command,
      labrob::SolutionMPC& sol,
      labrob::infoPinocchio& pinocchio_info
  );

  labrob::DesiredConfiguration des_configuration_;

  const labrob::walkingPlanner& get_walking_planner() const { return walkingPlanner_; }
  const labrob::MPC& get_mpc() const { return mpc_; }


 protected:
  pinocchio::Model robot_model_;
  pinocchio::Data robot_data_;
  pinocchio::FrameIndex right_leg4_idx_;
  pinocchio::FrameIndex left_leg4_idx_;

  double wheel_radius_;

  double controller_timestep_msec_;

  std::shared_ptr<labrob::WholeBodyController> whole_body_controller_ptr_;

private:

  double controller_frequency_;
  double t_msec_ = 0;


  labrob::walkingPlanner walkingPlanner_;
  labrob::MPC mpc_;


  // Log files:
  // std::ofstream mpc_timings_log_file_;
  std::ofstream state_log_file_;

}; 

} // end namespace labrob
