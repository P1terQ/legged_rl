//
// Created by luohx on 23-11-15.
//
#include <pinocchio/fwd.hpp>

#include "legged_rl_controllers/BipedController.h"

#include <std_msgs/Float32MultiArray.h>

#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <pluginlib/class_list_macros.hpp>

namespace legged {
bool BipedController::init(hardware_interface::RobotHW* robotHw, ros::NodeHandle& controllerNH) {
  // Load policy model and rl cfg
  if (!loadModel(controllerNH)) {
    ROS_ERROR_STREAM("[BipedController] Failed to load the model. Ensure the path is correct and accessible.");
    return false;
  }
  if (!loadRLCfg(controllerNH)) {
    ROS_ERROR_STREAM("[BipedController] Failed to load the rl config. Ensure the yaml is correct and accessible.");
    return false;
  }

  // Hardware interface
  auto* hybridJointInterface = robotHw->get<HybridJointInterface>();
  std::vector<std::string> jointNames = {"abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"};
  for (const auto& jointName : jointNames) {
    hybridJointHandles_.push_back(hybridJointInterface->getHandle(jointName));
  }
  imuSensorHandles_ = robotHw->get<hardware_interface::ImuSensorInterface>()->getHandle("unitree_imu");

  auto* contactInterface = robotHw->get<ContactSensorInterface>();
  std::vector<std::string> footNames = {"foot_L_Link", "foot_R_Link"};
  for (const auto& footName : footNames) {
    contactHandles_.push_back(contactInterface->getHandle(footName));
  }

  cmdVelSub_ = controllerNH.subscribe("/cmd_vel", 1, &BipedController::cmdVelCallback, this);
  gTStateSub_ = controllerNH.subscribe("/ground_truth/state", 1, &BipedController::stateUpdateCallback, this);
  joyInfoSub_ = controllerNH.subscribe("/joy", 1000, &BipedController::joyInfoCallback, this);
  switchCtrlClient_ = controllerNH.serviceClient<controller_manager_msgs::SwitchController>("/controller_manager/switch_controller");
  jointDebugPub_ = controllerNH.advertise<std_msgs::Float32MultiArray>("/debug_info", 1, true);
  obsDebugPub_ = controllerNH.advertise<std_msgs::Float32MultiArray>("/obs_debug_info", 1, true);

  return true;
}

void BipedController::starting(const ros::Time& time) {
  for (auto& hybridJointHandle : hybridJointHandles_) {
    initJointAngles_.push_back(hybridJointHandle.getPosition());
  }
  scalar_t durationSecs = 2.0;
  standDuration_ = durationSecs * 1000.0;
  standPercent_ += 1 / standDuration_;
  mode_ = Mode::LIE;

  loopCount_ = 0;
}

void BipedController::update(const ros::Time& time, const ros::Duration& period) {
  switch (mode_) {
    case Mode::LIE:
      handleLieMode();
      break;
    case Mode::STAND:
      handleStandMode();
      break;
    case Mode::WALK:
      handleWalkMode();
      break;
    default:
      ROS_ERROR_STREAM("Unexpected mode encountered: " << static_cast<int>(mode_));
      break;
  }

  loopCount_++;
}

void BipedController::handleLieMode() {
  if (standPercent_ < 1) {
    for (int j = 0; j < hybridJointHandles_.size(); j++) {
      scalar_t pos_des = initJointAngles_[j] * (1 - standPercent_) + defaultJointAngles_[j] * standPercent_;
      hybridJointHandles_[j].setCommand(pos_des, 0, 60, 4, 0);
    }
    standPercent_ += 1 / standDuration_;
  } else {
    mode_ = Mode::STAND;
  }
}

void BipedController::handleStandMode() {
  //  if (loopCount_ > 5000) {
  //    mode_ = Mode::WALK;
  //  }
}

void BipedController::handleWalkMode() {
  // compute observation & actions
  if (loopCount_ % robotCfg_.controlCfg.decimation == 0) {
    computeObservation();
    computeEncoder();
    computeActions();
    // limit action range
    scalar_t actionMin = -robotCfg_.clipActions;
    scalar_t actionMax = robotCfg_.clipActions;
    std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                   [actionMin, actionMax](scalar_t x) { return std::max(actionMin, std::min(actionMax, x)); });
  }

  // set action
  std_msgs::Float32MultiArray debugInfoArray;
  debugInfoArray.data.resize(hybridJointHandles_.size() * 6);
  vector_t jointPos(hybridJointHandles_.size()), jointVel(hybridJointHandles_.size());
  for (size_t i = 0; i < hybridJointHandles_.size(); ++i) {
    jointPos(i) = hybridJointHandles_[i].getPosition();
    jointVel(i) = hybridJointHandles_[i].getVelocity();
  }
  for (int i = 0; i < hybridJointHandles_.size(); i++) {
    scalar_t actionMin =
        jointPos(i) - defaultJointAngles_(i, 0) +
        (robotCfg_.controlCfg.damping * jointVel(i) - robotCfg_.controlCfg.user_torque_limit) / robotCfg_.controlCfg.stiffness;
    scalar_t actionMax =
        jointPos(i) - defaultJointAngles_(i, 0) +
        (robotCfg_.controlCfg.damping * jointVel(i) + robotCfg_.controlCfg.user_torque_limit) / robotCfg_.controlCfg.stiffness;
    actions_[i] = std::max(actionMin / robotCfg_.controlCfg.actionScale,
                           std::min(actionMax / robotCfg_.controlCfg.actionScale, (scalar_t)actions_[i]));
    scalar_t pos_des = actions_[i] * robotCfg_.controlCfg.actionScale + defaultJointAngles_(i, 0);
    hybridJointHandles_[i].setCommand(pos_des, 0, robotCfg_.controlCfg.stiffness, robotCfg_.controlCfg.damping, 0);

    lastActions_(i, 0) = actions_[i];

    debugInfoArray.data[i * 6] = pos_des;
    debugInfoArray.data[i * 6 + 1] = hybridJointHandles_[i].getPosition();
    debugInfoArray.data[i * 6 + 2] = debugInfoArray.data[i * 6] - debugInfoArray.data[i * 6 + 1];
    debugInfoArray.data[i * 6 + 3] = hybridJointHandles_[i].getVelocity();
    debugInfoArray.data[i * 6 + 4] = hybridJointHandles_[i].getEffort();
    debugInfoArray.data[i * 6 + 5] = robotCfg_.controlCfg.stiffness * (pos_des - debugInfoArray.data[i * 6 + 1]) -
                                     robotCfg_.controlCfg.damping * debugInfoArray.data[i * 6 + 3];
  }
  jointDebugPub_.publish(debugInfoArray);
}

void BipedController::computeActions() {
  // create input tensor object
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  std::vector<tensor_element_t> combined_obs;
  for (const auto& item : observations_) {
    combined_obs.push_back(item);
  }
  for (const auto& item : encoderOut_) {
    combined_obs.push_back(item);
  }
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, combined_obs.data(), combined_obs.size(),
                                                                   policyInputShapes_[0].data(), policyInputShapes_[0].size()));
  // run inference
  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues =
      policySessionPtr_->Run(runOptions, policyInputNames_.data(), inputValues.data(), 1, policyOutputNames_.data(), 1);

  for (int i = 0; i < actionsSize_; i++) {
    actions_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

void BipedController::computeEncoder() {
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, proprioHistoryBuffer_.data(), proprioHistoryBuffer_.size(),
                                                                   encoderInputShapes_[0].data(), encoderInputShapes_[0].size()));
  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues =
      encoderSessionPtr_->Run(runOptions, encoderInputNames_.data(), inputValues.data(), 1, encoderOutputNames_.data(), 1);

  for (int i = 0; i < encoderOutputSize_; i++) {
    encoderOut_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

void BipedController::computeObservation() {
  Eigen::Quaternion<scalar_t> quat;
  for (size_t i = 0; i < 4; ++i) {
    quat.coeffs()(i) = imuSensorHandles_.getOrientation()[i];
  }

  vector3_t zyx = quatToZyx(quat);
  matrix_t inverseRot = getRotationMatrixFromZyxEulerAngles(zyx).inverse();

  // linear velocity (base frame)
  vector3_t baseLinVel = inverseRot * baseLinVel_;
  // Angular velocity
  vector3_t baseAngVel(imuSensorHandles_.getAngularVelocity()[0], imuSensorHandles_.getAngularVelocity()[1],
                       imuSensorHandles_.getAngularVelocity()[2]);

  // Projected gravity
  vector3_t gravityVector(0, 0, -1);
  vector3_t projectedGravity(inverseRot * gravityVector);

  vector3_t zyx_(imu_orientation_offset[0], imu_orientation_offset[1], imu_orientation_offset[2]);
  matrix_t Rot_ = getRotationMatrixFromZyxEulerAngles(zyx_);
  baseAngVel = Rot_ * baseAngVel;
  projectedGravity = Rot_ * projectedGravity;

  // command
  vector3_t command = command_;

  // dof position and dof velocity
  vector_t jointPos(hybridJointHandles_.size()), jointVel(hybridJointHandles_.size());
  for (size_t i = 0; i < hybridJointHandles_.size(); ++i) {
    jointPos(i) = hybridJointHandles_[i].getPosition();
    jointVel(i) = hybridJointHandles_[i].getVelocity();
  }

  vector_t gait(4);
  gait << 2.0, 0.5, 0.5, 0.1;  // trot
  gait_index_ += 0.02 * gait(0);
  if (gait_index_ > 1.0) {
    gait_index_ = 0.0;
  }
  vector_t gait_clock(2);
  gait_clock << sin(gait_index_ * 2 * M_PI), cos(gait_index_ * 2 * M_PI);

  // actions
  vector_t actions(lastActions_);

  RLRobotCfg::ObsScales& obsScales = robotCfg_.obsScales;
  matrix_t commandScaler = Eigen::DiagonalMatrix<scalar_t, 3>(obsScales.linVel, obsScales.linVel, obsScales.angVel);

  vector_t obs(observationSize_);
  std_msgs::Float32MultiArray debugInfoArray;
  debugInfoArray.data.resize(observationSize_);
  // clang-format off
  obs << projectedGravity,
      baseAngVel,
      (jointPos - defaultJointAngles_) * obsScales.dofPos,
      jointVel * obsScales.dofVel,
      commandScaler * command, 0.625,
      actions,
      gait_clock,
      gait;
  // clang-format on
  for (size_t i = 0; i < observationSize_; i++) {
    debugInfoArray.data[i] = obs[i];
  }
  obsDebugPub_.publish(debugInfoArray);

  if (isfirstRecObs_) {
    int64_t inputSize =
        std::accumulate(encoderInputShapes_[0].begin(), encoderInputShapes_[0].end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    proprioHistoryBuffer_.resize(inputSize);
    for (size_t i = 0; i < obsHistoryLength_; i++) {
      proprioHistoryBuffer_.segment(i * observationSize_, observationSize_) = obs.cast<tensor_element_t>();
    }
    isfirstRecObs_ = false;
  }
  proprioHistoryBuffer_.head(proprioHistoryBuffer_.size() - observationSize_) =
      proprioHistoryBuffer_.tail(proprioHistoryBuffer_.size() - observationSize_);
  proprioHistoryBuffer_.tail(observationSize_) = obs.cast<tensor_element_t>();

  for (size_t i = 0; i < obs.size(); i++) {
    observations_[i] = static_cast<tensor_element_t>(obs(i));
    // std::cout << "observations" << i << " " << observations_[i] << std::endl;
  }
  // Limit observation range
  scalar_t obsMin = -robotCfg_.clipObs;
  scalar_t obsMax = robotCfg_.clipObs;
  std::transform(observations_.begin(), observations_.end(), observations_.begin(),
                 [obsMin, obsMax](scalar_t x) { return std::max(obsMin, std::min(obsMax, x)); });
}

bool BipedController::loadModel(ros::NodeHandle& nh) {
  ROS_INFO_STREAM("load policy model");

  std::string policyModelPath;
  std::string encoderModelPath;
  if (!nh.getParam("/policyModelPath", policyModelPath) || !nh.getParam("/encoderModelPath", encoderModelPath)) {
    ROS_ERROR_STREAM("Get policy path fail from param server, some error occur!");
    return false;
  }

  // create env
  onnxEnvPrt_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController"));
  // create session
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads(1);
  sessionOptions.SetInterOpNumThreads(1);

  Ort::AllocatorWithDefaultOptions allocator;
  // policy session
  std::cout << "load policy from" << policyModelPath.c_str() << std::endl;
  policySessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, policyModelPath.c_str(), sessionOptions);
  policyInputNames_.clear();
  policyOutputNames_.clear();
  policyInputShapes_.clear();
  policyOutputShapes_.clear();
  for (int i = 0; i < policySessionPtr_->GetInputCount(); i++) {
    policyInputNames_.push_back(policySessionPtr_->GetInputName(i, allocator));
    policyInputShapes_.push_back(policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::cerr << policySessionPtr_->GetInputName(i, allocator) << std::endl;
    std::vector<int64_t> shape = policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j != shape.size() - 1) {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  for (int i = 0; i < policySessionPtr_->GetOutputCount(); i++) {
    policyOutputNames_.push_back(policySessionPtr_->GetOutputName(i, allocator));
    std::cerr << policySessionPtr_->GetOutputName(i, allocator) << std::endl;
    policyOutputShapes_.push_back(policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::vector<int64_t> shape = policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j != shape.size() - 1) {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  // encoder session
  std::cout << "load encoder from" << encoderModelPath.c_str() << std::endl;
  encoderSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, encoderModelPath.c_str(), sessionOptions);
  encoderInputNames_.clear();
  encoderOutputNames_.clear();
  encoderInputShapes_.clear();
  encoderOutputShapes_.clear();
  for (int i = 0; i < encoderSessionPtr_->GetInputCount(); i++) {
    encoderInputNames_.push_back(encoderSessionPtr_->GetInputName(i, allocator));
    encoderInputShapes_.push_back(encoderSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::cerr << encoderSessionPtr_->GetInputName(i, allocator) << std::endl;
    std::vector<int64_t> shape = encoderSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j != shape.size() - 1) {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  for (int i = 0; i < encoderSessionPtr_->GetOutputCount(); i++) {
    encoderOutputNames_.push_back(encoderSessionPtr_->GetOutputName(i, allocator));
    std::cerr << encoderSessionPtr_->GetOutputName(i, allocator) << std::endl;
    encoderOutputShapes_.push_back(encoderSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::vector<int64_t> shape = encoderSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j != shape.size() - 1) {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  ROS_INFO_STREAM("Load Onnx model from successfully !!!");
  return true;
}

bool BipedController::loadRLCfg(ros::NodeHandle& nh) {
  RLRobotCfg::InitState& initState = robotCfg_.initState;
  RLRobotCfg::ControlCfg& controlCfg = robotCfg_.controlCfg;
  RLRobotCfg::ObsScales& obsScales = robotCfg_.obsScales;

  int error = 0;
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/L_HAA_joint", initState.L_HAA_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/L_HFE_joint", initState.L_HFE_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/L_KFE_joint", initState.L_KFE_joint));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/R_HAA_joint", initState.R_HAA_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/R_HFE_joint", initState.R_HFE_joint));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/R_KFE_joint", initState.R_KFE_joint));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/stiffness", controlCfg.stiffness));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/damping", controlCfg.damping));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/action_scale", controlCfg.actionScale));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/decimation", controlCfg.decimation));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/user_torque_limit", controlCfg.user_torque_limit));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/clip_scales/clip_observations", robotCfg_.clipObs));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/clip_scales/clip_actions", robotCfg_.clipActions));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/lin_vel", obsScales.linVel));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/ang_vel", obsScales.angVel));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/dof_pos", obsScales.dofPos));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/dof_vel", obsScales.dofVel));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/height_measurements", obsScales.heightMeasurements));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/actions_size", actionsSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/observations_size", observationSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/obs_history_length", obsHistoryLength_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/encoder_output_size", encoderOutputSize_));

  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/imu_orientation_offset/yaw", imu_orientation_offset[0]));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/imu_orientation_offset/pitch", imu_orientation_offset[1]));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/imu_orientation_offset/roll", imu_orientation_offset[2]));

  actions_.resize(actionsSize_);
  observations_.resize(observationSize_);
  encoderOut_.resize(encoderOutputSize_);
  std::cout << "actionsSize_ " << actionsSize_ << std::endl;
  std::cout << "observationSize_ " << observationSize_ << std::endl;
  std::cout << "encoderOutputSize_ " << encoderOutputSize_ << std::endl;
  std::cout << "obsHistoryLength_ " << obsHistoryLength_ << std::endl;

  command_.setZero();
  baseLinVel_.setZero();
  basePosition_.setZero();
  std::vector<scalar_t> defaultJointAngles{robotCfg_.initState.L_HAA_joint, robotCfg_.initState.L_HFE_joint,
                                           robotCfg_.initState.L_KFE_joint, robotCfg_.initState.R_HAA_joint,
                                           robotCfg_.initState.R_HFE_joint, robotCfg_.initState.R_KFE_joint};
  lastActions_.resize(actionsSize_);
  lastActions_.setZero();
  defaultJointAngles_.resize(defaultJointAngles.size());
  for (int i = 0; i < defaultJointAngles_.size(); i++) {
    defaultJointAngles_(i, 0) = defaultJointAngles[i];
  }
  return (error == 0);
}

void BipedController::cmdVelCallback(const geometry_msgs::Twist& msg) {
  command_(0) = msg.linear.x;
  command_(1) = msg.linear.y;
  command_(2) = msg.angular.z;
}

void BipedController::stateUpdateCallback(const nav_msgs::Odometry& msg) {
  baseLinVel_(0) = msg.twist.twist.linear.x;
  baseLinVel_(1) = msg.twist.twist.linear.y;
  baseLinVel_(2) = msg.twist.twist.linear.z;
}

void BipedController::joyInfoCallback(const sensor_msgs::Joy& msg) {
  if (msg.buttons[7] == 1) {
    std::cout << "You have pressed the start button!!!!" << std::endl;
    // set the string start_controllers to controllers/legged_controller
    switchCtrlSrv_.request.start_controllers = {"controllers/biped_controller"};
    switchCtrlSrv_.request.stop_controllers = {""};
    switchCtrlSrv_.request.strictness = switchCtrlSrv_.request.BEST_EFFORT;
    switchCtrlSrv_.request.start_asap = true;
    switchCtrlSrv_.request.timeout = 0.0;
    switchCtrlClient_.call(switchCtrlSrv_);
  } else if (msg.buttons[0] == 1) {
    std::cout << "You have pressed the stop button!!!!" << std::endl;
    // set the string stop_controllers to controllers/legged_controller
    switchCtrlSrv_.request.start_controllers = {""};
    switchCtrlSrv_.request.stop_controllers = {"controllers/biped_controller"};
    switchCtrlSrv_.request.strictness = switchCtrlSrv_.request.BEST_EFFORT;
    switchCtrlSrv_.request.start_asap = true;
    switchCtrlSrv_.request.timeout = 0.0;
    switchCtrlClient_.call(switchCtrlSrv_);
  }
  if (msg.buttons[1] == 1) {
    if (mode_ == Mode::STAND) {
      mode_ = Mode::WALK;
    }
  }
}

}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::BipedController, controller_interface::ControllerBase)
