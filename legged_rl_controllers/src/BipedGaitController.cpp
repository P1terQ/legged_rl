//
// Created by luohx on 23-11-15.
//
#include <pinocchio/fwd.hpp>

#include "legged_rl_controllers/BipedGaitController.h"

#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <pluginlib/class_list_macros.hpp>

namespace legged {
bool BipedGaitController::init(hardware_interface::RobotHW* robotHw, ros::NodeHandle& controllerNH) {
  // Load policy model and rl cfg
  if (!loadModel(controllerNH)) {
    ROS_ERROR_STREAM("[BipedGaitController] Failed to load the model. Ensure the path is correct and accessible.");
    return false;
  }
  if (!loadRLCfg(controllerNH)) {
    ROS_ERROR_STREAM("[BipedGaitController] Failed to load the rl config. Ensure the yaml is correct and accessible.");
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

  cmdVelSub_ = controllerNH.subscribe("/cmd_vel", 1, &BipedGaitController::cmdVelCallback, this);
  gTStateSub_ = controllerNH.subscribe("/ground_truth/state", 1, &BipedGaitController::stateUpdateCallback, this);

  return true;
}

void BipedGaitController::starting(const ros::Time& time) {
  loopCount_ = 0;
}

void BipedGaitController::update(const ros::Time& time, const ros::Duration& period) {
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
  for (int i = 0; i < hybridJointHandles_.size(); i++) {
    scalar_t pos_des = actions_[i] * robotCfg_.controlCfg.actionScale + defaultJointAngles_(i, 0);
    hybridJointHandles_[i].setCommand(pos_des, 0, robotCfg_.controlCfg.stiffness, robotCfg_.controlCfg.damping, 0);
    lastActions_(i, 0) = actions_[i];
  }

  loopCount_++;
}

void BipedGaitController::computeActions() {
  // create input tensor object
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  std::vector<tensor_element_t> combined_obs;
  for (const auto& item : encoderOut_) {
    combined_obs.push_back(item);
  }
  for (const auto& item : observations_) {
    combined_obs.push_back(item);
  }
  for (const auto& item : commands_) {
    combined_obs.push_back(item);
  }
  for (const auto& item : gaitGeneratorOut_) {
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

void BipedGaitController::computeEncoder() {
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

  Ort::MemoryInfo gaitGeneratorMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> gaitGeneratorInputValues;
  std::vector<tensor_element_t> combined_input;
  //   for (const auto& item : observations_) {
  //     combined_input.push_back(item);
  //   }
  for (const auto& item : proprioHistoryVector_) {
    combined_input.push_back(item);
  }
  for (const auto& item : commands_) {
    combined_input.push_back(item);
  }
  gaitGeneratorInputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(gaitGeneratorMemoryInfo, combined_input.data(),
                                                                                combined_input.size(), gaitGeneratorInputShapes_[0].data(),
                                                                                gaitGeneratorInputShapes_[0].size()));
  // for teacher student encoder
  //   gaitGeneratorInputValues.push_back(
  //       Ort::Value::CreateTensor<tensor_element_t>(gaitGeneratorMemoryInfo, proprioHistoryBuffer_.data(), proprioHistoryBuffer_.size(),
  //                                                  gaitGeneratorInputShapes_[0].data(), gaitGeneratorInputShapes_[0].size()));
  Ort::RunOptions gaitGeneratorRunOptions;
  std::vector<Ort::Value> gaitGeneratorOutputValues = gaitGeneratorSessionPtr_->Run(
      gaitGeneratorRunOptions, gaitGeneratorInputNames_.data(), gaitGeneratorInputValues.data(), 1, gaitGeneratorOutputNames_.data(), 1);

  tensor_element_t gaitGeneratorOutNorm = 0;
  for (int i = 0; i < gaitGeneratorOutputSize_; i++) {
    gaitGeneratorOut_[i] = *(gaitGeneratorOutputValues[0].GetTensorMutableData<tensor_element_t>() + i);
    gaitGeneratorOutNorm += gaitGeneratorOut_[i] * gaitGeneratorOut_[i];
  }
  // for gait generator
  for (int i = 0; i < gaitGeneratorOutputSize_; i++) {
    gaitGeneratorOut_[i] /= sqrt(gaitGeneratorOutNorm);
  }
}

void BipedGaitController::computeObservation() {
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
  // clang-format off
  obs << baseAngVel * obsScales.angVel,                     //
    projectedGravity,                                     //
    (jointPos - defaultJointAngles_) * obsScales.dofPos,  //
    jointVel * obsScales.dofVel,                          //
    actions;

  command = commandScaler * command;
  // clang-format on

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

  // clang-format on
  //   printf("observation\n");
  for (size_t i = 0; i < obs.size(); i++) {
    observations_[i] = static_cast<tensor_element_t>(obs(i));
    // std::cout << "observations" << i << " " << observations_[i] << std::endl;
  }
  for (size_t i = 0; i < command.size(); i++) {
    commands_[i] = static_cast<tensor_element_t>(command(i));
    // std::cout << "commands_" << i << " " << commands_[i] << std::endl;
  }
  for (size_t i = 0; i < proprioHistoryBuffer_.size(); i++) {
    proprioHistoryVector_[i] = static_cast<tensor_element_t>(proprioHistoryBuffer_(i));
    // std::cout << "proprioHistoryVector_" << i << " " << proprioHistoryVector_[i] << std::endl;
  }
  // Limit observation range
  scalar_t obsMin = -robotCfg_.clipObs;
  scalar_t obsMax = robotCfg_.clipObs;
  std::transform(observations_.begin(), observations_.end(), observations_.begin(),
                 [obsMin, obsMax](scalar_t x) { return std::max(obsMin, std::min(obsMax, x)); });
}

bool BipedGaitController::loadModel(ros::NodeHandle& nh) {
  ROS_INFO_STREAM("load policy model");

  std::string policyModelPath;
  std::string encoderModelPath;
  std::string gaitGeneratorModelPath;
  if (!nh.getParam("/policyModelPath", policyModelPath) || !nh.getParam("/encoderModelPath", encoderModelPath) ||
      !nh.getParam("/gaitGeneratorModelPath", gaitGeneratorModelPath)) {
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

  // gait generator session
  std::cout << "load gait generator from" << gaitGeneratorModelPath.c_str() << std::endl;
  gaitGeneratorSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, gaitGeneratorModelPath.c_str(), sessionOptions);
  gaitGeneratorInputNames_.clear();
  gaitGeneratorOutputNames_.clear();
  gaitGeneratorInputShapes_.clear();
  gaitGeneratorOutputShapes_.clear();
  for (int i = 0; i < gaitGeneratorSessionPtr_->GetInputCount(); i++) {
    gaitGeneratorInputNames_.push_back(gaitGeneratorSessionPtr_->GetInputName(i, allocator));
    gaitGeneratorInputShapes_.push_back(gaitGeneratorSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::cerr << gaitGeneratorSessionPtr_->GetInputName(i, allocator) << std::endl;
    std::vector<int64_t> shape = gaitGeneratorSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j != shape.size() - 1) {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  for (int i = 0; i < gaitGeneratorSessionPtr_->GetOutputCount(); i++) {
    gaitGeneratorOutputNames_.push_back(gaitGeneratorSessionPtr_->GetOutputName(i, allocator));
    std::cerr << gaitGeneratorSessionPtr_->GetOutputName(i, allocator) << std::endl;
    gaitGeneratorOutputShapes_.push_back(gaitGeneratorSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::vector<int64_t> shape = gaitGeneratorSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j != shape.size() - 1) {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  ROS_INFO_STREAM("Load Onnx model successfully !!!");
  return true;
}

bool BipedGaitController::loadRLCfg(ros::NodeHandle& nh) {
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
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/commands_size", commandsSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/obs_history_length", obsHistoryLength_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/encoder_output_size", encoderOutputSize_));
  error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/gait_generator_output_size", gaitGeneratorOutputSize_));
  encoderIntputSize_ = obsHistoryLength_ * observationSize_;

  actions_.resize(actionsSize_);
  observations_.resize(observationSize_);
  commands_.resize(commandsSize_);
  encoderOut_.resize(encoderOutputSize_);
  proprioHistoryVector_.resize(observationSize_ * obsHistoryLength_);
  gaitGeneratorOut_.resize(gaitGeneratorOutputSize_);
  std::cout << "actionsSize_ " << actionsSize_ << std::endl;
  std::cout << "observationSize_ " << observationSize_ << std::endl;
  std::cout << "commandsSize_ " << commandsSize_ << std::endl;
  std::cout << "encoderOutputSize_ " << encoderOutputSize_ << std::endl;
  std::cout << "gaitGeneratorOutputSize_ " << gaitGeneratorOutputSize_ << std::endl;
  std::cout << "obsHistoryLength_ " << obsHistoryLength_ << std::endl;

  command_.setZero();
  baseLinVel_.setZero();
  basePosition_.setZero();
  std::vector<scalar_t> defaultJointAngles{robotCfg_.initState.L_HAA_joint, robotCfg_.initState.L_HFE_joint,
                                           robotCfg_.initState.L_KFE_joint, robotCfg_.initState.R_HAA_joint,
                                           robotCfg_.initState.R_HFE_joint, robotCfg_.initState.R_KFE_joint};
  lastActions_.resize(actionsSize_);
  defaultJointAngles_.resize(defaultJointAngles.size());
  for (int i = 0; i < defaultJointAngles_.size(); i++) {
    defaultJointAngles_(i, 0) = defaultJointAngles[i];
  }
  return (error == 0);
}

void BipedGaitController::cmdVelCallback(const geometry_msgs::Twist& msg) {
  command_(0) = msg.linear.x;
  command_(1) = msg.linear.y;
  command_(2) = msg.angular.z;
}

void BipedGaitController::stateUpdateCallback(const nav_msgs::Odometry& msg) {
  baseLinVel_(0) = msg.twist.twist.linear.x;
  baseLinVel_(1) = msg.twist.twist.linear.y;
  baseLinVel_(2) = msg.twist.twist.linear.z;
}

}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::BipedGaitController, controller_interface::ControllerBase)
