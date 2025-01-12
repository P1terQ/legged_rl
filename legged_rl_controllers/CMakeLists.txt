cmake_minimum_required(VERSION 3.10)
project(legged_rl_controllers)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Find catkin macros and libraries
find_package(catkin REQUIRED
        COMPONENTS
        roscpp
        legged_common
        legged_interface
        legged_estimation
        controller_interface
        ocs2_legged_robot_ros
        ocs2_self_collision_visualization
        angles
        )

find_package(onnxruntime REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
pkg_check_modules(pinocchio REQUIRED pinocchio)

catkin_package(
        INCLUDE_DIRS
        include
        LIBRARIES
        ${PROJECT_NAME}
        CATKIN_DEPENDS
        roscpp
        legged_common
        legged_interface
        legged_estimation
        controller_interface
        ocs2_legged_robot_ros
        ocs2_self_collision_visualization
        angles
        DEPENDS
        onnxruntime
        DEPENDS
        OpenCV
)

###########
## Build ##
###########

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${pinocchio_INCLUDE_DIRS}
)

link_directories(
        ${pinocchio_LIBRARY_DIRS}
)

add_library(${PROJECT_NAME}
        src/RLControllerBase.cpp
        src/LeggedRLController.cpp
        )

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        onnxruntime
        )
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

add_library(student_policy_controller
        src/RLControllerBase.cpp
        src/StudentPolicyController.cpp
        )

target_link_libraries(student_policy_controller
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        onnxruntime
        )
set_property(TARGET student_policy_controller PROPERTY CXX_STANDARD 14)

add_library(amp_controller
        src/RLControllerBase.cpp
        src/AmpController.cpp
        )

target_link_libraries(amp_controller
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        onnxruntime
        )
set_property(TARGET amp_controller PROPERTY CXX_STANDARD 14)

add_library(parkour_controller
        src/RLControllerBase.cpp
        src/ParkourController.cpp
        )

target_link_libraries(parkour_controller
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        ${OpenCV_LIBRARIES}
        onnxruntime
        )
set_property(TARGET parkour_controller PROPERTY CXX_STANDARD 14)

add_library(trot_controller
        src/RLControllerBase.cpp
        src/TrotController.cpp
        )

target_link_libraries(trot_controller
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        onnxruntime
        )
set_property(TARGET trot_controller PROPERTY CXX_STANDARD 14)

add_library(aliengo_controller
        src/RLControllerBase.cpp
        src/AliengoController.cpp
        )

target_link_libraries(aliengo_controller
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        onnxruntime
        )
set_property(TARGET aliengo_controller PROPERTY CXX_STANDARD 14)

add_library(biped_controller
        src/BipedController.cpp
        )

target_link_libraries(biped_controller
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        onnxruntime
        )
set_property(TARGET biped_controller PROPERTY CXX_STANDARD 14)

add_library(biped_gait_controller
        src/BipedGaitController.cpp
        )

target_link_libraries(biped_gait_controller
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        onnxruntime
        )
set_property(TARGET biped_gait_controller PROPERTY CXX_STANDARD 14)


add_library(biped_vision_controller
        src/BipedController.cpp
        src/BipedVisionController.cpp
        )

target_link_libraries(biped_vision_controller
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        ${OpenCV_LIBRARIES}
        onnxruntime
        )
set_property(TARGET biped_vision_controller PROPERTY CXX_STANDARD 14)


#########################
###   CLANG TOOLING   ###
#########################
find_package(cmake_clang_tools QUIET)
if (cmake_clang_tools_FOUND)
    message(STATUS "Run clang tooling for target " ${PROJECT_NAME})
    add_clang_tooling(
            TARGETS ${PROJECT_NAME} legged_target_trajectories_publisher
            SOURCE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
            CT_HEADER_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
            CF_WERROR
    )
endif (cmake_clang_tools_FOUND)

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

# Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h"
        )

# Mark other files for installation
install(
        FILES legged_rl_controllers_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(
        FILES student_policy_controller_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(
        FILES amp_controller_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(
        FILES parkour_controller_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(
        FILES trot_controller_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(
        FILES biped_controller_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(
        FILES biped_vision_controller_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)