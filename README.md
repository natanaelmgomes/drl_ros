drl_ros
===========

# Disclaimer and License

The DRL package is supported under ROS Melodic (Ubuntu 18.04).

This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.

The source code is released under the **MIT License**.


# Package Overview

The package is the result of my Master Thesis (not published yet) and it is a work under development.
It consists of Webots simulation environment, control layer and a deep reinforcement learning module using convolutional neural network. It is intended to be used with a Univeral Robots UR3e, Robotiq Gripper 2f-85 and Intel RealSense D435.

Author: Natanael M Gomes

# Installation

TODO.

# Usage

To bring the Webots simulation to ROS open the Environment.wbl and run:

roslaunch ur_e_webots ur3e_joint_limited.launch
roslaunch gripper SimGripperKuka.launch
roslaunch integrator Camera.launch
roslaunch integrator WebotsSupervisor.launch
roslaunch integrator Watchdog.launch

To bring the real UR3e run:
roslaunch ur_robot_driver ur3e_bringup.launch limited:=true robot_ip:=192.168.56.2
roslaunch ur3_e_moveit_config ur3_e_moveit_planning_execution.launch limited:=true
roslaunch realsense2_camera rs_camera.launch

# Calibration

The calibration is done with a AruCo marker available in the package.
