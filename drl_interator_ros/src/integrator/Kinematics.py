#!/usr/bin/env python3.6

# Python
import numpy as np
import argparse
from math import pi, sin, cos, asin, acos, atan2, radians, sqrt
import cmath
import sys
import os
import time
import yaml

# imports for keyboard input
import select
import tty
import termios

# ROS
import rospy
import actionlib
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Trigger
from ur_dashboard_msgs.srv import GetRobotMode, GetSafetyMode

#
from scipy.spatial.transform import Rotation as Rot


class URKinematics(object):
    """
    Class to deal with UR kinematics
    # the two functions movej and movel tries to mimic the UR original behavior
    """

    def __init__(self, args):
        """
        Constructor
        """
        # Parameters
        self.debug = False
        self.is_sim = args.get('simulation')
        self.precision = args.get('precision') if args.get('precision') else 0.02  # standard precision 2cm

        # print(self.precision)
        # inital state is zeros, but will be updated asynchronous
        self.joints_state = [0] * 6

        # d (unit: mm)
        d1 = 0.15185
        d2 = d3 = 0
        d4 = 0.13105
        d5 = 0.08535
        d6 = 0.0921 + 0.1755 - 0.016 # there is 175.5mm between the robot and the gripper center
        # https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf

        # a (unit: mm)
        a1 = a4 = a5 = a6 = 0
        a2 = -0.24355
        a3 = -0.2132

        # List type of D-H parameter
        self.d = np.array([d1, d2, d3, d4, d5, d6])  # unit: mm
        self.a = np.array([a1, a2, a3, a4, a5, a6])  # unit: mm
        self.alpha = np.array([pi / 2, 0, 0, pi / 2, -pi / 2, 0])  # unit: radian

        # ------------------ old tf transformations.py functions
        # It is not included on tf2 see https://github.com/ros/geometry2/issues/222

        # epsilon for testing whether a number is close to zero
        self._EPS = np.finfo(float).eps * 4.0

        self.client = None

        rospy.Subscriber("/joint_states", JointState, self.callback)

        # print(self.is_sim)
        self.JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                            'wrist_2_joint', 'wrist_3_joint']
        self.p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.p1 = [pi, -2.26893, 2.26893, -pi / 2, -pi / 2, 0]
        self.p2 = [3.687479067620226, -0.8842977790133202, 0.6497038813419902,
                   -1.9608433912231122, -1.0204688666195634, -0.273701231943166]

        if self.is_sim:
            # simulated
            self.client = actionlib.SimpleActionClient('/follow_joint_trajectory', FollowJointTrajectoryAction)

        else:
            # Real
            self.client = actionlib.SimpleActionClient('/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                                                       FollowJointTrajectoryAction)
            # Try to init the robot
            rospy.wait_for_service('/ur_hardware_interface/dashboard/brake_release')
            try:
                self.service_brake_release = rospy.ServiceProxy('/ur_hardware_interface/dashboard/brake_release',
                                                                Trigger)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)

            rospy.wait_for_service('/ur_hardware_interface/dashboard/get_robot_mode')
            try:
                self.service_get_mode = rospy.ServiceProxy('ur_hardware_interface/dashboard/get_robot_mode',
                                                           GetRobotMode)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)

            rospy.wait_for_service('/ur_hardware_interface/dashboard/connect')
            try:
                self.service_connect = rospy.ServiceProxy('ur_hardware_interface/dashboard/connect', Trigger)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)

            rospy.wait_for_service('/ur_hardware_interface/dashboard/quit')
            try:
                self.service_quit = rospy.ServiceProxy('ur_hardware_interface/dashboard/quit', Trigger)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)
            # Safety mode
            # NORMAL = 1
            # REDUCED = 2
            # PROTECTIVE_STOP = 3
            # RECOVERY = 4
            # SAFEGUARD_STOP = 5
            # SYSTEM_EMERGENCY_STOP = 6
            # ROBOT_EMERGENCY_STOP = 7
            # VIOLATION = 8
            # FAULT = 9
            # VALIDATE_JOINT_ID = 10
            # UNDEFINED_SAFETY_MODE = 11
            # AUTOMATIC_MODE_SAFEGUARD_STOP = 12
            # SYSTEM_THREE_POSITION_ENABLING_STOP = 13
            rospy.wait_for_service('/ur_hardware_interface/dashboard/get_safety_mode')
            try:
                self.get_safety_mode = rospy.ServiceProxy('ur_hardware_interface/dashboard/get_safety_mode', GetSafetyMode)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)

            rospy.wait_for_service('/ur_hardware_interface/dashboard/close_safety_popup')
            try:
                self.close_safety_popup = rospy.ServiceProxy('ur_hardware_interface/dashboard/close_safety_popup', Trigger)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)

            rospy.wait_for_service('/ur_hardware_interface/dashboard/unlock_protective_stop')
            try:
                self.unlock_protective_stop = rospy.ServiceProxy('ur_hardware_interface/dashboard/unlock_protective_stop',
                                                             Trigger)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)
            # except Exception as e:
            #     rospy.logerr("General fail: %s" % e)

            # restart the dashboard connection is the best way to go
            self.service_quit()
            time.sleep(0.1)
            connected = self.service_connect()
            # rospy.logerr(connected)

            # get root mode
            mode = self.service_get_mode()

            # rospy.logerr(mode)

            # NO_CONTROLLER = -1
            # DISCONNECTED = 0
            # CONFIRM_SAFETY = 1
            # BOOTING = 2
            # POWER_OFF = 3
            # POWER_ON = 4
            # IDLE = 5
            # BACKDRIVE = 6
            # RUNNING = 7
            # UPDATING_FIRMWARE = 8

            # ensure the brakes are released
            if mode.robot_mode.mode != 7:
                brake = self.service_brake_release()

                if brake.success:
                    rospy.loginfo('Releasing breakes...')
                    mode = self.service_get_mode()
                    while mode.robot_mode.mode != 7:
                        mode = self.service_get_mode()
                        time.sleep(0.5)

                else:
                    rospy.logwarn('Please, release the brakes manually to continue')
                    mode = self.service_get_mode()
                    while mode.robot_mode.mode != 7:
                        mode = self.service_get_mode()
                        time.sleep(0.5)

            # service to stop the program
            rospy.wait_for_service('/ur_hardware_interface/dashboard/stop')
            try:
                self.service_stop = rospy.ServiceProxy('/ur_hardware_interface/dashboard/stop', Trigger)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)
            # service to start the program
            rospy.wait_for_service('/ur_hardware_interface/dashboard/play')
            try:
                self.service_play = rospy.ServiceProxy('/ur_hardware_interface/dashboard/play', Trigger)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)

            # also restart the program is the best way to go
            self.service_stop()
            time.sleep(0.1)
            obj = self.service_play()

            if obj.success:
                rospy.loginfo('Running program...')
            else:
                rospy.logwarn('Please, run the program manually to continue')
        # end of real

        rospy.loginfo("Waiting for UR server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to UR server")

        self.g0 = FollowJointTrajectoryGoal()
        self.g0.trajectory = JointTrajectory()
        self.g0.trajectory.joint_names = self.JOINT_NAMES
        self.g0.trajectory.points = [
            JointTrajectoryPoint(positions=self.p0, velocities=[0] * 6, time_from_start=rospy.Duration(0, 50000))]

        self.g1 = FollowJointTrajectoryGoal()
        self.g1.trajectory = JointTrajectory()
        self.g1.trajectory.joint_names = self.JOINT_NAMES
        self.g1.trajectory.points = [
            JointTrajectoryPoint(positions=self.p1, velocities=[0] * 6, time_from_start=rospy.Duration(0, 50000))]

        self.g2 = FollowJointTrajectoryGoal()
        self.g2.trajectory = JointTrajectory()
        self.g2.trajectory.joint_names = self.JOINT_NAMES
        self.g2.trajectory.points = [
            JointTrajectoryPoint(positions=self.p2, velocities=[0] * 6, time_from_start=rospy.Duration(0, 50000))]

        self.done = True

    def callback(self, data):
        """
        callback function to update the joints states
        """
        if self.is_sim:
            self.joints_state = data.position
        else:

            tmp0 = data.position[0]
            tmp1 = data.position[1]
            tmp2 = data.position[2]
            tmp3 = data.position[3]
            tmp4 = data.position[4]
            tmp5 = data.position[5]
            self.joints_state = [tmp2, tmp1, tmp0, tmp3, tmp4, tmp5]
        return

    def __del__(self):
        if not self.is_sim:
            self.service_play.close()
            self.service_stop.close()
            self.service_get_mode.close()
            self.service_brake_release.close()

    def send_goal(self, goal):
        """
        Receives 'zero' or 'stop' or a set of joints positions
        Args:
            goal: 'zero' or 'stop'
                  or
                  list of joints positions in radians
        Return:
            None
        """
        if goal == 'zero':
            self.client.send_goal(self.g0)
        elif goal == 'stop':
            self.client.send_goal(self.g1)
        elif goal == 'drop':
            self.client.send_goal(self.g2)
        else:
            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names = self.JOINT_NAMES
            g.trajectory.points = [
                JointTrajectoryPoint(positions=goal, velocities=[0] * 6, time_from_start=rospy.Duration(0, 5000000))]
            # print(g)
            self.client.send_goal(g)

    def wait_for_result(self, timeout=rospy.Duration(1)):
        """
        Repeat the client function
        """
        self.client.wait_for_result(timeout)

    def cancel_goal(self):
        """
        Repeat the client function
        """
        self.client.cancel_goal()

    @staticmethod
    def unit_vector(data, axis=None, out=None):
        """Return ndarray normalized by length, i.e. eucledian norm, along axis.
        >> v0 = np.random.random(3)
        >> v1 = unit_vector(v0)
        >> np.allclose(v1, v0 / np.linalg.norm(v0))
        True
        >> v0 = np.random.rand(5, 4, 3)
        >> v1 = unit_vector(v0, axis=-1)
        >> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
        >> np.allclose(v1, v2)
        True
        >> v1 = unit_vector(v0, axis=1)
        >> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
        >> np.allclose(v1, v2)
        True
        >> v1 = np.empty((5, 4, 3), dtype=np.float64)
        >> unit_vector(v0, axis=1, out=v1)
        >> np.allclose(v1, v2)
        True
        >> list(unit_vector([]))
        []
        >> list(unit_vector([1.0]))
        [1.0]
        """
        if out is None:
            data = np.array(data, dtype=np.float64, copy=True)
            if data.ndim == 1:
                data /= sqrt(np.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data * data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data

    def quaternion_matrix(self, quaternion):
        """Return homogeneous rotation matrix from quaternion.

        >> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
        >> np.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
        True

        """
        q = np.array(quaternion[:4], dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < self._EPS:
            return np.identity(4)
        q *= sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array((
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)

    def rotation_matrix(self, angle, direction, point=None):
        """Return matrix to rotate about axis defined by point and direction.
        >> angle = (random.random() - 0.5) * (2*pi)
        >> direc = np.random.random(3) - 0.5
        >> point = np.random.random(3) - 0.5
        >> R0 = rotation_matrix(angle, direc, point)
        >> R1 = rotation_matrix(angle-2*pi, direc, point)
        >> is_same_transform(R0, R1)
        True
        >> R0 = rotation_matrix(angle, direc, point)
        >> R1 = rotation_matrix(-angle, -direc, point)
        >> is_same_transform(R0, R1)
        True
        >> I = np.identity(4, np.float64)
        >> np.allclose(I, rotation_matrix(pi*2, direc))
        True
        >> np.allclose(2., np.trace(rotation_matrix(pi/2,
        ...                                                direc, point)))
        True
        """
        sina = sin(angle)
        cosa = cos(angle)
        direction = self.unit_vector(direction[:3])
        # rotation matrix around unit vector
        r = np.array(((cosa, 0.0, 0.0),
                      (0.0, cosa, 0.0),
                      (0.0, 0.0, cosa)), dtype=np.float64)
        r += np.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        r += np.array(((0.0, -direction[2], direction[1]),
                       (direction[2], 0.0, -direction[0]),
                       (-direction[1], direction[0], 0.0)),
                      dtype=np.float64)
        m = np.identity(4)
        m[:3, :3] = r
        if point is not None:
            # rotation not around origin
            point = np.array(point[:3], dtype=np.float64, copy=False)
            m[:3, 3] = point - np.dot(r, point)
        return m

    @staticmethod
    def quaternion_from_matrix(matrix):
        """Return quaternion from rotation matrix.

        >> R = rotation_matrix(0.123, (1, 2, 3))
        >> q = quaternion_from_matrix(R)
        >> np.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
        True

        """
        q = np.empty((4,), dtype=np.float64)
        m = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        t = np.trace(m)
        if t > m[3, 3]:
            q[3] = t
            q[2] = m[1, 0] - m[0, 1]
            q[1] = m[0, 2] - m[2, 0]
            q[0] = m[2, 1] - m[1, 2]
        else:
            i, j, k = 0, 1, 2
            if m[1, 1] > m[0, 0]:
                i, j, k = 1, 2, 0
            if m[2, 2] > m[i, i]:
                i, j, k = 2, 0, 1
            t = m[i, i] - (m[j, j] + m[k, k]) + m[3, 3]
            q[i] = t
            q[j] = m[i, j] + m[j, i]
            q[k] = m[k, i] + m[i, k]
            q[3] = m[k, j] - m[j, k]
        q *= 0.5 / sqrt(t * m[3, 3])
        return q

    # ----- Modified functions from https://github.com/littleblakew/UR_kinematics_solver
    # Auxiliary Functions

    def ur2ros(self, ur_pose):
        """Transform pose from UR format to ROS Pose format.
        Args:
            ur_pose: A pose in UR format [px, py, pz, rx, ry, rz]
            (type: list)
        Returns:
            An HTM (type: Pose).
        """

        # ROS pose
        ros_pose = Pose()

        # ROS position
        ros_pose.position.x = ur_pose[0]
        ros_pose.position.y = ur_pose[1]
        ros_pose.position.z = ur_pose[2]

        # Ros orientation
        angle = sqrt(ur_pose[3] ** 2 + ur_pose[4] ** 2 + ur_pose[5] ** 2)
        direction = [i / angle for i in ur_pose[3:6]]
        np_t = self.rotation_matrix(angle, direction)
        np_q = self.quaternion_from_matrix(np_t)
        ros_pose.orientation.x = np_q[0]
        ros_pose.orientation.y = np_q[1]
        ros_pose.orientation.z = np_q[2]
        ros_pose.orientation.w = np_q[3]

        return ros_pose

    def ros2np(self, ros_pose):
        """Transform pose from ROS Pose format to np.array format.
        Args:
            ros_pose: A pose in ROS Pose format (type: Pose)
        Returns:
            An HTM (type: np.array).
        """

        # orientation
        np_pose = self.quaternion_matrix([ros_pose.orientation.x, ros_pose.orientation.y,
                                          ros_pose.orientation.z, ros_pose.orientation.w])

        # position
        np_pose[0][3] = ros_pose.position.x
        np_pose[1][3] = ros_pose.position.y
        np_pose[2][3] = ros_pose.position.z

        return np_pose

    def np2ros(self, np_pose):
        """Transform pose from np.array format to ROS Pose format.
        Args:
            np_pose: A pose in np.array format (type: np.array)
        Returns:
            An HTM (type: Pose).
        """

        # ROS pose
        ros_pose = Pose()

        # ROS position
        ros_pose.position.x = np_pose[0, 3]
        ros_pose.position.y = np_pose[1, 3]
        ros_pose.position.z = np_pose[2, 3]

        # ROS orientation
        np_q = self.quaternion_from_matrix(np_pose)
        ros_pose.orientation.x = np_q[0]
        ros_pose.orientation.y = np_q[1]
        ros_pose.orientation.z = np_q[2]
        ros_pose.orientation.w = np_q[3]

        return ros_pose

    @staticmethod
    def select(q_sols, q_d, w=None):
        """Select the optimal solutions among a set of feasible joint value
           solutions.
        Args:
            q_sols: A set of feasible joint value solutions (unit: radian)
            q_d: A list of desired joint value solution (unit: radian)
            w: A list of weight corresponding to robot joints
        Returns:
            A list of optimal joint value solution.
        """

        if w is None:
            w = [1] * 6
        error = []
        for q in q_sols:
            error.append(sum([w[i] * (q[i] - q_d[i]) ** 2 for i in range(6)]))

        return q_sols[error.index(min(error))]

    # noinspection PyDeprecation
    def htm(self, i, theta):
        """Calculate the HTM between two links.
        Args:
            i: A target index of joint value.
            theta: A list of joint value solution. (unit: radian)
        Returns:
            An HTM of Link l w.r.t. Link l-1, where l = i + 1.
        """

        # noinspection PyDeprecation
        rot_z = np.matrix(np.identity(4))
        rot_z[0, 0] = rot_z[1, 1] = cos(theta[i])
        rot_z[0, 1] = -sin(theta[i])
        rot_z[1, 0] = sin(theta[i])

        trans_z = np.matrix(np.identity(4))
        trans_z[2, 3] = self.d[i]

        trans_x = np.matrix(np.identity(4))
        trans_x[0, 3] = self.a[i]

        rot_x = np.matrix(np.identity(4))
        rot_x[1, 1] = rot_x[2, 2] = cos(self.alpha[i])
        rot_x[1, 2] = -sin(self.alpha[i])
        rot_x[2, 1] = sin(self.alpha[i])

        a_i = rot_z * trans_z * trans_x * rot_x

        return a_i

    # Forward Kinematics
    # noinspection PyDeprecation
    def fwd_kin(self, theta, i_unit='r', o_unit='n'):
        """Solve the HTM based on a list of joint values.
        Args:
            theta: A list of joint values. (unit: radian)
            i_unit: Output format. 'r' for radian; 'd' for degree.
            o_unit: Output format. 'n' for np.array; 'p' for ROS Pose.
        Returns:
            The HTM of end-effector joint w.r.t. base joint
        """

        t_06 = np.matrix(np.identity(4))

        if i_unit == 'd':
            theta = [radians(i) for i in theta]

        for i in range(6):
            t_06 *= self.htm(i, theta)

        if o_unit == 'n':
            return t_06
        elif o_unit == 'p':
            return self.np2ros(t_06)

    @staticmethod
    def ros2ur(ros_pose):
        """Transform pose from ROS Pose format to UR format.
        Args:
            Pose object
        Returns:
            ur_pose: A pose in UR format [px, py, pz, rx, ry, rz]
            (type: list)
        """

        # UR list position
        ur_pose = [ros_pose.position.x, ros_pose.position.y, ros_pose.position.z]

        # Quaternion to Yaw, Pitch and Roll
        x = ros_pose.orientation.x
        y = ros_pose.orientation.y
        z = ros_pose.orientation.z
        w = ros_pose.orientation.w
        ll = (x ** 2 + y ** 2 + z ** 2 + w ** 2) ** 0.5
        w = w / ll
        x = x / ll
        y = y / ll
        z = z / ll
        roll = atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
        if roll < 0:
            roll += 2 * pi

        temp = w * y - z * x
        if temp >= 0.5:
            temp = 0.5
        elif temp <= -0.5:
            temp = -0.5
        else:
            pass
        pitch = asin(2 * temp)
        yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        if yaw < 0:
            yaw += 2 * pi

        ur_pose.append(yaw)
        ur_pose.append(pitch)
        ur_pose.append(roll)

        return ur_pose

    # ************************************************** INVERSE KINEMATICS
    # ----  https://github.com/mc-capolei/python-Universal-robot-kinematics

    # noinspection PyDeprecation
    def ah(self, n, th, c):
        t_a = np.matrix(np.identity(4), copy=False)
        t_a[0, 3] = np.matrix(self.a)[0, n - 1]
        t_d = np.matrix(np.identity(4), copy=False)
        t_d[2, 3] = np.matrix(self.d)[0, n - 1]
        rzt = np.matrix([[cos(th[n - 1, c]), -sin(th[n - 1, c]), 0, 0],
                         [sin(th[n - 1, c]), cos(th[n - 1, c]), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], copy=False)
        rxa = np.matrix([[1, 0, 0, 0],
                         [0, cos(np.matrix(self.alpha)[0, n - 1]), -sin(np.matrix(self.alpha)[0, n - 1]), 0],
                         [0, sin(np.matrix(self.alpha)[0, n - 1]), cos(np.matrix(self.alpha)[0, n - 1]), 0],
                         [0, 0, 0, 1]], copy=False)
        a_i = t_d * rzt * t_a * rxa
        return a_i

    # noinspection PyDeprecation
    def inv_kine(self, desired_pos):  # T60
        """
        Function to calculate inverse kinematics
        input HTM matrix

        """
        th = np.matrix(np.zeros((6, 8)))
        p_05 = (desired_pos * np.matrix([0, 0, - self.d[5], 1]).T - np.matrix([0, 0, 0, 1]).T)

        # **** theta1 ****

        psi = atan2(p_05[2 - 1, 0], p_05[1 - 1, 0])
        phi = acos(self.d[3] / sqrt(p_05[2 - 1, 0] * p_05[2 - 1, 0] + p_05[1 - 1, 0] * p_05[1 - 1, 0]))
        # The two solutions for theta1 correspond to the shoulder
        # being either left or right
        th[0, 0:4] = pi / 2 + psi + phi
        th[0, 4:8] = pi / 2 + psi - phi
        th = th.real

        # **** theta5 ****

        cl = [0, 4]  # wrist up or down
        for i in range(0, len(cl)):
            c = cl[i]
            t_10 = np.linalg.inv(self.ah(1, th, c))
            t_16 = t_10 * desired_pos
            th[4, c:c + 2] = + acos((t_16[2, 3] - self.d[3]) / self.d[5])
            th[4, c + 2:c + 4] = - acos((t_16[2, 3] - self.d[3]) / self.d[5])

        th = th.real

        # **** theta6 ****
        # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

        cl = [0, 2, 4, 6]
        for i in range(0, len(cl)):
            c = cl[i]
            t_10 = np.linalg.inv(self.ah(1, th, c))
            t_16 = np.linalg.inv(t_10 * desired_pos)
            th[5, c:c + 2] = atan2((-t_16[1, 2] / sin(th[4, c])), (t_16[0, 2] / sin(th[4, c])))

        th = th.real

        # **** theta3 ****
        cl = [0, 2, 4, 6]
        for i in range(0, len(cl)):
            c = cl[i]
            t_10 = np.linalg.inv(self.ah(1, th, c))
            t_65 = self.ah(6, th, c)
            t_54 = self.ah(5, th, c)
            t_14 = (t_10 * desired_pos) * np.linalg.inv(t_54 * t_65)
            p_13 = t_14 * np.matrix([0, - self.d[3], 0, 1]).T - np.matrix([0, 0, 0, 1]).T
            t3 = cmath.acos(
                (np.linalg.norm(p_13) ** 2 - self.a[1] ** 2 - self.a[2] ** 2) / (2 * self.a[1] * self.a[2]))  # norm ?
            th[2, c] = t3.real
            th[2, c + 1] = - t3.real

        # **** theta2 and theta 4 ****

        cl = [0, 1, 2, 3, 4, 5, 6, 7]
        for i in range(0, len(cl)):
            c = cl[i]
            t_10 = np.linalg.inv(self.ah(1, th, c))
            t_65 = np.linalg.inv(self.ah(6, th, c))
            t_54 = np.linalg.inv(self.ah(5, th, c))
            t_14 = (t_10 * desired_pos) * t_65 * t_54
            p_13 = t_14 * np.matrix([0, -self.d[3], 0, 1]).T - np.matrix([0, 0, 0, 1]).T

            # theta 2
            th[1, c] = - atan2(p_13[1], -p_13[0]) + asin(self.a[2] * sin(th[2, c]) / np.linalg.norm(p_13))
            # theta 4
            t_32 = np.linalg.inv(self.ah(3, th, c))
            t_21 = np.linalg.inv(self.ah(2, th, c))
            t_34 = t_32 * t_21 * t_14
            th[3, c] = atan2(t_34[1, 0], t_34[0, 0])
        th = th.real

        return th

    def go_to(self, pose, orientation=None):
        """
        Go to a requested pose, does not wait
        """
        if isinstance(pose, str):
            self.send_goal(pose)
            # if pose == 'zero':
            #     self.send_goal(self.g0)
            # elif pose == 'stop':
            #     self.send_goal(self.g1)
            # elif pose == 'drop':
            #     self.send_goal(self.g2)

        if type(pose) is list:
            if len(pose) == 3:
                if len(self.check_working_space([pose])) == 0:
                    rospy.logerr('Position out of reach')
                else:
                    ros_pose = Pose()
                    ros_pose.position.x = pose[0]
                    ros_pose.position.y = pose[1]
                    ros_pose.position.z = pose[2]
                    if orientation is None:
                        ros_pose.orientation = self.fwd_kin(self.joints_state, i_unit='r', o_unit='p').orientation
                    else:
                        ros_pose.orientation = orientation
                    pose = ros_pose

        if type(pose) == type(Pose()):
            point = self.check_working_space([[pose.position.x, pose.position.y, pose.position.z]])
            if len(point) == 0:
                rospy.logerr('Position out of reach')
            else:
                a = self.inv_kine(self.ros2np(pose))
                a = a.transpose().tolist()
                joints_inv_kine = self.select(a, self.joints_state)
                # print(pose)
                self.send_goal(joints_inv_kine)

    def go_to_and_wait(self, pose):
        """
        Go to a requested pose and wait
        """
        self.client.cancel_all_goals()

        self.go_to(pose)
        self.wait_for_result(rospy.Duration(3))

    @staticmethod
    def insert_points(lista):
        """
        Receives a list of points in 3D space and return a new list with more points and half of distance between them.
        The list is always in the same order, the new points are inserted between the old ones.
        Args:
            lista: a list of points in 3D space
        Return:
            new list of points in 3D space with half os distance between them
        """
        # divide
        nova_lista = [lista[0]]
        anterior = lista[0]
        for i in range(1, len(lista), 1):
            inter = (anterior + lista[i]) / 2
            nova_lista.append(inter)
            nova_lista.append(lista[i])
            anterior = lista[i]
        return nova_lista

    def trajectory(self, desired_position, current_position=None):
        """
        creates a linear trajectory to a given point
        """
        if current_position is None:
            current_position = self.fwd_kin(self.joints_state, i_unit='r', o_unit='p')
        current_position = np.array(
            [current_position.position.x, current_position.position.y, current_position.position.z,
             current_position.orientation.x, current_position.orientation.y,
             current_position.orientation.z, current_position.orientation.w])
        desired_position = np.array(desired_position)
        current_position = np.array(current_position)
        lista = [current_position, desired_position]
        while np.linalg.norm(lista[0][0:2] - lista[1][0:2]) > 0.005:
            lista = self.insert_points(lista)

        return lista

    def movel(self, destino, orientation=None, speed=None):
        """
        Move to destination following a line
        args:
            destino: a list of xyz defining a point in space
            orientation: a quaternion defining the orientation
        """
        self.client.cancel_all_goals()
        # to create uniform movements lets define speed to calculate time
        if speed is None:
            speed = 0.4  # m/s

        cur_pose = self.fwd_kin(self.joints_state, i_unit='r', o_unit='p')
        if orientation is None:
            orientation = cur_pose.orientation

        if type(destino) == type(Pose()):
            desired_position = np.array(
                [destino.position.x, destino.position.y, destino.position.z, destino.orientation.x,
                 destino.orientation.y, destino.orientation.z, destino.orientation.w])
            tmp1 = np.array([cur_pose.position.x, cur_pose.position.y, cur_pose.position.z])
            tmp2 = np.array([destino.position.x, destino.position.y, destino.position.z])

        else:
            desired_position = np.array([*destino, orientation.x, orientation.y, orientation.z, orientation.w])
            tmp1 = np.array([cur_pose.position.x, cur_pose.position.y, cur_pose.position.z])
            tmp2 = np.array(destino)

        distance = np.linalg.norm(tmp1 - tmp2)
        if distance <= 0:
            nanoseconds = 1_000_000
        else:
            nanoseconds = 1_000_000_000 * distance / speed

        # check if it is possible to reach
        if len(self.check_working_space([tmp2.tolist()])) == 0:
            rospy.logerr('Position %s out of reach' % np.array_str(tmp2, precision=2, suppress_small=True))
            return False

        # cur_pose = np.array([cur_pose.position.x,cur_pose.position.y,cur_pose.position.z])

        lista = self.trajectory(desired_position, cur_pose)  # returns a list of np array with position and orientation

        pose = Pose()

        g = FollowJointTrajectoryGoal()
        g.trajectory = JointTrajectory()
        g.trajectory.joint_names = self.JOINT_NAMES

        joints_inv_kine = self.joints_state
        g.trajectory.points = []
        n = len(lista)

        if self.debug:
            os.chdir('/home/ubuntu/Documents/debug')
            new_lista = []
            for item in lista:
                new_lista.append(np.array_str(item, precision=2, suppress_small=True))
            with open(time.strftime("%Y%m%d-%H%M%S")+' lista.yaml', 'w') as file:
                yaml.dump(new_lista, file)

        for i, point in enumerate(lista):
            pose.position.x = point[0]
            pose.position.y = point[1]
            pose.position.z = point[2]
            pose.orientation.x = point[3]
            pose.orientation.y = point[4]
            pose.orientation.z = point[5]
            pose.orientation.w = point[6]


            a = self.inv_kine(self.ros2np(pose))
            a = a.transpose().tolist()
            joints_inv_kine = self.select(a, joints_inv_kine)

            t = int(i * nanoseconds / n)
            g.trajectory.points.append(JointTrajectoryPoint(positions=joints_inv_kine, velocities=[0] * 6,
                                                            time_from_start=rospy.Duration(0, t)))

        # Callback goal
        self.done = False

        self.client.send_goal(g, done_cb=self.done_callback)
        start_time = time.time()
        while not self.done and time.time() - start_time < 3:  # 3 seconds timeout
            time.sleep(0.05)

        if not self.done:
            rospy.logerr('Canceling movement')
        self.client.cancel_all_goals()
        return self.done

    def done_callback(self, result,a):
        # print(result)
        # print(a)
        self.done = True

    def movej(self, destino: list, orientation=None):
        """
        Move to destination as fast as possible
        """
        if len(self.check_working_space([destino])) == 0:
            rospy.logerr('Position out of reach')
            return

        pose = Pose()
        pose.position.x = destino[0]
        pose.position.y = destino[1]
        pose.position.z = destino[2]

        if orientation == None:
            orientation = self.fwd_kin(self.joints_state, i_unit='r', o_unit='p').orientation
        # else:
        pose.orientation = orientation
        self.go_to_and_wait(pose)

    def check_working_space(self, lista):
        """
        Check if the list of points are inside the working space
        Args:
            lista: a list of points in 3D space
        Return:
            lista: a new list of points with points outside the working space removed
        """

        # first limitation, avoid the cylinder around the z axis
        def space1(point):
            if point[0] ** 2 + point[1] ** 2 < 0.17 ** 2:
                return True
            else:
                return False

        # second limitation, avoid points too far away
        def space2(point):
            if point[0] ** 2 + point[1] ** 2 + point[2] ** 2 > 0.52 ** 2:
                return True
            else:
                return False

        # third limitation, don't collide with the table
        def space3(point):
            if point[2] < 0.0:
                return True
            else:
                return False

        for point in lista:
            if space1(point): lista.remove(point)
            if space2(point): lista.remove(point)
            if space3(point): lista.remove(point)

        return lista


def main(args):
    def is_data():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    print('Python version:', sys.version)
    print('')
    print('Running on:')
    print(os.getcwd())

    rx = Rot.from_euler('x', 5, degrees=True)
    ry = Rot.from_euler('y', 5, degrees=True)
    rz = Rot.from_euler('z', 5, degrees=True)

    step = 0.01  # 1cm
    old_settings = termios.tcgetattr(sys.stdin)
    rospy.init_node("ur_kinematics", anonymous=False, disable_signals=True)

    kin = URKinematics(args)

    try:
        tty.setcbreak(sys.stdin.fileno())
        rospy.loginfo('Kinematics test.')
        rospy.loginfo('Press: ')
        rospy.loginfo('qa->z ws->x  ed->y')
        rospy.loginfo('t -> stop')
        rospy.loginfo('p -> current pose')
        rospy.loginfo('ESC to exit.')

        while not rospy.is_shutdown():

            if is_data():

                joints = kin.joints_state
                pos = kin.fwd_kin(joints, i_unit='r', o_unit='p')
                r = Rot.from_quat([pos.orientation.x, pos.orientation.y, pos.orientation.z, pos.orientation.w], True)

                c = sys.stdin.read(1)

                if c == '\x1b': break  # x1b is ESC
                if c == 'q': pos.position.z += step
                if c == 'a': pos.position.z -= step
                if c == 'w': pos.position.x += step
                if c == 's': pos.position.x -= step
                if c == 'e': pos.position.y += step
                if c == 'd': pos.position.y -= step
                if c == '1': r = r * rx
                if c == '7': r = r * rx.inv()
                if c == '2': r = r * ry
                if c == '8': r = r * ry.inv()
                if c == '3': r = r * rz
                if c == '9': r = r * rz.inv()
                if c == 'p': print(pos)

                r = r.as_quat()
                pos.orientation.x = r[0]
                pos.orientation.y = r[1]
                pos.orientation.z = r[2]
                pos.orientation.w = r[3]
                kin.movel(pos)
                # time.sleep(0.1)

                if c == 't': kin.go_to_and_wait('stop')
                if c == 'l': kin.go_to_and_wait('drop')

                if c == 'z':
                    kin.movel([0.25, 0.0, 0.1])
                    kin.movel([0.25, 0.0, 0.0], speed=0.1)
                if c == 'b':
                    os.chdir('/home/ubuntu/ur_ws/src/integrator/config')
                    with open('coordinates.yaml') as file:
                        data = yaml.load(file, Loader=yaml.FullLoader)
                        a = data['coord_a'][0]
                        b = data['coord_a'][1]
                        c = data['coord_d'][0]
                        d = data['coord_d'][1]

                    coord_a = [a, b, 0.06]
                    coord_a_ = [a, b, 0.05]
                    coord_b = [a, d, 0.06]
                    coord_b_ = [a, d, 0.05]
                    coord_c = [c, b, 0.06]
                    coord_c_ = [c, b, 0.05]
                    coord_d = [c, d, 0.06]
                    coord_d_ = [c, d, 0.05]

                    kin.movel(coord_b, speed=0.1)
                    # input('pause')
                    kin.movel(coord_b_, speed=0.1)
                    # input('pause')
                    kin.movel(coord_b, speed=0.1)
                    kin.movel(coord_a, speed=0.1)
                    kin.movel(coord_a_, speed=0.1)
                    kin.movel(coord_a, speed=0.1)
                    # kin.movel(coord_b, speed=0.1)
                    kin.movel(coord_c, speed=0.1)
                    kin.movel(coord_c_, speed=0.1)
                    kin.movel(coord_c, speed=0.1)

                    kin.movel(coord_d, speed=0.1)
                    kin.movel(coord_d_, speed=0.1)
                    kin.movel(coord_d, speed=0.1)

                    # kin.movel(coord_a, speed=0.1)
                    kin.movel([a,d,0.1], speed=0.1)
                    # kin.go_to_and_wait('stop')
                if c == 'c':
                    x = float(input('Enter x: '))
                    y = float(input('Enter y: '))
                    z = float(input('Enter z: '))
                    point = kin.check_working_space([[x, y, z]])
                    if len(point) == 0:
                        rospy.logerr('Position out of reach')
                    else:
                        kin.movel(*point)
                if c == 'j':
                    print(kin.joints_state)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        kin.send_goal('stop')
        rospy.loginfo("Position stop")
        try:
            kin.wait_for_result(rospy.Duration(3))
        except KeyboardInterrupt:
            kin.cancel_goal()
            raise


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Deep reinforcement learning in PyTorch.')

    #
    parser.add_argument('--real', dest='is_sim', action='store_false', default=True,
                        help='Real or simulated, default is simulated.')
    parser.add_argument('--gpu', dest='is_cuda', action='store_true', default=False, help='GPU mode, default is CPU.')
    parser.add_argument('--test', dest='is_test', action='store_true', default=False,
                        help='Testing or training, default is training.')

    #
    args_parser = parser.parse_args()

    # hyperparameters
    args = {}

    # convert to dictionary
    args['simulation'] = args_parser.is_sim
    args['testing'] = args_parser.is_test

    main(args)
