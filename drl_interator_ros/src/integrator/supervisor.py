#!/usr/bin/env python3

# from __future__ import print_function

import os
import sys
import random
import time
import numpy as np

# imports for keyboard input
import select
import tty
import termios

# library to perform the rotation
from scipy.spatial.transform import Rotation as Rot

from pynput.mouse import Button, Controller

from controller import Supervisor
import rospy
from std_msgs.msg import Int8
from integrator.msg import BlockPose
from integrator.srv import SupervisorGrabService, SupervisorPositionService


class WebotsSupervisor(object):
    """Control the Webots simulation"""

    def __init__(self):
        self.number_of_blocks = 5
        # define the controller name we want to connect to
        os.environ['WEBOTS_ROBOT_NAME'] = 'supervisor'
        # get the controller
        self.supervisor = Supervisor()
        # get timestep of the simulation
        self.timestep = int(self.supervisor.getBasicTimeStep())

        # position and rotation of the cobot in the simulation
        self.ur3e_position = [0.69, 0.74, 0]
        self.ur3e_rotation = Rot.from_rotvec(-(np.pi / 2) * np.array([1.0, 0.0, 0.0]))

        # get all blocks of the simulation
        self.blocks = []
        for i in range(self.number_of_blocks):
            self.blocks.append(self.supervisor.getFromDef("block{0}".format(i)))

        # get the position of the end-effector
        self.endEffector = self.supervisor.getFromDef("gps")

        # initialize the variable to grab objects
        self.grabbed = []
        for i in range(self.number_of_blocks): self.grabbed.append(False)
        self.is_grabbed = False

        # get the positions field for all blocks
        self.block_positions_field = []
        self.block_rotations_field = []
        for i in range(self.number_of_blocks):
            # get the fields
            self.block_positions_field.append(self.blocks[i].getField("translation"))
            self.block_rotations_field.append(self.blocks[i].getField("rotation"))

        # get the name of the blocks
        self.block_names = []
        for i in range(self.number_of_blocks):
            self.block_names.append(self.blocks[i].getField("name").getSFString())

        self.is_recording = False

        self.table_color_field = self.supervisor.getFromDef("table").getField('trayAppearance').getSFNode().getField('baseColor')

        rospy.Subscriber('touchsensor_status', Int8, self.touch_callback, queue_size=1)
        self.touchsensors = 0

    def touch_callback(self, data):
        self.touchsensors = data.data

    def grab_service(self, data):
        """
        Service call to grab and release objects in the simulation
        """
        c = data.obj

        if c == 'c':
            self.clean_table()

            return True, self.number_of_blocks

        for i in range(self.number_of_blocks):
            if c == str(i):
                if self.is_grabbed:
                    print('Please, release the other object')
                    return False, self.number_of_blocks
                else:
                    self.grabbed[i] = True
                    self.is_grabbed = True
                    return True, self.number_of_blocks

        if c == 'r':
            for i in range(self.number_of_blocks):
                self.grabbed[i] = False
                self.is_grabbed = False
                self.blocks[i].resetPhysics()
            return True, self.number_of_blocks

        if c == 'n':
            return True, self.number_of_blocks

        if c == 'reset':
            # self.supervisor.simulationReset()
            # self.supervisor.simulationRevert()

            mouse = Controller()
            # print(mouse.position)
            mouse.position = (971, 96)

            mouse.click(Button.left, 1)
            return True, self.number_of_blocks

        if c == 'clean':
            self.clean_table()
            return True, self.number_of_blocks

        if c == 'prepare':
            self.prepare_table(self.number_of_blocks-1)
            return True, self.number_of_blocks

        if c == 'prepare1':
            self.prepare_table(1)
            return True, self.number_of_blocks

        if c == 'prepare2':
            self.prepare_table(2)
            return True, self.number_of_blocks

        if c == 'prepare3':
            self.prepare_table(3)
            return True, self.number_of_blocks

        if c == 'prepare4':
            self.prepare_table(4)
            return True, self.number_of_blocks

        if c == 'start_video':
            timestr = time.strftime("%Y%m%d-%H%M%S")
            file = '/home/ubuntu/Videos/' + timestr + ' Simulation.mp4'
            self.supervisor.movieStartRecording(file, 854, 480, quality=97)
            self.is_recording = True
            return True

        if c == 'stop_video':
            if self.is_recording:
                self.supervisor.movieStopRecording()
                self.is_recording = False
                return not self.supervisor.movieFailed()

        return False, self.number_of_blocks

    def position_service(self, data):
        """
        Service call to the position of objects in the simulation
        """
        c = int(data.obj)

        block_msgs = BlockPose()

        # get the name of the block
        block_msgs.name = self.block_names[c]

        # get the position and orientation of the block
        block_msgs.position = np.array(self.block_positions_field[c].getSFVec3f())
        block_msgs.rotation = self.block_rotations_field[c].getSFRotation()

        # change the reference of the block
        # rotation
        rot_vec = block_msgs.rotation[3] * np.array(
            [block_msgs.rotation[0], block_msgs.rotation[1], block_msgs.rotation[2]]
        )
        block_msgs.rotation = Rot.from_rotvec(rot_vec) * self.ur3e_rotation
        # position
        block_msgs.position = block_msgs.position - np.array(self.ur3e_position)
        block_msgs.position = self.ur3e_rotation.inv().apply(block_msgs.position)
        block_msgs.position = Rot.from_rotvec(np.pi * np.array([0.0, 0.0, 1.0])).apply(block_msgs.position)
        # if i == 1: print(block_msgs[i].position)
        # convert to lists
        block_msgs.rotation = block_msgs.rotation.as_quat()
        block_msgs.position = block_msgs.position.tolist()

        return block_msgs

    def clean_table(self):
        """
        Remove all objects from table
        """
        self.table_color_field.setSFColor([0.6, 0.6, 0.6])
        for i in range(self.number_of_blocks):
            if not self.grabbed[i]:
                # get position above the basket
                position = [0.3, 1.2, 0.55]

                # set the position of the block
                self.block_positions_field[i].setSFVec3f(position)
                self.blocks[i].resetPhysics()
                self.supervisor.step(40 * self.timestep)

    def position_collision(self, positions, position):
        """
        Check if there is chance of collisions between objects
        """
        collision = False
        for pos in positions:
            if (pos[0]-position[0]) ** 2 + (pos[2]-position[2]) ** 2 < 0.09 ** 2: collision = True
        return collision

    def prepare_table(self, obj_num):
        """
        Place objects in the table
        """
        # self.clean_table()
        for i in range(self.number_of_blocks):
            if not self.grabbed[i]:
                # get position above the basket
                position = [-i, 1, 1]

                # set the position of the block
                self.block_positions_field[i].setSFVec3f(position)
                self.blocks[i].resetPhysics()

        positions = []

        for i in range(1, obj_num+1, 1):  # start, stop, step
            # print(i)
            if not self.grabbed[i]:
                # get position middle of the workspace
                y = 0.773

                if len(positions) == 0:
                    x = random.uniform(0.250, 0.530)
                    z = random.uniform(-0.140, 0.140)

                    # center = [0.390, 0.773, 0.0]
                    position = [x, y, z]
                    positions.append(position)

                    # set the position of the block
                    if obj_num == 1: i = random.randint(1,4)
                    self.block_positions_field[i].setSFVec3f(position)
                    self.blocks[i].resetPhysics()
                    # self.supervisor.step(40 * self.timestep)
                else:
                    x = random.uniform(0.250, 0.530)
                    z = random.uniform(-0.140, 0.140)
                    if i==3: y = 0.76

                    # estimate a position
                    position = [x, y, z]
                    # check if there are collisions
                    while self.position_collision(positions, position):
                        # if affirmative create another estimate and check again
                        # print(position)
                        x = random.uniform(0.250, 0.530)
                        z = random.uniform(-0.140, 0.140)
                        position = [x, y, z]
                    positions.append(position)

                    # set the position of the block
                    self.block_positions_field[i].setSFVec3f(position)
                    self.blocks[i].resetPhysics()
                    # self.supervisor.step(40 * self.timestep)
        r = random.uniform(0.0, 1.0)
        g = random.uniform(0.0, 1.0)
        b = random.uniform(0.0, 1.0)
        self.table_color_field.setSFColor([r, g, b])


def main():
    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    print('Python version:', sys.version)
    print('')
    print('Running on:')
    print(os.getcwd())

    # rotation to keep the object in the same orientation as on the table
    invert_rotation = Rot.from_rotvec(np.pi * np.array([0, 0, 1]))
    # Initialize the ROS Node
    rospy.init_node('webots_supervisor', anonymous=False)

    start_time = time.time()
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        print('Supervisor test.')
        print('Press: number to grab and r to release')
        print('Press: c to clean the table')
        print('Press: g to reset the simulation')
        print('ESC to exit.')

        supervisor = WebotsSupervisor()
        # Initialize the ROS Services
        service1 = rospy.Service('supervisor_grab_service', SupervisorGrabService, supervisor.grab_service)
        service2 = rospy.Service('supervisor_position_service', SupervisorPositionService, supervisor.position_service)

        # just in case, lets reset the physics
        supervisor.supervisor.simulationResetPhysics()

        while supervisor.supervisor.step(supervisor.timestep) != -1 and not rospy.is_shutdown():
            if isData():
                c = sys.stdin.read(1)
                if c == '\x1b': break  # x1b is ESC
                for i in range(supervisor.number_of_blocks):
                    if c == str(i):
                        if supervisor.is_grabbed: print('Please, release the other object')
                        else:
                            supervisor.grabbed[i] = True
                            supervisor.is_grabbed = False

                if c == 'r':
                    for i in range(supervisor.number_of_blocks):
                        supervisor.grabbed[i] = False
                        supervisor.is_grabbed = False
                        supervisor.blocks[i].resetPhysics()

                if c == 'c':
                    supervisor.clean_table()

                if c == 'g':
                    from pynput.mouse import Button, Controller
                    mouse = Controller()
                    mouse.position = (829, 96)
                    mouse.click(Button.left, 1)
                if c == 'p':
                    supervisor.prepare_table()


            if supervisor.touchsensors != 1:
                start_time = time.time()
            if time.time() - start_time > 3.0:
                rospy.logwarn('Collision detected')
                for i in range(supervisor.number_of_blocks):
                    if not supervisor.grabbed[i]:
                        # set the position of the block
                        supervisor.block_positions_field[i].setSFVec3f([-i, 1, 1])
                        supervisor.blocks[i].resetPhysics()

            for i in range(supervisor.number_of_blocks):
                if supervisor.grabbed[i]:
                    # get position and orientation of the end-effector
                    end_effector_position = supervisor.endEffector.getPosition()
                    end_effector_rotation = np.array(supervisor.endEffector.getOrientation()).reshape(3, 3)
                    # convert to axis-rotation
                    rot_matrix = Rot.from_matrix(end_effector_rotation)
                    rot_matrix = rot_matrix * invert_rotation
                    if i == 0:  # the calibration block
                        extra_rot = Rot.from_rotvec((-np.pi/5) * np.array([1, 0, 0]))
                        rot_matrix = rot_matrix * extra_rot
                    rot_vect = [*rot_matrix.as_rotvec().tolist(), rot_matrix.magnitude()]

                    # set the position and the rotation of the block
                    supervisor.block_positions_field[i].setSFVec3f(end_effector_position)
                    supervisor.block_rotations_field[i].setSFRotation(rot_vect)
                    # reset the Physics so the block does not fall
                    supervisor.blocks[i].resetPhysics()

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    # webots has reset
    print('---------------- Webots has reset! ----------------')
    print('Waiting for respawn')
    print('')




if __name__ == '__main__':
    main()
