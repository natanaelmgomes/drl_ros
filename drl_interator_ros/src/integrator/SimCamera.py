#!/usr/bin/env python3

import os
import sys
import numpy as np
import time
from matplotlib import pyplot as plt

# Webots
from controller import Robot

# ROS
import rospy
from integrator.srv import SimImageCameraService, SimDepthCameraService
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# opencv
import cv2


class SimCamera(object):
    """Acquire image and depth from a camera sensor and publish on ROS"""

    def __init__(self):
        # webots
        os.environ['WEBOTS_ROBOT_NAME'] = 'camera'
        self.robot = Robot()
        self.bridge = CvBridge()
        self.timestep = int(self.robot.getBasicTimeStep())
        # timestep is 4ms equals to 250FPS
        # nopub 4ms aprox 250FPS avg 0.5x simulation time
        # nopub 33ms aprox 30FPS avg 4x simulation time
        # nopub 67ms aprox 15FPS avg 7x simulation time
        # nopub 100ms aprox 10FPS avg 8x simulation time
        # nopub 500ms aprox 2FPS avg 8.5x simulation time

        self.camera = self.robot.getDeviceByIndex(1)
        self.depthcamera = self.robot.getDeviceByIndex(2)


        # ROS
        rospy.init_node('camera_server', anonymous=False)
        self.service_image = rospy.Service('image_camera_service', SimImageCameraService, self.getImage)
        self.service_depth = rospy.Service('depth_camera_service', SimDepthCameraService, self.getDepth)

        self.flag = False

    def getImage(self, data):
        """
        respond to image request
        """
        # enable the camera
        self.camera.enable(self.timestep)

        # take a time step
        # for i in range(5):
        self.robot.step(self.timestep)

        # get the image
        self.camera.getImage()
        self.camera.saveImage('tmp-img.png', 100)
        img = cv2.imread('tmp-img.png')
        # os.remove('tmp-img.png')
        self.camera_msg = self.bridge.cv2_to_imgmsg(img)

        # disable the camera to save processing power
        self.camera.disable()

        self.flag = True

        # send the image
        rospy.loginfo('Sending Image ')
        return self.camera_msg

    def getDepth(self, data):
        """
        respond to depth image request
        """
        # enable the camera
        self.depthcamera.enable(self.timestep)

        # take a time step
        # for i in range(5):
        self.robot.step(self.timestep)

        # get the depth image
        self.depthcamera.getRangeImage()
        self.depthcamera.saveImage('tmp-dep.png', 100)

        dep = self.depthcamera.getRangeImageArray()
        # print(type(dep))
        # print(len(dep),len(dep[0]))
        dep = np.array(dep).transpose()

        self.depthcamera_msg = self.bridge.cv2_to_imgmsg(dep)

        # disable the camera to save processing power
        self.depthcamera.disable()

        # send the image
        rospy.loginfo('Sending Depth')
        return self.depthcamera_msg

    def update(self):
        """
        publish the image and depth
        """
        # image
        return
        self.camera.getImage()
        self.camera.saveImage('tmp-img.png', 100)
        img = cv2.imread('tmp-img.png')
        self.camera_msg = self.bridge.cv2_to_imgmsg(img)
        self.camera_pub.publish(self.camera_msg)

        # depth
        self.depthcamera.getRangeImage()
        dep = self.depthcamera.getRangeImageArray()

        dep = np.array(dep).transpose()

        self.depthcamera_msg = self.bridge.cv2_to_imgmsg(dep)
        self.depthcamera_pub.publish(self.depthcamera_msg)


def main():

    print('Python version:', sys.version)
    print('')
    print('Running on:')
    print(os.getcwd())

    camera = SimCamera()
    while camera.robot.step(camera.timestep) != -1 and not rospy.is_shutdown():
        camera.update()


    print('---------------- Webots has reset! ----------------')
    print('Waiting for respawn')
    print('')


if __name__ == '__main__':
    main()