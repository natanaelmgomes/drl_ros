#!/usr/bin/env python3.6

import time
import numpy as np
import cv2
import rospy
# from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from integrator.srv import SupervisorGrabService, WatchdogService

# imports for keyboard input
import sys
import select
import tty
import termios

class Watchdog(object):
    """
    Class to implement a watchdog for the simulation
    """

    def __init__(self, args=None):
        """
        Constructor
        """
        self.ros_service = rospy.Service('watchdog_service', WatchdogService, self.callback)

        # init timer
        self.time = time.time()

        # the watchdog starts off
        self.enabled = False
        rospy.loginfo('Watchdog disabled')

    def __del__(self):
        """
        Collector
        """
        self.ros_service.shutdown('Collector')

    def service(self, obj):
        """
        Send request execute the supervisor service
        """
        # connect to supervisor service, execute and disconnect
        rospy.wait_for_service('supervisor_grab_service')
        try:
            self.grab_service = rospy.ServiceProxy('supervisor_grab_service', SupervisorGrabService)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        rst = self.grab_service(obj)
        self.grab_service.close()

        return rst
    
    def callback(self, obj=None):
        """
        Callback function to receive the heartbeat
        """
        if not self.enabled:
            rospy.loginfo('Watchdog enabled')
            self.enabled = True

        self.time = time.time()
        return True

    def watch(self, obj=None):
        """
        Callback function to receive the heartbeat
        """
        # if passes more than a minute send a reset
        if (time.time() - self.time) > 60.0 and self.enabled:
            self.service('reset') # reset simulation
            # self.service('clean') # remove everything from the table
            self.time = time.time()


def main():
    rospy.init_node('watchdog', anonymous=False)
    dog = Watchdog()

    def is_data():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not rospy.is_shutdown():
            if is_data():
                c = sys.stdin.read(1)
                if c == '\x1b': break  # x1b is ESC
                if c == 'd':
                    dog.enabled = False
                    rospy.loginfo('Watchdog disabled')
            time.sleep(1)
            dog.watch()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == '__main__':
    main()













