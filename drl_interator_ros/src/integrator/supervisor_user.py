#!/usr/bin/env python3.6

import numpy as np
import cv2
import rospy
# from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from integrator.srv import SupervisorGrabService, SupervisorPositionService

class SupervisorUser(object):
    """
    Class to use the supervisor from ROS service
    """

    def __init__(self, args):
        """
        Constructor
        """

    def service(self, obj):
        """
        Send request to grab one of the blocks
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
    
    def get_position(self, obj):
        """
        Get the position of one of the blocks
        """
        # connect to supervisor service, execute and disconnect
        rospy.wait_for_service('supervisor_position_service')
        try:
            self.position_service = rospy.ServiceProxy('supervisor_position_service', SupervisorPositionService)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        rst = self.position_service(obj)
        self.position_service.close()
        return rst

def main():
    supervisor = SupervisorUser()

    while not rospy.is_shutdown():
        # supervisor.grab('r')
        
        print(supervisor.get_position('0'))
        


if __name__ == '__main__':
    main()