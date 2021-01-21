#!/usr/bin/env python3.6

import numpy as np
import cv2
import rospy
import time
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
from gripper.srv import GripperService
from std_msgs.msg import String
from gripper.cmodel_urcap import RobotiqCModelURCap

class GripperUser(object):
    """
    Class to use the gripper from ROS service
    """

    def __init__(self, args):
        """
        Constructor
        """
        self.args = args
        if self.args['simulation']:
            rospy.Subscriber("/gripper_status", String, self.callback)
            self.gripper_status = 'UNKNOWN'
        else:
            self.hardware = RobotiqCModelURCap('192.168.56.2')

    def close(self):
        """
        Send request close the gripper and check an object is detected
        """
        if self.args['simulation']:
            # connect to gripper service, execute and disconnect
            rospy.wait_for_service('gripper_service')
            try:
                self.gripper_service = rospy.ServiceProxy('gripper_service', GripperService)
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
            self.gripper_service('close')
            self.gripper_service.close()

            self.gripper_status = 'UNKNOWN'
            i=0
            time.sleep(0.1)
            while self.gripper_status != 'AT_DEST' and self.gripper_status != 'STOPPED_INNER_OBJECT':
                # print(i)
                # print(self.gripper_status)
                i += 1
                pass

            # return false if there is no object
            if self.gripper_status == 'AT_DEST':
                return False

            # return true if there is object
            if self.gripper_status == 'STOPPED_INNER_OBJECT':
                return True
        else:
            rst = self.hardware.try_to_grasp()
            return rst


    
    def open(self):
        """
        Open the gripper
        """
        if self.args['simulation']:
            # connect to gripper service, execute and disconnect
            rospy.wait_for_service('gripper_service')
            try:
                self.gripper_service = rospy.ServiceProxy('gripper_service', GripperService)
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
            self.gripper_service('open')
            self.gripper_service.close()
        else:
            self.hardware.prepare_to_grasp()

    def open_and_wait(self):
        """
        Open the gripper and wait to complete the movement
        """
        if self.args['simulation']:
            rospy.wait_for_service('gripper_service')
            try:
                self.gripper_service = rospy.ServiceProxy('gripper_service', GripperService)
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
            self.gripper_service('open')
            self.gripper_service.close()


            while self.gripper_status != 'AT_DEST' and self.gripper_status != 'STOPPED_OUTER_OBJECT':
                # print(self.gripper_status)
                pass
        else:
            self.hardware.prepare_to_grasp()

    def callback(self, data):
        """
        update the gripper status
        """
        # print(data.data)
        self.gripper_status = data.data

def main():
    gripper = GripperUser()
    rospy.init_node('gripper_user_test', anonymous=False)
    gripper.open()
    gripper.close()
    gripper.open()
    gripper.close()


    while not rospy.is_shutdown():
        # supervisor.grab('r')
        print(gripper.gripper_status)


        
        pass


if __name__ == '__main__':
    main()