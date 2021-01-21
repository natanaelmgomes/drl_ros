#!/usr/bin/env python3.6

# Python
import time
import os
import yaml
import numpy as np
import argparse
from collections import deque

# ROS
import rospy
from cv_bridge import CvBridge
from integrator.srv import SimImageCameraService, SimDepthCameraService
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

# opencv
import cv2

# RealSense
import pyrealsense2 as rs2

# we want to save some files in different places
original = os.getcwd()
home = '/home/ubuntu'
config = '/home/ubuntu/ur_ws/src/integrator/config'


class Camera(object):
    """
    Class to handle the camera image from ROS service
    """

    def __init__(self, args):
        """
        Constructor
        """
        self.args = args
        # connect to image service
        self.is_sim = args.get('simulation')

        # init the deque before the node subscribe to the camera
        self.distances = deque(maxlen=10)

        # load img size
        os.chdir(config)
        with open('img_size.yaml') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            self.h = data['size'][0]
            self.w = data['size'][1]

        if not self.is_sim:
            # not simulated
            self.color_intrinsics = None
            self.depth_intrinsics = None

            # TODO msg = rospy.wait_for_message("/camera/depth/color/points", PointCloud2, timeout=None)
            rospy.Subscriber('/camera/color/image_raw', Image, self.imageColorCallback, queue_size=1)
            rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.imageColorInfoCallback, queue_size=1)

            rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.imageDepthCallback, queue_size=1)
            rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.imageDepthInfoCallback, queue_size=1)

        # get the cv_bridge object
        self.bridge = CvBridge()
        self.iscallibrated = False
        self.delta = 25

        # check for calibration file
        try:
            os.chdir(config)
            if self.is_sim:
                # simulated
                with open('camera_position_simulation.yaml') as file:

                    data = yaml.load(file, Loader=yaml.FullLoader)
                    self.ponto_a = data['ponto a']
                    self.ponto_b = data['ponto b']
                    self.ponto_c = data['ponto c']
                    self.ponto_d = data['ponto d']

                    group = np.vstack([self.ponto_a, self.ponto_b, self.ponto_c, self.ponto_d])

                    self.min_u = group[:, 0].min() - self.delta
                    self.max_u = group[:, 0].max() + self.delta
                    self.min_v = group[:, 1].min() - self.delta
                    self.max_v = group[:, 1].max() + self.delta


                    self.ponto_a_cropped = (self.ponto_a[0] - self.min_u, self.ponto_a[1] - self.min_v)
                    self.ponto_b_cropped = (self.ponto_b[0] - self.min_u, self.ponto_b[1] - self.min_v)
                    self.ponto_c_cropped = (self.ponto_c[0] - self.min_u, self.ponto_c[1] - self.min_v)
                    self.ponto_d_cropped = (self.ponto_d[0] - self.min_u, self.ponto_d[1] - self.min_v)

                    self.iscallibrated = True

                    file.close()
                with open('zero_distances_simulation.yaml') as file:
                    self.dep_raw_cropped_warped_zero = np.array(yaml.load(file, Loader=yaml.FullLoader))
                    file.close()
            else:
                # not simulated
                with open('camera_position_real.yaml') as file:

                    data = yaml.load(file, Loader=yaml.FullLoader)
                    # print(data)
                    self.ponto_a = data['ponto a']
                    self.ponto_b = data['ponto b']
                    self.ponto_c = data['ponto c']
                    self.ponto_d = data['ponto d']

                    group = np.vstack([self.ponto_a, self.ponto_b, self.ponto_c, self.ponto_d])

                    self.min_u = group[:, 0].min() - self.delta
                    self.max_u = group[:, 0].max() + self.delta
                    self.min_v = group[:, 1].min() - self.delta
                    self.max_v = group[:, 1].max() + self.delta

                    self.ponto_a_cropped = (self.ponto_a[0] - self.min_u, self.ponto_a[1] - self.min_v)
                    self.ponto_b_cropped = (self.ponto_b[0] - self.min_u, self.ponto_b[1] - self.min_v)
                    self.ponto_c_cropped = (self.ponto_c[0] - self.min_u, self.ponto_c[1] - self.min_v)
                    self.ponto_d_cropped = (self.ponto_d[0] - self.min_u, self.ponto_d[1] - self.min_v)

                    self.iscallibrated = True

                    file.close()
                with open('zero_distances_real.yaml') as file:
                    self.dep_raw_cropped_warped_zero = np.array(yaml.load(file, Loader=yaml.FullLoader))
                    file.close()
        except FileNotFoundError:
            rospy.logerr("File not accessible, camera not calibrated")

    def update(self, save=False, dir=home):
        """
        Function to update the image and depth obtained
        """
        if self.is_sim:
            # connect to the service, get the image and disconnect.
            rospy.wait_for_service('image_camera_service')
            try:
                self.image_service = rospy.ServiceProxy('image_camera_service', SimImageCameraService, persistent=True)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)
            img_obj = self.image_service()
            self.image_service.close()

            # connect to the service, get the image and disconnect.
            rospy.wait_for_service('depth_camera_service')
            try:
                self.depth_service = rospy.ServiceProxy('depth_camera_service', SimDepthCameraService, persistent=True)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)
            dep_obj = self.depth_service()
            self.depth_service.close()

            self.img = self.bridge.imgmsg_to_cv2(img_obj.image)

            self.dep_raw = np.asarray(self.bridge.imgmsg_to_cv2(dep_obj.depth))

            # reshape the image to match the depth image size
            h, w = self.dep_raw.shape
            self.img = cv2.resize(self.img, (w, h))
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

            # convert to Z16 format
            # the value is the distance in millimeters, data type is INT16
            self.dep_raw = np.array(self.dep_raw * 1000, dtype=np.int16)

        # convert the depth from mm to [0, 255], makes easier to visualization
        tmp = self.dep_raw.astype(np.float)
        maximum = tmp.max()
        tmp = (tmp / maximum) * 255
        self.dep = tmp.astype(np.uint8)
        self.dep = cv2.cvtColor(self.dep, cv2.COLOR_GRAY2RGB)

        # show it and save it
        if save:
            os.chdir(dir)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite('Pictures/' + timestr + ' Image.png', cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            cv2.imwrite('Pictures/' + timestr + ' Depth.png', cv2.cvtColor(self.dep, cv2.COLOR_BGR2RGB))
            rospy.loginfo('Images saved to Pictures folder')

        if self.iscallibrated:
            self.img_croped = self.img[self.min_v:self.max_v, self.min_u:self.max_u, :]
            self.dep_croped = self.dep[self.min_v:self.max_v, self.min_u:self.max_u]

            # Create the transformation from two sets of points
            pts1 = np.float32([self.ponto_a, self.ponto_b, self.ponto_c, self.ponto_d])
            pts2 = np.float32([[self.delta, self.delta],
                               [self.w-self.delta, self.delta],
                               [self.delta, self.h-self.delta],
                               [self.w-self.delta, self.h-self.delta]])

            # Apply Perspective Transform Algorithm
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.img_croped_warped = cv2.warpPerspective(self.img, matrix, (self.w, self.h))
            self.dep_croped_warped= cv2.warpPerspective(self.dep, matrix, (self.w, self.h))
            # second transformation for the raw depth
            pts2 = np.float32([[0, 0],
                               [self.w, 0],
                               [0, self.h],
                               [self.w, self.h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.dep_raw_croped_warped = cv2.warpPerspective(self.dep_raw, matrix, (self.w, self.h))

    def imageColorCallback(self, data):
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except Exception as e:
            rospy.logerr(e)
            rospy.logerr('Error on color callback')

    def imageColorInfoCallback(self, cameraInfo):
        try:
            if self.color_intrinsics:
                return
            self.color_intrinsics = rs2.intrinsics()
            self.color_intrinsics.width = cameraInfo.width
            self.color_intrinsics.height = cameraInfo.height
            self.color_intrinsics.ppx = cameraInfo.K[2]
            self.color_intrinsics.ppy = cameraInfo.K[5]
            self.color_intrinsics.fx = cameraInfo.K[0]
            self.color_intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.color_intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.color_intrinsics.model = rs2.distortion.kannala_brandt4
            self.color_intrinsics.coeffs = [i for i in cameraInfo.D]
        except Exception as e:
            print(e)
            rospy.logerr('Error on color info callback')

    def imageDepthCallback(self, data):
        try:
            self.dep_raw = self.bridge.imgmsg_to_cv2(data, data.encoding)
            if self.iscallibrated:
                # Create the transformation from two sets of points
                pts1 = np.float32([self.ponto_a, self.ponto_b, self.ponto_c, self.ponto_d])
                pts2 = np.float32([[0, 0],
                                   [self.w, 0],
                                   [0, self.h],
                                   [self.w, self.h]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                self.dep_raw_croped_warped = cv2.warpPerspective(self.dep_raw, matrix, (self.w, self.h))

                self.distances.append(self.dep_raw_croped_warped)
        except Exception as e:
            rospy.logerr(e)
            rospy.logerr('Error on depth callback')

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.depth_intrinsics:
                return
            self.depth_intrinsics = rs2.intrinsics()
            self.depth_intrinsics.width = cameraInfo.width
            self.depth_intrinsics.height = cameraInfo.height
            self.depth_intrinsics.ppx = cameraInfo.K[2]
            self.depth_intrinsics.ppy = cameraInfo.K[5]
            self.depth_intrinsics.fx = cameraInfo.K[0]
            self.depth_intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.depth_intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.depth_intrinsics.model = rs2.distortion.kannala_brandt4
            self.depth_intrinsics.coeffs = [i for i in cameraInfo.D]
        except Exception as e:
            rospy.logerr(e)
            rospy.logerr('Error on depth info callback')

    def show_point_cloud(self):
        """
        Function to show the PointCloud
        """
        import open3d as open3d

        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(open3d.geometry.Image(self.img),
                                                                     open3d.geometry.Image(self.dep_raw))
        # it looses the colors with the create function, so we add again
        rgbd.color = open3d.geometry.Image(self.img)

        # create a point cloud
        pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd,
                                        open3d.camera.PinholeCameraIntrinsic(
                                            open3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault))

        # print(list(pcd.points))
        point_cloud_array = np.asarray(pcd.points)
        print(point_cloud_array.shape)
        print(point_cloud_array[0,0])

        vis = open3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        print(vis.get_picked_points())
        # pcd.

        # flip the image
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # show the result
        # open3d.visualization.draw_geometries([pcd])

    def show_img_dep(self, save=True):
        """
        function to show the image and depth in memory
        """
        # create a superposition to check how good the imgs overlap
        alpha = 0.1
        beta = 1.0 - alpha
        dst = cv2.addWeighted(self.img, alpha, self.dep, beta, 0.0)

        # show it and save it
        os.chdir(home)
        cv2.imshow("Image", cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        cv2.imshow("Depth", cv2.cvtColor(self.dep, cv2.COLOR_BGR2RGB))
        if save:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite('Pictures/' + timestr+' Image.png', cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            cv2.imwrite('Pictures/' + timestr + ' Depth.png', cv2.cvtColor(self.dep, cv2.COLOR_BGR2RGB))
        cv2.imshow("Superposition", cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        cv2.waitKey()
        cv2.destroyAllWindows()

    def show_img_dep_cropped(self, save=True):
        """
        function to show the cropped image and depth in memory
        """

        if not self.iscallibrated:
            rospy.logerr('Please, calibrate.')
            return
        try:
            self.img_croped
        except AttributeError:
            rospy.logerr('Please, update the images')
            return

        cv2.line(self.img_croped, self.ponto_a_cropped, self.ponto_b_cropped, (0, 255, 0))
        cv2.line(self.img_croped, self.ponto_a_cropped, self.ponto_c_cropped, (0, 255, 0))
        cv2.line(self.img_croped, self.ponto_b_cropped, self.ponto_d_cropped, (0, 255, 0))
        cv2.line(self.img_croped, self.ponto_c_cropped, self.ponto_d_cropped, (0, 255, 0))

        cv2.imshow("Image cropped", cv2.cvtColor(self.img_croped, cv2.COLOR_BGR2RGB))
        cv2.imshow("Input", cv2.cvtColor(self.img_croped_warped, cv2.COLOR_BGR2RGB))
        cv2.imshow("Raw Depth", self.dep_raw_croped_warped/self.dep_raw_croped_warped.max())

        if save:
            os.chdir(home)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite('Pictures/' + timestr+' Image_cropped.png', cv2.cvtColor(self.img_croped, cv2.COLOR_BGR2RGB))
            cv2.imwrite('Pictures/' + timestr + ' Depth_cropped.png', cv2.cvtColor(self.dep_croped, cv2.COLOR_BGR2RGB))
            cv2.imwrite('Pictures/' + timestr+' Image_cropped_warp.png', cv2.cvtColor(self.img_croped_warped, cv2.COLOR_BGR2RGB))
            cv2.imwrite('Pictures/' + timestr + ' Depth_cropped_warp.png', cv2.cvtColor(self.dep_croped_warped, cv2.COLOR_BGR2RGB))

    def calibrate_position(self, show=False):
        """
        Function to calibrate the position of the working space
        """
        # calibration coordinates
        os.chdir(config)
        with open('coordinates.yaml') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            a = data['coord_a'][0]
            b = data['coord_a'][1]
            c = data['coord_d'][0]
            d = data['coord_d'][1]

        rospy.loginfo('Starting calibration')
        coord_a = [a, b, 0.04]
        coord_a_p = [a, b, 0.01]
        coord_b = [a, d, 0.04]
        coord_b_p = [a, d, 0.01]
        coord_c = [c, b, 0.04]
        coord_c_p = [c, b, 0.01]
        coord_d = [c, d, 0.04]
        coord_d_p = [c, d, 0.01]

        from integrator.Kinematics import URKinematics
        from integrator.srv import SupervisorGrabService
        from gripper.cmodel_urcap import RobotiqCModelURCap
        from cv2 import aruco

        # load the kinematics module
        kin = URKinematics(self.args)
        time.sleep(0.5)
        kin.go_to_and_wait('stop')
        kin.movel([(a+c)/2,(b+d)/2,0.1], speed=0.5)

        if self.is_sim:
            # connect to image service
            rospy.wait_for_service('supervisor_grab_service')
            try:
                supervisor = rospy.ServiceProxy('supervisor_grab_service', SupervisorGrabService)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)
                return

            # release any other block
            # release previour object and clean the table
            supervisor('r')
            supervisor('c')

            # grab the aruco block
            supervisor('0')
        else:
            rospy.logwarn('Please place the aruco block in the gripper')
            gripper = RobotiqCModelURCap('192.168.56.2')
            # print(gripper.has_object())
            if not gripper.has_object():
                gripper.prepare_to_grasp()
                # rospy.loginfo('Press Enter to continue.')
                input('Press Enter to continue.')
                while not gripper.try_to_grasp():
                    gripper.prepare_to_grasp()
                    input('Try again')

        input('Press Enter to continue.')

        # move to the first position
        kin.movel(coord_a, speed=0.5)
        kin.movel(coord_a_p, speed=0.5)

        # get the images
        time.sleep(0.5)
        self.update()

        # analyse it
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if not ids:
            rospy.logerr('Aruco not detected')
            return
        self.ponto_a = (int(corners[0][0][:, 0].mean()), int(corners[0][0][:, 1].mean()))

        if show:
            cv2.drawMarker(self.img, self.ponto_a, (0, 255, 0))
            cv2.imshow("Image", cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            cv2.waitKey()
            cv2.destroyAllWindows()

        # move the arm
        kin.movel(coord_b, speed=0.5)
        kin.movel(coord_b_p, speed=0.5)

        # get the images
        time.sleep(0.5)
        self.update()

        # analyse it
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if not ids:
            rospy.logerr('Aruco not detected')
            return
        self.ponto_b = (int(corners[0][0][:, 0].mean()), int(corners[0][0][:, 1].mean()))

        if show:
            cv2.drawMarker(self.img, self.ponto_a, (0, 255, 0))
            cv2.drawMarker(self.img, self.ponto_b, (0, 255, 0))
            cv2.imshow("Image", cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            cv2.waitKey()
            cv2.destroyAllWindows()

        # move the arm
        kin.movel(coord_c, speed=0.5)
        kin.movel(coord_c_p, speed=0.5)

        # get the images
        time.sleep(0.5)
        self.update()

        # analyse it
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if not ids:
            rospy.logerr('Aruco not detected')
            return
        self.ponto_c = (int(corners[0][0][:, 0].mean()), int(corners[0][0][:, 1].mean()))

        if show:
            cv2.drawMarker(self.img, self.ponto_a, (0, 255, 0))
            cv2.drawMarker(self.img, self.ponto_b, (0, 255, 0))
            cv2.drawMarker(self.img, self.ponto_c, (0, 255, 0))
            cv2.imshow("Image", cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            cv2.waitKey()
            cv2.destroyAllWindows()

        # move the arm
        kin.movel(coord_d, speed=0.5)
        kin.movel(coord_d_p, speed=0.5)

        # get the images
        time.sleep(0.5)
        self.update()

        # analyse it
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if not ids:
            rospy.logerr('Aruco not detected')
            return
        self.ponto_d = (int(corners[0][0][:, 0].mean()), int(corners[0][0][:, 1].mean()))

        if show:
            cv2.drawMarker(self.img, self.ponto_a, (0, 255, 0))
            cv2.drawMarker(self.img, self.ponto_b, (0, 255, 0))
            cv2.drawMarker(self.img, self.ponto_c, (0, 255, 0))
            cv2.drawMarker(self.img, self.ponto_d, (0, 255, 0))
            cv2.imshow("Image", cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            cv2.waitKey()
            cv2.destroyAllWindows()

        # move the arm
        kin.go_to_and_wait('stop')

        if self.is_sim:
            kin.go_to_and_wait('drop')
            supervisor('r')
            supervisor('c')
            time.sleep(0.1)
            os.chdir(config)
            with open('camera_position_simulation.yaml', 'w') as file:
                dic = {'ponto a': self.ponto_a, 'ponto b': self.ponto_b, 'ponto c': self.ponto_c, 'ponto d': self.ponto_d}
                yaml.dump(dic, file)

        else:
            kin.movel(coord_c, speed=0.5)
            input('Press Enter to continue.')
            os.chdir(config)
            with open('camera_position_real.yaml', 'w') as file:
                dic = {'ponto a': self.ponto_a, 'ponto b': self.ponto_b, 'ponto c': self.ponto_c, 'ponto d': self.ponto_d}
                yaml.dump(dic, file)

        group = np.vstack([self.ponto_a, self.ponto_b, self.ponto_c, self.ponto_d])

        self.min_u = group[:, 0].min() - self.delta
        self.max_u = group[:, 0].max() + self.delta
        self.min_v = group[:, 1].min() - self.delta
        self.max_v = group[:, 1].max() + self.delta

        self.ponto_a_cropped = (self.ponto_a[0] - self.min_u, self.ponto_a[1] - self.min_v)
        self.ponto_b_cropped = (self.ponto_b[0] - self.min_u, self.ponto_b[1] - self.min_v)
        self.ponto_c_cropped = (self.ponto_c[0] - self.min_u, self.ponto_c[1] - self.min_v)
        self.ponto_d_cropped = (self.ponto_d[0] - self.min_u, self.ponto_d[1] - self.min_v)

        self.iscallibrated = True

        kin.go_to_and_wait('stop')
        if self.is_sim:
            self.update()
            self.dep_raw_cropped_warped_zero = self.dep_raw_croped_warped
            os.chdir(config)
            with open('zero_distances_simulation.yaml', 'w') as file:
                yaml.dump(self.dep_raw_cropped_warped_zero.tolist(), file)
        else:
            time.sleep(0.7)
            self.dep_raw_cropped_warped_zero = np.mean(self.distances,axis=0)
            os.chdir(config)
            with open('zero_distances_real.yaml', 'w') as file:
                yaml.dump(self.dep_raw_cropped_warped_zero.tolist(), file)

        rospy.loginfo('Calibration successful')

    def get_z(self, position):
        """
        Estimate the z position based on depth information
        """
        os.chdir(config)
        with open('coordinates.yaml') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            a = data['coord_a'][0]
            b = data['coord_a'][1]
            c = data['coord_d'][0]
            d = data['coord_d'][1]

        u = int(((position[0] - a) / (c - a)) * self.w)
        v = int(((position[1] - b) / (d - b)) * self.h)

        if self.is_sim:
            z = float(self.dep_raw_cropped_warped_zero[u, v] - self.dep_raw_croped_warped[u, v]) / 1000
        else:
            z = float(self.dep_raw_cropped_warped_zero[u, v] - np.mean(self.distances,axis=0)[u, v]) / 1000
        z *= 0.8

        if z > 0.10:
            z = 0.10
        elif z >= 0.03 and z < 0.10:
            z -= 0.03
        elif z < 0.03:
            z /= 2
        if z < 0.01:
            z = 0.005

        return z


def main(args):
    rospy.init_node("camera_user_test", anonymous=False)
    camera = Camera(args)
    time.sleep(0.1)

    if not camera.iscallibrated:
        camera.calibrate_position()
    c = '0'
    while  (c != 'y' and c != 'n'):
        c = input('Recalibrate the camera? [y/n] ')
        print(c)

    if c == 'y':
        camera.calibrate_position(show=True)

    while not rospy.is_shutdown():

        camera.update()
        camera.show_img_dep_cropped()

        cv2.imshow("Depth", cv2.cvtColor(camera.dep, cv2.COLOR_BGR2RGB))
        if camera.iscallibrated:
            cv2.drawMarker(camera.img, camera.ponto_a, (0, 255, 0))
            cv2.drawMarker(camera.img, camera.ponto_b, (0, 255, 0))
            cv2.drawMarker(camera.img, camera.ponto_c, (0, 255, 0))
            cv2.drawMarker(camera.img, camera.ponto_d, (0, 255, 0))
            cv2.imshow("Image", cv2.cvtColor(camera.img, cv2.COLOR_BGR2RGB))
        else:
            cv2.imshow("Image", cv2.cvtColor(camera.img, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print('')
        print(camera.get_z((0.25,0)))


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Deep reinforcement learning in PyTorch.')

    #
    parser.add_argument('--real', dest='is_sim', action='store_false', default=True, help='Real or simulated, default is simulated.')
    parser.add_argument('--gpu', dest='is_cuda', action='store_true', default=False, help='GPU mode, default is CPU.')
    parser.add_argument('--test', dest='is_test', action='store_true', default=False, help='Testing or training, default is training.')

    #
    args_parser = parser.parse_args()

    # hyperparameters
    args = {    }

    # convert to dictionary
    args['simulation'] = args_parser.is_sim
    args['testing'] = args_parser.is_test

    main(args)
