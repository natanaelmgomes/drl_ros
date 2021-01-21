#!/usr/bin/env python3.6

# Python
import sys
import os
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import yaml
# from scipy import ndimage

# opencv
import cv2

# Pytorch
import PIL.Image as Image
#from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

# ROS
import rospy
from integrator.camera import Camera
from integrator.Kinematics import URKinematics
from integrator.supervisor_user import SupervisorUser
from integrator.gripper_user import GripperUser
from integrator.msg import BlockPose

# This folder
from memory import ReplayMemory, KinReplayMemory


class Manager(object):

    def __init__(self, args):
        """
        Args:
            args - dictionary:
                device: torch device, cuda or cpu
                lr: learning rate
                weight_decay: weight decay
        """
        self.args=args
        self.device = args['device']
        # print(args['training'])
        if not args['training']:
            if args['kinematic']:
                self.memory = KinReplayMemory(args)
            else:
                self.memory = ReplayMemory(args)
            self.camera = Camera(args)
            self.supervisor = SupervisorUser(args)
            self.robot = URKinematics(args)
            self.gripper = GripperUser(args)

        # prepare the data normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transformation = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        with open('/home/ubuntu/ur_ws/src/integrator/config/coordinates.yaml') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            self.min_x = data['coord_a'][0]
            self.min_y = data['coord_a'][1]
            self.max_x = data['coord_d'][0]
            self.max_y = data['coord_d'][1]

    def get_images(self):

        # get the image from the environment
        self.camera.update()

        # get RGB
        rgb_raw_ = self.camera.img_croped_warped
        rgb_tns_ = Image.fromarray(rgb_raw_)

        rgb_tns_ = self.transformation(rgb_tns_).unsqueeze(0).to(self.args['device'])

        # get Depth
        dep_raw_ = self.camera.dep_croped_warped
        dep_tns_ = Image.fromarray(dep_raw_)
        dep_tns_ = self.transformation(dep_tns_).unsqueeze(0).to(self.args['device'])

        return rgb_tns_, dep_tns_, rgb_raw_, dep_raw_

    def estimar_valores_q(self, rgb=None):
        """
        Estimate the reward result from position of the objects

        The input to the CNN is 2 x 214 x 214
        The output is 112 by 112

        ponto a # 0.16, -0.22
        ponto b # 0.16, 0.22
        ponto c # 0.44, -0.22
        ponto d # 0.44, 0.22

        """
        obj = self.supervisor.service('n')
        n = obj.n
        # print(n)
        blocks = []
        for i in range(1,n,1):
            pose = self.supervisor.get_position(str(i))
            blocks.append(pose)

        from skimage.transform import ProjectiveTransform
        h = 112
        w = 112
        t = ProjectiveTransform()
        src = np.asarray([[0.16, -0.14], [0.16, 0.14], [0.44, -0.14], [0.44, 0.14]])
        dst = np.asarray([[0, 0], [0, w], [h, 0], [h, w]])
        if not t.estimate(src, dst): raise Exception("estimate failed")

        max = np.zeros((h, w))
        curve = np.zeros((h, w))
        for block in blocks:
            # print(block)
            x = block.pose.position[0]
            y = block.pose.position[1]
            a = t((x,y))
            # print(a)
            u = int(a[0][0])
            v = int(a[0][1])

            for i in range(h):
                for j in range(w):
                    curve[i,j] = np.sqrt((u-i)**2 + (v-j)**2)
            curve = np.clip(curve, 0, 20)
            curve /= curve.max()
            curve = 1 - curve

            if u < 0: u = 0
            if v < 0: v = 0
            if u > h-1: u = h-1
            if v > w-1: v = w-1
            max[u,v] = 1
            max = np.maximum(max, curve)

        return max

    def draw_from_q_values(self, rgb, q_values, attempt=None):
        """
        receive a rgb image and a numpy array with q_values and draw a superposition
        """
        fig = plt.figure(figsize=(8,6))

        delta = int(self.camera.delta / 1.1)
        tmp = cv2.copyMakeBorder(q_values, delta, delta, delta, delta, cv2.BORDER_CONSTANT, None, np.NaN)
        # print(tmp)
        prob_plot = cv2.resize(tmp, (rgb.shape[0], rgb.shape[1] ))

        plt.imshow(rgb)


        if not attempt is None:
        #     print('teste de desenho')
        #     # h, w = q_values.shape[:2]
            tmp = np.zeros(q_values.shape, q_values.dtype)
        #     # tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
        #     # tmp = np.dstack((tmp, np.zeros((h, w), dtype=np.uint8) + 255))
        #     # print(tmp[0,0,0])
            cv2.drawMarker(tmp, attempt, (225))
        #     # cv2.imshow("Image", cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        #     plt.imshow(tmp, cmap='Reds')
        #     # plt.show()
        #     # cv2.waitKey()
        #     # cv2.destroyAllWindows()
        #     # tmp[tmp == 0] = np.NaN
            tmp = cv2.copyMakeBorder( tmp, delta, delta, delta, delta, cv2.BORDER_CONSTANT, None, np.NaN)
            tmp = cv2.resize(tmp, (rgb.shape[0], rgb.shape[1]))
            plt.imshow(tmp, alpha=0.2)

        plt.imshow(prob_plot, vmin=0.0, vmax=1.0, alpha=0.3)
        plt.axis('off')
        plt.colorbar()
        # plt.show()
        # redraw the canvas
        fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

        plt.close(fig)

        # cv2.imshow("Test", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        return img

    def coletar_dados_simulados(self):
        """
        Collect data from the simulation to train the CNN
        """
        # self.supervisor.service('r')         # release all objects
        self.supervisor.service('clean')     # clear the workspace
        self.supervisor.service('prepare')   # prepare some objects with random position

        kwargs = {'success': None,
                'simulated': True,
                'generated': True}

        rgb_tns, dep_tns, rgb_raw, dep_raw = self.get_images()

        q_table = self.estimar_valores_q(rgb_raw)


def main(args):

    print('Python version:', sys.version)
    print('')
    print('Running on:')
    print(os.getcwd())

    rospy.init_node('Gerador', anonymous=False)
    gerador = Manager(args)

    for i in range(10):
        gerador.coletar_dados_simulados()



if __name__ == '__main__':
    # hyperparameters
    args = {
        'epoch_num': 5,  # Número de épocas.
        'lr': 1e-2,  # Taxa de aprendizado.
        'weight_decay': 8e-4,  # Penalidade L2 (Regularização).
        'batch_size': 1,  # Tamanho do batch.
        'testing': False, # no exploration
        # RL
        'gamma' : 0.999,
        'eps_start' : 0.9,      # initial randomness
        'eps_end' : 0.05,       # final randomness
        'eps_decay' : 200,      # exponential decay
        'target_update' : 10,
        'grasp_reward': 1,
        'proportional_reward': 0.25
    }

    # convert to dictionary
    args['simulation'] = True
    args['device'] = torch.device('cpu')
    args['testing'] = False

    main(args)