#!/usr/bin/env python3.6

import os
import yaml
import pickle
# from collections import OrderedDict
import numpy as np
# from scipy import ndimage
import torch
# import torch.nn as nn
from torch.utils.data import Dataset
# from torch.autograd import Variable
# import torchvision
import cv2

# import matplotlib.pyplot as plt
import time
import rospy
import random


class ReplayMemory(Dataset):
    """
    Class derivated from the PyTorch Dataset to handle the batch execution.
    It is added a function to include new data in the dataset.
    """
    def __init__(self, args):
        """
        Constructor
        Args:
            args dictionary:
                directory: the folder where the data is located
                device: torch device, cuda or cpu
        """
        directory = args.get('directory')
        if directory: self.directory = directory
        else: self.directory = '/home/ubuntu/Documents/data'

        self.device =  args.get('device') if args.get('device') else torch.device('cpu')
        # get all folders
        self.folders = [ f.path for f in os.scandir(self.directory) if f.is_dir() ]

    def __getitem__(self, idx):
        """
        return a single sample with result
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        folder = self.folders[idx]

        # recover the data from disk
        rgb_tns = torch.load(os.path.join(folder, 'rgb_tensor.pt')).to(self.device)
        dep_tns = torch.load(os.path.join(folder, 'dep_tensor.pt')).to(self.device)
        q_values = torch.load(os.path.join(folder, 'q_values_tensor.pt')).to(self.device)

        others_file = os.path.join(folder, 'data.yaml')
        # load the kwargs dictionary from file
        with open(others_file) as file:
            arguments = yaml.load(file, Loader=yaml.FullLoader)

        return rgb_tns, dep_tns, q_values, arguments

    def __len__(self):
        return len(self.folders)

    def add(self, rgb_tns, dep_tns, q_values, imgs, kwargs):
        """
        Save more data to the folder
        Args:
            rgb_tns: PyTorch tensor with color image transformed
            dep_tns: PyTorch tensor with depth image transformed
            q_values: PyTorch tensor with desired Q values
            rgb_raw: Raw color image
            dep_raw: Raw depth image
            kwargs: dictionary with:
                success: if have been an attempt was it successfull?
                simulated: was it from simulation?
                generated: was it generated ?
        """

        new_dir = os.path.join(self.directory, time.strftime("%Y%m%d-%H%M%S"))

        try:
            os.mkdir(new_dir)
        except OSError:
            print("Creation of the directory %s failed" % new_dir)
        else:
            rgb_tns_file = os.path.join(new_dir, 'rgb_tensor.pt')
            dep_tns_file = os.path.join(new_dir, 'dep_tensor.pt')
            q_values_file = os.path.join(new_dir, 'q_values_tensor.pt')
            rgb_raw_file = os.path.join(new_dir, 'rgb_raw.png')
            dep_raw_file = os.path.join(new_dir, 'dep_raw.png')
            others_file = os.path.join(new_dir, 'data.yaml')

            # save the tensors
            torch.save(rgb_tns, rgb_tns_file)
            torch.save(dep_tns, dep_tns_file)
            torch.save(q_values, q_values_file)

            # save the images
            for i, img in enumerate(imgs):
                file = os.path.join(new_dir, 'img' + str(i) + '.png')
                cv2.imwrite(file, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # save the success
            with open(others_file, 'w') as file:

                yaml.dump(kwargs, file)

            # add to the list of folders
            self.folders.append(new_dir)


class KinReplayMemory(Dataset):
    """
    Class derivated from the PyTorch Dataset to handle the batch execution.
    It is added a function to include new data in the dataset.
    """
    def __init__(self, args):
        """
        Constructor
        Args:
            args dictionary:
                directory: the folder where the data is located
                device: torch device, cuda or cpu
        """
        directory = args.get('directory')
        if directory: self.directory = directory
        else: self.directory = '/home/ubuntu/Documents/kin-data'

        self.device =  args.get('device') if args.get('device') else torch.device('cpu')
        # get all folders
        self.folders = sorted ([ f.path for f in os.scandir(self.directory) if f.is_dir() ])

    def __getitem__(self, idx):
        """
        return a single sample with result
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        folder = self.folders[idx]

        # recover the data from disk
        rgb_tns = torch.load(os.path.join(folder, 'rgb_tensor.pt')).to(self.device)
        dep_tns = torch.load(os.path.join(folder, 'dep_tensor.pt')).to(self.device)
        new_rgb_tns = torch.load(os.path.join(folder, 'new_rgb_tensor.pt')).to(self.device)
        new_dep_tns = torch.load(os.path.join(folder, 'new_dep_tensor.pt')).to(self.device)
        q_values = torch.load(os.path.join(folder, 'q_values_tensor.pt')).to(self.device)
        pose = torch.load(os.path.join(folder, 'pose.pt')).to(self.device)
        new_pose = torch.load(os.path.join(folder, 'new_pose.pt')).to(self.device)

        others_file = os.path.join(folder, 'data.yaml')
        # load the kwargs dictionary from file
        with open(others_file) as file:
            arguments = yaml.load(file, Loader=yaml.FullLoader)

        return [pose, rgb_tns, dep_tns, new_pose, new_rgb_tns, new_dep_tns, q_values, arguments]

    def __len__(self):
        return len(self.folders)

    def add(self, pose, rgb_tns, dep_tns, q_values, new_pose, new_rgb_tns, new_dep_tns, imgs, arguments):
        """
        Save more data to the folder
        Args:
            rgb_tns: PyTorch tensor with color image transformed
            dep_tns: PyTorch tensor with depth image transformed
            q_values: PyTorch tensor with desired Q values
            rgb_raw: Raw color image
            dep_raw: Raw depth image
            kwargs: dictionary with:
                success: if have been an attempt was it successfull?
                simulated: was it from simulation?
                generated: was it generated ?
        """

        new_dir = os.path.join(self.directory, time.strftime("%Y%m%d-%H%M%S") + '_' + str(random.randint(1, 1000)))

        try:
            os.mkdir(new_dir)
        except OSError as e:
            rospy.logerr("Creation of the directory %s failed" % new_dir)
            rospy.logerr(e)
        else:
            rgb_tns_file = os.path.join(new_dir, 'rgb_tensor.pt')
            dep_tns_file = os.path.join(new_dir, 'dep_tensor.pt')
            q_values_file = os.path.join(new_dir, 'q_values_tensor.pt')
            new_rgb_tns_file = os.path.join(new_dir, 'new_rgb_tensor.pt')
            new_dep_tns_file = os.path.join(new_dir, 'new_dep_tensor.pt')
            pose_file = os.path.join(new_dir, 'pose.pt')
            new_pose_file = os.path.join(new_dir, 'new_pose.pt')


            # save the tensors
            torch.save(rgb_tns, rgb_tns_file)
            torch.save(dep_tns, dep_tns_file)
            torch.save(q_values, q_values_file)
            torch.save(new_rgb_tns, new_rgb_tns_file)
            torch.save(new_dep_tns, new_dep_tns_file)
            torch.save(pose, pose_file)
            torch.save(new_pose, new_pose_file)

            # save the images
            for i, img in enumerate(imgs):
                file = os.path.join(new_dir, 'img' + str(i) + '.png')
                cv2.imwrite(file, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # save the kwargs
            others_file = os.path.join(new_dir, 'data.yaml')
            with open(others_file, 'w') as file:
                yaml.dump(arguments, file)

            # add to the list of folders
            self.folders.append(new_dir)


if __name__ == '__main__':
    args = { 'directory': '/home/ubuntu/Documents/test',
             'device': torch.device('cpu')}
    dataset_tester = ReplayMemory(args)

    random_tensor = torch.randn((2,3,224,224))
    random_numpy = np.random.randint(0, 255, (224,224,3),np.uint8)

    dataset_tester.add(random_tensor, random_tensor, random_tensor, random_numpy, random_numpy, True)

    a = 0
    for [rgb, dep], q_value in dataset_tester:
        a += 1
        print(a)
        print(rgb.size())
        print(dep.size())
        print(q_value.size())