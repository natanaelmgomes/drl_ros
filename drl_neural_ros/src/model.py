#!/usr/bin/env python3.6

from collections import OrderedDict
import numpy as np
# from scipy import ndimage
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

import rospy


# import matplotlib.pyplot as plt
# import time


class reinforcement_module(nn.Module):

    def __init__(self, args, model=None):
        """
        Args:
            args - dictionary:
                device: torch device, cuda or cpu
                lr: learning rate
                weight_decay: weight decay
        """
        super(reinforcement_module, self).__init__()
        self.model_name = model

        if model == 'densenet': # Densenet
            # for the laptop used it is better to save memory when running on gpu
            if args['device'] == torch.device('cuda'):
                memory_efficient = True
            else:
                memory_efficient = False
            # use pre-trained densenet121 to estimate the features
            self.color_features = torchvision.models.densenet.densenet121(pretrained = True, memory_efficient = memory_efficient).to(args['device']).features
            self.depth_features = torchvision.models.densenet.densenet121(pretrained = True, memory_efficient = memory_efficient).to(args['device']).features
            n_features = 2 * 1024

        elif model == 'resnext':
            # use the ResNeXt-50-32x4d to estimate the features resnext50_32x4d2 = nn.Sequential(*list(resnext50_32x4d.children())[:-2])
            color_features = torchvision.models.resnext50_32x4d(pretrained=True).to(args['device'])
            depth_features = torchvision.models.resnext50_32x4d(pretrained=True).to(args['device'])
            self.color_features = nn.Sequential(*list(color_features.children())[:-2])
            self.depth_features = nn.Sequential(*list(depth_features.children())[:-2])
            n_features = 2 * 2048

        elif model == 'mobilenet':
            # use the RMNASNet 1.0 to estimate the features
            self.color_features = torchvision.models.mobilenet_v2(pretrained = True).to(args['device']).features
            self.depth_features = torchvision.models.mobilenet_v2(pretrained = True).to(args['device']).features
            n_features = 2 * 1280

        elif model == 'mnasnet':
            # use the RMNASNet 1.0 to estimate the features
            self.color_features = torchvision.models.mnasnet1_0(pretrained = True).to(args['device']).layers
            self.depth_features = torchvision.models.mnasnet1_0(pretrained = True).to(args['device']).layers
            n_features = 2 * 1280
        else:
            raise Exception('Model name not recognized.')

        # each feature net gives n features that will be concatenated in channels
        self.net = nn.Sequential(OrderedDict([
            ('neural-norm0', nn.BatchNorm2d(n_features)),
            ('neural-relu0', nn.ReLU(inplace=True)),
            ('neural-conv0', nn.Conv2d(n_features, 64, kernel_size=1, stride=1, bias=False)),
            ('neural-norm1', nn.BatchNorm2d(64)),
            ('neural-relu1', nn.ReLU(inplace=True)),
            ('neural-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
            ('neural-upsam', nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True))
        ])).to(args['device'])

        # multiple learning rate
        ph = 0.1 # factor to reduce the learning rate of the pre-trained net
        self.optimizer = optim.Adam([
            {'params': self.color_features.parameters(), 'lr':args['lr']*ph, 'weight_decay': args['weight_decay']*ph},
            {'params': self.depth_features.parameters(), 'lr':args['lr']*ph, 'weight_decay': args['weight_decay']*ph},
            {'params': self.net.parameters(), 'lr': args['lr'], 'weight_decay': args['weight_decay']}], lr=0)


        self.criterion = torch.nn.SmoothL1Loss(reduction = 'sum')  # Huber loss

        # Initialize network weights
        # https://arxiv.org/pdf/1502.01852.pdf
        for m in self.named_modules():
            if 'neural-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    # print(m[1])
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

    def forward(self, input_color_data, input_depth_data):

        # Compute features
        color_features = self.color_features(input_color_data)
        depth_features = self.depth_features(input_depth_data)
        # each feature net gives 1024 features with 7x7 that will be concatenated in channels
        features = torch.cat((color_features, depth_features), dim=1)

        # print(color_features.size())  # torch.Size([1, 1024, 7, 7])
        # print(depth_features.size())  # torch.Size([1, 1024, 7, 7])
        # print(features.size())        # torch.Size([1, 2048, 7, 7])

        # Pass through the net to estimate the Q-values
        q_values = self.net(features)

        return q_values

class kinematic_reinforcement_module(nn.Module):

    def __init__(self, args):
        """
        Args:
            args - dictionary:
                device: torch device, cuda or cpu
                lr: learning rate
                weight_decay: weight decay
        """
        super(kinematic_reinforcement_module, self).__init__()

        self.args = args

        # for the laptop used it is better to save memory when running on gpu
        if args['device'] == torch.device('cuda'):
            memory_efficient = True
        else:
            memory_efficient = False

        # use pre-trained densenet121 to estimate the features
        self.color_features = torchvision.models.densenet.densenet121(pretrained = True, memory_efficient = memory_efficient).to(args['device']).features
        self.depth_features = torchvision.models.densenet.densenet121(pretrained = True, memory_efficient = memory_efficient).to(args['device']).features

        # each feature net gives 1024 features that will be concatenated in channels
        # the output of this layer is 112 x 112 = 12544
        self.net = nn.Sequential(OrderedDict([
            ('neural-norm0', nn.BatchNorm2d(2048)),
            ('neural-relu0', nn.ReLU(inplace=True)),
            ('neural-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('neural-norm1', nn.BatchNorm2d(64)),
            ('neural-relu1', nn.ReLU(inplace=True)),
            ('neural-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
            ('neural-upsam', nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True))
        ])).to(args['device'])

        self.kinnet = nn.Sequential(OrderedDict([
            ('kinnet-linear0', nn.Linear(12551, 4096)), # input = 12544 + 7 = 12551
            ('kinnet-relu0', nn.ReLU(inplace=True)),
            ('kinnet-linear1', nn.Linear(4096, 1024)),
            ('kinnet-relu1', nn.ReLU(inplace=True)),
            ('kinnet-linear2', nn.Linear(1024, 14))
        ])).to(args['device'])

        # multiple learning rate
        ph = 0.1 # factor to reduce the learning rate of the pre-trained net
        self.optimizer = optim.Adam([
            {'params': self.color_features.parameters(), 'lr':args['lr']*ph, 'weight_decay': args['weight_decay']*ph},
            {'params': self.depth_features.parameters(), 'lr':args['lr']*ph, 'weight_decay': args['weight_decay']*ph},
            {'params': self.net.parameters(), 'lr': args['lr'], 'weight_decay': args['weight_decay']},
            {'params': self.kinnet.parameters(), 'lr': args['lr'], 'weight_decay': args['weight_decay']}], lr=0)

        self.criterion = torch.nn.SmoothL1Loss(reduction = 'sum')  # Huber loss

        # Initialize network weights
        for m in self.named_modules():
            if 'neural-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    # print(m[1])
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

    def forward(self, input_color_data, input_depth_data, pose):
        """
        Args:
            input_color_data: tensor with color image
            input_depth_data: tensor with depth image
            pose: object Pose() of the current pose of the tool
        """
        # Compute features
        color_features = self.color_features(input_color_data)
        depth_features = self.depth_features(input_depth_data)
        # each feature net gives 1024 features with 7x7 that will be concatenated in channels
        # print(color_features.size())
        # print(depth_features.size())

        image_features = torch.cat((color_features, depth_features), dim=1)

        # print(image_features.size())

        # Pass through the net to estimate the Q-values
        q_values = self.net(image_features)
        q_values = torch.flatten(q_values, start_dim=1).squeeze()

        # print(q_values.size()[0])
        # print(pose.size())
        if q_values.size()[0] == 12544:
            features = torch.cat([q_values, pose])
        else:
            features = torch.cat([q_values, pose], dim=1)

        q_values_kin = self.kinnet(features.float())

        return q_values_kin