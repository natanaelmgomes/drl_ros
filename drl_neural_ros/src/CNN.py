#!/usr/bin/env python3.6

# Python
from collections import OrderedDict
import os
import random
import math
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import cv2
import shlex, subprocess
import yaml

# Pytorch
import PIL.Image as Image
#from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter


# ROS
import rospy
from integrator.camera import Camera
from integrator.Kinematics import URKinematics
from integrator.supervisor_user import SupervisorUser
from integrator.gripper_user import GripperUser
from integrator.srv import WatchdogService

# this folder
from model import reinforcement_module
from memory import ReplayMemory
from manager import Manager

# other
from skimage.transform import ProjectiveTransform

def check_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = c - a  # free inside cache
    print('total: ' + str(t / 1024 / 1024 / 1024))
    print('reservado: ' + str(c / 1024 / 1024 / 1024))
    print('alocado: ' + str(a / 1024 / 1024 / 1024))
    print('livre: ' + str(f / 1024 / 1024 / 1024))

def ImagetoTensor(img):
    """convert a numpy array of shape HWC to CHW tensor"""
    img = img.transpose((2, 0, 1)).astype(np.float32)
    tensor = torch.from_numpy(img).float()
    return tensor/255.0

def choose_action(args, q_values):
    """
    The input to the CNN is 2 x 3 x 214 x 214
    The output is 112 by 112

    ponto a # 0.16, -0.22
    ponto b # 0.16, 0.22
    ponto c # 0.44, -0.22
    ponto d # 0.44, 0.22

    The action is epsilon-greedy with decay

    """

    sample = random.random()
    eps_threshold = args['eps_end'] + (args['eps_start'] - args['eps_end']) * math.exp(-1. * args['epoch'] / args['eps_decay'])

    q_values = q_values.cpu().detach().numpy().squeeze()
    h, w = q_values.shape
    if sample > eps_threshold or args['testing']:
        u, v = np.unravel_index(q_values.argmax(), q_values.shape)
        rospy.loginfo('Fair attempt')
        fair = True
    else:
        u = int(random.uniform(0, h))
        v = int(random.uniform(0, w))
        rospy.loginfo('Random attempt')
        fair = False

    from skimage.transform import ProjectiveTransform
    t = ProjectiveTransform()
    src = np.asarray([[0, 0], [0, w], [h, 0], [h, w]])
    with open('/home/ubuntu/ur_ws/src/integrator/config/coordinates.yaml') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        a = data['coord_a'][0]
        b = data['coord_a'][1]
        c = data['coord_d'][0]
        d = data['coord_d'][1]
    dst = np.asarray([[a, b], [a, d], [c, b], [c, d]])
    if not t.estimate(src, dst): raise Exception("estimate failed")

    a = t((u, v)).squeeze()
    x = a[0]
    y = a[1]

    return x, y, u, v, fair

def grasp(args, x, y, manager):
    """
    given a coordenate (x, y) attempt to grasp and gives a bool as result
    """
    # if args['epoch'] % 10 == 0: manager.robot.go_to_and_wait('stop')
    # time.sleep(0.1)
    try:
        manager.robot.movel([0.21, 0, 0.10])
        if not args['simulation']: time.sleep(0.1)
        if x < 0.21: manager.robot.movel([x, y, 0.10], speed=0.1)
        else:  manager.robot.movel([x, y, 0.10])
        if not args['simulation']: time.sleep(0.1)
        manager.gripper.open_and_wait()
        if not args['simulation']: time.sleep(0.1)
        z = manager.camera.get_z((x, y))
        # z = 0.02
        manager.robot.movel([x, y, z], speed=0.1)
        if not args['simulation']: time.sleep(0.1)
        rst = manager.gripper.close()
        manager.robot.movel([x, y, 0.10])
        if not args['simulation']: time.sleep(0.1)

        if rst:
            # manager.robot.go_to_and_wait('drop')
            x = random.uniform(manager.min_x, manager.max_x)
            y = random.uniform(manager.min_y, manager.max_y)
            manager.robot.movel([x,y,0.10])
            if not args['simulation']: time.sleep(0.1)
            manager.robot.movel([x, y, z + 0.005], speed=0.1)
            manager.gripper.open()
            if not args['simulation']: time.sleep(1)

        manager.robot.movel([0.21, 0, 0.10])
        manager.gripper.open()
    except Exception as e:
        rospy.logerr(e)
        rospy.logerr((x,y))
        manager.supervisor.service('reset')
        call_watchdog()
        time.sleep(5)
        manager.robot.go_to_and_wait('stop')
        return False
    # if args['epoch'] % 10 == 0: manager.robot.go_to_and_wait('stop')
    return rst

def show_and_save(rgb_raw_, q_values, camera):
    plt.figure()
    plt.imshow(rgb_raw_)
    # cv2.imshow("Image", cv2.cvtColor(rgb_raw_, cv2.COLOR_BGR2RGB))
    delta = int(camera.delta / 1.8)
    q_values = q_values.cpu().detach().numpy().squeeze()
    tmp = cv2.copyMakeBorder(q_values, delta, delta, delta, delta, cv2.BORDER_CONSTANT, None, np.NaN)
    prob_plot = cv2.resize(tmp, (rgb_raw_.shape[0], rgb_raw_.shape[1]))
    plt.imshow(prob_plot, alpha=0.3)
    plt.axis('off')
    plt.colorbar()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.chdir('/home/ubuntu')
    # cv2.imwrite('Pictures/' + timestr + ' heatmap.png', cv2.cvtColor(self.img_croped, cv2.COLOR_BGR2RGB))
    plt.savefig('Pictures/' + timestr + ' heatmap.png', facecolor='w', dpi=300)
    # plt.show()

def call_watchdog():
    rospy.wait_for_service('watchdog_service')
    try:
        watchdog_service = rospy.ServiceProxy('watchdog_service', WatchdogService)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
    rst = watchdog_service(True)
    watchdog_service.close()
    return rst

def generate(args, manager):
    rospy.loginfo('Preparing to generate %s data points for training' % args['epoch_num'])
    # prepare simulation env
    rospy.loginfo('Init simulation')
    manager.supervisor.service('r')  # release all objects
    # manager.supervisor.service('reset')
    time.sleep(3)
    manager.robot.go_to_and_wait('stop')
    # manager.gripper.close()
    # manager.gripper.open()
    # manager.gripper.open_and_wait()
    manager.supervisor.service('clean')
    rospy.loginfo('Simulation set')

    args['directory'] = '/home/ubuntu/Documents/data-generated'
    manager.memory = ReplayMemory(args)
    n = 0
    for i in range(args['epoch_num']):
        if i%500 == 0: n += 1
        rospy.loginfo(' --- --- Iteration %s --- ---' % args['epoch'])
        args['epoch'] += 1
        rospy.loginfo('Randomizing items')
        manager.supervisor.service('prepare' + str(n))
        rospy.loginfo('Items set')
        rospy.loginfo('Acquire images')
        rgb, dep, rgb_raw, dep_raw = manager.get_images()  # observation
        q_values = torch.tensor(manager.estimar_valores_q()).unsqueeze(0).unsqueeze(0)
        img_gen = manager.draw_from_q_values(rgb_raw, q_values.cpu().detach().numpy().squeeze())

        # save all data generated
        kwargs = {'success': True,
                  'simulated': True,
                  'generated': True}
        manager.memory.add(rgb, dep, q_values, [rgb_raw, dep_raw, img_gen], kwargs)
        rospy.loginfo('Data saved')

def train_with_generated(manager, model, writer=None):
    args['directory'] = '/home/ubuntu/Documents/data-generated'
    # args['batch_size'] = 20
    manager.memory = ReplayMemory(args)

    data_loader = DataLoader(manager.memory,
                              batch_size=args['batch_size'],
                              shuffle=True)
    rospy.loginfo(
        'Preparing to trains for {0} epochs with batch size: {1}'.format(len(data_loader), args['batch_size']))
    # forward_time = []
    # backward_time = []
    for k, batch in enumerate(data_loader):
        rospy.loginfo('------ Epoch  {0} / {1}  ------'.format(k, len(data_loader)))

        # unpack data
        rgb, dep, q_values, kwargs = batch
         # = imgs
        rgb = rgb.squeeze()
        dep = dep.squeeze()
        q_values = q_values.squeeze().unsqueeze(1)

        # forward
        start_time = time.time()
        q_values_pred = model(rgb, dep)
        seconds = time.time() - start_time
        # forward_time.append(seconds)
        writer.add_scalar('Train/Forward', seconds, k)
        # rospy.loginfo("---- forward ---- %s seconds ----" % seconds)

        # backward
        start_time = time.time()
        loss = model.criterion(q_values, q_values_pred)
        writer.add_scalar('Train/Loss', loss.item(), k)
        rospy.loginfo('LOSS: ' + str(loss.item()))
        # perdas.append(loss.item())
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        seconds = time.time() - start_time
        # backward_time.append(seconds)
        writer.add_scalar('Train/Backward', seconds, k)

    return  model

def train_with_all_data(manager, model):
    args['directory'] = '/home/ubuntu/Documents/data'
    # args['batch_size'] = 10
    manager.memory = ReplayMemory(args)

    data_loader = DataLoader(manager.memory,
                              batch_size=args['batch_size'],
                              shuffle=True)
    rospy.loginfo(
        'Preparing to trains for {0} epochs with batch size: {1}'.format(len(data_loader), args['batch_size']))
    forward_time = []
    backward_time = []
    for k, batch in enumerate(data_loader):
        rospy.loginfo('------ Epoch  {0} / {1}  ------'.format(k, len(data_loader)))

        # unpack data
        imgs, q_values = batch
        rgb, dep = imgs
        rgb = rgb.squeeze()
        dep = dep.squeeze()
        q_values = q_values.squeeze().unsqueeze(1)

        # forward
        start_time = time.time()
        q_values_pred = model(rgb, dep)

        seconds = time.time() - start_time
        forward_time.append(seconds)
        rospy.loginfo("---- forward ---- %s seconds ----" % seconds)

        # backward
        start_time = time.time()
        loss = model.criterion(q_values, q_values_pred)
        rospy.loginfo('LOSS: ' + str(loss.item()))
        # perdas.append(loss.item())
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        seconds = time.time() - start_time
        backward_time.append(seconds)
        rospy.loginfo("---- backward ---- %s seconds ----" % seconds)

        np_forward = np.array(forward_time)
        np_backward = np.array(backward_time)

        rospy.loginfo(
            "-- forward -- mean -- {:2.3f} seconds -- +- {:2.3f} --".format(np_forward.mean(), np_forward.std()))
        rospy.loginfo(
            "-- backward -- mean -- {:2.3f} seconds -- +- {:2.3f} --".format(np_backward.mean(), np_backward.std()))


    return model

def train(args, model, manager, writer):
    save = False
    # Start training only if certain number of samples is already saved

    if len(manager.memory) < args['min_replay_memory']:
        return model

    rospy.loginfo('Training.')

    data_loader = DataLoader(manager.memory,
                              batch_size=args['batch_size'],
                              shuffle=True)

    for k, batch in enumerate(data_loader):
        # rospy.loginfo('------ Epoch  {0} / {1}  ------'.format(k, len(data_loader)))

        # unpack data
        # pose, rgb, dep, new_pose, new_rgb, new_dep, q_values, kwargs = batch
        rgb, dep, q_values, kwargs = batch

        rgb = rgb.squeeze()
        dep = dep.squeeze()

        # q_values = q_values.squeeze() #.unsqueeze(1)

        # new_q_values = new_q_values.squeeze().unsqueeze(1)

        # Get current states from minibatch, then query NN model for Q values
        # current_states = [rgb, dep, pose]
        # with torch.no_grad():
        #     current_qs_list = model(*current_states)

        # forward
        start_time = time.time()
        q_values_pred = model(rgb, dep)
        seconds = time.time() - start_time
        writer.add_scalar('Train/Forward', seconds, args['epoch'])

        q_values = q_values_pred.clone().detach().to(args['device'])

        # update the q values
        for k in range(len(kwargs)):

            u = kwargs['attempt(u,v)'][0][k]
            v = kwargs['attempt(u,v)'][1][k]

            rst = kwargs['success'][k]
            if rst:
                reward = args['grasp_reward']
            else:
                reward = 0

            # update q values
            q_values[k, :, :, :] = update_q_values(args, q_values[k, :, :, :].detach(), u, v, rst)

        # backward
        start_time = time.time()
        loss = model.criterion(q_values, q_values_pred)
        rospy.loginfo('LOSS: ' + str(loss.item()))
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        seconds = time.time() - start_time

        writer.add_scalar('Train/Loss', loss.item(), args['epoch'])
        writer.add_scalar('Train/Backward', seconds, args['epoch'])

        break

    if save:
        # save the model
        rospy.loginfo('Saving the model')
        new_dir = os.path.join('/home/ubuntu/Documents/kin-models', time.strftime("%Y%m%d-%H%M%S"))
        os.mkdir(new_dir)
        file = os.path.join(new_dir, 'model.pt')
        torch.save(model, file)

    return model

def test_model(args, manager, model):
    # return 0
    rospy.loginfo(' --- --- Evaluating model --- ---')

    rst = []
    previous_arg_testing = args['testing']
    args['testing'] = True
    for i in range(10):
        call_watchdog()
        # rospy.loginfo('Randomizing items')
        if args['simulation']:
            manager.supervisor.service('prepare' + str(random.randint(1, 4)))
        # rospy.loginfo('Items set')
        rgb, dep, rgb_raw, dep_raw = manager.get_images()  # observation
        q_values_pred = model(rgb, dep)
        q_values = q_values_pred.clone().detach()
        x, y, u, v, fair = choose_action(args, q_values_pred)
        g = grasp(args, x, y, manager)
        rst.append(g)
        if rst[i]:
            rospy.loginfo('Grasp success')
            reward = args['grasp_reward']
        else:
            rospy.loginfo('Grasp fail')
            reward = 0
        if not args['simulation']: time.sleep(0.2)

        # # save all data generated
        # update q values
        q_values[0,:,:,:] = update_q_values(args, q_values[0,:,:,:].detach(), u, v, rst)

        img_pred = manager.draw_from_q_values(rgb_raw, q_values_pred.cpu().detach().numpy().squeeze(), attempt=(v,u))
        # img_res = manager.draw_from_q_values(rgb_raw, q_values.cpu().detach().numpy().squeeze())
        kwargs = {'success': g,
                  'simulated': False,
                  'generated': False,
                  'attempt(u,v)': (int(u), int(v)),
                  'attempt(x,y)': (float(x), float(y)),
                  'model': model.model_name,
                  'fair attempt': fair}

        manager.memory.add(rgb, dep, q_values_pred, [rgb_raw, dep_raw, img_pred], kwargs)
    args['testing'] = previous_arg_testing
    return np.mean(rst)

def update_q_values(args, q_values, u, v, rst):
    """
    Update the Q-values with coordinates of the attempt and if it was success of fail
    """

    for i in range(q_values.size()[1]):
        for j in range(q_values.size()[2]):
            distance = np.sqrt((u - i) ** 2 + (v - j) ** 2)
            if distance < 20:
                value = args['rl_lr'] * (1 / (distance + args['grasp_reward']))
                # value = (-distance/20.0) + reward
                # print(q_values[0, 0, i, j])
                if rst:
                    q_values[0, i, j] += value
                else:
                    q_values[0, i, j] -= value
                # print(q_values[0, 0, i, j])
                # torch.clamp
            # q_values[k,:,:,:].clamp(min = 0.0, max = 1.0)
            if q_values[0, i, j] > 1.0: q_values[0, i, j] = 1.0
            if q_values[0, i, j] < 0.0: q_values[0, i, j] = 0.0

    return q_values


def main(args):

    rospy.init_node('Neural', anonymous=False)


    if args['simulation']:
        # forward_time = []
        # backward_time = []

        model_names = ['mnasnet',
                       'resnext',
                       'mobilenet',
                       'densenet']

        for model_name in model_names:
            if model_name == 'mnasnet' or 'mobilenet': args['device'] = torch.device('cuda')
            else: args['device'] = torch.device('cpu')
            manager = Manager(args)
            rospy.loginfo('Model: ' + model_name)
            if args['device'] == torch.device('cuda'):
                rospy.loginfo('Running on CUDA')
            else:
                rospy.loginfo('Running on CPU')
            model = reinforcement_module(args, model_name)
            writer = SummaryWriter('/home/ubuntu/Documents/Tensorboard5/' + model_name)

            args['epoch'] = 1
            writer.add_scalar('Test/Acc', test_model(args, manager, model), 0)


            for i in range(args['epoch_num']):
                manager.robot.go_to_and_wait('stop')
                call_watchdog()

                rospy.loginfo(' --- --- Epoch %s --- ---' % args['epoch'])
                eps_threshold = args['eps_end'] + (args['eps_start'] - args['eps_end']) * math.exp(
                    -1. * args['epoch'] / args['eps_decay'])
                writer.add_scalar('Train/Epsilon', eps_threshold, args['epoch'])

                # rospy.loginfo('Randomizing items')
                manager.supervisor.service('prepare'+str(random.randint(1, 4)))
                # rospy.loginfo('Items set')
                rgb, dep, rgb_raw, dep_raw = manager.get_images()  # observation
                # rospy.loginfo('Images acquired')


                with torch.no_grad():
                    q_values_pred = model(rgb, dep)
                    q_values = q_values_pred.clone().detach()
                # rospy.loginfo('')
                # seconds = time.time() - start_time
                # forward_time.append(seconds)
                # rospy.loginfo("---- forward ---- %s seconds ----" % seconds)
                # writer.add_scalar('Train/Forward', seconds, args['epoch'])

                q_values = torch.tensor(manager.estimar_valores_q()).unsqueeze(0).unsqueeze(0)
                x, y, u, v, fair = choose_action(args, q_values)
                # rst = False

                # rospy.loginfo('Test the grasp')
                # if x > 0.2: continue
                rst = grasp(args, x, y, manager)


                img_pred = manager.draw_from_q_values(rgb_raw, q_values_pred.cpu().detach().numpy().squeeze(), attempt=(v,u))
                img_res = manager.draw_from_q_values(rgb_raw, q_values.cpu().detach().numpy().squeeze())


                writer.add_image('Predicted', ImagetoTensor(img_pred), args['epoch'])
                writer.add_image('Result', ImagetoTensor(img_res), args['epoch'])

                args['epoch'] += 1
                if rospy.is_shutdown(): return

            rospy.loginfo('Saving the model')
            new_dir = os.path.join('/home/ubuntu/Documents/models', time.strftime("%Y%m%d-%H%M%S") + ' ' + model_name)
            os.mkdir(new_dir)
            file = os.path.join(new_dir, 'model.pt')
            torch.save(model, file)

    else:
        # real robot
        base_folder = '/home/ubuntu/Documents/models'
        models = [
            '20210108-140316 resnext',
            '20210108-141440 densenet',
            '20210108-135306 mnasnet',
            '20210108-140616 mobilenet']

        for model_name in models:
            if 'mnasnet' in model_name or 'mobilenet' in model_name: args['device'] = torch.device('cuda')
            else: args['device'] = torch.device('cpu')
            manager = Manager(args)
            if args['device'] == torch.device('cuda'):
                rospy.loginfo('Running on CUDA')
            else:
                rospy.loginfo('Running on CPU')
            rospy.loginfo('Model: %s' % model_name)
            manager.robot.go_to_and_wait('stop')
            file = os.path.join(base_folder, model_name)
            file = os.path.join(file, 'model.pt')
            if args['device'] == torch.device('cuda'):
                rospy.loginfo('Running on CUDA')
                model = torch.load(file, map_location="cuda:0")
            else:
                rospy.loginfo('Running on CPU')
                model = torch.load(file)

            writer = SummaryWriter('/home/ubuntu/Documents/Tensorboard7/' + model_name)

            results = []
            args['epoch'] = 1
            while args['epoch'] < args['epoch_num']:
                rospy.loginfo(' --- --- Epoch %s --- ---' % args['epoch'])
                # eps_threshold = args['eps_end'] + (args['eps_start'] - args['eps_end']) * math.exp(
                #     -1. * args['epoch'] / args['eps_decay'])
                # writer.add_scalar('Train/Epsilon', eps_threshold, args['epoch'])
                rgb, dep, rgb_raw, dep_raw = manager.get_images()  # observation
                # rospy.loginfo('Images acquired')

                q_values_pred = model(rgb, dep)
                q_values = q_values_pred.clone().detach().to(args['device'])

                x, y, u, v, fair = choose_action(args, q_values_pred)

                rst = grasp(args, x, y, manager)
                if rst:
                    rospy.loginfo('Grasp success')
                    # reward = args['grasp_reward']
                else:
                    rospy.loginfo('Grasp fail')
                    # reward = 0

                #update q values
                q_values[0,:,:,:] = update_q_values(args, q_values[0,:,:,:].detach(), u, v, rst)


                results.append(rst)
                if len(results) == 10:
                    writer.add_scalar('Test/Acc_10', np.mean(results), args['epoch'])
                    results = []

                writer.add_scalar('Test/Acc_1', int(rst), args['epoch'])

                loss = model.criterion(q_values, q_values_pred)
                # print(loss.device)
                rospy.loginfo('LOSS: ' + str(loss.item()))
                model.optimizer.zero_grad()
                loss.backward()

                model.optimizer.step()

                writer.add_scalar('Train/Loss', loss.item(), args['epoch'])

                img_pred = manager.draw_from_q_values(rgb_raw, q_values_pred.cpu().detach().numpy().squeeze(), attempt=(v,u))
                img_after = manager.draw_from_q_values(rgb_raw, q_values.cpu().detach().numpy().squeeze())

                writer.add_image('Predicted', ImagetoTensor(img_pred), args['epoch'])
                writer.add_image('Result', ImagetoTensor(img_after), args['epoch'])

                cv2.imshow("Predicted", cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
                cv2.imshow("Result", cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
                cv2.imshow("Depth", cv2.cvtColor(manager.camera.dep, cv2.COLOR_BGR2RGB))
                cv2.drawMarker(manager.camera.img, manager.camera.ponto_a, (0, 255, 0))
                cv2.drawMarker(manager.camera.img, manager.camera.ponto_b, (0, 255, 0))
                cv2.drawMarker(manager.camera.img, manager.camera.ponto_c, (0, 255, 0))
                cv2.drawMarker(manager.camera.img, manager.camera.ponto_d, (0, 255, 0))
                cv2.imshow("Image Full", cv2.cvtColor(manager.camera.img, cv2.COLOR_BGR2RGB))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

                # save all data generated
                kwargs = {'success': rst,
                          'simulated': False,
                          'generated': False,
                          'attempt(u,v)': (int(u), int(v)),
                          'attempt(x,y)': (float(x), float(y)),
                          'fair attempt': fair,
                          'model': model_name}

                manager.memory.add(rgb, dep, q_values, [rgb_raw, dep_raw, img_pred, img_after], kwargs)
                args['epoch'] += 1
                if rospy.is_shutdown(): return

            # save the model
            new_dir = os.path.join('/home/ubuntu/Documents/models', time.strftime("%Y%m%d-%H%M%S") + ' ' + model_name)
            os.mkdir(new_dir)
            file = os.path.join(new_dir, 'model.pt')
            torch.save(model, file)

    writer.close()

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Deep reinforcement learning in PyTorch.')

    parser.add_argument('--real', dest='is_sim', action='store_false', default=True, help='Real or simulated, default is simulated.')
    parser.add_argument('--gpu', dest='is_cuda', action='store_true', default=False, help='GPU mode, default is CPU.')
    parser.add_argument('--test', dest='is_test', action='store_true', default=False, help='Testing only.')
    parser.add_argument('--train', dest='is_train', action='store_true', default=False, help='Training only')

    args_parser = parser.parse_args()

    # hyperparameters
    args = {
        'epoch_num': 100,  # Número de épocas.
        'epoch': 0,  # Número de épocas.
        'lr': 1e-3,  # Taxa de aprendizado.
        'rl_lr': 0.7,  # Taxa de aprendizado.
        'weight_decay': 8e-5,  # Penalidade L2 (Regularização).
        'batch_size': 10,  # Tamanho do batch.
        'gamma' : 0.99,
        'eps_start' : 0.9,      # initial randomness
        'eps_end' : 0.05,       # final randomness
        'eps_decay' : 100,      # exponential decay
        'target_update' : 10,
        'grasp_reward': 1,
        'proportional_reward': 0.25,
        'min_replay_memory': 20
    }

    # convert to dictionary
    args['simulation'] = args_parser.is_sim
    args['device'] = torch.device('cuda') if args_parser.is_cuda else torch.device('cpu')
    args['testing'] = args_parser.is_test
    args['training'] = args_parser.is_train
    args['kinematic'] = False

    main(args)