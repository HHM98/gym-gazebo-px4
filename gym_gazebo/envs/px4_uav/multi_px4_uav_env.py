#!/usr/bin/env python
from __future__ import division

import gym
import rospy
import roslaunch
import time
import numpy as np
import socket
import subprocess
import os
import pickle
import random

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

from gym.utils import seeding


class MultiPx4UavEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # number of uav
        self.uav_count = 3
        self.ports = [19881, 19882, 19883]

        gazebo_env.GazeboEnv.__init__(self, "MultiPx4Uav-v0.launch")
        rospy.wait_for_service('/gazebo/unpause_physics', 30)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        self.action_space = spaces.Discrete(7)  # U, D, F, B, L, R
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.radius = 1

        cmd = 'python /home/huhaomeng/gym-gazebo/gym_gazebo/envs/px4_uav/multi_mavros_ctrl_server.py {0} {1} &'
        for x in range(0, self.uav_count):
            self.ports[x] = random.randint((19881 + x * 1000), (19881 + (x + 1) * 1000))
            os.system(cmd.format(x, self.ports[x]))
            print('@env@ uav{0} ctrl server started port number: {1}'.format(x, self.ports[x]))

        time.sleep(5)
        self.pos = np.array([0, 0, 0])

    def step(self, action):
        margin = 2
        data = str(action).split('#')
        print('@env@ step cmd {0} data {1}'.format(action, str(data)))
        uav_number = int(data[0])
        act_code = int(data[1])

        cmd = ''
        if act_code == 0:  # xPlus
            cmd = 'moveXPlus' + '#' + str(margin)
        if act_code == 1:  # xMin
            cmd = 'moveXMin' + '#' + str(margin)
        if act_code == 2:  # yPlus
            cmd = 'moveYPlus' + '#' + str(margin)
        if act_code == 3:  # yMin
            cmd = 'moveYMin' + '#' + str(margin)
        if act_code == 4:  # up
            cmd = 'moveUp' + '#' + str(margin)
        if act_code == 5:  # down
            cmd = 'moveDown' + '#' + str(margin)
        if act_code == 6:  # stay
            cmd = 'stay' + '#' + str(margin)
        data = self.send_msg_get_return(uav_number, cmd)

        self.pos = [data[0], data[1], data[2]]
        lidar_ranges = data[3:]

        reward = 0
        done = False
        # calculate reward and check done status
        if uav_number == 2 and act_code == 6:
            done = True

        state = data
        print('@env@ observation:' + str(state))
        print('@env@ reward:' + str(reward))
        print('@env@ done:' + str(done))
        return state, reward, done, {}

    def reset(self):
        print('@env@ Resets the state of the environment and returns an initial observation.')
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            # reset_proxy.call()
            self.reset_proxy()
            print('@env@ reset model place')
        except rospy.ServiceException as e:
            print ("@env@ /gazebo/reset_world service call failed")

        time.sleep(3)
        for x in range(0, self.uav_count):
            self.send_msg_get_return(x, 'reset')
        time.sleep(3)

        data = [[], [], []]
        for x in range(0, self.uav_count):
            data[x] = self.send_msg_get_return(x, 'takeoff')
        state = data
        return state

    def send_msg_get_return(self, uav_number, msg):
        ctrl_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False
        while not connected:
            try:
                # print('@env@ try to connect with ctrl server')
                ctrl_client.connect(('localhost', self.ports[int(uav_number)]))
                connected = True
                print('@env@ connected with ctrl server port:{0}'.format(self.ports[int(uav_number)]))
            except BaseException as e:
                print('@env@[Error] ' + str(e))
                time.sleep(1)
                pass
        try:
            print('@env@ sending msg: ' + str(msg))
            ctrl_client.send(msg)
            data = pickle.loads(ctrl_client.recv(1024))
            print('@env@ send msg ' + msg + ' get return: ' + str(data))
            # done = True
        except BaseException as e:
            print ('@env@[Error] ' + str(e))
            time.sleep(1)
        ctrl_client.close()
        return data

    def stop_ctrl_server(self):
        r_msg = self.send_msg_get_return('shutdown')
        print('@env@ ' + r_msg + ' ctrl_server shutdown')

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


