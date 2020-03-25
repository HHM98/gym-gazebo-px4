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

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

from gym.utils import seeding
class SingelPx4UavEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        self.des = [170, 0, 15]

        gazebo_env.GazeboEnv.__init__(self, "SinglePx4Uav-v0.launch")
        rospy.wait_for_service('/gazebo/unpause_physics', 30)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        self.action_space = spaces.Discrete(7) # U, D, F, B, L, R
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.radius = 1
        
        os.system('python /home/huhaomeng/gym-gazebo/gym_gazebo/envs/px4_uav/mavros_ctrl_server.py &')
        
        # self.server_process = subprocess.Popen('python','/home/huhaomeng/gym-gazebo/gym_gazebo/envs/px4_uav/mavros_ctrl_server.py')
        print('@env@ ctrl server started')
        time.sleep(5)
        
        self.pos = np.array([0,0,0])

    def step(self, action):
        margin = 2
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        old_position = [self.pos[0],
                        self.pos[1],
                        self.pos[2]]

        cmd = ''
        if action == 0: #xPlus
            cmd = 'moveXPlus' + '#' + str(margin)
        if action == 1: #xMin
            cmd = 'moveXMin' + '#' + str(margin)
        if action == 2: #yPlus
            cmd = 'moveYPlus' + '#' + str(margin)
        if action == 3: #yMin
            cmd = 'moveYMin' + '#' + str(margin)
        if action == 4: #up
            cmd = 'moveUp' + '#' + str(margin)
        if action == 5: #down
            cmd = 'moveDown' + '#' + str(margin)
        if action == 6: #stay
            cmd = 'stay' + '#' + str(margin)
        data = self.sendMsgGetReturn(cmd)
        self.pos = [data[0], data[1], data[2]]
        lidar_ranges = data[3:]
   
        print('@env@ data' + str(data))

        reward = 0
        done = False # done check

        # finish reward
        if self.is_at_position(self.des[0], self.des[1], self.des[2], 
                               self.pos[0], self.pos[1], self.pos[2], 
                               self.radius):
            done = True
            reward = reward + 1000
        # move reward
        reward = reward + 2 * self.cal_distence(old_position, self.pos, self.des)

        # danger reward
        for i in lidar_ranges:
            if i < 1.5:
                reward = -1000
                done = True
            elif i <= 6 :
                reward = reward - 1 / (i - 1.6)
                
        # fail reward
        if (self.pos[0] < -50 or
            self.pos[0] > 150 or
            np.abs(self.pos[1]) > 50 or
            self.pos[2] > 40 or
            self.pos[2] < 1):
            reward = reward - 1000
            done = True

        state = np.append(data, self.des) # process needed

        print('@env@ observation:' + str(state))
        print('@env@ reward:' + str(reward))
        print('@env@ done:' + str(done))
        return state, reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        print('@env@ Resets the state of the environment and returns an initial observation.')

        rospy.wait_for_service('/gazebo/reset_world')
        try:
            #reset_proxy.call()
            self.reset_proxy()
            print('@env@ reset model place')
        except (rospy.ServiceException) as e:
            print ("@env@ /gazebo/reset_world service call failed")

        self.sendMsgGetReturn('reset')
        print('@env@ sleep 10s')
        time.sleep(10)
        # print('@env@ sleep 10s over')
        # Unpause simulation to make observation
        # print('Unpause simulation to make observation')
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.unpause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/unpause_physics service call failed")

        # reset ctrl and get observation
        # print('@env@ send takeoff msg')
        data = self.sendMsgGetReturn('takeoff')
        print ('@env@ takeoff init state:' + str(data))



        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")

        state = np.append(data, self.des)
        return state

    def set_des(self, destination):
        self.des = destination
            
    def is_at_position(self, tx, ty, tz, x, y, z, offset):
        """offset:meters"""
        desired = np.array((tx, ty, tz))
        pos = np.array((x, y, z))
        return np.linalg.norm(desired - pos) < offset

    def cal_distence(self, old_position, new_position, destination):
        old_distance = np.sqrt(np.square(destination[0] - old_position[0]) + np.square(destination[1] - old_position[1]) + np.square(destination[2] - old_position[2]))
        
        new_distance = np.sqrt(np.square(destination[0] - new_position[0]) + np.square(destination[1] - new_position[1]) + np.square(destination[2] - new_position[2]))
        
        return old_distance - new_distance

    def stop_ctrl_server(self):
        r_msg = self.sendMsgGetReturn('shutdown')
        print('@env@ ' + r_msg +' ctrl_server shutdown')

    def sendMsgGetReturn(self, msg):
        ctrl_client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        connected = False
        while not connected:
            try:
                # print('@env@ try to connect with ctrl server')
                ctrl_client.connect(('localhost',19881))
                connected = True
                print('@env@ connected with ctrl server')
            except BaseException as e:
                print('@env@[Error] ' + str(e))
                time.sleep(1)
                pass

        # done = False
        # while not done:
        # print('@env@ send msg: ' + msg)
        try:
                ctrl_client.send(msg)
                data = pickle.loads(ctrl_client.recv(1024))
                print('@env@ send msg '+ msg + ' get return: ' + str(data))
                # done = True
        except BaseException as e:
                time.sleep(1)
                print ('@env@[Error] ' + str(e))
        ctrl_client.close()
        return data


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]




