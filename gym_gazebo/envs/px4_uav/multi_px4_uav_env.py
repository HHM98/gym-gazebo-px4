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
import memory
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
        self.ports = [19881, 19883, 19885]

        gazebo_env.GazeboEnv.__init__(self, "MultiPx4Uav-v0.launch")
        rospy.wait_for_service('/gazebo/unpause_physics', 30)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        self.des = np.array([0, 0, 0])
        self.action_space = spaces.Discrete(7)  # U, D, F, B, L, R
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.radius = 10

        cmd = 'python /home/huhaomeng/gym-gazebo/gym_gazebo/envs/px4_uav/multi_mavros_ctrl_server.py {0} {1} &'
        for x in range(0, self.uav_count):
            os.system(cmd.format(x, self.ports[x]))
            # print('@env@ uav{0} ctrl server started port number: {1}'.format(x, self.ports[x]))

        time.sleep(5)
        self.positions = np.empty((self.uav_count, 3))
        self.old_distance_values = np.zeros(self.uav_count)
        self.old_danger_values = np.zeros(self.uav_count)

    def step(self, actions):
        margin = 2
        all_uav_data = []
        rewards = np.zeros(self.uav_count)
        done = False
        done_reason = ''
        old_positions = self.positions.copy()
        dones = np.empty(self.uav_count, bool)

        step_start_time = time.time()
        # execute cmd action for every uav and get new data
        for idx in range(self.uav_count):
            act_code = int(actions[idx])

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
            self.send_msg_get_return(idx, cmd)
        # wait for move
        time.sleep(0.35)
        #get state
        for idx in range(self.uav_count):
            new_uav_date = self.send_msg_get_return(idx, 'state')
            self.positions[idx] = np.array([new_uav_date[0], new_uav_date[1], new_uav_date[2]]).copy()
            all_uav_data.append(new_uav_date)
        execute_action_finish_time = time.time()

        # process origin data
        all_uav_rel_position = self.get_all_uav_rel_pos()

        all_uav_data = self.process_origin_data(all_uav_data, actions[0], all_uav_rel_position)

        all_uav_distance = self.get_all_uav_distance(all_uav_rel_position)

        # execute reward for each uav
        for idx in range(self.uav_count):
            uav_done = False
            uav_data = all_uav_data[idx]
            uav_pos = self.positions[idx]
            uav_old_pos = old_positions[idx]
            # print '@env@ uav_{0} \n data: {1} \n pos: {2} \n o_pos: {3}'.format(
            #     idx, uav_data, uav_pos, uav_old_pos
            # )
            reward = 0
            # leader
            if idx == 0:
                # transition reward
                dist_change = self.cal_distance(uav_old_pos, uav_pos, self.des)
                reward += 4 * dist_change
                # print 'distance change {0} transition reward: {1}'.format(dist_change, reward)

            # follower
            else:
                # distance with leader reward
                old_distance_with_leader = self.cal_distance(old_positions[idx], old_positions[0], old_positions[0])
                distance_with_leader = all_uav_distance[0][idx]
                old_distance_value, useless = self.distance_value(old_distance_with_leader)
                dist_value, uav_done = self.distance_value(distance_with_leader, uav_done)
                if uav_done and done_reason == '':
                    done_reason = 'distance'
                dist_reward = dist_value - old_distance_value
                if distance_with_leader > 5:
                    dist_reward += 1.5 * (old_distance_with_leader - distance_with_leader)
                reward += 1.5 * dist_reward
                # print 'old_dis_with_leader {0} new dis_with_leader is {1} transition reward: {2}'.format(
                #     old_distance_with_leader, distance_with_leader, dist_reward
                # )

            # laser danger punishment
            laser_data = uav_data[3:3 + 9]
            danger_value = 0
            for i in laser_data:
                if i < 1:
                    danger_value -= 100
                    uav_done = True
                    # print '@env@ uav_{0} done because laser'.format(idx)
                elif i <= 6:
                    danger_value -= 6 / (i - 1)
            danger_reward = danger_value - self.old_danger_values[idx]
            reward += danger_reward
            if uav_done and done_reason == '':
                done_reason = 'laser_danger'
            # print 'old danger_value: {0} new danger_value: {1} danger_reward {2}'.format(
            #     self.old_danger_values[idx], danger_value, danger_reward
            # )
            self.old_danger_values[idx] = 2 * danger_value

            # distance reward
            distance_value = 0
            for uav in range(1, self.uav_count):
                if uav != idx:
                    rel_dis = all_uav_distance[idx][uav]
                    value, uav_done = self.distance_value(rel_dis, uav_done)
                    if rel_dis < 5:
                        reward -= 2
                    distance_value += value
                    # print 'distance with uav_{0}: {1} is {2} \n reward {3}'.format(
                    #     uav, self.positions[uav],  rel_dis, dist_reward
                    # )
            dist_reward = distance_value - self.old_distance_values[idx]
            reward += dist_reward
            if uav_done and done_reason == '':
                done_reason = 'distance'
            # print 'uav_{0} \n old_distance_value: {1} distance_value: {2} \n distance reward: {3}'.format(
            #     idx, self.old_distance_values[idx], distance_value, dist_reward
            # )
            self.old_distance_values[idx] = distance_value

            # finish reward
            if idx == 0 and self.is_at_position(self.des, uav_pos, 10):
                # print '@env@ done because finish'
                uav_done = True
                reward += 20
            if uav_done and done_reason == '':
                done_reason = 'finish'

            # out of map punishment
            if (uav_pos[0] < -50 or
                    uav_pos[0] > 50 or
                    np.abs(uav_pos[1]) > 50 or
                    uav_pos[2] > 40 or
                    uav_pos[2] < 4):
                reward -= 50
                uav_done = True
                # print '@env@ uav_{0} done because out of map'.format(idx)
            if uav_done and done_reason == '':
                done_reason = 'out'

            rewards[idx] = reward
            dones[idx] = uav_done

            if uav_done:
                done = True
        # print('@env@ observation:' + str(all_uav_data))
        states = self.std_data(all_uav_data)
        # states = all_uav_data
        # step_finish_time = time.time()
        # print 'exec action cost time: {0} \n full step cost time: {1}'.format(
        #     execute_action_finish_time - step_start_time,
        #     step_finish_time - step_start_time
        # )
        return states, np.sum(rewards), done, {'rewards': rewards, 'dones': dones, 'done_reason': done_reason}

    def reset(self):
        print('@env@ Resets the state of the environment and returns an initial observation.')

        data = []
        for x in range(0, self.uav_count):
            self.send_msg_get_return(x, 'takeoff')

        time.sleep(5)

        for idx in range(self.uav_count):
            data.append(self.send_msg_get_return(idx, 'state'))

        # set uav positions
        for idx in range(self.uav_count):
            self.positions[idx] = data[idx][:3].copy()
        # add extra
        all_uav_rel_position = self.get_all_uav_rel_pos()
        states = self.process_origin_data(data, 6, all_uav_rel_position)
        states = self.std_data(states)
        return states

    def send_msg_get_return(self, uav_number, msg):
        ctrl_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ctrl_client.settimeout(15)
        connected = False
        data = None
        while not connected:
            try:
                # print('@env@ try to connect with ctrl server')
                ctrl_client.connect(('localhost', self.ports[int(uav_number)]))
                connected = True
                # print('@env@ connected with ctrl server port:{0}'.format(self.ports[int(uav_number)]))
            except BaseException as e:
                print('@env@[Error] ' + str(e))
                time.sleep(1)
                pass
        try:
            # print('@env@ sending msg: ' + str(msg))
            ctrl_client.send(msg)
            data = pickle.loads(ctrl_client.recv(1024))
            # print('@env@ send msg {0} to uav_{1} get return: {2}'.format(msg, uav_number, data))
            # done = True
        except BaseException as e:
            print ('@env@[Error] sending message:' + str(e))
            time.sleep(0.5)
        ctrl_client.close()

        if data is None:
            data = 'false'
        return data

    def stop_ctrl_server(self):
        for x in range(0, self.uav_count):
            r_msg = self.send_msg_get_return(x, 'shutdown')
        # print('@env@ ctrl_server shutdown' + str(r_msg))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # calculate pos2's relative position about pos1
    def cal_relative_position(self, pos1, pos2):
        relative_pos = np.empty(len(pos1))
        for idx in range(len(pos1)):
            relative_pos[idx] = pos2[idx] - pos1[idx]
        # print 'rel pos of {0} and {1} is {2}'.format(pos1, pos2, relative_pos)
        return relative_pos

    def get_all_uav_rel_pos(self):
        # calculate all relative position
        relative_position = np.empty((self.uav_count, self.uav_count, 3))
        for uav1_idx in range(self.uav_count):
            for uav2_idx in range(self.uav_count):
                relative_position[uav1_idx][uav2_idx] = self.cal_relative_position(
                    self.positions[uav1_idx], self.positions[uav2_idx]
                )
        return relative_position

    def process_origin_data(self, states, leader_action, rel_positions):
        for idx in range(self.uav_count):
            if idx == 0:
                # change pos to rel pos
                states[0][:3] = self.cal_relative_position(self.des, self.positions[idx])
                # add follower rel pos
                for uav in range(1, self.uav_count):
                    states[0] = np.append(states[0], rel_positions[0][uav].copy())
            else:
                # change pos to rel pos about leader position
                states[idx][:3] = rel_positions[0][idx].copy()
                # add other follower uav rel pos
                for uav in range(1, self.uav_count):
                    if uav != idx:
                        states[idx] = np.append(states[idx], rel_positions[idx][uav].copy())
                states[idx] = np.append(states[idx], np.array([leader_action]))
        # print ('@env@ process data over: ' + str(states))
        return states

    def std_data(self, states):
        for idx in range(self.uav_count):
            # standardization self rel pos
            # leader uav
            if idx == 0:
                for pos_idx in range(3):
                    states[idx][pos_idx] = (states[idx][pos_idx] + 50) / 100
            # follower
            else:
                for pos_idx in range(3):
                    states[idx][pos_idx] = (states[idx][pos_idx] + 12) / 24

            # standardization laser data
            for laser_idx in range(3, 3 + 9):
                if states[idx][laser_idx] > 10 or states[idx][laser_idx] == np.inf:
                    states[idx][laser_idx] = 1
                else:
                    states[idx][laser_idx] = (states[idx][laser_idx] + 0.2) / 9.8

            # standardization rel pos with other uav
            end_idx = len(states[idx])
            if idx != 0:
                end_idx -= 1
            for pos_idx in range(3 + 9, end_idx):
                states[idx][pos_idx] = (states[idx][pos_idx] + 12) / 24
        return states

    def set_des(self, destination):
        self.des = destination

    def cal_distance(self, position, new_position=None, destination=None):
        if destination is None:
            destination = [0, 0, 0]
        if new_position is None:
            new_position = [0, 0, 0]
        old_distance = np.sqrt(
            np.square(destination[0] - position[0]) + np.square(destination[1] - position[1]) + np.square(
                destination[2] - position[2]))
        new_distance = np.sqrt(
            np.square(destination[0] - new_position[0]) + np.square(destination[1] - new_position[1]) + np.square(
                destination[2] - new_position[2]))
        # print 'distance change between {0} and {1} about {2}'.format(position, new_position, destination)
        return old_distance - new_distance

    def distance_value(self, distance, done=False):
        value = 0
        if distance < 0.5:
            # print '@env@ done because distance: ' + str(distance)
            done = True
            value -= 100
        elif 4 < distance < 8:
            value += 4 * (2 - np.abs(distance - 6))
        elif 0.5 <= distance <= 4:
            value -= np.power((5.5 - distance), 3)
        else:
            value -= np.power((distance - 7), 2)
        return value, done

    def is_at_position(self, desired, pos, offset):
        """offset:meters"""
        desired = np.array(desired)
        pos = np.array(pos)
        return np.linalg.norm(desired - pos) < offset

    def get_all_uav_distance(self, all_uav_rel_pos):
        all_uav_rel_dis = np.empty((self.uav_count, self.uav_count))
        for uav1_idx in range(self.uav_count):
            for uav2_idx in range(self.uav_count):
                distance = self.cal_distance(all_uav_rel_pos[uav1_idx][uav2_idx])
                all_uav_rel_dis[uav1_idx, uav2_idx] = distance
        return all_uav_rel_dis


if __name__ == '__main__':

    pass
