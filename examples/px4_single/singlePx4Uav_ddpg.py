#!/usr/bin/env python
import gym
from gym import wrappers
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import ddpg
import random
import numpy as np


def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]


def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print(file)
        os.unlink(file)


def restart_env(env):
    env.stop_ctrl_server()
    env.close()

    tmp = os.popen("ps -Af").read()
    gzclient_count = tmp.count('gzclient')
    gzserver_count = tmp.count('gzserver')
    roscore_count = tmp.count('roscore')
    rosmaster_count = tmp.count('rosmaster')
    px4_count = tmp.count('px4')
    mavros_count = tmp.count('mavros_node')
    if gzclient_count > 0:
        os.system("killall -9 gzclient")
    if gzserver_count > 0:
        os.system("killall -9 gzserver")
    if rosmaster_count > 0:
        os.system("killall -9 rosmaster")
    if roscore_count > 0:
        os.system("killall -9 roscore")
    if px4_count > 0:
        os.system("killall -9 px4")
    if mavros_count > 0:
        os.system("killall -9 mavros_node")

    if (gzclient_count or gzserver_count or roscore_count or rosmaster_count or px4_count or mavros_count > 0):
        os.wait()

    env = gym.make('SingelPx4Uav-v0')
    env._max_episode_steps = steps
    env = gym.wrappers.Monitor(env, outdir, force=not False, resume=False)


if __name__ == '__main__':
    des_list = np.array([[40, 30, 30], [35, -20, 20], [-20, 35, 10], [-20, -40, 25]])

    env = gym.make('SinglePx4Uav-v0')

    outdir = '/home/huhaomeng/px4_train/gazebo_gym_experiments'
    path = '/home/huhaomeng/px4_train/weights/px4_nav_ddpg_'

    epochs = 1000
    steps = 1000
    updateTargetNetwork = 5000
    explorationRate = 1
    minibatch_size = 32
    learnStart = 32
    learningRate = 0.0001
    discountFactor = 0.99
    memorySize = 1000000
    network_inputs = 12
    network_outputs = 3
    network_structure = [30, 30]
    current_epoch = 0
    max_margin = 3

    ddpg = ddpg.DDPG(network_inputs, network_outputs, memorySize, learningRate, discountFactor, learnStart, max_margin)
    ddpg.init_net_works(network_structure)

    restart_cnt = 0
    stepCounter = 0

    for epoch in range(1, epochs + 1):
        restart_cnt += 1
        random_des = random.randint(0, 3)
        env.env.set_des(des_list[random_des])
        print ('set des: ' + str(des_list[random_des]))
        observation = env.reset()

        done = False
        episode_step = 0

        while not done:
            action = ddpg.get_action(observation.reshape(1, network_inputs), explorationRate)

            n_observation, reward, done, info = env.step(action)
            if stepCounter % 100 == 0 or done == True or episode_step == 1:
                print('EP: ' + str(epoch) + ' step:' + str(stepCounter) + ' episode:' + str(episode_step))
                print('@env@ ob:' + str(observation))
                print('@env@ des' + str(des_list[random_des]))
                print('@env@ reward:' + str(reward))
                print('@env@ done:' + str(done))

            ddpg.add_memory(observation, action, reward, n_observation, done)

            if stepCounter >= learnStart:
                ddpg.learn_on_batch(minibatch_size)

            observation = n_observation

            if done:
                if episode_step < 2 or restart_cnt > 5:
                    restart_env(env)
                    restart_cnt = 0

                print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(
                    steps) + " Episode steps   Exploration=" + str(round(explorationRate, 2)))

            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                ddpg.update_target()
                print("updating target network")
            episode_step += 1
        restart_cnt += 1
        explorationRate *= 0.996
        explorationRate = max(0.10, explorationRate)

    env.close()
