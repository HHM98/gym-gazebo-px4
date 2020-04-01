#!/usr/bin/env python
import gym
from gym import wrappers
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import signal
import deepq
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
    env = gym.wrappers.Monitor(env, outdir, force=not continue_execution, resume=continue_execution)


def load_leader_weights(leader_path, epoch, _outdir):
    resume_path = leader_path + epoch
    weights_path = resume_path + '.h5'
    monitor_path = resume_path
    params_json = resume_path + '.json'

    with open(params_json) as file:
        d = json.load(file)
        learnStart = d.get('learnStart')
        learningRate = d.get('learningRate')
        discountFactor = d.get('discountFactor')
        memorySize = d.get('memorySize')
        network_inputs = d.get('network_inputs')
        network_outputs = d.get('network_outputs')
        network_structure = d.get('network_structure')

    leader_deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
    leader_deepQ.initNetworks(network_structure)

    leader_deepQ.loadWeights(weights_path)

    clear_monitor_files(_outdir)
    copy_tree(monitor_path, _outdir)


if __name__ == '__main__':
    env = gym.make('MultiPx4Uav-v0')
    print ('sleep 5s')
    time.sleep(5)
    data = env.reset()

    while True:
        cmd = input('input cmd please\n')
        print ('get input' + cmd)
        if cmd == 'reset':
            env.reset()
        else:
            env.step(cmd)

    print ('@rl@ ' + str(data))
    time.sleep(9999999)

    outdir = '/home/huhaomeng/px4_train/gazebo_gym_experiments'

    # load leader deepQ network as leader_deepQ
    leader_weights_path = '/home/huhaomeng/px4_train/weights/px4_nav_dqn_ep'
    leader_weights_epoch = '50'
    load_leader_weights(leader_weights_path, leader_weights_epoch, outdir)

    path = '/home/huhaomeng/px4_train/wrights/multi_px4_dan_ep'
    # init follower uav
    contine_excution = False
    # choose the epoch
    resume_epoch = '50'
    resume_path = path + resume_epoch
    weights_path = resume_path + '.h5'
    monitor_path = resume_path
    params_json = resume_path + '.json'

    if not contine_excution:
        # Each time we take a sample and update our weights it is called a mini-batch.
        # Each time we run through the entire dataset, it's called an epoch.
        # PARAMETER LIST
        epochs = 1000
        steps = 1000
        updateTargetNetwork = 10000
        explorationRate = 1
        minibatch_size = 64
        learnStart = 64
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 100000
        network_inputs = 12
        network_outputs = 7
        network_structure = [30, 30]
        current_epoch = 0

        # init and create deepQ
        deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)

    else:
        # load weights, monitor info and parameter info,
        with open(params_json) as outfile:
            d = json.load(outfile)
            epochs = d.get('epochs')
            steps = d.get('steps')
            updateTargetNetwork = d.get('updateTargetNetwork')
            explorationRate = d.get('explorationRate')
            minibatch_size = d.get('minibatch_size')
            learnStart = d.get('learnStart')
            learningRate = d.get('learningRate')
            discountFactor = d.get('discountFactor')
            memorySize = d.get('memorySize')
            network_inputs = d.get('network_inputs')
            network_outputs = d.get('network_outputs')
            network_structure = d.get('network_structure')
            current_epoch = d.get('current_epoch')

        deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)

        deepQ.loadWeights(weights_path)

        clear_monitor_files(outdir)
        copy_tree(monitor_path, outdir)

    env._max_episode_steps = steps # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir, force=not contine_excution, resume=contine_excution)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()

    # iteration from current epoch
    for epoch in range(current_epoch + 1, epochs + 1, 1):
        print ('do something')

    env.close()