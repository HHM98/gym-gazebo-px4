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
import pickle
import liveplot
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

    new_env = gym.make('MultiPx4Uav-v0')
    time.sleep(3)
    env._max_episode_steps = steps
    env.env = new_env

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
    des_list = np.array([[40, 35, 30], [35, -15, 20], [-40, 30, 10], [-35, -35, 25]])

    env = gym.make('MultiPx4Uav-v0')
    print ('sleep 3s')
    time.sleep(3)

    # while True:
    #     cmd = input('input cmd please\n')
    #     print ('get input' + cmd)
    #     if cmd == 'reset':
    #         env.reset()
    #     else:
    #         env.step(cmd)
    #
    # print ('@rl@ ' + str(data))
    # time.sleep(9999999)

    outdir = '/home/huhaomeng/px4_train/multi_px4/gazebo_gym_experiments'
    path = '/home/huhaomeng/px4_train/multi_px4/wrights/multi_px4_dan_ep'
    plotter = liveplot.LivePlot(outdir)

    uav_count = 3

    continue_execution = True
    # choose the epoch
    resume_epoch = '175'
    resume_path = path + resume_epoch
    weights_path = [resume_path + '_leader.h5', resume_path + '_follower.h5']
    monitor_path = resume_path
    params_json = resume_path + '.json'

    # init networks
    if not continue_execution:
        # Each time we take a sample and update our weights it is called a mini-batch.
        # Each time we run through the entire dataset, it's called an epoch.
        # PARAMETER LIST
        epochs = 350
        steps = 1000
        updateTargetNetwork = 5000
        explorationRate = 1
        minibatch_size = 32
        learnStart = 64
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 100000
        network_inputs = 9 + 3 * uav_count
        network_outputs = 7
        network_structure = [50, 50]
        current_epoch = 0

        # init and create deepQ
        leader_deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize,
                                   discountFactor, learningRate, learnStart)
        leader_deepQ.initNetworks(network_structure)

        follower_deepQ = deepq.DeepQ((network_inputs - 2), network_outputs, memorySize,
                                     discountFactor, learningRate, learnStart)
        follower_deepQ.initNetworks(network_structure)

        leader_deepQ.memory.liveplot = plotter

    # load weights, monitor info and parameter info,
    else:
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

        uav_deepQs = []
        leader_deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize,
                                   discountFactor, learningRate, learnStart)
        follower_deepQ = deepq.DeepQ(network_inputs - 2, network_outputs, memorySize,
                                     discountFactor, learningRate, learnStart)
        leader_deepQ.initNetworks(network_structure)
        leader_deepQ.loadWeights(weights_path[0])

        follower_deepQ.initNetworks(network_structure)
        follower_deepQ.loadWeights(weights_path[1])

        with open(resume_path + 'leader_memory.pkl', 'rb') as openfile:
            leader_deepQ.memory = pickle.load(openfile)
        with open(resume_path + 'follower_memory.pkl', 'rb') as openfile:
            follower_deepQ.memory = pickle.load(openfile)
        clear_monitor_files(outdir)
        copy_tree(monitor_path, outdir)

    plotter = leader_deepQ.memory.liveplot
    env._max_episode_steps = steps  # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir, force=not continue_execution, resume=continue_execution)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = [0, 0, 0]

    start_time = time.time()
    restart_cnt = 0

    # iteration from current epoch
    for epoch in range(current_epoch + 1, epochs + 1, 1):
        restart_cnt += 1
        # random choose destination
        random_des_idx = random.choice([3])
        env.env.set_des(des_list[random_des_idx])
        print ('set des: ' + str(des_list[random_des_idx]))

        observations = env.reset()
        # print ('origin obs: ' + str(observations))
        cumulated_reward = [0, 0, 0]
        done = False
        episode_step = 0
        # run until done
        while not done:
            action_start_time = time.time()

            actions = np.empty(uav_count)
            for idx in range(0, uav_count):
                if idx == 0:
                    qValues = leader_deepQ.getQValues(observations[idx])
                    actions[idx] = int(leader_deepQ.selectAction(qValues, explorationRate))
                else:
                    qValues = follower_deepQ.getQValues(observations[idx])
                    actions[idx] = int(follower_deepQ.selectAction(qValues, explorationRate))

            n_observations, reward, done, info = env.step(actions)
            rewards = info['rewards']
            dones = info['dones']

            if stepCounter % 200 == 0 or done or episode_step == 0:
                print('@env@ ob:' + str(observations))
                print('@env@ des:' + str(des_list[random_des_idx]))
                print('@env@ actions:' + str(actions))
                print('@env@ reward:' + str(reward))
                print('@env@ rewards:' + str(rewards))
                print('@env@ done:' + str(done))
                print('EP:' + str(epoch) + ' step:' + str(stepCounter) + ' episode:'
                      + str(episode_step) + ' er:' + str(explorationRate))

            action_finish_time = time.time()
            # print 'action cost time: ' + str(action_finish_time - action_start_time)

            for idx in range(uav_count):
                cumulated_reward[idx] += rewards[idx]
                if idx == 0:
                    if not ('nan' in (str(observations) + str(actions) + str(n_observations))):
                        leader_deepQ.addMemory(observations[idx], actions[idx], rewards[idx] + 0.25 * reward, n_observations[idx], dones[idx])
                    if stepCounter >= learnStart:
                        leader_deepQ.learnOnMiniBatch(minibatch_size, stepCounter > updateTargetNetwork)
                else:
                    follower_deepQ.addMemory(observations[idx], actions[idx], rewards[idx] + 0.25 * reward, n_observations[idx], dones[idx])
                    if stepCounter >= learnStart:
                        follower_deepQ.learnOnMiniBatch(minibatch_size, stepCounter > updateTargetNetwork)

                highest_reward[idx] = max(highest_reward[idx], cumulated_reward[idx])
            # print 'learn cost time: ' + str(learn_finish_time - learn_start_time)

            observations = n_observations

            if done:
                # record cumulated_reward and done reason
                leader_deepQ.episode_record(cumulated_reward, info['done_reason'])

                # restart env when env broken and used 5 times
                restart_env(env)

                last100Scores[last100ScoresIndex] = episode_step
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(
                        steps) + " Episode steps   Exploration=" + str(round(explorationRate, 2)))
                else:
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(
                        steps) + " Episode steps - last100 Steps : " + str(
                        (sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(
                        cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (
                               h, m, s))

                # save model weights and monitoring data every 25 epochs.
                if epoch % 25 == 0:
                    leader_deepQ.saveModel(path + str(epoch) + '_leader.h5')
                    follower_deepQ.saveModel(path + str(epoch) + '_follower.h5')
                    env._flush()
                    copy_tree(outdir, path + str(epoch))
                    parameter_keys = ['epochs', 'steps', 'updateTargetNetwork', 'explorationRate', 'minibatch_size',
                                      'learnStart', 'learningRate', 'discountFactor', 'memorySize',
                                      'network_inputs', 'network_outputs', 'network_structure', 'current_epoch']
                    parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size,
                                        learnStart, learningRate, discountFactor, memorySize, network_inputs,
                                        network_outputs, network_structure, epoch]
                    parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                    with open(path + str(epoch) + '.json', 'w') as outfile:
                        json.dump(parameter_dictionary, outfile)
                    with open(path + str(epoch) + 'leader_memory.pkl', 'w') as outfile:
                        pickle.dump(leader_deepQ.memory, outfile, True)
                    with open(path + str(epoch) + 'follower_memory.pkl', 'w') as outfile:
                        pickle.dump(follower_deepQ.memory, outfile, True)
            episode_step += 1
            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                leader_deepQ.updateTargetNetwork()
                follower_deepQ.updateTargetNetwork()
                print 'updating target network'

        explorationRate *= 0.992
        explorationRate = max(0.1, explorationRate)

        if epoch % 10 == 0:
            plotter.plot(env)

    env.close()
