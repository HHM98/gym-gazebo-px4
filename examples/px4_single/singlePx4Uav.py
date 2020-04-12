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

""" def render():
    render_skip = 0 #Skip first X episodes.
    render_interval = 50 #Show render Every Y episodes.
    render_episodes = 10 #Show Z episodes every rendering.

    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
        env.render(close=True) """


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


if __name__ == '__main__':
    des_list = np.array([[40, 30, 30], [35, -20, 20], [-20, 35, 10], [-20, -40, 25]])

    env = gym.make('SinglePx4Uav-v0')

    outdir = '/home/huhaomeng/px4_train/gazebo_gym_experiments'
    path = '/home/huhaomeng/px4_train/weights/px4_nav_dqn_ep'

    continue_execution = True
    # fill this if continue_execution=True
    resume_epoch = '50'  # change to epoch to continue from
    resume_path = path + resume_epoch
    weights_path = resume_path + '.h5'
    monitor_path = resume_path
    params_json = resume_path + '.json'

    if not continue_execution:
        # Each time we take a sample and update our weights it is called a mini-batch.
        # Each time we run through the entire dataset, it's called an epoch.
        # PARAMETER LIST
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

    env._max_episode_steps = steps  # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir, force=not continue_execution, resume=continue_execution)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()
    counter = 0

    # iterating from 'current epoch'.
    for epoch in range(current_epoch + 1, epochs + 1, 1):
        counter += 1
        random_int = random.randint(0, 3)
        env.env.set_des(des_list[random_int])
        print ('set des: ' + str(des_list[random_int]))
        observation = env.reset()

        cumulated_reward = 0
        done = False
        episode_step = 0
        # print('observation: ' + str(observation))
        # run until env returns done
        while not done:
            qValues = deepQ.getQValues(observation)

            action = deepQ.selectAction(qValues, explorationRate)

            n_observation, reward, done, info = env.step(action)

            if stepCounter % 200 == 0 or done == True or episode_step == 0:
                print('EP:' + str(epoch) + ' step:' + str(stepCounter) + ' episode:' + str(episode_step))
                print('@env@ ob:' + str(observation))
                print('@env@ des' + str(des_list[random_int]))
                print('@env@ reward:' + str(reward))
                print('@env@ done:' + str(done))

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            deepQ.addMemory(observation, action, reward, n_observation, done)

            if stepCounter >= learnStart:
                if stepCounter <= updateTargetNetwork:
                    deepQ.learnOnMiniBatch(minibatch_size, False)
                else:
                    deepQ.learnOnMiniBatch(minibatch_size, True)

            observation = n_observation

            if done:
                # restart env when env broken
                if episode_step < 2 or counter > 5:
                    restart_env(env)
                    counter = 0

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

                if (epoch) % 50 == 0:
                    # save model weights and monitoring data every 50 epochs.
                    deepQ.saveModel(path + str(epoch) + '.h5')
                    env._flush()
                    copy_tree(outdir, path + str(epoch))
                    # save simulation parameters.
                    parameter_keys = ['epochs', 'steps', 'updateTargetNetwork', 'explorationRate', 'minibatch_size',
                                      'learnStart', 'learningRate', 'discountFactor', 'memorySize',
                                      'network_inputs', 'network_outputs', 'network_structure', 'current_epoch']
                    parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size,
                                        learnStart, learningRate, discountFactor, memorySize, network_inputs,
                                        network_outputs, network_structure, epoch]
                    parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                    with open(path + str(epoch) + '.json', 'w') as outfile:
                        json.dump(parameter_dictionary, outfile)
            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print("updating target network")

            episode_step += 1

        explorationRate *= 0.996  # epsilon decay
        explorationRate = max(0.10, explorationRate)

    env.close()
