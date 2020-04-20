import pickle
import Queue
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

path = '/home/huhaomeng/px4_train/multi_px4/wrights/multi_px4_dan_ep350leader_memory.pkl'
with open(path, 'rb') as file:
    _memory = pickle.load(file)
    reward = []

    # styling options
    matplotlib.rcParams['toolbar'] = 'None'
    plt.style.use('ggplot')
    plt.xlabel("Episodes")
    plt.ylabel('reward')
    fig = plt.gcf().canvas.set_window_title('reward_graph')
    reward_his = []

    cnt = 0

    ave = 0
    ave_record = []
    queue = Queue.Queue()
    queue_sum = 0

    for rewards in _memory.rewards_record:
        reward = np.sum(rewards)
        # if np.abs(reward) > 2000:
        #     reward = ave
        queue.put(reward)
        queue_sum += reward
        if queue.qsize() > 15:
            queue_sum -= queue.get()
        ave = (reward + ave * cnt) / (cnt + 1)
        near_ave = queue_sum / queue.qsize()
        cnt += 1
        ave_record.append([ave, near_ave])

    print 'hello'
    plt.plot(ave_record)
    # plt.plot(ave_record)
    plt.pause(0.000001)
