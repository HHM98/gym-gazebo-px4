import pickle
import Queue
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

path = '/home/huhaomeng/px4_train/single_px4/weights/px4_nav_dqn_ep500memory.pkl'
with open(path, 'rb') as file:
    _memory = pickle.load(file)

    # styling options
    matplotlib.rcParams['toolbar'] = 'None'
    plt.style.use('ggplot')
    plt.xlabel("Episodes")
    plt.ylabel('reward')
    fig = plt.gcf().canvas.set_window_title('reward_graph')

    cnt = 0

    reason_count = [0, 0, 0, 0]
    rate_record = []
    queue_sum = 0
    resent_rate = False
    q = Queue.Queue()

    for reason in _memory.done_reason:
        cnt += 1.0
        q.put(reason)
        if reason is '':
            reason_count[0] += 1
        elif reason in 'laser_danger':
            reason_count[1] += 1
        elif reason in 'out of map':
            reason_count[2] += 1
        elif reason in 'finish':
            reason_count[3] += 1

        rate_cnt = cnt
        # cal recent rate
        if resent_rate and cnt > 50:
            abandon_reason = q.get()

            if abandon_reason is '':
                reason_count[0] -= 1
            elif abandon_reason in 'laser_danger':
                reason_count[1] -= 1
            elif abandon_reason in 'out of map':
                reason_count[2] -= 1
            elif abandon_reason in 'finish':
                reason_count[3] -= 1

            rate_cnt = 50.0

        timeout_rate = reason_count[0] / rate_cnt
        laser_rate = reason_count[1] / rate_cnt
        out_rate = reason_count[2] / rate_cnt
        finish_rate = reason_count[3] / rate_cnt
        rate_record.append([timeout_rate, laser_rate, out_rate, finish_rate])


    print 'hello'
    plt.plot(rate_record)
    # plt.plot(ave_record)
    plt.pause(0.000001)
