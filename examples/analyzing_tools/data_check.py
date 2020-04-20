import numpy as np
import pickle
import memory

def reverse_std(states, idx):
    # standardization self rel pos
    # leader uav
    if idx == 0:
        for pos_idx in range(3):
            states[pos_idx] = states[pos_idx] * 100 - 50
    # follower
    else:
        for pos_idx in range(3):
            states[pos_idx] = states[pos_idx] * 24 - 12

    # standardization laser data
    for laser_idx in range(3, 3 + 9):
        states[laser_idx] = states[laser_idx]*9.8 - 0.2

    # standardization rel pos with other uav
    end_idx = len(states)
    if idx != 0:
        end_idx -= 1
    for pos_idx in range(3 + 9, end_idx):
        states[pos_idx] = states[pos_idx]*24 - 12

path = '/home/huhaomeng/px4_train/multi_px4/wrights/multi_px4_dan_ep350leader_memory.pkl'
with open(path, 'rb') as file:
    _memory = pickle.load(file)
    reward = []
    while True:
        batch = _memory.getMiniBatch(128)
        cnt = 0
        for b in batch:
            reverse_std(b['state'], 1)
            reverse_std(b['newState'], 1)
            print b['state']
            act_code = int(b['action'])
            margin = ''
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
            print cmd
            print b['newState']
            print b['reward']
            if b['isFinal'] == True:
                print 'shit'


    print 'hh'