import gym
import rospy
import roslaunch
import time
import numpy as np

from .mavros_ctrl_common import MavrosCtrlCommon
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from gym.utils import seeding

class SingelPx4UavEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        print('init_env')
        gazebo_env.GazeboEnv.__init__(self, "SinglePx4Uav-v0.launch")
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        

        self.action_space = spaces.Discrete(6) # U, D, F, B, L, R
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        print('inti uav cmd class')
        rospy.init_node('gym', anonymous=True)
        self.uav_cmd = MavrosCtrlCommon()
        self.uav_cmd.setUp()
        print('uav cmd setuped')
        self.uav_cmd.getReady()
        while True:
            self.uav_cmd.moveUp()
        print('uav cmd getReady')

    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        if action == 0: #xPlus
            self.uav_cmd.moveUp()
        
        if action == 1: #xMin
            self.uav_cmd.moveXMin()

        if action == 2: #yPlus
            self.uav_cmd.moveYPlus()

        if action == 3: #yMin
            self.uav_cmd.moveYMin()

        if action == 4: #up
            self.uav_cmd.moveUp()

        if action == 5: #down
            self.uav_cmd.moveDown()

        data = None
        # TODO: get scan data from env

        reward = 0
        state = data # process needed
        done = False # check needed
        return state, reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        data = None
        # TODO: get scan data from env

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")
        state = data # process needed
        return state
            

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



