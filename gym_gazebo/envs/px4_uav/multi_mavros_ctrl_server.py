#!/usr/bin/env python
# need two args 1. uav_number 2. port_number
from __future__ import division

import sys
import rospy
import math
import numpy as np
import inspect
import ctypes
import socket
import time
import pickle

from geometry_msgs.msg import PoseStamped, Quaternion
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, State, \
    WaypointList
from mavros_msgs.srv import CommandBool, ParamGet, SetMode, WaypointClear, \
    WaypointPush
from std_msgs.msg import Header
from pymavlink import mavutil
from sensor_msgs.msg import NavSatFix, LaserScan
from threading import Thread
from tf.transformations import quaternion_from_euler


class MavrosCtrlCommon():
    def __init__(self, number, *args):
        self.uav_number = str(number)
        self.uav_prefix = 'uav' + self.uav_number + '/'
        pass

    def setUp(self):
        print('@ctrl_server@ uav{0} setUp'.format(self.uav_number))
        self.pos = PoseStamped()
        self.altitude = Altitude()
        self.extended_state = ExtendedState()
        self.global_position = NavSatFix()
        self.home_position = HomePosition()
        self.local_position = PoseStamped()
        self.misson_wp = WaypointList()
        self.state = State()
        self.scan = LaserScan()
        self.mav_type = None
        self.ready = False

        # Target offset radius
        self.radius = 0.25

        self.sub_topics_ready = {
            key: False
            for key in [
                'alt', 'ext_state', 'global_pos', 'home_pos', 'local_pos',
                'mission_wp', 'state', 'scan'
            ]
        }

        # ROS services
        service_timeout = 60
        rospy.loginfo("waiting for ROS services")

        try:
            rospy.wait_for_service(self.uav_prefix + 'mavros/param/get', service_timeout)
            rospy.wait_for_service(self.uav_prefix + 'mavros/cmd/arming', service_timeout)
            rospy.wait_for_service(self.uav_prefix + 'mavros/mission/push', service_timeout)
            rospy.wait_for_service(self.uav_prefix + 'mavros/mission/clear', service_timeout)
            rospy.wait_for_service(self.uav_prefix + 'mavros/set_mode', service_timeout)
            rospy.loginfo("ROS services are up")
        except rospy.ROSException:
            print("@ctrl_server@ failed to connect to services")
        self.get_param_srv = rospy.ServiceProxy(self.uav_prefix + 'mavros/param/get',
                                                ParamGet)
        self.set_arming_srv = rospy.ServiceProxy(self.uav_prefix + 'mavros/cmd/arming',
                                                 CommandBool)
        self.set_mode_srv = rospy.ServiceProxy(self.uav_prefix + 'mavros/set_mode',
                                               SetMode)
        self.wp_clear_srv = rospy.ServiceProxy(self.uav_prefix + 'mavros/mission/clear',
                                               WaypointClear)
        self.wp_push_srv = rospy.ServiceProxy(self.uav_prefix + 'mavros/mission/push',
                                              WaypointPush)

        # ROS subscribers
        self.alt_sub = rospy.Subscriber(self.uav_prefix + 'mavros/altitude',
                                        Altitude,
                                        self.altitude_callback)
        self.ext_state_sub = rospy.Subscriber(self.uav_prefix + 'mavros/extended_state',
                                              ExtendedState,
                                              self.extended_state_callback)
        self.global_pos_sub = rospy.Subscriber(self.uav_prefix + 'mavros/global_position/global',
                                               NavSatFix,
                                               self.global_position_callback)
        self.home_pos_sub = rospy.Subscriber(self.uav_prefix + 'mavros/home_position/home',
                                             HomePosition,
                                             self.home_position_callback)
        self.local_pos_sub = rospy.Subscriber(self.uav_prefix + 'mavros/local_position/pose',
                                              PoseStamped,
                                              self.local_position_callback)
        self.mission_wp_sub = rospy.Subscriber(self.uav_prefix + 'mavros/mission/waypoints',
                                               WaypointList,
                                               self.mission_wp_callback)
        self.state_sub = rospy.Subscriber(self.uav_prefix + 'mavros/state', State,
                                          self.state_callback)
        self.get_scan_srv = rospy.Subscriber('iris_rplidar_' + self.uav_number + '/laser/scan',
                                             LaserScan,
                                             self.scan_callback)

        # ROS publisers
        self.pos_setpoint_pub = rospy.Publisher(
            self.uav_prefix + 'mavros/setpoint_position/local', PoseStamped, queue_size=1)

        # send setpoints in seperate thread
        self.pos_thread = Thread(target=self.send_pos, args=())
        self.pos_thread.daemon = True
        self.pos_thread.start()

        print('@ctrl_server@ setUp over')
        pass

    def getReady(self):
        print('@ctrl_server@ uav{0} getReady and takeoff'.format(self.uav_number))
        """Test offboard position control"""

        # make sure the simulation is ready to start the mission
        self.wait_for_topics(60)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,
                                   10, -1)

        self.set_mode("OFFBOARD", 5)
        self.set_arm(True, 5)
        self.reach_position(0, 0, 15, 5)

        self.ready = True

    def shutDown(self):
        self.stop_thread(self.pos_thread)

    def _async_raise(self, tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(self, thread):
        self._async_raise(thread.ident, SystemExit)
        print('@ctrl_server@ stop_thread')

    def tearDown(self):
        pass

    # Callback functions
    def altitude_callback(self, data):
        self.altitude = data

        if not self.sub_topics_ready['alt'] and not math.isnan(data.amsl):
            self.sub_topics_ready['alt'] = True

    def extended_state_callback(self, data):
        self.extended_state = data

        if not self.sub_topics_ready['ext_state']:
            self.sub_topics_ready['ext_state'] = True

    def global_position_callback(self, data):
        self.global_position = data

        if not self.sub_topics_ready['global_pos']:
            self.sub_topics_ready['global_pos'] = True

    def home_position_callback(self, data):
        self.home_position = data

        if not self.sub_topics_ready['home_pos']:
            self.sub_topics_ready['home_pos'] = True

    def local_position_callback(self, data):
        self.local_position = data

        if not self.sub_topics_ready['local_pos']:
            self.sub_topics_ready['local_pos'] = True

    def mission_wp_callback(self, data):
        self.mission_wp = data

        if not self.sub_topics_ready['mission_wp']:
            self.sub_topics_ready['mission_wp'] = True

    def state_callback(self, data):
        self.state = data

        if not self.sub_topics_ready['state'] and data.connected:
            self.sub_topics_ready['state'] = True

    def scan_callback(self, data):
        self.scan = data

        if not self.sub_topics_ready['scan']:
            self.sub_topics_ready['scan'] = True

    # Helper methods
    def send_pos(self):
        rate = rospy.Rate(30)  # Hz
        self.pos.header = Header()
        self.pos.header.frame_id = "base_footprint"

        while not rospy.is_shutdown():
            self.pos.header.stamp = rospy.Time.now()
            self.pos_setpoint_pub.publish(self.pos)
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def is_at_position(self, x, y, z, offset):
        """offset:meters"""
        desired = np.array((x, y, z))
        pos = np.array((self.local_position.pose.position.x,
                        self.local_position.pose.position.y,
                        self.local_position.pose.position.z))
        return np.linalg.norm(desired - pos) < offset

    def reach_position(self, x, y, z, timeout):
        """timeout(int): seconds"""
        # set a position setpoint
        self.pos.pose.position.x = x
        self.pos.pose.position.y = y
        self.pos.pose.position.z = z

        # For demo purposes we will lock yaw/heading to north.
        yaw_degrees = 0  # North
        yaw = math.radians(yaw_degrees)
        quaternion = quaternion_from_euler(0, 0, yaw)
        self.pos.pose.orientation = Quaternion(*quaternion)

        # dose it reach the position in 'time' seconds?
        loop_freq = 100  # Hz
        rate = rospy.Rate(loop_freq)
        for i in xrange(int(timeout * loop_freq)):
            if self.is_at_position(self.pos.pose.position.x,
                                   self.pos.pose.position.y,
                                   self.pos.pose.position.z, self.radius):
                break
            try:
                rate.sleep()
            except rospy.ROSException:
                pass

    def set_arm(self, arm, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        for i in xrange(timeout * loop_freq):
            if self.state.armed == arm:
                break
            else:
                try:
                    res = self.set_arming_srv(arm)
                    if not res.success:
                        rospy.logerr("failed to send arm command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                print(e)

    def set_mode(self, mode, timeout):

        """mode: PX4 mode string, timeout(int): seconds"""
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        for i in xrange(timeout * loop_freq):
            if self.state.mode == mode:
                rospy.logerr('no need to set mode')
                break
            else:
                try:
                    res = self.set_mode_srv(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                print(e)

    def wait_for_topics(self, timeout):
        """wait for simulation to be ready, make sure we're getting topic info
        from all topics by checking dictionary of flag values set in callbacks,
        timeout(int): seconds"""
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        for i in xrange(timeout * loop_freq):
            if all(value for value in self.sub_topics_ready.values()):
                rospy.loginfo("simulation topics ready | seconds: {0} of {1}".
                              format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                print(e)

    def wait_for_landed_state(self, desired_landed_state, timeout, index):
        loop_freq = 10  # Hz
        rate = rospy.Rate(loop_freq)
        for i in xrange(timeout * loop_freq):
            if self.extended_state.landed_state == desired_landed_state:
                rospy.loginfo("landed state confirmed | seconds: {0} of {1}".
                              format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                print(e)

    def wait_for_vtol_state(self, transition, timeout, index):
        """Wait for VTOL transition, timeout(int): seconds"""
        loop_freq = 10  # Hz
        rate = rospy.Rate(loop_freq)
        for i in xrange(timeout * loop_freq):
            if transition == self.extended_state.vtol_state:
                rospy.loginfo("transitioned | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                print(e)

    def clear_wps(self, timeout):
        """timeout(int): seconds"""
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        for i in xrange(timeout * loop_freq):
            if not self.mission_wp.waypoints:
                rospy.loginfo("clear waypoints success | seconds: {0} of {1}".
                              format(i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.wp_clear_srv()
                    if not res.success:
                        rospy.logerr("failed to send waypoint clear command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                print(e)

    def send_wps(self, waypoints, timeout):
        """waypoints, timeout(int): seconds"""
        if self.mission_wp.waypoints:
            rospy.loginfo("FCU already has mission waypoints")

        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        wps_sent = False
        wps_verified = False
        for i in xrange(timeout * loop_freq):
            if not wps_sent:
                try:
                    res = self.wp_push_srv(start_index=0, waypoints=waypoints)
                    wps_sent = res.success
                    if wps_sent:
                        rospy.loginfo("waypoints successfully transferred")
                except rospy.ServiceException as e:
                    rospy.logerr(e)
            else:
                if len(waypoints) == len(self.mission_wp.waypoints):
                    rospy.loginfo("number of waypoints transferred: {0}".
                                  format(len(waypoints)))
                    wps_verified = True

            if wps_sent and wps_verified:
                rospy.loginfo("send waypoints success | seconds: {0} of {1}".
                              format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                print(e)

    def wait_for_mav_type(self, timeout):
        """Wait for MAV_TYPE parameter, timeout(int): seconds"""
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        res = False
        for i in xrange(timeout * loop_freq):
            try:
                res = self.get_param_srv('MAV_TYPE')
                if res.success:
                    self.mav_type = res.value.integer
                    rospy.loginfo(
                        "MAV_TYPE received | type: {0} | seconds: {1} of {2}".
                            format(mavutil.mavlink.enums['MAV_TYPE'][self.mav_type]
                                   .name, i / loop_freq, timeout))
                    break
            except rospy.ServiceException as e:
                rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                print(e)

    def moveOnce(self, cmd, margin):
        if cmd == 'moveUp':
            mcc.moveUp(margin)
        elif cmd == 'moveDown':
            mcc.moveDown(margin)
        elif cmd == 'moveXPlus':
            mcc.moveXPlus(margin)
        elif cmd == 'moveXMin':
            mcc.moveXMin(margin)
        elif cmd == 'moveYPlus':
            mcc.moveYPlus(margin)
        elif cmd == 'moveYMin':
            mcc.moveYMin(margin)
        elif cmd == 'stay':
            pass

    def getState(self):
        data = np.array([self.local_position.pose.position.x,
                         self.local_position.pose.position.y,
                         self.local_position.pose.position.z])
        data = np.append(data, self.scan.ranges)
        # a string state date
        return data

    def moveUp(self, margin=1):
        if not self.ready:
            self.getReady()
        self.reach_position(self.local_position.pose.position.x,
                            self.local_position.pose.position.y,
                            self.local_position.pose.position.z + margin,
                            0)

    def moveDown(self, margin=1):
        if not self.ready:
            self.getReady()
        self.reach_position(self.local_position.pose.position.x,
                            self.local_position.pose.position.y,
                            self.local_position.pose.position.z - margin,
                            0)

    def moveXPlus(self, margin=1):
        if not self.ready:
            self.getReady()
        self.reach_position(self.local_position.pose.position.x + margin,
                            self.local_position.pose.position.y,
                            self.local_position.pose.position.z,
                            0)

    def moveXMin(self, margin=1):
        if not self.ready:
            self.getReady()
        self.reach_position(self.local_position.pose.position.x - margin,
                            self.local_position.pose.position.y,
                            self.local_position.pose.position.z,
                            0)

    def moveYPlus(self, margin=1):
        if not self.ready:
            self.getReady()
        self.reach_position(self.local_position.pose.position.x,
                            self.local_position.pose.position.y + margin,
                            self.local_position.pose.position.z,
                            0)

    def moveYMin(self, margin=1):
        if not self.ready:
            self.getReady()
        self.reach_position(self.local_position.pose.position.x,
                            self.local_position.pose.position.y - margin,
                            self.local_position.pose.position.z,
                            0)

    def returnHomePosition(self):
        if not self.ready:
            self.getReady()
        self.reach_position(self.home_position.position.x,
                            self.home_position.position.y,
                            self.home_position.position.z,
                            5)

    def land(self):
        self.set_mode("AUTO.LAND", 2)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,
                                   45, 0)
        self.set_arm(False, 5)

    def land_mode(self):
        self.set_mode("AUTO.LAND", 2)


    def disarm(self):
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,
                                   45, 0)
        self.set_arm(False, 5)

    def test_flight(self):
        rospy.loginfo("run mission")
        positions = ((0, 0, 20), (50, 50, 20), (50, -50, 20), (-50, -50, 20),
                     (0, 0, 20))

        for i in xrange(len(positions)):
            self.reach_position(positions[i][0], positions[i][1],
                                positions[i][2], 30)

        self.set_mode("AUTO.LAND", 5)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,
                                   45, 0)
        self.set_arm(False, 5)


if __name__ == '__main__':
    number = sys.argv[1]
    port = sys.argv[2]

    rospy.init_node('ctrl_server', anonymous=True)
    mcc = MavrosCtrlCommon(number)
    mcc.setUp()
    time.sleep(5)
    # mcc.test_flight()

    # create a server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', int(port)))
    server.listen(10)
    # print('@ctrl_server@ ready port {0}'.format(port))
    over = False

    while True:
        conn, addr = server.accept()
        # rospy.loginfo('mavros_ctrl_server port:{0} connected with:{1}'.format(port, str(addr)))
        try:
            # print('@ctrl_server@ waiting cmd')
            data = conn.recv(1024).split('#')
            # print('@ctrl_server@ get cmd ' + str(data))
            # get cmd content
            cmd = data[0]
            margin = 1
            if len(data) > 1 and len(data) < 2:
                margin = int(data[1])
            # execute cmd
            #   env reset 
            r_msg = ''
            # print('@ctrl_server@ executing cmd: ' + cmd)
            if cmd == 'reset':
                mcc.land_mode()
                r_msg = 'reset'
            #   the env is killed
            elif cmd == 'takeoff':
                mcc.getReady()
                r_msg = mcc.getState()
                if mcc.uav_number == '1':
                    r_msg[0] += 6
                elif mcc.uav_number == '2':
                    r_msg[1] += 6
            elif cmd == 'disarm':
                mcc.land()
                r_msg = 'disarm'
            elif cmd == 'shutdown':
                mcc.shutDown()
                over = True
                r_msg = 'recv shutdown'
            elif cmd == 'state':
                r_msg = mcc.getState()
                if mcc.uav_number == '1':
                    r_msg[0] += 6
                elif mcc.uav_number == '2':
                    r_msg[1] += 6
            else:
                mcc.moveOnce(cmd, margin)
                r_msg = 'roger'
            # print('@ctrl_server@ uav_{0} execute {1} over, return msg {2}'.format(mcc.uav_number, cmd, r_msg))
            conn.send(pickle.dumps(r_msg))
        except BaseException as e:
            print('@ctrl_server@' + str(e))
            time.sleep(3)
            # conn.send('invaild cmd')

        if over:
            break
