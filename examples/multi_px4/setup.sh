cd ~/px4/Firmware/
source Tools/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd):$(pwd)/Tools/sitl_gazebo
export PX4_SIM_SPEED_FACTOR=8
cd ~/gym-gazebo/examples/multi_px4
