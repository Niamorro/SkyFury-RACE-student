#!/bin/bash

cd `dirname $0`
dir=`pwd`

source /opt/ros/humble/setup.bash

trap 'kill 0' SIGINT

ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://10.9.8.7:15016@10.9.8.1:15011 -p tgt_system:=1 &
./control.py

kill 0
wait
