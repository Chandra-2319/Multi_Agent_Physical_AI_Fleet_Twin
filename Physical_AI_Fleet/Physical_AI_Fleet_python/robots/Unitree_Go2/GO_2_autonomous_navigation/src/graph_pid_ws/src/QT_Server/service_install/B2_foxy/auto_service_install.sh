#!/bin/bash
WorkDir=$(cd $(dirname $0); pwd)

read -p "Please confirm if the ROS workspace directory is /home/unitree/graph_pid_ws ? (Y/n) " input
set -e
if [ "$input" = "y" ] || [ "$input" = "Y" ]; then
	cp $WorkDir/unitree_slam.service /usr/lib/systemd/system
	cp $WorkDir/0_QTServer.sh ../../0_QTServer.sh
	chmod 644 /usr/lib/systemd/system/unitree_slam.service
	systemctl daemon-reload
	systemctl start unitree_slam.service
	systemctl enable unitree_slam.service
	echo "success!"
else
	echo "fail!"
fi

	
	

