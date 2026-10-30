#!/bin/bash
# Copies the repo onto the Go2's Jetson, mirroring its filesystem layout.
set -e
R="$(cd "$(dirname "$0")/.." && pwd)"
sudo mkdir -p /unitree/module /unitree/lib/unitree_slam
sudo rsync -a "$R/src/graph_pid_ws/" /unitree/module/graph_pid_ws/
sudo rsync -a "$R/src/rocky_demos/" /unitree/lib/unitree_slam/rocky/
cp "$R/scripts/site_manager.sh" ~/ && chmod +x ~/site_manager.sh
mkdir -p ~/unitree_sites/hackathon && rsync -a "$R/sites/" ~/unitree_sites/hackathon/
sed 's/<PEER_IP>/YOUR_LAPTOP_IP/' "$R/config/cyclonedds_wifi.xml.example" > ~/cyclonedds_wifi_test.xml
echo "Edit ~/cyclonedds_wifi_test.xml with your peer IPs, then build (see README)."
