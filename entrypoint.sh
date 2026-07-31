#!/bin/bash
set -e

# Create log directory
mkdir -p /var/log/supervisor

# Source ROS 2 environment for any interactive shells
echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
echo "source /opt/reachy_ws/install/setup.bash" >> ~/.bashrc

echo "================================================"
echo "  Reachy 1.2 Simulator"
echo "================================================"
echo "  JupyterLab  → http://localhost:8888"
echo "  RViz2/noVNC → http://localhost:6080"
echo "  SDK gRPC    → localhost:50051"
echo "================================================"

# Hand off to supervisord — manages all services
exec /usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf
