# Reachy 1.2 Simulator Docker Image
# Target: linux/amd64 (macOS host compatible)
# ROS 2: Foxy on Ubuntu 20.04
# Ports: 8888 (JupyterLab), 6080 (RViz2 via noVNC), 50051 (gRPC)
#
# ASSUMPTION: reachy_sdk_server fake mode launched via:
#   ros2 launch reachy_sdk_server reachy_sdk_server.launch.py fake:=true
# Confirm with Siva if the launch file name or fake param differs.

FROM --platform=linux/amd64 ros:foxy

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1 \
    ROS_DISTRO=foxy \
    REACHY_WS=/opt/reachy_ws

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    # ROS 2 visualization
    ros-foxy-rviz2 \
    ros-foxy-robot-state-publisher \
    ros-foxy-joint-state-publisher \
    ros-foxy-joint-state-publisher-gui \
    ros-foxy-xacro \
    # Virtual display stack
    xvfb \
    fluxbox \
    x11vnc \
    websockify \
    # Process supervisor
    supervisor \
    # Utilities
    git \
    wget \
    curl \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# ── noVNC (web VNC client) ─────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/novnc/noVNC /opt/novnc \
    && ln -sf /opt/novnc/vnc.html /opt/novnc/index.html

# ── Python packages ────────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── ROS 2 workspace: reachy_sdk_server_2021 + reachy-description ──────────────
RUN mkdir -p ${REACHY_WS}/src && \
    cd ${REACHY_WS}/src && \
    # Reachy 1.2 (2021) SDK server — contains reachy_bringup package
    # Launched via: ros2 launch reachy_bringup reachy.launch.py fake:=true start_sdk_server:=true
    git clone --depth 1 https://github.com/pollen-robotics/reachy_sdk_server_2021.git && \
    # gRPC / protobuf API definitions (confirmed by Siva 2026-07-31)
    # NOTE: named reachy2-sdk-api but used by the 2021 server — verify if reachy-sdk-api is needed instead
    git clone --depth 1 https://github.com/pollen-robotics/reachy2-sdk-api.git && \
    # URDF / robot description for RViz2 display
    git clone --depth 1 https://github.com/pollen-robotics/reachy-description.git

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    rosdep update --rosdistro ${ROS_DISTRO} && \
    rosdep install --from-paths ${REACHY_WS}/src --ignore-src -r -y && \
    cd ${REACHY_WS} && \
    colcon build --symlink-install && \
    echo "source ${REACHY_WS}/install/setup.bash" >> /etc/bash.bashrc

# ── Notebooks ──────────────────────────────────────────────────────────────────
COPY notebooks/ /notebooks/

# ── RViz2 config ───────────────────────────────────────────────────────────────
COPY rviz/ /opt/rviz_config/

# ── supervisord config ─────────────────────────────────────────────────────────
COPY supervisord.conf /etc/supervisor/conf.d/reachy.conf

# ── Entrypoint ─────────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8888   # JupyterLab
EXPOSE 6080   # noVNC → RViz2
EXPOSE 50051  # Reachy SDK gRPC server

ENTRYPOINT ["/entrypoint.sh"]
