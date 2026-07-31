# Reachy 1.2 Simulator Docker Image
# Target: linux/amd64 (macOS host compatible)
# ROS 2: Foxy on Ubuntu 20.04
# Ports: 8888 (JupyterLab), 6080 (RViz2 via noVNC), 50051 (gRPC)

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
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# ── noVNC (web VNC client) ─────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/novnc/noVNC /opt/novnc \
    && ln -sf /opt/novnc/vnc.html /opt/novnc/index.html

# ── Python packages ────────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── ROS 2 workspace: SDK server + Reachy 1.2 (2021) URDF description ──────────
# reachy_description (underscore) is the public Foxy package with reachy.URDF
# and all .dae meshes. Earlier attempts used reachy-description (hyphen) — wrong.
RUN mkdir -p ${REACHY_WS}/src && \
    cd ${REACHY_WS}/src && \
    git clone --depth 1 https://github.com/pollen-robotics/reachy_sdk_server_2021.git && \
    git clone --depth 1 https://github.com/pollen-robotics/reachy_description.git

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    rosdep update --rosdistro ${ROS_DISTRO} && \
    rosdep install --from-paths ${REACHY_WS}/src --ignore-src -r -y \
        --skip-keys "gazebo gazebo11 rviz" && \
    cd ${REACHY_WS} && \
    colcon build --symlink-install && \
    echo "source ${REACHY_WS}/install/setup.bash" >> /etc/bash.bashrc

# ── Fake gRPC server + ROS joint state bridge ─────────────────────────────────
COPY fake_reachy_server.py /opt/fake_reachy_server.py
COPY joint_state_bridge.py /opt/joint_state_bridge.py

# ── Notebooks ──────────────────────────────────────────────────────────────────
COPY notebooks/ /notebooks/

# ── RViz2 config ───────────────────────────────────────────────────────────────
COPY rviz/ /opt/rviz_config/

# ── supervisord config ─────────────────────────────────────────────────────────
COPY supervisord.conf /etc/supervisor/conf.d/reachy.conf

# ── Entrypoint ─────────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# JupyterLab
EXPOSE 8888
# noVNC → RViz2
EXPOSE 6080
# Reachy SDK gRPC server
EXPOSE 50051

ENTRYPOINT ["/entrypoint.sh"]
