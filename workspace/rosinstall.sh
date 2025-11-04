set -e
. /etc/os-release
add-apt-repository -y universe || true
apt-get update
apt-get install -y --no-install-recommends curl gnupg lsb-release

# Add ROS 2 apt repo + key (HTTPS + keyring)
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
https://packages.ros.org/ros2/ubuntu ${VERSION_CODENAME} main" \
  > /etc/apt/sources.list.d/ros2.list

apt-get update
apt-get install -y ros-humble-desktop python3-colcon-common-extensions ros-dev-tools

# Make ROS auto-load in future shells
echo '[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash' >> /etc/bash.bashrc
source /opt/ros/humble/setup.bash

# Smoke check
which ros2 && ros2 --help | head -n 5 && echo "ROS OK"
