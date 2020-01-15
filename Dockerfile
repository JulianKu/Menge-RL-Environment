FROM ros:melodic-robot

WORKDIR /home/ros

# Install system libraries
RUN apt-get update && \
    apt-get install -y sudo vim apt-utils ninja-build python-catkin-tools python-wstool python-rosdep ros-melodic-catkin ros-melodic-geometry2 ros-melodic-moveit-msgs net-tools python3-dev python3-numpy python3-pip python3-yaml

RUN 

WORKDIR /home/ros/src
COPY menge_ros ./menge_ros
COPY menge_scene_creation ./menge_scene_creation


WORKDIR /home/ros

RUN apt-get update && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src --rosdistro=${ROS_DISTRO} -y

# Build catkin workspace
RUN catkin config --extend /opt/ros/melodic && \
    catkin build


