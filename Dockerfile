FROM fnndsc/ubuntu-python3:latest

ENV DEBIAN_FRONTEND=noninteractive

# install packages
RUN apt update && apt install -q -y \
    apt-utils \
    sudo \    
    build-essential \
    dirmngr \
    gnupg2 \
    lsb-release \
    wget \
    git \
    less \
    python-pip \
    tmux \
    vim \
    bash-completion \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# setup keysq
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key \
    C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt update && apt install -y \
    python-rosdep \
    python-rosinstall \
    python-wstool \
    python-rosinstall-generator \
    python-vcstools \
    python-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# install ros packages
ENV ROS_DISTRO melodic
RUN apt update && apt install -y \
    ros-melodic-ros-base=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

# install additional packages
RUN apt update && apt install -y \
    net-tools libsdl-image1.2-dev libsdl-dev libsdl-ttf2.0-dev libtinyxml-dev

RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo

RUN pip3 install --upgrade pip

# Install some python packages
RUN pip3 install pyyaml rospkg triangle numpy argparse scikit-image

RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc


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

# Enable root ssh access for development purposes (e.g.SFTP via CLion)
#   RUN chmod +x src/setup_ssh.sh
#   RUN bash src/setup_ssh.sh
#   RUN sudo service ssh start

# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

RUN touch /root/.bashrc && \
    echo 'source /home/ros/devel/setup.bash' >> /root/.bashrc
