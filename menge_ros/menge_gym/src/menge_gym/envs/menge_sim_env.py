#!/usr/bin/env python3

import gym
from gym import spaces
import numpy as np
from os import path
import rospy as rp
import rosnode
from geometry_msgs.msg import PoseArray, PoseStamped, Twist
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Bool
from menge_srv.srv import RunSim, CmdVel, CmdVelResponse
from .utils.ros import obstacle2array, marker2array, ROSHandle  # ,pose2array, launch
from .utils.params import match_in_xml, goal2array, get_robot_initial_position
from .utils.info import *
from .utils.tracking import Sort, KalmanTracker
from .utils.format import format_array
from typing import List, Union


class MengeGym(gym.Env):
    """
    Custom gym environment for the Menge_ROS simulation
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MengeGym, self).__init__()

        self.config = None

        # Environment variables
        self.time_limit = None
        self.time_step = None
        self.randomize_attributes = None
        self.robot_sensor_range = None

        # Simulation scenario variables
        self.scenario_xml = None
        self.scene_xml = None
        self.behavior_xml = None
        self.initial_robot_pos = None
        self.goals_array = None
        self.goal = None

        # Robot variables
        self.robot_radius = None
        self.robot_range = None
        self.robot_speed_sampling = None
        self.robot_rotation_sampling = None

        # Reward variables
        self.success_reward = None
        self.collision_penalty_crowd = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        self.collision_penalty_obs = None
        self.clearance_dist = None
        self.clearance_penalty_factor = None

        # Observation variables
        self._crowd_poses = []  # type: List[np.ndarray]
        self._robot_poses = []  # type: List[np.ndarray]
        self._static_obstacles = np.array([], dtype=float)
        self.rob_tracker = None
        self.ped_tracker = Sort()
        self.combined_state = np.array([], dtype=float)

        # Action variables
        self._velocities = None
        self._angles = None
        self._action = None  # type: Union[None, np.ndarray]

        # Schedule variables
        self.case_size = None
        self.case_counter = None

        # ROS
        self.roshandle = None
        self._sim_pid = None
        self._pub_run = None
        self._step_done = None
        self._cmd_vel_srv = None
        self._advance_sim_srv = None
        self.ros_rate = None
        self._rate = None

    def configure(self, config):

        self.config = config

        # Environment
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes  # randomize humans' radius and preferred speed

        # Simulation
        self.scenario_xml = config.sim.scenario

        if path.isfile(self.scenario_xml):
            self._initialize_from_scenario()
        else:
            self.robot_radius = config.robot.radius
            self.robot_range = config.robot.sensor_range
            # TODO: make scenario from this
            assert path.isfile(self.scenario_xml), 'No valid scenario_xml specified' # as scenario generation not implemented yet
        # sample first goal
        self.sample_goal(exclude_initial=True)

        # Reward
        self.success_reward = config.reward.success_reward
        self.collision_penalty_crowd = config.reward.collision_penalty_crowd
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.collision_penalty_obs = config.reward.collision_penalty_obs
        self.clearance_dist = config.reward.clearance_dist
        self.clearance_penalty_factor = config.reward.clearance_dist_penalty_factor

        # Robot
        v_max = config.robot.v_pref
        rotation_constraint = config.robot.rotation_constraint
        num_speeds = config.robot.action_space.speed_samples
        num_angles = config.robot.action_space.rotation_samples
        self.robot_speed_sampling = config.robot.action_space.speed_sampling
        self.robot_rotation_sampling = config.robot.action_space.rotation_sampling
        # action space
        # from paper RGL for CrowdNav --> 6 speeds [0, v_pref] and 16 headings [0, 2*pi)
        if self.robot_speed_sampling == 'exponential':
            # exponentially distributed speeds (distributed between 0 and v_max)
            self._velocities = np.geomspace(1, v_max + 1, num_speeds + 1, endpoint=True) - 1
        elif self.robot_speed_sampling == 'linear':
            self._velocities = np.linspace(0, v_max, num_speeds + 1, endpoint=True)
        else:
            raise NotImplementedError

        if self.robot_rotation_sampling == 'linear':
            # linearly distributed angles
            # make num_angles odd to ensure null action (0 --> not changing steering)
            num_angles = num_angles // 2 * 2 + 1
            # angles between -45° und +45° in contrast to paper between -pi and +pi
            self._angles = np.linspace(-rotation_constraint, rotation_constraint, num_angles, endpoint=True)
        elif self.robot_rotation_sampling == 'exponential':
            min_angle_increment = 1             # (in deg)
            min_angle_increment *= np.pi / 180  # (in rad)
            positive_angles = np.geomspace(min_angle_increment, rotation_constraint, num_angles//2, endpoint=True)
            self._angles = np.concatenate((-positive_angles[::-1], [0], positive_angles))
        else:
            raise NotImplementedError

        self.action_space = spaces.MultiDiscrete([num_speeds, num_angles])

        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        self.ros_rate = config.ros.rate

    def _initialize_from_scenario(self):
        scenario_xml = self.scenario_xml
        scenario_dir = path.split(scenario_xml)[0]

        scene_xml = match_in_xml(scenario_xml, attrib_name='scene')
        if not path.isabs(scene_xml):
            self.scene_xml = path.join(scenario_dir, scene_xml)
        assert path.isfile(self.scene_xml), 'Scene file specified in scenario_xml non-existent'

        # extract robot radius from behavior_xml file
        self.robot_radius = float(
            match_in_xml(self.scene_xml, tag="Common", attrib_name="r", constraints={"external": "1"}))

        behavior_xml = match_in_xml(scenario_xml, attrib_name='behavior')
        if not path.isabs(behavior_xml):
            self.behavior_xml = path.join(scenario_dir, behavior_xml)
        assert path.isfile(self.behavior_xml), 'Behavior file specified in scenario_xml non-existent'

        # extract goal set from behavior file
        goals = match_in_xml(self.behavior_xml, tag="Goal", return_all=True)
        self.goals_array = np.array(list(map(goal2array, goals)))

    def sample_goal(self, exclude_initial: bool = False):
        """
        sample goal from available goals and set "goal" attribute accordingly

        :param exclude_initial:     bool, if True exclude a goal from sampling
                                          if the robot's initial position lies within this goal
        """
        goals_array = self.goals_array
        if exclude_initial:
            if self.initial_robot_pos is None:
                self.initial_robot_pos = get_robot_initial_position(self.scene_xml)
            # if initial robot position falls within goal, exclude this goal from sampling
            dist_rob_goals = np.linalg.norm(goals_array[:, :2] - self.initial_robot_pos, axis=1) - goals_array[:, 2] \
                             - self.robot_radius
            # mask out respective goal(s)
            mask = dist_rob_goals > 0
            goals_array = goals_array[mask]

        self.goal = goals_array[np.random.randint(len(goals_array))]

    def setup_ros_connection(self):
        rp.loginfo("Initializing ROS")
        self.roshandle = ROSHandle()

        # TODO: rviz config not loaded properly (workaround: start rviz seperately via launch file etc.)
        # visualization = True
        # if visualization:
        #     # Get rviz configuration file from "menge_vis" package
        #     rviz_path = path.join(path.join(rospkg.RosPack().get_path("menge_vis"), "rviz"), "menge_ros.rviz")
        #     # Start rviz rosnode
        #     self.roshandle.start_rosnode('rviz', 'rviz', launch_cli_args={"d": rviz_path})

        rp.on_shutdown(self.close)

        rp.loginfo("Start Menge simulator node")
        # launch_cli_args = {'project': self.scenario_xml,
        #                    'timeout': self.timeout,
        #                    'timestep': self.time_step}
        # self._sim_process = start_roslaunch_file('menge_vis', 'menge.launch', launch_cli_args)
        cli_args = {'p': self.scenario_xml,
                    'd': self.time_limit,
                    't': self.time_step}
        self._sim_pid = self.roshandle.start_rosnode('menge_sim', 'menge_sim', cli_args)
        rp.sleep(5)
        # self._sim_process = launch('menge_sim', 'menge_sim', cli_args)

        # simulation controls
        rp.logdebug("Set up publishers and subscribers")
        rp.init_node('MengeSimEnv', log_level=rp.DEBUG)
        self._pub_run = rp.Publisher('run', Bool, queue_size=1)
        self._step_done = False

        # rp.Subscriber("crowd_pose", PoseArray, self._crowd_pose_callback)
        rp.Subscriber("crowd_expansion", MarkerArray, self._crowd_expansion_callback, queue_size=50)
        rp.Subscriber("laser_static_end", PoseArray, self._static_obstacle_callback, queue_size=50)
        rp.Subscriber("pose", PoseStamped, self._robot_pose_callback, queue_size=50)
        rp.Subscriber("done", Bool, self._done_callback, queue_size=50)

        # self._cmd_vel_pub = rp.Publisher('/cmd_vel', Twist, queue_size=50)
        self._cmd_vel_srv = rp.Service('cmd_vel_srv', CmdVel, self._cmd_vel_srv_handler)
        self._advance_sim_srv = rp.ServiceProxy('advance_simulation', RunSim)

        # initialize time
        # self._run_duration = rp.Duration(self.time_step)
        self._rate = rp.Rate(self.ros_rate)

    # def _crowd_pose_callback(self, msg: PoseArray):
    #     rp.logdebug('Crowd Pose subscriber callback called')
    #     # transform PoseArray message to numpy array
    #     pose_array = np.array(list(map(pose2array, msg.poses)))
    #     self._crowd_poses.append(pose_array)

    def _crowd_expansion_callback(self, msg: MarkerArray):
        rp.logdebug('Crowd Expansion subscriber callback called')
        # transform MarkerArray message to numpy array
        marker_array = np.array(list(map(marker2array, msg.markers)))
        self._crowd_poses.append(marker_array.reshape(-1, 4))

    def _static_obstacle_callback(self, msg: PoseArray):
        rp.logdebug('Static Obstacle subscriber callback called')
        # transform PoseArray message to numpy array
        self._static_obstacles = np.array(list(map(obstacle2array, msg.poses))).reshape(-1, 2)

    def _robot_pose_callback(self, msg: PoseStamped):
        rp.logdebug('Robot Pose subscriber callback called')
        # extract 2D pose and orientation from message
        robot_pose = msg.pose
        robot_x = robot_pose.position.x
        robot_y = robot_pose.position.y
        robot_omega = 2 * np.arccos(robot_pose.orientation.w)

        # update list of robot poses + pointer to current position
        self._robot_poses.append(np.array([robot_x, robot_y, robot_omega]).reshape(-1, 3))

    def _done_callback(self, msg: Bool):
        rp.logdebug('Done message received')
        self._step_done = msg.data

    def _cmd_vel_srv_handler(self, request):
        # in menge_ros the published angle defines an angle increment
        if self._action is not None:
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = self._velocities[self._action[0]]  # vel_action
            cmd_vel_msg.linear.y = 0
            cmd_vel_msg.linear.z = 0
            cmd_vel_msg.angular.x = 0
            cmd_vel_msg.angular.y = 0
            cmd_vel_msg.angular.z = self._angles[self._action[1]]  # angle_action
            return CmdVelResponse(True, cmd_vel_msg)
        else:
            return CmdVelResponse(False, Twist())

    def step(self, action: np.ndarray):
        rp.logdebug("Performing step in the environment")

        # # only keep most recent poses before updating simulation
        # self._crowd_poses = list(self._crowd_poses[-1])
        # self._robot_poses = list(self._robot_poses[-1])

        self._take_action(action)

        self.roshandle.log_output()

        reward, done, info = self._get_reward_done_info()

        # in first iteration, initialize Kalman Tracker for robot
        if not self.rob_tracker:
            self.rob_tracker = KalmanTracker(self._robot_poses[0][:, :3])

        # update velocities
        for (robot_pose, crowd_pose) in zip(self._robot_poses, self._crowd_poses):
            rp.logdebug("Robot Pose (Shape %r):\n %r" % (robot_pose.shape, robot_pose))
            rp.logdebug("Crowd Pose (Shape %r):\n %r" % (crowd_pose.shape, crowd_pose))
            # state = np.concatenate((robot_pose[:, :3], crowd_pose[:, :3]), axis=0)
            ped_trackers = self.ped_tracker.update(crowd_pose[:, :3])
            self.rob_tracker.predict()
            self.rob_tracker.update(robot_pose[:, :3])
        rob_tracker = self.rob_tracker.get_state()
        trackers = np.concatenate((np.concatenate((rob_tracker, [[0]]), axis=1).reshape(1, -1), ped_trackers), axis=0)
        combined_state = trackers[trackers[:, -1].argsort()]
        self.combined_state = combined_state

        ob = (combined_state, self._static_obstacles, self.goal)

        # reset last poses
        self._crowd_poses = []
        self._robot_poses = []

        return ob, reward, done, info

    def _take_action(self, action: np.ndarray):
        """
        execute one time step within the environment
        """
        rp.logdebug("Taking action")

        self._action = action

        rp.logdebug("Calling Service")

        # Not sure why, but advance_sim service kept getting stuck (not reaching simulator node)
        # sleep is a working hack around this, not nice though
        self._rate.sleep()
        # advance simulation by one step
        while not self._advance_sim_srv(1):
            rp.logwarn("Simulation not paused, service failed")
            self._pub_run.publish(Bool(data=False))
        rp.logdebug("Service called")
        # wait for response from simulation, in the meantime publish cmd_vel
        while not self._step_done or not self._crowd_poses or not self._robot_poses:
            # rp.logdebug("Publishing cmd_vel message")
            # self._cmd_vel_pub.publish(vel_msg)
            rp.logdebug('Simulation not done yet')
            rp.logdebug('Done %r, #Crowd %d, #Rob %d' %
                        (self._step_done, len(self._crowd_poses), len(self._robot_poses)))
            self._rate.sleep()
        rp.logdebug('Done %r, #Crowd %d, #Rob %d' %
                    (self._step_done, len(self._crowd_poses), len(self._robot_poses)))
        self._step_done = False
        self._action = None

        # self._pub_run.publish(Bool(data=True))
        # current_time = start_time = rp.Time.now()
        # while current_time <= start_time + self._run_duration:
        #     self._cmd_vel_pub.publish(vel_msg)
        #     current_time = rp.Time.now()
        # self._pub_run.publish(Bool(data=False))

    def _get_reward_done_info(self) -> (float, bool, object):
        """
        compute reward and other information from current state

        :return:
            reward, done, info
        """

        # crowd_pose = [x, y, omega, r]
        recent_crowd_pose = self._crowd_poses[-1]

        # obstacle_position = [x, y]
        obstacle_position = self._static_obstacles

        # robot_pose = [x, y, omega]
        recent_robot_pose = self._robot_poses[-1]

        robot_radius = self.robot_radius
        goal = self.goal

        crowd_distances = np.linalg.norm(recent_crowd_pose[:, :2] - recent_robot_pose[:, :2], axis=1)
        crowd_distances -= recent_crowd_pose[:, -1]
        crowd_distances -= robot_radius

        obstacle_distances = np.linalg.norm(obstacle_position - recent_robot_pose[:, :2], axis=1)
        obstacle_distances -= robot_radius

        # compute distance to closest pedestrian
        if crowd_distances.size == 0:
            # if no pedestrian, set to infinity
            d_min_crowd = np.inf
        else:
            d_min_crowd = crowd_distances.min()

        # compute distance to closest static obstacle
        if obstacle_distances.size == 0:
            # if no obstacles, set to infinity
            d_min_obstacle = np.inf
        else:
            d_min_obstacle = obstacle_distances.min()

        d_goal = np.linalg.norm(recent_robot_pose[:, :2] - goal[:2]) - robot_radius - goal[-1]

        # sim node terminated
        if '/menge_sim' not in rosnode.get_node_names():
            reward = 0
            done = True
            info = Timeout()
        # collision with crowd
        elif d_min_crowd < 0:
            reward = self.collision_penalty_crowd
            done = True
            info = Collision('Crowd')
        # collision with obstacle
        elif d_min_obstacle < 0:
            reward = self.collision_penalty_obs
            done = True
            info = Collision('Obstacle')
        # goal reached
        elif d_goal < 0:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        # too close to people
        elif d_min_crowd < self.discomfort_dist:
            # adjust the reward based on FPS
            reward = (d_min_crowd - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Discomfort(d_min_crowd)
        # too close to obstacles
        elif d_min_obstacle < self.clearance_dist:
            # adjust the reward based on FPS
            reward = (d_min_obstacle - self.clearance_dist) * self.clearance_penalty_factor * self.time_step
            done = False
            info = Clearance(d_min_obstacle)
        else:
            reward = 0
            done = False
            info = Nothing()
        return reward, done, info

    def reset(self):
        """
        reset the state of the environment to an initial state

        :return: initial observation (ob return from step)
        """
        rp.loginfo("Env reset - Shutting down simulation process")
        # self._sim_process.terminate()
        # self._sim_process.wait()
        self.roshandle.terminateOne(self._sim_pid)
        # self._sim_process.shutdown()

        rp.loginfo("Env reset - Starting new simulation process")
        # launch_cli_args = {'project': self.scenario_xml,
        #                    'timeout': self.timeout,
        #                    'timestep': self.time_step}
        # self._sim_process = start_roslaunch_file('menge_vis', 'menge.launch', launch_cli_args)
        cli_args = {'p': self.scenario_xml,
                    'd': self.time_limit,
                    't': self.time_step}
        self._sim_pid = self.roshandle.start_rosnode('menge_sim', 'menge_sim', cli_args)
        rp.sleep(5)
        # self._sim_process = launch('menge_sim', 'menge_sim', cli_args)

        # Sample new goal
        self.sample_goal(exclude_initial=True)

        # perform idle action and return observation
        return self.step(np.array([0, np.median(range(self.action_space.nvec[1]))], dtype=np.int32))[0]

    def render(self, mode='human', close=False):
        """
        render environment information to screen
        """
        if close:
            self.close()
        trackers = self.combined_state
        if len(trackers):
            rp.loginfo('Tracked Objects')
            trackers_str = format_array(trackers[:, :-1],
                                        row_labels=trackers[:, -1].astype(int),
                                        col_labels=['x', 'y', 'omega', 'x_dot', 'y_dot', 'omega_dot'])
            rp.loginfo('\n' + trackers_str)
        else:
            rp.logwarn("No objects tracked")
        rp.loginfo('\nNumber of static obstacles: %d\n' % len(self._static_obstacles))

    def close(self):
        """
        close the environment
        """

        rp.loginfo("Env close - Shutting down simulation process and killing roscore")
        self.roshandle.terminate()
        # self._sim_process.shutdown()
