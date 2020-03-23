"""
Never Modify this file! Always copy the settings you want to change to your local file.
"""


import numpy as np


class Config(object):
    def __init__(self):
        pass


class BaseEnvConfig(object):
    env = Config()
    env.time_limit = 30
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 500
    env.train_size = np.iinfo(np.uint32).max - 2000

    reward = Config()
    reward.success_reward = 1
    reward.collision_penalty_crowd = -0.25
    reward.discomfort_dist = 0.2
    reward.discomfort_penalty_factor = 0.5
    reward.collision_penalty_obs = -0.25
    reward.clearance_dist = 0.2
    reward.clearance_dist_penalty_factor = 0.5

    sim = Config()
    sim.scenario = '/home/julian/Desktop/test_data/test.xml'
    sim.human_num = 30
    sim.randomize_attributes = True

    humans = Config()
    humans.visible = True
    humans.policy = 'orca'
    humans.radius = 0.3
    humans.v_pref = 1
    humans.sensor = 'coordinates'

    robot = Config()
    robot.visible = True
    robot.policy = 'none'
    robot.radius = 0.3
    robot.v_pref = 1
    robot.sensor_range = 50
    robot.fov = 1.7174  # field of view (FOV) in rad
    robot.sensor_resolution = 0.0017453  # in rad
    robot.rotation_constraint = np.pi / 3

    robot.action_space = Config()
    robot.action_space.kinematics = 'holonomic'
    robot.action_space.speed_samples = 5
    robot.action_space.rotation_samples = 16
    robot.action_space.speed_sampling = 'exponential'
    robot.action_space.rotation_sampling = 'linear'

    ros = Config()
    ros.rate = 10

    def __init__(self, debug=False):
        if debug:
            self.env.val_size = 1
            self.env.test_size = 1

