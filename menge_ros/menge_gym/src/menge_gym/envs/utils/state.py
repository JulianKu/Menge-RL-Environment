import torch
import numpy as np
from typing import Tuple


class FullState(object):
    def __init__(self, state: np.ndarray):
        assert state.size % 11 == 0, "FullState needs to be composed of " \
                                     "p_x, p_y, angle_z, r, v_x, v_y, v_angular, g_x, g_y, r_goal and v_pref"
        state = state.reshape(-1, 11)
        self.state = state
        self.observable_state = state[:, :7]

        self.position = state[:, :2]
        self.angle = state[:, 2]
        self.radius = state[:, 3]
        self.velocity = state[:, 4:6]
        self.angular_velocity = state[:, 6]
        self.goal_position = state[:, 7:9]
        self.goal_radius = state[:, 9]
        self.v_pref = state[:, 10]

    def __add__(self, other: np.ndarray):
        return other.reshape(-1, 11) + self.state

    def __str__(self):
        return ' '.join([str(x) for x in self.state])

    def __len__(self):
        return len(self.state)

    def to_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.state)

    def get_observable_state(self):
        return ObservableState(self.state[:, :7])


class ObservableState(object):
    def __init__(self, state: np.ndarray):
        assert state.size % 7 == 0, "ObservableState needs to be composed of " \
                                    "p_x, p_y, angle_z, r, v_x, v_y and v_angular"
        state = state.reshape(-1, 7)
        self.state = state

        self.position = state[:, :2]
        self.angle = state[:, 2]
        self.radius = state[:, 3]
        self.velocity = state[:, 4:6]
        self.angular_velocity = state[:, 6]

    def __add__(self, other):
        return other.reshape(-1, 7) + self.state

    def __str__(self):
        return ' '.join([str(x) for x in self.state])

    def __len__(self):
        return len(self.state)

    def to_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.state)


class ObstacleState(object):
    def __init__(self, state: np.ndarray):
        assert state.size % 2 == 0, "ObstacleState needs to be composed of p_x and p_y"
        state = state.reshape(-1, 2)
        self.state = state
        self.position = state[:, :2]

    def __add__(self, other):
        return other.reshape(-1, 2) + self.state

    def __str__(self):
        return ' '.join([str(x) for x in self.state])

    def __len__(self):
        return len(self.state)

    def to_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.state)


class JointState(object):
    def __init__(self, robot_state, human_states, obstacles):
        assert isinstance(robot_state, FullState)
        assert isinstance(human_states, ObservableState)
        assert isinstance(obstacles, ObstacleState)

        self.robot_state = robot_state
        self.human_states = human_states
        self.obstacles = obstacles

    def to_tensor(self, add_batch_size=False, device=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        robot_state_tensor = self.robot_state.to_tensor()
        human_states_tensor = self.human_states.to_tensor()
        obstacle_tensor = self.obstacles.to_tensor()

        if add_batch_size:
            robot_state_tensor = robot_state_tensor.unsqueeze(0)
            human_states_tensor = human_states_tensor.unsqueeze(0)
            obstacle_tensor = obstacle_tensor.unsqueeze(0)

        if device is not None:
            robot_state_tensor.to(device)
            human_states_tensor.to(device)
            obstacle_tensor.to(device)

        return robot_state_tensor, human_states_tensor, obstacle_tensor


def tensor_to_joint_state(state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    robot_state, human_states, obstacles = state

    robot_state = robot_state.squeeze().data.numpy()
    robot_state = FullState(robot_state)
    human_states = human_states.squeeze(0).data.numpy()
    human_states = ObservableState(human_states)
    obstacle_states = obstacles.squeeze(0).data.numpy()
    obstacle_states = ObstacleState(obstacle_states)

    return JointState(robot_state, human_states, obstacle_states)
