import random

import hydra
import numpy as np
from calvin.calvin_env.calvin_env.envs.play_table_env import PlayTableSimEnv
from gym import spaces


class SlideEnv(PlayTableSimEnv):
    def __init__(
        self,
        target_tasks: list,
        seed: int = 0,
        tasks: dict = {},
        **kwargs,
    ):
        super(SlideEnv, self).__init__(seed=seed, **kwargs)
        # For this example we will modify the observation to
        # only retrieve the end effector pose
        self.target_tasks = target_tasks
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(39,))
        # We can use the task utility to know if the task was executed correctly
        self.tasks = hydra.utils.instantiate(tasks)

    def reset(self):
        obs = super().reset()

        # task setup
        if "close_drawer" in self.target_tasks:  # move_door_rel = 0.12
            self.scene.doors[1].reset(random.uniform(0.12, 0.2))
        if "move_slider_left" in self.target_tasks:  # move_door_rel = 0.15
            self.scene.doors[0].reset(random.uniform(0, 0.12))
        if "turn_on_led" in self.target_tasks:
            self.scene.lights[1].reset(0)
        if "turn_on_lightbulb" in self.target_tasks:
            self.scene.lights[0].reset(0)
        if "lift_blue_block_table" in self.target_tasks:
            pass # blue block is already initialized to be on the table in the config

        self.start_info = self.get_info()
        return obs

    def get_obs(self):
        """Overwrite robot obs to only retrieve end effector position"""
        robot_obs, robot_info = self.robot.get_observation()
        # return robot_obs[:7]
        scene_obs = self.scene.get_obs()
        obs = np.concatenate([robot_obs, scene_obs])
        return obs

    def _success(self):
        """Returns a boolean indicating if the task was performed correctly"""
        current_info = self.get_info()

        successes = []
        for task in self.target_tasks:
            task_info = self.tasks.get_task_info_for_set(
                self.start_info, current_info, [task]
            )
            successes.append(task in task_info)

        return all(successes)

    def _reward(self):
        """Returns the reward function that will be used
        for the RL algorithm"""
        reward = int(self._success()) * 10
        r_info = {"reward": reward}
        return reward, r_info

    def _termination(self):
        """Indicates if the robot has reached a terminal state"""
        success = self._success()
        done = success
        d_info = {"success": success}
        return done, d_info

    def step(self, action):
        """Performing a relative action in the environment
        input:
            action: 7 tuple containing
                    Position x, y, z.
                    Angle in rad x, y, z.
                    Gripper action
                    each value in range (-1, 1)

                    OR
                    8 tuple containing
                    Relative Joint angles j1 - j7 (in rad)
                    Gripper action
        output:
            observation, reward, done info
        """
        # Transform gripper action to discrete space
        env_action = action.copy()
        env_action[-1] = (int(action[-1] >= 0) * 2) - 1

        obs, reward, done, info = super().step(env_action)
        done, d_info = self._termination()
        info.update(d_info)
        return obs, reward, False, info
