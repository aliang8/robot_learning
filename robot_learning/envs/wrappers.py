from collections import deque

import cv2
import gymnasium
import gymnasium.spaces as spaces
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.25,
    "azimuth": 145,
    "elevation": -25.0,
    "lookat": np.array([0.0, 0.65, 0.0]),
}
DEFAULT_SIZE = 224


class ImageObservationWrapper(gymnasium.Wrapper):
    def __init__(self, env, image_shape=(64, 64)):
        super().__init__(env)
        self.image_shape = image_shape
        # Set observation space to match the rendered image dimensions
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(image_shape[0], image_shape[1], 3), dtype=np.uint8
        )

        mujoco_renderer = MujocoRenderer(
            model=env.model,
            data=env.data,
            default_cam_config=DEFAULT_CAMERA_CONFIG,
            width=DEFAULT_SIZE,
            height=DEFAULT_SIZE,  # TODO: doesn't accept these inputs, maybe a version mismatch,
        )
        self.env.unwrapped.mujoco_renderer = mujoco_renderer

    def reset(self, **kwargs):
        # Reset the environment and capture the initial image observation
        res = self.env.reset()

        if isinstance(res, tuple):
            info = res[1]
            info["state"] = res[0]
            return self._get_image_obs(), info

        return self._get_image_obs()

    def step(self, action):
        # Take a step in the environment
        step_res = self.env.step(action)

        # Return the image observation instead of the state

        if len(step_res) == 4:
            obs, reward, done, info = step_res
            # add obs into info
            info["state"] = obs
            return self._get_image_obs(), reward, done, info
        else:
            obs, reward, done, terminated, info = step_res
            # add obs into info
            info["state"] = obs
            return self._get_image_obs(), reward, done, terminated, info

    def _get_image_obs(self):
        # Render the environment to capture the current image
        image = self.env.render()

        # Resize the image if needed
        if self.image_shape is not None and image.shape[:2] != self.image_shape:
            # from PIL import Image

            # image = Image.fromarray(image).resize(self.image_shape)
            image = cv2.resize(image, self.image_shape)
            image = np.array(image)

        return image


class FlipImageObsWrapper(gymnasium.Wrapper):
    """
    Flips the image upside down
    """

    def __init__(self, env):
        super(FlipImageObsWrapper, self).__init__(env)
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        return self._process_obs(obs), reward, done, terminated, info

    def _process_obs(self, obs):
        return np.flipud(obs)


class ActionRepeatWrapper(gymnasium.Wrapper):
    """
    Repeat the action for a fixed number of steps
    """

    def __init__(self, env, repeat=2):
        super(ActionRepeatWrapper, self).__init__(env)
        self.repeat = repeat

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        step_reward = 0
        done = False
        for _ in range(self.repeat):
            obs, reward, done, terminated, info = self.env.step(action)
            step_reward += reward

        return obs, step_reward, done, terminated, info


class FrameStackWrapper(gymnasium.Wrapper):
    """
    Stack multiple frames together
    """

    def __init__(self, env, num_frames=4):
        super(FrameStackWrapper, self).__init__(env)
        self.num_frames = num_frames
        self.frames = deque([], maxlen=num_frames)

        # Update the observation space to include the stacked frames
        low = np.repeat(self.observation_space.low, num_frames, axis=-1)
        high = np.repeat(self.observation_space.high, num_frames, axis=-1)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, terminated, info

    def _get_obs(self):
        return np.concatenate(self.frames, axis=-1)


# TODO: not able to get both the state and image when i call reset with the sb wrappers
# # gym
# import gym


# class ImageObservationWrapperv2(gym.Wrapper):
#     def __init__(self, env, image_shape=(64, 64)):
#         super().__init__(env)

#         self.image_shape = image_shape
#         # Set observation space to match the rendered image dimensions
#         self.observation_space = gym.spaces.Box(
#             low=0, high=255, shape=(image_shape[0], image_shape[1], 3), dtype=np.uint8
#         )

#     def reset(self, **kwargs):
#         self.env.reset(**kwargs)

#         return self._get_image_obs()

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         info["state"] = obs
#         return self._get_image_obs(), reward, done, info

#     def _get_image_obs(self):
#         image = self.env.render(mode='rgb_array')

#         if self.image_shape is not None and image.shape[:2] != self.image_shape:
#             image = cv2.resize(image, self.image_shape)
#         return image

#     def get_obs(self):
#         return self.env.get_obs()
