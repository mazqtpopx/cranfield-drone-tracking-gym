#Pygame is necessary for the human game environment
# import pygame
# import pygame.freetype

import gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed, explained_variance

from sb3_contrib.ppo_recurrent import RecurrentPPO

import os
from typing import Optional
from typing import Literal

import os 

import drone_tracking_gym


import matplotlib.pyplot as plt
import numpy as np



# SEG_LOSS_COEF = 0.01
LEARNING_RATE = 0.000075
RUN_NAME = f"Tracking_Dynamic_lr_{LEARNING_RATE}_loaded_model"
LOAD_MODEL = False
LOAD_MODEL_PATH = "saved_models/Tracking_Basic_lr_0.0001"


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.save_latest_path = os.path.join(log_dir, "last_model")
        self.best_mean_reward = -np.inf


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          #save latest model
          self.model.save(self.save_latest_path)
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
        
    def normalize(self, img):
        return 1 -(img-np.min(img))/(np.max(img)-np.min(img))

    def viz_feats(self, observations):
        a = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0)
        b = nn.ReLU()
        a = a.to("cuda")
        b = b.to("cuda")
        c = a(observations)
        feats_detached = c.detach().cpu().numpy()
        feats_avgd = np.sum(feats_detached[0], axis=0)

        feats_avgd = self.normalize(feats_avgd)
        plt.imshow(feats_avgd)
        plt.savefig('features_lay_1.png')
        plt.colorbar()
        plt.show()



def make_env():
    log_dir = "log_dirs/" + RUN_NAME
    env = gym.make("blendtorch-drone_tracking-v0", real_time=False)
    env = Monitor(env, log_dir)
    return env


if __name__ == "__main__":
    log_dir = "log_dirs/" + RUN_NAME

    os.makedirs(log_dir, exist_ok=True)

    env_id = "blendtorch-drone_tracking-v0"
    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(env, f"videos/{RUN_NAME}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)

    obs = env.reset()
    #for continuous 
    obs, reward, done, info = env.step([[0,0,0]])

    reward = 0

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )


    lstm_states = None
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((1,), dtype=bool)


    tensorboard_dir = f"tensorboard_logging/{RUN_NAME}"
    os.makedirs(tensorboard_dir, exist_ok=True)
    if LOAD_MODEL: 
        model = RecurrentPPO.load(LOAD_MODEL_PATH, env, learning_rate=LEARNING_RATE, verbose=1, tensorboard_log=tensorboard_dir, device="cuda")
    else:
        model = RecurrentPPO("CnnLstmPolicy", env, learning_rate=LEARNING_RATE,policy_kwargs=policy_kwargs, n_steps=256, batch_size=256,verbose=1, tensorboard_log=tensorboard_dir, device="cuda")

    

    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
    timesteps = 5000000
    model.learn(total_timesteps=timesteps, callback=callback)

    model.save("./saved_models/" + RUN_NAME)

