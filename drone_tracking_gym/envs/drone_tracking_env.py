from pathlib import Path
import numpy as np
from gym import spaces
from blendtorch import btt

CONTINUOUS = True
OUTPUT_MASK = False
#"basic_tracking"
#"dynamic_tracking"
#"obstacle_tracking"
DRONE_SCENARIO = "dynamic_tracking"

class DroneTrackingEnv(btt.env.OpenAIRemoteEnv):
    def __init__(self, render_every=1, real_time=True, seed=None, rank=0):

        super().__init__(version="0.0.1")
        start_port= 11000 + rank * 10 #change the start port depending on the rank
        if DRONE_SCENARIO == "obstacle_tracking":
            scene_name = "drone_tracking_obstacles.blend"
        else:
            scene_name = "drone_tracking.blend"

        self.launch(
            scene=Path(__file__).parent / scene_name,
            script=Path(__file__).parent / "drone_tracking.blend.py",
            real_time=real_time,
            render_every=1
            # drone_scenario=DRONE_SCENARIO
            # seed=seed,
            # start_port=start_port,
        )


        if CONTINUOUS:
            self.action_space = spaces.Box(
                np.array([-1, -1, -1]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # PAN, TILT, ZOOM
        else:
            self.action_space = spaces.Discrete(7)
        self.obs = np.zeros((3,160,160))


        # self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]))

        if OUTPUT_MASK:
            self.observation_space = spaces.Box(low=0, high=1, shape=(4, 160, 160), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(3, 160, 160), dtype=np.float32)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(160, 160, 4), dtype=np.uint8)
    
    def step(self, action):
        assert self._env, "Environment not running."
        obs, reward, done, info = self._env.step(action)
        self.obs = obs
        return obs, reward, done, info

    def render(self, mode="human"):
        #need to convert to int8
        frame = np.transpose(self.obs, axes=[1,2,0]) * 255
        frame = frame.astype(np.uint8)
        frame = frame[:,:,::-1]
        return frame

    # def seed(self, seed):
    #     return self._env.seed(seed)
