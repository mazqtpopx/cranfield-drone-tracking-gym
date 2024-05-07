from gym.envs.registration import register

register(id="blendtorch-drone_tracking-v0", entry_point="drone_tracking_gym.envs:DroneTrackingEnv")