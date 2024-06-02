# cranfield-drone-tracking-gym

A gym environment for training reinforcement learning agents to autonomously control pan-tilt-zoom cameras to track drones. 

Requirements: 
-Blender (tested on 3.4) 
-Python (tested on 3.9)
pip packages:
-gym
-stable baselines 3
-stable baselines 3 conrtib (for PPO LSTM) 
-blendtorch: follow instllation instructions https://github.com/cheind/pytorch-blender

TBA: Cranfield is undergoing a data repository change. In the meantime the two Blender files are hosted on google drive. 

To get the blender environment working: 
Download the two blender files from [google drive](https://drive.google.com/drive/folders/1zzx-V_QKEmDVkUEARsH-7NaOgS4mtEk5?usp=sharing). Extract them to your cranfield-drone-tracking-gym/drone_tracking_gym/envs/ repository
Update the line 38 from your drone_tracking.path.append("E:/Repos/PTZ_drone_tracking/drone_tracking_gym/envs/") with your local path

To change drone scenarios, change the DRONE_SCENARIO in drone_tracking_gym/envs/drone_tracking_env.py and drone_tracking_gym/envs/drone_tracking.blend.py


Basic tracking:
<p align="center">
    <img width=100% src="https://github.com/mazqtpopx/cranfield-drone-tracking-gym/imgs/basic_tracking.gif">
</p>

Obstacle tracking:
<p align="center">
    <img width=100% src="https://github.com/mazqtpopx/cranfield-drone-tracking-gym/imgs/obstacle_tracking.gif">
</p>

