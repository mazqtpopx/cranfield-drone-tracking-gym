# cranfield-drone-tracking-gym

## Examples
Basic tracking:
<p align="center">
    <img width=25% src="https://github.com/mazqtpopx/cranfield-drone-tracking-gym/blob/main/imgs/basic_tracking.gif">
</p>

Dynamic tracking:
<p align="center">
    <img width=25% src="https://github.com/mazqtpopx/cranfield-drone-tracking-gym/blob/main/imgs/dynamic_tracking.gif">
</p>

Obstacle tracking:
<p align="center">
    <img width=25% src="https://github.com/mazqtpopx/cranfield-drone-tracking-gym/blob/main/imgs/obstacle_tracking.gif">
</p>


## Installation
A gym environment for training reinforcement learning agents to autonomously control pan-tilt-zoom cameras to track drones. 

Requirements: 
- Blender (tested on 3.4) 
- Python (tested on 3.9)

pip packages:
- gym
- stable baselines 3
- stable baselines 3 contrib (for PPO LSTM) 
- blendtorch: follow instllation instructions https://github.com/cheind/pytorch-blender

TBA: Cranfield is undergoing a data repository change. In the meantime the two Blender files are hosted on google drive. 

To get the blender environment working: 
Download the two blender files from [google drive](https://drive.google.com/drive/folders/1zzx-V_QKEmDVkUEARsH-7NaOgS4mtEk5?usp=sharing). Extract them to your cranfield-drone-tracking-gym/drone_tracking_gym/envs/ repository

If you wish to test the models, you can download the pre-trained model weights [here](https://drive.google.com/drive/folders/1XipW5XxtcUODThjmtj5xKh4epXWISXuG?usp=sharing)

- Update the line 38 from your drone_tracking.path.append("E:/Repos/PTZ_drone_tracking/drone_tracking_gym/envs/") with your local path
- To change drone scenarios, change the DRONE_SCENARIO in drone_tracking_gym/envs/drone_tracking_env.py and drone_tracking_gym/envs/drone_tracking.blend.py
