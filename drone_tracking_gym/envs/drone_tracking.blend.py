#this is the code the controls the blender environment 
#based on structure of pytorch-blender 
#https://github.com/cheind/pytorch-blender/tree/develop/examples/control/cartpole_gym/envs
import bpy
import gpu  # noqa
import bgl

from blendtorch import btb

import os
#from simple_pid import PID
import math

import gym
from gym import spaces
import numpy as np
import sys

#Pygame is necessary for the human game environment
#import pygame
#import pygame.freetype

#Blender maths - e.g. contains Vector data type
import mathutils
from mathutils import Vector

import logging


# from OpenGL.GL import glGetTexImage

import cv2
import random
from simple_pid import PID

from sys import path
#Append this directory to your /PTZ_drone_tracking/drone_tracking_env/envs/
path.append("E:/Repos/PTZ_drone_tracking/drone_tracking_gym/envs/")
#Internal libs
from DroneControl import DroneControl, DroneControl_DynamicTracking, DroneControl_ObstacleTracking
from CameraControl import CameraControl
from OffscreenRenderer import OffScreenRenderer
from Reward import RewardCalculator

STATE_W = 160
STATE_H = 160

WINDOW_W = 160
WINDOW_H = 160

CONTINUOUS = True
OUTPUT_MASK = False
CIRCULAR_GRADIENT_REWARD = True

DRONE_SCENARIO = "dynamic_tracking"

#bpy.data.scenes["Scene"].node_tree.nodes["File Output"].base_path


def gamma(x, coeff=2.2):
    """Return sRGB (gamme encoded o=i**(1/coeff)) image.
    This gamma encodes linear colorspaces as produced by Blender
    renderings.
    Params
    ------
    x: HxWxC uint8 array
        C either 3 (RGB) or 4 (RGBA)
    coeff: scalar
        correction coefficient
    Returns
    -------
    y: HxWxC uint8 array
        Gamma encoded array.
    """
    y = x[..., :3].astype(np.float32) / 255
    y = np.uint8(255.0 * y ** (1 / 2.2))
    if x.shape[-1] == 3:
        return y
    else:
        return np.concatenate((y, x[..., 3:4]), axis=-1)

# def find_first_view3d():
#     """Helper function to find first space view 3d and associated window region.
#     The three returned objects are useful for setting up offscreen rendering in
#     Blender.
#     Returns
#     -------
#     area: object
#         Area associated with space view.
#     window: object
#         Window region associated with space view.
#     space: bpy.types.SpaceView3D
#         Space view.
#     """
#     areas = [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"]
#     assert len(areas) > 0
#     area = areas[0]
#     region = sorted(
#         [r for r in area.regions if r.type == "WINDOW"],
#         key=lambda x: x.width,
#         reverse=True,
#     )[0]
#     spaces = [s for s in areas[0].spaces if s.type == "VIEW_3D"]
#     assert len(spaces) > 0
#     return area, spaces[0], region





    




#OLD--------------------------------------------------------------
        #find the min/max of the bounding box around the target object
        #start_total = (int(pixels[:,0].min()),int(pixels[:,1].min()))
        #end_total = (int(pixels[:,0].max()),int(pixels[:,1].max()))
        
        #we need to clip the max/min with the size of the image though
                
        #find the area        
        #x_len = (int(pixels[:,0].max()) - int(pixels[:,0].min()))
        #y_len = (int(pixels[:,1].max()) - int(pixels[:,1].min()))
        #object_area = x_len * y_len
        #print(f"{object_area=}")
        #img_area = bpy.data.scenes["Scene"].render.resolution_x * bpy.data.scenes["Scene"].render.resolution_y
        ##print(f"{img_area=}")
        #object_area_percentage = (object_area / img_area) * 100
        #print(f"{object_area_percentage=}")

        #return object_area_percentage


class DroneTrackingEnv_Base(btb.env.BaseEnv):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}
    def __init__(self, agent, drone_name, camera_name, dt, scenario, scenario_length = 300):
        super().__init__(agent)
        print("Hello custom env")
        """Init the PTZ tracking environment

        Args: 
            render_mode - 'human', 'AI'
            drone_name - name of the drone as it appears in the editor/collection
            camera_name - name of the camera as it appears in the editor/collection
            dt - timestep (difference in time between each step)
        """
        # self.render_every = 1
        
        # self.display_stats = False
        
        self.renderer = OffScreenRenderer(camera_name, mode="rgb")
        # RENDER_STYLE = "MATERIAL"
        RENDER_STYLE = "RENDERED"
        self.renderer.set_render_style(RENDER_STYLE, True)

        self.scenario = scenario
        if self.scenario == "basic_tracking":
            self.drone = DroneControl(drone_name, dt)
        elif self.scenario == "dynamic_tracking":
            self.drone = DroneControl_DynamicTracking(drone_name, dt)
        elif self.scenario == "obstacle_tracking":
            self.drone = DroneControl_ObstacleTracking(drone_name, dt)
        
        self.camera = CameraControl(camera_name)
        
        self.reward_calc = RewardCalculator(drone_name)
        
        print("Checkpoint1")
        # Define action and observation space
        #nb: continuous is a global var defined at top of file
        if CONTINUOUS:
            self.action_space = (
                np.array([-1, -1, -1]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # PAN, TILT, ZOOM
        else:       
            # In discrete space we have 6 actions - nothing, right, left, up, down, zoom in, zoom out
            self.action_space = spaces.Discrete(7)
        
        # image as observation space:
        self.observation_space = spaces.Box(low=0, high=255, shape=
                        (STATE_W, STATE_H, 3), dtype=np.uint8)

        self.rgb_array = np.zeros([STATE_W,STATE_H,3],dtype=np.uint8)
                        
        self.step_reward = 0
        self.total_reward = 0
        self.prev_reward = 0
        
        self.current_step = 0

        #this needs to be commented out - otherwise it causes frame skipping for some reason
        bpy.data.scenes["Scene"].frame_current = self.current_step

        # Timing - not sure if its actually needed but implement anyway
        self.dt = dt
        self.t = 0

        self.scenario_length = scenario_length
        if self.scenario == "obstacle_tracking":
            self.scenario_length = 1000

        self.current_step_img = None
        self.terminate = False
        # self.scenario = scenario
        self.reset()
        print("Custom env init finished")
        return
    
    
    #we dont actually want to reset anything in the environment...
    def _env_reset(self):
        return self._env_post_step()
    
    def _env_prepare_step(self, action):
        self.step(action)

        self.current_step_img = self.render()
        if CIRCULAR_GRADIENT_REWARD:
            self.step_reward, self.terminate = self._calculate_reward_circular_gradient()
        else:
            self.step_reward, self.terminate = self._calculate_reward_basic()
        return
    
    
    def _env_post_step(self):       
        #We terminate either after reaching 300 steps or after 
        #the object goes out of viewport - in which case self.terminate is set to true
        done = False
        if self.current_step > self.scenario_length:
            done = True

        #but only do this when were on the basic tracking scenario
        # if self.terminate and self.scenario == "basic_tracking": 
        if self.terminate:
            done = True



        if self.current_step_img is not None:
            #we get the mask & concat it with the image so the 4th channel is the mask
            mask = self.reward_calc.get_mask_image()
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            obs_img = cv2.cvtColor(self.current_step_img, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("E:/Repos/PTZ_drone_tracking/debug_imgs/obs"+str(self.current_step)+".png", obs_img)

            #if output mask, stack the obs and mask frames - otherwise output the RGB image only
            if OUTPUT_MASK:
                obs_img = np.dstack((self.current_step_img, mask))

            #convert from 0-255 (for all channels)
            obs_img = obs_img / 255.0

            #lastly, reorder the array so its (c,w,h) as per SB3 standard
            #doing the reordering based on 
            #https://stackoverflow.com/questions/43829711/what-is-the-correct-way-to-change-image-channel-ordering-between-channels-first
            #Hence, the output should be something like (4,640,640)
            #with the first 3 channels being RGB, and fourth being the mask!
            #NB: the mask should only be used for training loss ground truth,
            #it should not be used for trainin per se
            obs_img = np.moveaxis(obs_img, -1, 0)
            
            #Use this for debugging if the images/masks are as expected
            LOCAL_DEBUG = False
            if LOCAL_DEBUG:
                #DEBUG
                #check max/min values - should be between 0-1...
                local_obs = obs_img[:,:,1]
                local_obs_2 = obs_img[:,:,:3]
                local_mask = obs_img[:,:,3]
                #get the first three channels 
                cv2.imshow("local_obs", local_obs_2)
                cv2.imshow("local_MASK", local_mask)
                # cv2.imshow("raw_img", self.current_step_img)
                cv2.waitKey(1)

        else:
            obs_img = self.current_step_img

        return dict(obs=(obs_img), reward=self.step_reward, done=done)

    def _restart(self):
        self.reset()

        
    def reset(self):
        # print("Resetting the environment!")
        # Reset the state of the environment to an initial state
        self.reward = 0
        self.step_reward = 0
        self.prev_reward = 0
        self.total_reward = 0
        self.current_step = 0
        self.t = 0
        self.terminate = False
        
        self.drone.reset()
        self.camera.reset()

        bpy.data.scenes["Scene"].frame_current = self.current_step


        #Why do we need to return the rgb array? I guess for blendtorch interface.?
        return self.rgb_array
    
    
    def step(self, action):
        # print("We are stepping")
        #step the drone
        self.drone.step()
                
        #deal with the action
        if action is not None:
            if CONTINUOUS:
                self.camera.pan(action[0], self.dt)
                self.camera.tilt(action[1], self.dt)
                self.camera.zoom(action[2], self.dt)
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                if action==1:
                    self.camera.right()
                elif action==2:
                    self.camera.left()
                elif action==3:
                    self.camera.up()
                elif action==4:
                    self.camera.down()
                elif action==5:
                    self.camera.zoom_in()
                elif action==6:
                    self.camera.zoom_out()
                
        
        #Update the time/step
        self.t += self.dt     
        self.current_step += 1
        
        #this needs to be commented out - otherwise it causes frame skipping for some reason
        # bpy.data.scenes["Scene"].frame_current = self.current_step

        # print(f"{self.current_step=}")
        return

    def render(self):
        BUFER_GAMMA_ENABLED = False
        buffer = self.renderer.render_nparray()
        #NB: two ways to render, but render_nparray should be quicker in principle
        # buffer = self.renderer.render()
        img_size = (bpy.data.scenes["Scene"].render.resolution_x, bpy.data.scenes["Scene"].render.resolution_y)
        buffer_2 = np.ascontiguousarray(buffer)
        output_img = buffer_2
        if BUFER_GAMMA_ENABLED:
            buffer_gamma = gamma(buffer_2)
            #if output is empty, instead generate a black np array
            if buffer_gamma is None:
                buffer_gamma = np.zeros([img_size[0],img_size[1],3],dtype=np.uint8)
            output_img = buffer_gamma
        return output_img

    def load_image(self, dir, filename):
        filepath = os.path.join(dir, filename)
        
    def __calc_area(self, arr):
        return (arr > 0).sum()
    
    
    #Returns area occupied by mask as a % of image.
    #Input: Mask of type blender image
    def __calc_mask_area_percentage(self, mask):
        #convert mask to np array first
        #(assume we take the blender image as an input)
        np_mask = np.asarray(mask.pixels)
        mask_area = self.__calc_area(np_mask)
        total_area = np_mask.size
        return mask_area/total_area
        

    def _calculate_reward_circular_gradient(self):
        step_reward = 0 
        step_reward = self.reward_calc.calc_area_occupied_by_sphere_gradient_map()

        step_reward /= 200

        if (self.scenario == "obstacle_tracking") and (self.current_step > 300):
            step_reward *= 2

        terminate = False
        if step_reward < 0.04:
            terminate = True
            if self.scenario == "dynamic_tracking":
                # step_reward -= 2
                step_reward -= 1500
            elif self.scenario == "basic_tracking":
                step_reward -= 500
            elif self.scenario == "obstacle_tracking":
                step_reward -= 500


        self.prev_reward = self.reward
        self.step_reward = step_reward
        self.total_reward += step_reward
        return step_reward, terminate

    def _calculate_reward(self):
        area_percantage = self.reward_calc.calc_area_occupied_by_sphere()
        step_reward = 0
        #if the sphere exists in the area (assume it does)
        #and the area is greater than 10% - give +10 reward
        if area_percantage > 50:
            step_reward += 3
        elif area_percantage > 30:
            step_reward += 2
        elif area_percantage > 10:
            step_reward += 1
        elif area_percantage > 0.5:
            step_reward += 0.1
        else:
            step_reward -= 3
            
        
        self.prev_reward = self.reward
        self.step_reward = step_reward
        self.total_reward += step_reward
            
            
        return step_reward
            

    #here we calculate reward in a differen way:
    #if object is in viewport, add zero reward
    #if object is not in the viewport, takeaway 100 reward
    def _calculate_reward_basic(self):
        area_percantage = self.reward_calc.calc_area_occupied_by_sphere()
        step_reward = 0 

        terminate = False
        if area_percantage < 1:
            terminate = True
            step_reward -= 100
        elif area_percantage < 20:
            step_reward = 0
        else:
            step_reward += 1

        self.prev_reward = self.reward
        self.step_reward = step_reward
        self.total_reward += step_reward
            
        return step_reward, terminate



class DroneTrackingEnv_Old(btb.env.BaseEnv):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}
    def __init__(self, agent, render_mode, drone_name, camera_name, dt, scenario="tracking_basic"):
        super().__init__(agent)
        print("Hello custom env")
        """Init the PTZ tracking environment

        Args: 
            render_mode - 'human', 'AI'
            drone_name - name of the sphere as it appears in the editor/collection
            camera_name - name of the camera as it appears in the editor/collection
            dt - timestep (difference in time between each step)
        """
        self.render_every = 1
        
        self.display_stats = False
        self.renderer = OffScreenRenderer(camera_name, mode="rgb")
        # RENDER_STYLE = "MATERIAL"
        RENDER_STYLE = "RENDERED"
        self.renderer.set_render_style(RENDER_STYLE, True)
        
        self.render_mode = render_mode
        print(f"Selected render mode: {render_mode}")
        
        #image data. Used to define the observation space
        #width/height - can modify these later.
        
        #self.screen_width = 600
        #self.screen_height = 400
        #self.n_channels = 3
        #self.observation_img_dtype = np.uint8
        self.drone = DroneControl(drone_name, dt)

        #self.drone.set_acceleration(np.array([-0.020,0.010,0.010]))
        
        self.camera = CameraControl(camera_name)
        
        self.reward_calc = RewardCalculator(drone_name)
        
        print("Checkpoint1")
        # Define action and observation space
        #nb: continuous is a global var defined at top of file
        if CONTINUOUS:
            self.action_space = (
                np.array([-1, -1, -1]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # PAN, TILT, ZOOM
        else:       
            # We have 6 actions - nothing, right, left, up, down, zoom in, zoom out
            self.action_space = spaces.Discrete(7)
        
        
        
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=
                        (STATE_W, STATE_H, 3), dtype=np.uint8)


        self.rgb_array = np.zeros([STATE_W,STATE_H,3],dtype=np.uint8)
                        
        self.reward_range = (0, 100)
        self.step_reward = 0
        self.total_reward = 0
        self.prev_reward = 0
        
        self.current_step = 0

        #this needs to be commented out - otherwise it causes frame skipping for some reason
        bpy.data.scenes["Scene"].frame_current = self.current_step

        #the other current step is broken but needed to run the sim? 
        #so we use this for stopping criteria
        # self.real_current_step = 0

        self.clock = None

        #the rendered image from the camera
        self.screen = None
        self.state = None
        
        self.dt = dt
        self.t = 0
        self.current_step_img = None
        self.terminate = False
        self.scenario = scenario
        self.reset()
        print("Custom env init finished")
        


    #we dont actually want to reset anything in the environment...
    def _env_reset(self):
        # print("WE ARE RESETTING!!!!!!!!!")
        # self.current_step = 0
        # self.reset()
        # self.current_step_img = self.render("rgb_image")
        # return dict(
        #     obs=self.current_step_img,
        #     reward=0.,
        #     done=False
        # )
        return self._env_post_step()


        # return self._env_post_step()
        # print("WE ARE RESETTING!!!!!!!!!")
        # print("Starting reset")
        # self.reset()
        # print("Finishing Reset")
        # return self._env_post_step()

    def _env_prepare_step(self, action):
        # print("Preparing step")
        # if self.scenario == "tracking-basic":
        #     self.step(action)
        # elif self.scanrio == "tracking-occluded":
        #     return
        # elif self.scanrio == "tracking-complex":
        #     return
        # else:
        #     return #throw exception: scenario not specified
        self.step(action)

        self.current_step_img = self.render("rgb_image")
        if CIRCULAR_GRADIENT_REWARD:
            self.step_reward, self.terminate = self._calculate_reward_circular_gradient()
        else:
            self.step_reward, self.terminate = self._calculate_reward_basic()
        return
        
        
        # print(f"{self.current_step_img=}")
        # print("Finishing step")
        # return


    def _env_post_step(self):
        # print("Post step")
        #we want to return the image as the observaiton
        # observation = self.render("rgb_image")
        # 
        
        # if "rgb_array" not in self.ctx:
        #     print("RGB ARRAY NOT CALLED YET")
        #     self.ctx["rgb_array"] = self.renderer.render()
        # self.current_step_img = self.render("rgb_image")
        
        #We terminate either after reaching 300 steps or after 
        #the object goes out of viewport - in which case self.terminate is set to true
        done = False


        if self.current_step > self.scenario_length:
            done = True

        if self.terminate:
            done = True

        print(f"{self.current_step=} {self.step_reward=}, {done=}")

        # OUTPUT_MASK = False

        

        if self.current_step_img is not None:
            
            #we want to get the mask & concat it with the image so the 4th channel is the mask
            mask = self.reward_calc.get_mask_image()
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            #convertfrom 0-255 to 0-1 range
            # mask = mask / 255.0
            #....lastly add the '1' dimension so that we can concatenate (going from (w,h) to (w,h,1))
            # mask = np.expand_dims(mask, axis=2)



            obs_img = cv2.cvtColor(self.current_step_img, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("E:/Repos/PTZ_drone_tracking/debug_imgs/obs"+str(self.current_step)+".png", obs_img)
            #if output mask, stack the obs and mask frames
            if OUTPUT_MASK:
                obs_img = np.dstack((self.current_step_img, mask))

            #convert from 0-255 (for all channels)
            obs_img = obs_img / 255.0

            #lastly, reorder the array so its (c,w,h) as per SB3 standard
            #doing the reordering based on 
            #https://stackoverflow.com/questions/43829711/what-is-the-correct-way-to-change-image-channel-ordering-between-channels-first
            #Hence, the output should be something like (4,640,640)
            #with the first 3 channels being RGB, and fourth being the mask!
            #NB: the mask should only be used for training loss ground truth,
            #it should not be used for trainin per se
            obs_img = np.moveaxis(obs_img, -1, 0)
            
            LOCAL_DEBUG = False
            if LOCAL_DEBUG:
                #DEBUG
                #check max/min values - should be between 0-1...
                local_obs = obs_img[:,:,1]
                local_obs_2 = obs_img[:,:,:3]
                local_mask = obs_img[:,:,3]
                #get the first three channels 
                cv2.imshow("local_obs", local_obs_2)
                cv2.imshow("local_MASK", local_mask)
                # cv2.imshow("raw_img", self.current_step_img)
                cv2.waitKey(1)

        else:
            obs_img = self.current_step_img

        return dict(obs=(obs_img), reward=self.step_reward, done=done)

    def _restart(self):
        self.reset()

    def reset(self):
        # print("WE ARE RESETTING THE ENVIRONMENT!!!!!!!!!")
        # Reset the state of the environment to an initial state
        self.reward = 0
        self.step_reward = 0
        self.prev_reward = 0
        self.total_reward = 0
        self.current_step = 0
        self.t = 0
        self.terminate = False
        
        self.drone.reset()
        self.camera.reset()

        bpy.data.scenes["Scene"].frame_current = self.current_step
        
        # if self.render_mode == "human" or self.render_mode == "offscreen_human" or self.render_mode == "opencv_human":
        #     obs = self.render(self.render_mode)
        # self.rgb_array = self.render("rgb_array")


        return self.rgb_array
 

    def step(self, action):
        # print("We are stepping")

        #randomize sphere movement
        #self.drone.update_acceleration(np.random.rand(1,3))
        self.drone.step()
#        self._move_camera(action)
        
        #self.current_step += 1
        #print("Hello!")
        # print(f"{action=}")
        
        
        if action is not None:
            if CONTINUOUS:
                self.camera.pan(action[0], self.dt)
                self.camera.tilt(action[1], self.dt)
                self.camera.zoom(action[2], self.dt)
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                if action==1:
                    self.camera.right()
                elif action==2:
                    self.camera.left()
                elif action==3:
                    self.camera.up()
                elif action==4:
                    self.camera.down()
                elif action==5:
                    self.camera.zoom_in()
                elif action==6:
                    self.camera.zoom_out()
                
        
        # self.render(self.render_mode)
        # print(f"{self.current_step=}")
        
                
        
        
        self.t += self.dt     
        
        self.current_step += 1
        #this needs to be commented out - otherwise it causes frame skipping for some reason
        # bpy.data.scenes["Scene"].frame_current = self.current_step




        # self.real_current_step += 1
        # print(f"{self.current_step=}")
        # print(f"{self.real_current_step=}")
        #return step_reward, terminated, truncated, {} 
        #terminated - not needed...
        #truncated - finished? again not needed. 
        return
        
 
    def render(self, render_mode):
#         #Initialize the pygame environment
#         if self.screen is None and (self.render_mode == "human" or self.render_mode == "offscreen_human"):
#             print("init pygame")
#             pygame.init()
#             pygame.display.init()
#             self.screen = pygame.display.set_mode((bpy.data.scenes["Scene"].render.resolution_x, bpy.data.scenes["Scene"].render.resolution_y))
#             print("init pygame complete")
#         if (self.render_mode == "opencv_human"):
#             import time
#             self.start_time = time.time()
#         if self.clock is None:
#             #init clock for FPS calc
#             self.clock = pygame.time.Clock()
#         if self.display_stats:
#             #self.surf = pygame.Surface((640, 640))
#             GAME_FONT = pygame.freetype.SysFont("comicsansms", 24)

# #            pygame.font.init()
        

        if render_mode == "rgb_image":
            BUFER_GAMMA_ENABLED = False
            buffer = self.renderer.render_nparray()
            # buffer = self.renderer.render()
            img_size = (bpy.data.scenes["Scene"].render.resolution_x, bpy.data.scenes["Scene"].render.resolution_y)
            buffer_2 = np.ascontiguousarray(buffer)
            output_img = buffer_2
            if BUFER_GAMMA_ENABLED:
                buffer_gamma = gamma(buffer_2)
                #if output is empty, instead generate a black np array
                if buffer_gamma is None:
                    buffer_gamma = np.zeros([img_size[0],img_size[1],3],dtype=np.uint8)
                output_img = buffer_gamma
            return output_img

        # #Display the blender image on the pygame screen as the background
        # if render_mode == "human":
        #     # Render the Blender image
        #     bpy.ops.render.render(animation=False, write_still=False, use_viewport=True)             
        #     camera_image_dir = "C:\\tmp\\camera_image"
        #     img_filename = "Image" + str(self.current_step).zfill(4) + ".png"
        #     camera_image_filepath = os.path.join(camera_image_dir, img_filename)
        #     print(f"{camera_image_filepath=}")
        #     img_camera = bpy.data.images.load(camera_image_filepath, check_existing=False)
            
        #     pygame_img = pygame.image.load(camera_image_filepath)
        #     self.screen.blit(pygame_img, (0,0))
        #     pygame.display.flip()
        
            
            
        # if render_mode == "offscreen_human":
        #     #print("Start rendr")
        #     buffer = self.renderer.render_nparray()
        #     img_size = (bpy.data.scenes["Scene"].render.resolution_x, bpy.data.scenes["Scene"].render.resolution_y)
            
        #     import time

        #     start = time.process_time()



            
        #     #print(f"{buffer=}")
        #     #print(f"{buffer.flags=}")
        #     #buffer_2 = buffer.copy(order='C')
        #     buffer_2 = np.ascontiguousarray(buffer)
            
            
        #     buffer_gamma = gamma(buffer_2)
            
        #     #print(f"{buffer_2.flags=}")
        #     pygame_img = pygame.image.frombuffer(buffer_gamma, img_size, "RGBA")
        #     #print("End rendr")
            
        #     if self.display_stats:
        #         #render reward text
        #         text_surface, rect = GAME_FONT.render(f"reward = {self.reward}", (255, 255, 255))
        #         #renderfps text 
        #         fps_text, rect =  GAME_FONT.render(f"fps: {self.clock.get_fps():.1f}", (255, 255, 255))
                
        #     #Display the buffer
        #     self.screen.blit(pygame_img, (0,0))
        #     #Display reward text
        #     self.screen.blit(text_surface, (40, 40))
        #     #Dislay the FPS 
        #     self.screen.blit(fps_text, (40, 80))
        #     #Flip the display
        #     pygame.display.flip()
            
        #     end = time.process_time()
        #     print(f"time to convert to c-order: {end - start}")
        #     #Tick the clock (for fps calc)
        #     self.clock.tick()



        # #faster to render on its own, but slower to process actions using cv2.waitkey 
        # #80 fps on its own,
        # #20 fps with cv2.waitkey()
        # if render_mode == "opencv_human":
        #     # buffer = self.renderer.render()
        #     img_size = (bpy.data.scenes["Scene"].render.resolution_x, bpy.data.scenes["Scene"].render.resolution_y)
            
            
        #     print("FPS: ", self.current_step / (time.time() - self.start_time))

            
        #     fps_text = f"fps: {self.clock.get_fps():.1f}"
        #     #out_img = cv2.putText(buffer, fps_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        #     stats_img = np.zeros((200,200,3), dtype=np.uint8)
        #     stats_img = cv2.putText(stats_img, fps_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        #     cv2.imshow("game", buffer)
        #     cv2.imshow("stats", stats_img)
        #     print(fps_text)
        #     self.clock.tick()
            
        #     k = cv2.waitKey(33)


        return

    def load_image(self, dir, filename):
        filepath = os.path.join(dir, filename)
        
        
    
    def __calc_area(self, arr):
        return (arr > 0).sum()

    #Returns area occupied by mask as a % of image.
    #Input: Mask of type blender image
    def __calc_mask_area_percentage(self, mask):
        #convert mask to np array first
        #(assume we take the blender image as an input)
        np_mask = np.asarray(mask.pixels)
        mask_area = self.__calc_area(np_mask)
        total_area = np_mask.size
        return mask_area/total_area
        

    def _calculate_reward_circular_gradient(self):
        step_reward = 0 
        step_reward = self.reward_calc.calc_area_occupied_by_sphere_gradient_map()

        step_reward /= 200
        terminate = False
        if step_reward < 0.2:
            terminate = True
            step_reward -= 500

        self.prev_reward = self.reward
        self.step_reward = step_reward
        self.total_reward += step_reward
        return step_reward, terminate

    def _calculate_reward(self):
        area_percantage = self.reward_calc.calc_area_occupied_by_sphere()
        step_reward = 0
        #if the sphere exists in the area (assume it does)
        #and the area is greater than 10% - give +10 reward
        if area_percantage > 50:
            step_reward += 3
        elif area_percantage > 30:
            step_reward += 2
        elif area_percantage > 10:
            step_reward += 1
        elif area_percantage > 0.5:
            step_reward += 0.1
        else:
            step_reward -= 3
            
        
        self.prev_reward = self.reward
        self.step_reward = step_reward
        self.total_reward += step_reward
            
            
        return step_reward
            

    #here we calculate reward in a differen way:
    #if object is in viewport, add zero reward
    #if object is not in the viewport, takeaway 100 reward
    def _calculate_reward_basic(self):
        area_percantage = self.reward_calc.calc_area_occupied_by_sphere()
        step_reward = 0 

        terminate = False
        if area_percantage < 1:
            terminate = True
            step_reward -= 100
        elif area_percantage < 20:
            step_reward = 0
        else:
            step_reward += 1

        #red_cube_training_fixed2 reward
        #spoiler - it didnt work
        # terminate = False
        # if area_percantage < 0.5:
        #     step_reward -= 100
        #     terminate = True
        # elif area_percantage < 5:
        #     step_reward -= 10
        # elif area_percantage < 25:
        #     step_reward -= 1
        # else:
        #     step_reward += 1


        self.prev_reward = self.reward
        self.step_reward = step_reward
        self.total_reward += step_reward
            
        return step_reward, terminate
        
        
    def _calculate_reward_old(self):
        #get the composite image
        mask_dir = "C:\\tmp\\mask"
        camera_image_dir = "C:\\tmp\\camera_image"
        img_filename = "Image" + str(self.current_step).zfill(4) + ".png"
        
        mask_filepath = os.path.join(mask_dir, img_filename)
        camera_image_filepath = os.path.join(camera_image_dir, img_filename)
        
        #load camera viewport image and the mask
        img_mask = bpy.data.images.load(mask_filepath, check_existing=False)
        img_camera = bpy.data.images.load(camera_image_filepath, check_existing=False)
        
        mask_area_percentage = self.__calc_mask_area_percentage(img_mask)
        
        
        step_reward = 0
        #if the sphere exists in the area (assume it does)
        #and the area is greater than 10% - give +10 reward
        if mask_area_percentage > 0.03:
            step_reward += 3
        elif mask_area_percentage > 0.01:
            step_reward += 2
        elif mask_area_percentage > 0:
            step_reward += 1
        else:
            step_reward -= 3
            
        self.prev_reward = self.reward
        self.reward += step_reward
            
            
        return step_reward
            
        
        


#     def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
#         img_stream.seek(0)
#         img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
#         return cv2.imdecode(img_array, cv2_img_flag)
 
 




# def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
#     img_stream.seek(0)
#     img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)


#     return cv2.imdecode(img_array, cv2_img_flag)

# class BoundingArea():
#     """Defines the bounding area of the drone. This is the area in which the drone can fly. 
#     If the drone flies outside of these bounds, something happens.
    
#     Atrributes:
#         min: array in format [x_min, y_min, z_min]
#         max: array in format [x_max, y_max, z_max]

#     In the future, we should expand this class to include other shapes
#     """
#     def __init__(self, min, max):
#         self.min = min
#         self.max = max

#         self.x_min = min[0]
#         self.x_max = max[0]
        
#         self.y_min = min[1]
#         self.y_max = max[1]

#         self.z_min = min[2]
#         self.z_max = max[2]

# class Position():
#     """XYZ position
    
#     Atrributes:
#         position: np array of form np.array([x,y,z])

#     """
#     def __init__(self, position):
#         #probably not needed
#         self.position = position

#         self.x = position[0]        
#         self.y = position[1]
#         self.z = position[2]



# class DroneTrackingEnv(btb.env.BaseEnv):
#     """Defines the environment mechanics and renders the drone on viewport
#     Controls the action of the camera
        
#     Attributes:
#         drone_name: name of the drone as it appears in the Blender menu
#         bounding_area: area in which the drone can fly. This should be of data type BoundingArea()
#     """
#     def __init__(self, agent, bounding_area, render_mode, sphere_name, camera_name, dt):
#         super().__init__(agent)
#         self.drone = Drone(drone_name="Drone", bounding_area=bounding_area)
#         self.bounding_area = bounding_area

#         self.display_stats = False
#         self.renderer = OffScreenRenderer(camera_name, mode="rgba")
#         RENDER_STYLE = "MATERIAL"
#         #RENDER_STYLE = "RENDERED"
#         self.renderer.set_render_style(RENDER_STYLE, True)
        
#         self.render_mode = render_mode


#         self.camera = CameraControl(camera_name)
        
#         self.reward_calc = RewardCalculator(drone_name="Drone")

                
#         # Define action and observation space
#         if CONTINUOUS:
#             self.action_space = (
#                 np.array([-1, -1, -1]).astype(np.float32),
#                 np.array([+1, +1, +1]).astype(np.float32),
#             )  # PAN, TILT, ZOOM
#         else:       
#             # We have 6 actions - nothing, right, left, up, down, zoom in, zoom out
#             self.action_space = spaces.Discrete(7)
        
        
#         # Example for using image as input:
#         self.observation_space = spaces.Box(low=0, high=255, shape=
#                         (STATE_W, STATE_H, 3), dtype=np.uint8)
                        
#         self.reward_range = (0, 100)
#         self.reward = 0
#         self.prev_reward = 0
        
#         self.current_step = 0
#         bpy.data.scenes["Scene"].frame_current = self.current_step

#         self.clock = None

#         #the rendered image from the camera
#         self.screen = None           
#         self.state = None
        
#         self.dt = dt
#         self.t = 0
        
    
#     def _env_reset(self):
#         self.__reset()
#         return

#     def _env_prepare_step(self, action):
#         self.__step(action)
#         return

#     def _env_post_step(self):
#         return



    

# #In this scenario, the cube starts in the camras viewport 
# class Cube()


        

#This sets up the relevant parameters to help with the rendering process
def setup(fps):
    #set a high clipping distance
    bpy.data.cameras["Camera"].clip_end = 1000
    bpy.data.scenes["Scene"].render.fps = 30
    return
    #Set persistent data to true - cashes scene data on first render
    #This greatly reduces the render time on subsequent frames! 
    #bpy.context.scene.render.use_persistent_data = True
    
    #Set the output img to png format
    #bpy.context.scene.render.image_settings.file_format = 'PNG'
    
    


#bpy.data.cameras["Camera"].dof.focus_object = bpy.data.objects[DRONE_NAME]


#location
#bpy.data.objects["Sphere"].location

#bpy.data.objects["Camera"].location
        
def main():
    #setup blender env
    fps = 30
    setup(fps)
    args, remainder = btb.parse_blendtorch_args()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render-every", default=None, type=int)
    parser.add_argument("--real-time", dest="realtime", action="store_true")
    parser.add_argument("--no-real-time", dest="realtime", action="store_false")
    # parser.add_argument("--drone_scenario", dest="scenario")
    # parser.add_argument("--scenario", dest="scenario")
    envargs = parser.parse_args(remainder)

    # print({f"{envargs.scenario=}"})

    print(f"{envargs.realtime=}")
    # print(f"{envargs.scenario=}")

    agent = btb.env.RemoteControlledAgent(
        args.btsockets["GYM"], real_time=envargs.realtime
    )
    print(f"{agent=}")
    
    env = DroneTrackingEnv_Base(agent, "Sphere", "Camera", 1/fps, DRONE_SCENARIO)
    # env = DroneTrackingEnv_DynamicTracking(agent, )
    #env.attach_default_renderer(every_nth=envargs.render_every)
    env.run(frame_range=(1, 10000), use_animation=True)


main()
