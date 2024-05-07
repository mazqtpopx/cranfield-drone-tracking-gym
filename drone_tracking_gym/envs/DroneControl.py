import bpy
import numpy as np
import mathutils
import random

from SecondOrderSystem import SecondOrderSystem

class DroneControl():
    def __init__(self, drone_name, timestep):
        self.drone = bpy.data.objects[drone_name]
        self.timestep = timestep
        
        self.reset()

        #self.pos_min_limits = (0,0,0)
        #self.pos_max_limits = (100,100,100)
        self.__update_position_of_rendered_object()
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        #MOVE THIS TO CONFIG INIT
        RANDOM_STARTING_POSITION = False

        #RANDOM POSITION
        if RANDOM_STARTING_POSITION:
            pos_init = np.random.rand(1,3)[0]
            pos_init[0] *= 50
            pos_init[1] *= 50
            pos_init[2] *= 30
        else:
            #CONSTANT POSITION IN FRONT OF CAMERA
            pos_init = np.zeros((1,3))[0]
            pos_init[0] = 0
            pos_init[1] = 15
            pos_init[2] = 0

        self.pos_prev = pos_init
        self.pos_curr = pos_init
        
        # self.vel_init = np.array([-15.0,0.0,2.0]) * 0.1
        # self.vel_init = np.random.rand(1,3)[0] * 0.5
        self.vel_init = np.random.uniform(low=-1.0, high=1.0, size=(1,3))[0] * 0.2
        self.vel_prev = self.vel_init
        self.vel_curr = self.vel_init
        
        # self.acc_init = np.array([0.0,0.0,0.0])
        # self.acc_prev = self.acc_init
        # self.acc_curr = self.acc_init
    
    
    def step(self):
        self.current_step += 1
        VELOCITY_ENABLED = True

        BROWNIAN = False
        
        if VELOCITY_ENABLED:
            if BROWNIAN:
                self.update_position_brownian()
            else:
                self.__update_position()
                # self.update_velocity()
            
            
        self.__update_position_of_rendered_object()
        

        return 

    def update_position_brownian(self):
        self.pos_prev = self.pos_curr
        # diff = np.random.uniform(low=-1, high=1)
        diff = np.random.rand(1,3)[0] * 1
        self.pos_curr = self.pos_prev + diff
        # print(f"{self.pos_curr=} {self.pos_prev=} {diff=}")
        self.__check_boundaries()


    
    def set_acceleration(self, acc):
        self.acc_curr = acc
    
    def update_acceleration(self,acc):
        self.acc_curr = self.acc_prev + acc
        
    def update_velocity(self):
        # print("WTF")
        # print(f"{self.vel_prev=} {self.acc_curr=} {self.timestep=}")
        self.vel_prev = self.vel_curr
        self.vel_curr = self.vel_prev + self.acc_curr * self.timestep
        
        
    def __update_position(self):
        self.pos_prev = self.pos_curr
        self.pos_curr = self.pos_curr + self.vel_curr
        # self.pos_curr = self.pos_prev + self.vel_curr * self.timestep
        self.__check_boundaries()
        
        
    
    def __update_position_of_rendered_object(self):
        #NB: we have to convert from np array to Blender vector
        self.drone.location = mathutils.Vector(self.pos_curr)

    def __check_boundaries(self):
        #move these bounds to config init
        if self.pos_curr[0] > 300:
            print("CLIPPING")
            self.pos_curr[0] = 300
            self.vel_curr[0] = -self.vel_curr[0]
        if self.pos_curr[0] < -300:
            print("CLIPPING")
            self.pos_curr[0] = -300
            self.vel_curr[0] = -self.vel_curr[0]
        if self.pos_curr[1] > 300:
            print("CLIPPING")
            self.pos_curr[1] = 300
            self.vel_curr[1] = -self.vel_curr[1]
        if self.pos_curr[1] < -300:
            print("CLIPPING")
            self.pos_curr[1] = -300
            self.vel_curr[1] = -self.vel_curr[1]
        if self.pos_curr[2] > 50:
            print("CLIPPING")
            self.pos_curr[2] = 50
            self.vel_curr[2] = -self.vel_curr[2]
        if self.pos_curr[2] < -50:
            print("CLIPPING")
            self.pos_curr[2] = -50
            self.vel_curr[2] = -self.vel_curr[2]


class DroneControl_DynamicTracking():
    #change_pos_target_every updates the target every n steps
    def __init__(self, drone_name, timestep, change_pos_target_every_n_steps=30):
        self.drone = bpy.data.objects[drone_name]
        self.timestep = timestep
        self.change_pos_target_every_n_steps = change_pos_target_every_n_steps
        #In here we treat the drone as a second order system in the xyz axix
        self.x_model = SecondOrderSystem(0.002, 0.2, 0.5, 0.0, 1.0)
        self.y_model = SecondOrderSystem(0.002, 0.2, 0.5, 0.0, 1.0)
        self.z_model = SecondOrderSystem(0.002, 0.2, 0.5, 0.0, 1.0)
        #We're going to ignore the rotation axis and keep them static
        self.target_pos = np.zeros(3)
        # self.actual_pos = np.zeros(3)
        self.current_step = 0

        # pos_init = np.zeros((1,3))[0]
        # pos_init[0] = 0
        # pos_init[1] = 15
        # pos_init[2] = 0
        # self.set_target_pos(pos_init)
        self.reset()

    def reset(self):
        self.current_step = 0
        #MOVE THIS TO CONFIG INIT
        RANDOM_STARTING_POSITION = False

        #RANDOM POSITION
        if RANDOM_STARTING_POSITION:
            pos_init = np.random.rand(1,3)[0]
            pos_init[0] *= 50
            pos_init[1] *= 50
            pos_init[2] *= 30
        else:
            #CONSTANT POSITION IN FRONT OF CAMERA
            pos_init = np.zeros((1,3))[0]
            pos_init[0] = 0
            pos_init[1] = 15
            pos_init[2] = 0

        self.pos_curr = pos_init
        
        self.x_model.reset(pos_init[0])
        self.y_model.reset(pos_init[1])
        self.z_model.reset(pos_init[2])

        self.__update_position_of_rendered_object()
        # self.vel_init = np.array([-15.0,0.0,2.0]) * 0.1
        # self.vel_init = np.random.rand(1,3)[0] * 0.5
        # self.vel_init = np.random.uniform(low=-1.0, high=1.0, size=(1,3))[0] * 0.2
        # self.vel_prev = self.vel_init
        # self.vel_curr = self.vel_init
    
                
    #set target pos in the format of a np array of size 3 [x,y,z]
    #nb: this is unused as we use set_random_target_pos instead
    #but this can be used as a manual way of setting the target
    def set_target_pos(self, target_pos):
        self.target_pos = target_pos

    #sets random target pos within bounds [+-50,+-50,+-30]
    def set_random_target_pos(self):
        self.target_pos = np.random.uniform(low=-1.0, high=1.0, size=(1,3))[0]
        self.target_pos[0] *= 50
        self.target_pos[1] *= 50
        self.target_pos[2] *= 30

    def step(self):
        # print(f"Steping")
        print(f"{self.current_step=}")
        self.current_step += 1
        if self.current_step % self.change_pos_target_every_n_steps == 0:
            print(f"{self.current_step=} == {self.change_pos_target_every_n_steps=}")
            print(f"CHANGING POSITION TARGET")
            self.set_random_target_pos()

        self.__update_position()
        return 
    
    def __update_position(self):
        actual_x = self.x_model.step(self.target_pos[0])
        actual_y = self.y_model.step(self.target_pos[1])
        actual_z = self.z_model.step(self.target_pos[2])


        self.pos_curr = np.array([actual_x, actual_y, actual_z])
        #pos_curr is used to display the drone in __update_position_of_rendered_object
        # self.pos_curr = self.actual_pos

        self.__update_position_of_rendered_object()

    def __update_position_of_rendered_object(self):
        #NB: we have to convert from np array to Blender vector
        self.drone.location = mathutils.Vector(self.pos_curr)


#In this scenario, we fly the drone behind one of the tree trunks. 
#Then the drone cane wait for x amount of steps (0-100)
#Finally, the drone is flown in a random direction
#Each of the tree trunks is centered at 25,25 25,-25, -25,25, -25,-25
class DroneControl_ObstacleTracking():
    #change_pos_target_every updates the target every n steps
    def __init__(self, drone_name, timestep, change_pos_target_every_n_steps=30):
        self.drone = bpy.data.objects[drone_name]
        self.timestep = timestep
        self.change_pos_target_every_n_steps = change_pos_target_every_n_steps
        #In here we treat the drone as a second order system in the xyz axix
        self.x_model = SecondOrderSystem(0.0015, 0.2, 0.5, 0.0, 1.0)
        self.y_model = SecondOrderSystem(0.0015, 0.2, 0.5, 0.0, 1.0)
        self.z_model = SecondOrderSystem(0.0015, 0.2, 0.5, 0.0, 1.0)
        #We're going to ignore the rotation axis and keep them static
        self.target_pos = np.zeros(3)
        # self.actual_pos = np.zeros(3)
        self.current_step = 0

        self.first_wait_step = 150
        print("selected obstacle tracking")

        # pos_init = np.zeros((1,3))[0]
        # pos_init[0] = 0
        # pos_init[1] = 15
        # pos_init[2] = 0
        # self.set_target_pos(pos_init)
        self.reset()

    def reset(self):
        self.current_step = 0
        #MOVE THIS TO CONFIG INIT
        RANDOM_STARTING_POSITION = False

        #RANDOM POSITION
        if RANDOM_STARTING_POSITION:
            pos_init = np.random.rand(1,3)[0]
            pos_init[0] *= 50
            pos_init[1] *= 50
            pos_init[2] *= 30
        else:
            #CONSTANT POSITION IN FRONT OF CAMERA
            pos_init = np.zeros((1,3))[0]
            pos_init[0] = 0
            pos_init[1] = 15
            pos_init[2] = 0

        self.pos_curr = pos_init
        
        self.x_model.reset(pos_init[0])
        self.y_model.reset(pos_init[1])
        self.z_model.reset(pos_init[2])

        #pick the random tree to fly to
        self.random_tree = random.randint(1,4)

        #wait between [0,100] steps
        wait_time = random.randint(0,60)
        self.final_wait_step = self.first_wait_step + wait_time
        
        
        # self.random_tree = 4
        if self.random_tree == 1:
            self.target_pos = [25,25,4]
        elif self.random_tree == 2:
            self.target_pos = [-25,25,4]
        elif self.random_tree == 3:
            self.target_pos = [25,-25,4]
        elif self.random_tree == 4:
            self.target_pos = [-25,-25,4]


        self.__update_position_of_rendered_object()
        # self.vel_init = np.array([-15.0,0.0,2.0]) * 0.1
        # self.vel_init = np.random.rand(1,3)[0] * 0.5
        # self.vel_init = np.random.uniform(low=-1.0, high=1.0, size=(1,3))[0] * 0.2
        # self.vel_prev = self.vel_init
        # self.vel_curr = self.vel_init
    
                
    #set target pos in the format of a np array of size 3 [x,y,z]
    #nb: this is unused as we use set_random_target_pos instead
    #but this can be used as a manual way of setting the target
    def set_target_pos(self, target_pos):
        self.target_pos = target_pos

    #sets random target pos within bounds [+-50,+-50,+-30]
    # def set_random_target_pos(self):
    #     self.target_pos = np.random.uniform(low=-1.0, high=1.0, size=(1,3))[0]
    #     self.target_pos[0] *= 50
    #     self.target_pos[1] *= 50
    #     self.target_pos[2] *= 30

    def set_random_target_pos(self):
        self.target_pos = np.random.uniform(low=-1.0, high=1.0, size=(1,3))[0]
        self.target_pos[0] *= 30
        self.target_pos[1] *= 30
        self.target_pos[2] *= 15


    def step(self):
        # print(f"Steping")
        print(f"{self.current_step=}")
        self.current_step += 1
        # if self.current_step % self.change_pos_target_every_n_steps == 0:
        #     print(f"{self.current_step=} == {self.change_pos_target_every_n_steps=}")
        #     print(f"CHANGING POSITION TARGET")
        #     self.set_random_target_pos()
        #wait
        if self.current_step == self.first_wait_step:
            #enter the waiting state... do nothing
            #self.final_waiting_step = 
            print("waiting")
        elif self.current_step == self.final_wait_step:
            self.set_random_target_pos()

        self.__update_position()
        return 
    
    def __update_position(self):
        actual_x = self.x_model.step(self.target_pos[0])
        actual_y = self.y_model.step(self.target_pos[1])
        actual_z = self.z_model.step(self.target_pos[2])


        self.pos_curr = np.array([actual_x, actual_y, actual_z])
        #pos_curr is used to display the drone in __update_position_of_rendered_object
        # self.pos_curr = self.actual_pos

        self.__update_position_of_rendered_object()

    def __update_position_of_rendered_object(self):
        #NB: we have to convert from np array to Blender vector
        self.drone.location = mathutils.Vector(self.pos_curr)