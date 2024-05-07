import bpy
import math
from SecondOrderSystem import SecondOrderSystem



class CameraControl():
    def __init__(self, camera_name):
        self.camera_object = bpy.data.objects[camera_name]
        self.camera = bpy.data.cameras[camera_name]
        #define the max pan/tilt rotation speed (in radians/frame)
        self.max_rotation_speed = 0.023 #for 640x640, use 0.1 #for 160x160 use 0.02 (it's affected by)
        #define the max pan/tilt acceleration (in radians^2/frame)
        self.max_rotation_acc = 24.0

        self.pan_acc = 0.0
        self.pan_vel = 0.0
        self.tilt_acc = 0.0
        self.tilt_vel = 0.0

        #define the max zoom in/out speed (in focal_length/frame)
        self.max_zoom_speed = 1
        self.min_focal_length = 15
        self.max_focal_length = 300

        self.max_pan = 150
        self.max_pan = math.radians(160)
        self.min_pan = math.radians(20)

        # self.tilt_pid = PID(0.8, 0.0003, 0.00001, setpoint=0, sample_time=1/24)
        # self.pan_pid = PID(0.8, 0.0003, 0.00001, setpoint=0, sample_time=1/24)
        # self.zoom_pid = PID(0.8, 0.0003, 0.00001, setpoint=0, sample_time=1/24)

        self.tilt_model = SecondOrderSystem(0.05, 1.0, 1.0, 0.0, 1.0)
        self.pan_model = SecondOrderSystem(0.05, 1.0, 1.0, 0.0, 1.0)
        self.zoom_model = SecondOrderSystem(0.05, 1.5, 1.0, 0.0, 1.0)

        # self.tilt_pid(0)
        # self.pan_pid(0)
        # self.zoom_pid(0)

        self.reset()
        
    def reset(self):
        #reset position of blender object
        self.camera_object.location[0] = 0.0
        self.camera_object.location[1] = 0.0
        self.camera_object.location[2] = 0.0
        #reset rotation of blender object
        self.camera_object.rotation_euler[0] = math.radians(90)
        self.camera_object.rotation_euler[1] = math.radians(0)
        self.camera_object.rotation_euler[2] = math.radians(0)
        #reset focal length of blender object
        self.camera.lens = 20

        #reset pan/tilt class vel/acc
        self.pan_acc = 0.0
        self.pan_vel = 0.0
        self.tilt_acc = 0.0
        self.tilt_vel = 0.0

        
        
    def pan(self, pan_vel_request, dt):
        """control: camera pan (equiv. yaw)
        
        Args: 
            p (-1..1): target pan position.  
        """



        # #calculate the acceleration (find abs diff between request and current vel)
        # acc_request = (pan_vel_request - self.pan_vel)/dt
        # #find the min between request and max rotation acc
        # acc = np.clip(acc_request, -self.max_rotation_acc, self.max_rotation_acc)
        # # acc = min(abs(acc_request), self.max_rotation_acc)
        # if abs(acc_request) > self.max_rotation_acc:
        #     print(f"CLIPPING ACC (pan). {acc_request=} is larger than the {self.max_rotation_acc=}")
        # #calc the velocity
        # self.pan_vel = acc * dt * self.max_rotation_speed


        # pan_vel_target = pan_vel_request * self.max_rotation_speed

        # pan_vel_actual = self.pan_pid(pan_vel_target)
        pan_vel_actual = self.pan_model.step(pan_vel_request) * self.max_rotation_speed

        #Update the camera rotation with the velocity
        self.camera_object.rotation_euler[2] += pan_vel_actual
        
        
        
    def tilt(self, tilt_vel_request, dt):
        """control: camera tilt (equiv. pitch)
        
        Args: 
            tilt_vel_request (-1..1): target tilt position.  
        """
        # acc_request = (tilt_vel_request - self.pan_vel)/dt
        # #find the min between request and max rotation acc
        # acc = np.clip(acc_request, -self.max_rotation_acc, self.max_rotation_acc)
        # # acc = min(abs(acc_request), self.max_rotation_acc)
        # if abs(acc_request) > self.max_rotation_acc:
        #     print(f"CLIPPING ACC (tilt). {acc_request=} is larger than the {self.max_rotation_acc=}")
        # self.tilt_vel = acc * dt * self.max_rotation_speed

        # self.tilt_vel = tilt_vel_request * self.max_rotation_speed
        # self.camera_object.rotation_euler[0] += self.tilt_vel
        # tilt_vel_target = tilt_vel_request * self.max_rotation_speed

        # tilt_vel_actual = self.tilt_pid(tilt_vel_target)
        tilt_vel_actual = self.tilt_model.step(tilt_vel_request) * self.max_rotation_speed

        # tilt_pos_actual = self.tilt_pid(self.tilt_pos_target)

        self.camera_object.rotation_euler[0] += tilt_vel_actual

        if self.camera_object.rotation_euler[0] > self.max_pan:
            self.camera_object.rotation_euler[0] = self.max_pan
        if self.camera_object.rotation_euler[0] < self.min_pan:
            self.camera_object.rotation_euler[0] = self.min_pan
        
        
    
    def zoom(self, zoom_request, dt): 
        """control: camera zoom
        
        Args: 
            z (-1..1): target zoom position.  
        """
        #reduce the max_rotation_speed as the camera zooms in
        #0.02 at 20 mm
        #0.002 at 200 mm
        #y=mx+c, m = -0.0001, x = zoom, c = 0.02
        #saturate it at 0.002 rad/frame
        # self.max_rotation_speed = max(-0.0001*self.camera.lens + 0.026, 0.01)
        #get rid of this bs

        # print(f"zoom: {z}")
        # zoom_target = zoom_request * self.max_zoom_speed
        # zoom_vel_actual = self.zoom_pid(zoom_target)
        zoom_vel_actual = self.zoom_model.step(zoom_request) * self.max_zoom_speed

        self.camera.lens += zoom_vel_actual
        
        #cap min focal length
        if self.camera.lens < self.min_focal_length:
            self.camera.lens = self.min_focal_length
        if self.camera.lens > self.max_focal_length:
            self.camera.lens = self.max_focal_length

     
    #Discrete actions  
    def up(self):
        self.camera_object.rotation_euler[0] += math.radians(3)
            
    def down(self):
        self.camera_object.rotation_euler[0] -= math.radians(3)
        
    def right(self):
        self.camera_object.rotation_euler[2] -= math.radians(3)
                
    def left(self):
        self.camera_object.rotation_euler[2] += math.radians(3)
        
    def zoom_in(self):
        self.camera.lens += 3

    def zoom_out(self):
        self.camera.lens -= 3