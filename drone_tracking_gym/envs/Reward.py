import bpy

import numpy as np
import math
import cv2

#Largely based on the pytorch-blender utils an Camera code (add link)
class RewardCalculator():
    def __init__(self, drone_name):
        self.sph = bpy.data.objects[drone_name]
        self.cnt = 0

        # self.reward_map = self._generate_reward_map()
            
    def dehom(self, x):
        """Return de-homogeneous coordinates by perspective division."""
        return x[..., :-1] / x[..., -1:]

    def hom(self, x, v=1.0):
        """Convert to homogeneous coordinates in the last dimension."""
        return np.concatenate((x, np.full((x.shape[0], 1), v, dtype=x.dtype)), -1)


    def ndc_to_pixel(self, ndc, origin="upper-left"):
        """Converts NDC coordinates to pixel values
        Params
        ------
        ndc: Nx3 array
            Normalized device coordinates.
        origin: str
            Pixel coordinate orgin. Supported values are `upper-left` (OpenCV) and `lower-left` (OpenGL)
        Returns
        -------
        xy: Nx2 array
            Camera pixel coordinates
        """
        assert origin in ["upper-left", "lower-left"]
        w = bpy.data.scenes["Scene"].render.resolution_x 
        h = bpy.data.scenes["Scene"].render.resolution_y
        xy = np.atleast_2d(ndc)[:, :2]
        xy = (xy + 1) * 0.5
        if origin == "upper-left":
            xy[:, 1] = 1.0 - xy[:, 1]
        return xy * np.array([[w, h]])

    def is_object_in_front_of_camera(self):
        LOCAL_DEBUG = False
        #assume camera XY pos is (0,0)
        cam = bpy.context.scene.camera
        z_angle = bpy.data.objects["Camera"].rotation_euler[2]
        x_sph = self.sph.location[0]
        y_sph = self.sph.location[1]

        # print(f"{z_angle=}{x_sph=}{y_sph=}")

        #convert theta 
        theta = math.degrees(z_angle)%360

        if LOCAL_DEBUG:
            print(f"{z_angle=}{theta=}{y_sph=}{x_sph=}{(x_sph * math.tan(z_angle))=}")

        if 0 < theta < 90:
            if y_sph > x_sph * math.tan(z_angle): 
                return True
        if 90 < theta < 180:
            if y_sph < x_sph * math.tan(z_angle): 
                return True
        if 180 < theta < 270:
            if y_sph < x_sph * math.tan(z_angle): 
                return True
        if 270 < theta < 360:
            if y_sph > x_sph * math.tan(z_angle): 
                return True
        else:
            return False
        #if else, object is behind the camera!


    def get_sphere_contours(self):
        img_size = (bpy.data.scenes["Scene"].render.resolution_x, bpy.data.scenes["Scene"].render.resolution_y)
        proj_mat = bpy.context.scene.camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=img_size[0], y=img_size[1])
        view_mat = bpy.context.scene.camera.matrix_world.normalized().inverted()

        #get the XYZ coords of the sphere
        dg = bpy.context.evaluated_depsgraph_get()
        xyz = []
        eval_obj = self.sph.evaluated_get(dg)
        #This one is for bbox - we want all of the pixels for better accuracy
        #xyz_obj = [(eval_obj.matrix_world @ Vector(c)) for c in eval_obj.bound_box]
        xyz_obj = [(eval_obj.matrix_world @ v.co) for v in eval_obj.data.vertices]
        #print(f"{xyz_obj=}")
        xyz.extend(xyz_obj)

        #xyz coords
        xyz = np.stack(xyz)

        #convert to ndc 
        xyz = np.atleast_2d(xyz)
        xyzw = self.hom(xyz, 1.0)
        m = np.asarray(proj_mat @ view_mat)
        ndc = self.dehom(xyzw @ m.T)

        pixels = self.ndc_to_pixel(ndc)
        
        cnts = ObjectContours(pixels, img_size, )
        return cnts

    def calc_area_occupied_by_sphere_gradient_map(self):
        cnts = self.get_sphere_contours()

        #FIRST WE WANT TO DO A CHECK IF SPHERE IS IN FRONT
        #OR BEHIND THE CAMERA. ONLY SHOW IF THE SPHERE (or other object)
        #IS IN FRONT OF THE CAMERA (otherwise we get a glitch
        #where we can see both the sphere and its '180 reflection' 
        #when looking directly in the opposite direction of the sphere)

        #only calculate if object is in front of the camera!!!
        #This is based on calculation:
        #if y_object > x_object*tan(cam_z_angle)
        #object is in front of camera, else it is not
        if self.is_object_in_front_of_camera():
            # print("object in front of camera!!!!")
            reward = cnts.calculate_area_occupied_circular_gradient()


        else: 
            # print("object behind of camera!!!!")
            reward = 0.0
        return reward


    #Returns a percentage (between 0 and 1) of the area occupied by the sphere or tagert obj) in the scene
    def calc_area_occupied_by_sphere(self):
        #getting rid of this debug flag - as we are putting a seperate funct that returns the mask
        #image instead
        LOCAL_DEBUG = False

        cnts = self.get_sphere_contours()

        #FIRST WE WANT TO DO A CHECK IF SPHERE IS IN FRONT
        #OR BEHIND THE CAMERA. ONLY SHOW IF THE SPHERE (or other object)
        #IS IN FRONT OF THE CAMERA (otherwise we get a glitch
        #where we can see both the sphere and its '180 reflection' 
        #when looking directly in the opposite direction of the sphere)

        #only calculate if object is in front of the camera!!!
        #This is based on calculation:
        #if y_object > x_object*tan(cam_z_angle)
        #object is in front of camera, else it is not
        if self.is_object_in_front_of_camera():
            # print("object in front of camera!!!!")
            object_area_percentage = cnts.calculate_area_occupied()
            #getting rid of this debug flag - as we are putting a seperate funct that returns the mask
            #image instead
            if LOCAL_DEBUG:
                debug_img = self.get_mask_image()
                cv2.imshow("REWARD DEBUG", debug_img)
                self.cnt += 1
                cv2.imwrite("E:/Repos/PTZ_drone_tracking/debug_imgs/mask"+str(self.cnt)+".png", debug_img)
                cv2.waitKey(1)
        else: 
            # print("object behind of camera!!!!")
            object_area_percentage = 0.0
        return object_area_percentage
    
    def get_mask_image(self):
        cnts = self.get_sphere_contours()
        mask = cnts.draw_convex_hull()
        return mask
    



#We should move the image to be defined at __init__ stage 
#but, we only want to define the image size at __init__
#image_size input:
#   (w,h) 
class ObjectContours():
    def __init__(self, pixels, image_size):
        self.pixels = pixels
        self.image_size = image_size
        self.reward_map = self.__generate_reward_map()
        
    #Goes through the pixel contours and draws the outline of the object
    #on the given image
    def draw_on_image(self, image):
        cv2.drawContours(image, [self.pixels.astype(int)], -1, color=(255, 255, 255), thickness=cv2.FILLED)
        cv2.imshow("test", image)
        cv2.waitkey()

    #Uses scipy to draw the convex hull - this is an old method 
    #and just for reference. It was replaced with draw_convex_hull
    #which uses opencv and is easier to follow
    #Im leaving it in just incase I need to use the scipy method 
    #in the future
    def draw_convex_hull_scipy(self,image):
        for px in self.pixels:
            #print(f"{px=}")
            cv2.circle(image, (int(px[0]), int(px[1])), 1, (0,0,255), 1)
        
        from scipy.spatial import ConvexHull, convex_hull_plot_2d
        hull = ConvexHull(self.pixels)
        #print(f"{hull=}")
        #print(f"{hull.vertices=}")
        
        hull_points = np.zeros([len(self.pixels), 2])
        #print(f"{hull_points=}")
        i = 0

        for simplex in hull.simplices:
            # print(f"{simplex=}")
            p = (self.pixels[simplex[0],0], self.pixels[simplex[1],1])
            # print(f"{p=}")
            # print(f"{self.pixels[simplex,0].astype(int)=}")
            cv2.line(image, (self.pixels[simplex[0],0].astype(int), self.pixels[simplex[0],1].astype(int)), (self.pixels[simplex[1],0].astype(int), self.pixels[simplex[1],1].astype(int)), (255,255,255), 2)
        
        # print(f"{self.pixels[241,0]=}")
        # print(f"{self.pixels[241,1]=}")
            
        cv2.imshow("test", image)
        cv2.waitkey()

        
    #draws convex hull around the pixels. 
    def draw_convex_hull(self):
        hull = self.__get_convex_hull()
        w = bpy.data.scenes["Scene"].render.resolution_x
        h = bpy.data.scenes["Scene"].render.resolution_y
        output_img = np.zeros((w, h, 3), dtype=np.uint8)
        #if hull contains points, we can draw it on the image
        if hull.shape[0] > 0:
            cv2.drawContours(output_img, [hull], -1, color=(255, 255, 255), thickness=cv2.FILLED)
        #cv2.imshow("test", image)
        #cv2.waitKey()
        # output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        return output_img
        
        #draws convex hull around the pixels. 
    def draw_convex_hull_reward(self):
        hull = self.__get_convex_hull()
        w = bpy.data.scenes["Scene"].render.resolution_x
        h = bpy.data.scenes["Scene"].render.resolution_y
        output_img = np.zeros((w, h, 3), dtype=np.float32)
        #if hull contains points, we can draw it on the image
        if hull.shape[0] > 0:
            cv2.drawContours(output_img, [hull], -1, color=(1.0, 1.0, 1.0), thickness=cv2.FILLED)
        #cv2.imshow("test", image)
        #cv2.waitKey()
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        # print(f"{output_img.shape=}")

        return output_img
        

    #calculates the area occupied by the pixel 
    def calculate_area_occupied(self):
        LOCAL_DEBUG = False
        # DRAW_IMAGE = False
        hull = self.__get_convex_hull()
        
        area = cv2.contourArea(hull)
        
        if LOCAL_DEBUG:
            print(f"{area=}")
        
        #return the area of the object
        return area
        

    #calculates the area occupied by the hull, multipled by the gradient reward map
    def calculate_area_occupied_circular_gradient(self):
        LOCAL_DEBUG = True
        hull_img = self.draw_convex_hull_reward()
        reward_array = self.reward_map * hull_img
        reward = np.sum(reward_array)


        if LOCAL_DEBUG:
            # debug_img = self.get_mask_image()
            cv2.imshow("HULL DEBUG", hull_img)
            cv2.waitKey(1)
            cv2.imshow("REWARD MAP", self.reward_map)
            cv2.waitKey(1)
            cv2.imshow("REWARD DEBUG", reward_array)
            # self.cnt += 1
            # cv2.imwrite("E:/Repos/PTZ_drone_tracking/debug_imgs/mask"+str(self.cnt)+".png", debug_img)
            cv2.waitKey(1)
        
        return reward

    
    #Generates a circular gradient - an illustration can be found in reward_map_illustration.py
    def __generate_reward_map(self):
        arr = np.zeros((self.image_size[0],self.image_size[1],1), dtype=np.float32)
        x_axis = np.linspace(1.0, -1.0, self.image_size[0])[:, None]
        y_axis = np.linspace(1.0, -1.0, self.image_size[1])[None, :]

        arr = np.sqrt(x_axis ** 2 + y_axis ** 2)

        # Normalize the gradient to the range of 0 to 1
        arr = (arr - arr.min()) / (arr.max() - arr.min())

        # Adjust the gradient so that the center is 1.0 and the edges are 0.0
        arr = 1 - arr

        return arr





        
    #Gets convext hull (outline) of the pixel points.
    #Points which fall outside boundaries of the image
    #are deleted
    def __get_convex_hull(self):
        LOCAL_DEBUG = False
        hull = cv2.convexHull(self.pixels.astype(int))
        if LOCAL_DEBUG:
            print(f"{hull=}")
            print(f"{hull.shape=}")
        
        #check if point exitst in the bounds of the image
        #we should be able to parallelize this
        #find out the indexes of the points we want to delete
        to_delete = []
        i = 0
        for point in hull:
            if LOCAL_DEBUG:
                print(f"{point=}")
            if not self.__does_point_exists_within_image(point[0]):
                to_delete.append(i)
            i += 1

        #delete the points based on their indexes
        hull = np.delete(hull, to_delete, 0)
        if LOCAL_DEBUG:
            print(f"{hull=}")
        
        return hull
        
    #Caluclates if a point falls within boundaries of the image
    def __does_point_exists_within_image(self, point):
        # print(f"{point=}")
        x = point[0]
        y = point[1]
        
        xmin = 0
        ymin = 0
        xmax = self.image_size[0]
        ymax = self.image_size[1]
        
        inside = False
        if (x >= xmin and x <= xmax):
            if (y >= ymin and y <= ymax):
                inside = True
        
        return inside