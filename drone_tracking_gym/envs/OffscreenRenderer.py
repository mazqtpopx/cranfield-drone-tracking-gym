import bpy
import gpu
import bgl

from OpenGL.GL import glGetTexImage

import numpy as np

def find_first_view3d():
    """Helper function to find first space view 3d and associated window region.
    The three returned objects are useful for setting up offscreen rendering in
    Blender.
    Returns
    -------
    area: object
        Area associated with space view.
    window: object
        Window region associated with space view.
    space: bpy.types.SpaceView3D
        Space view.
    """
    areas = [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"]
    assert len(areas) > 0
    area = areas[0]
    region = sorted(
        [r for r in area.regions if r.type == "WINDOW"],
        key=lambda x: x.width,
        reverse=True,
    )[0]
    spaces = [s for s in areas[0].spaces if s.type == "VIEW_3D"]
    assert len(spaces) > 0
    return area, spaces[0], region



#Modified from https://github.com/cheind/pytorch-blender/blob/develop/pkg_blender/blendtorch/btb/offscreen.py
#NB: this is the 'default_renderer' as called in the 'attach_default_renderer' function
#namely used in the cartpole example
class OffScreenRenderer:
    def __init__(self, camera_name, mode="rgba", origin="upper-left"):
        # print("Hello offscreen")
        assert mode in ["rgba", "rgb"]
        assert origin in ["upper-left", "lower-left"]
#        self.camera = camera or Camera()
        self.camera = bpy.context.scene.camera
        self.offscreen = gpu.types.GPUOffScreen(self.shape[1], self.shape[0])
        self.area, self.space, self.region = find_first_view3d()
        self.handle = None
        self.origin = origin
        self.channels = 4 if mode == "rgba" else 3
        self.buffer = np.zeros((self.channels, self.shape[0], self.shape[1]), dtype=np.uint8, order='C')
        self.mode = bgl.GL_RGBA if mode == "rgba" else bgl.GL_RGB
        self.current_img = self.buffer
        
        
    @property
    def shape(self):
        return [bpy.data.scenes["Scene"].render.resolution_x, bpy.data.scenes["Scene"].render.resolution_y]

    def render(self):
        return self.current_img

    def render_buffer(self):
        """Render the scene and return image as buffer.
        Returns
        -------
        image: HxWxD array
            where D is 4 when `mode=='RGBA'` else 3.
        """
        # self.buffer = None
        with self.offscreen.bind():
            proj_matrix = self.camera.calc_matrix_camera(
                bpy.context.evaluated_depsgraph_get(), x=bpy.data.scenes["Scene"].render.resolution_x, y=bpy.data.scenes["Scene"].render.resolution_y
            )
            self.offscreen.draw_view3d(
                bpy.context.scene,
                bpy.context.view_layer,
                self.space,  # bpy.context.space_data
                self.region,  # bpy.context.region
                self.camera.matrix_world.normalized().inverted(),
                proj_matrix,
            )

            bgl.glActiveTexture(bgl.GL_TEXTURE0)
            bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.offscreen.color_texture)

            #self.offscreen.draw_callback_px

            # np.asarray seems slow, because bgl.buffer does not support the python buffer protocol
            # bgl.glGetTexImage(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGB, bgl.GL_UNSIGNED_BYTE, self.buffer)
            # https://docs.blender.org/api/blender2.8/gpu.html
            # That's why we use PyOpenGL at this point instead.
            #On buffer protocol:
            #https://blender.stackexchange.com/questions/176548/how-do-you-use-the-gpu-offscreen-render-api-in-2-8
            #(check comments from Mr epic fail)
            #https://blender.stackexchange.com/questions/221110/fastest-way-copying-from-bgl-buffer-to-numpy-array
            
            glGetTexImage(
                bgl.GL_TEXTURE_2D, 0, self.mode, bgl.GL_UNSIGNED_BYTE, self.buffer
            )

        buffer = self.buffer
        if self.origin == "upper-left":
            buffer = np.flipud(buffer)
            
        #print(f"{buffer.flags=}")
        return buffer
    
    #returns an np array instead of the buffer
    def render_nparray(self):
        """Render the scene and return image as buffer.
        Returns
        -------
        image: HxWxD array
            where D is 4 when `mode=='RGBA'` else 3.
        """
        with self.offscreen.bind():
            proj_matrix = self.camera.calc_matrix_camera(
                bpy.context.evaluated_depsgraph_get(), x=bpy.data.scenes["Scene"].render.resolution_x, y=bpy.data.scenes["Scene"].render.resolution_y
            )
            self.offscreen.draw_view3d(
                bpy.context.scene,
                bpy.context.view_layer,
                self.space,  # bpy.context.space_data
                self.region,  # bpy.context.region
                self.camera.matrix_world.normalized().inverted(),
                proj_matrix,
            )
            
            ndarray = np.zeros((self.shape[0], self.shape[1], self.channels), dtype=np.uint8, order='C')

            bgl.glActiveTexture(bgl.GL_TEXTURE0)
            bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.offscreen.color_texture)
            
            buffer = bgl.Buffer(bgl.GL_BYTE, ndarray.shape, ndarray)

            mode = bgl.GL_RGBA
            if self.channels == 3:
                mode = bgl.GL_RGB

            bgl.glGetTexImage(bgl.GL_TEXTURE_2D, 0, mode, bgl.GL_UNSIGNED_BYTE, buffer)
            bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)
            flipped_ndarray = ndarray[::-1, :, :]


        # print(f"{flipped_ndarray=}")
        self.current_img = flipped_ndarray
        #print(f"{flipped_ndarray.flags=}")
        return flipped_ndarray

    def set_render_style(self, shading="RENDERED", overlays=False):
        self.space.shading.type = shading
        self.space.overlay.show_overlays = overlays

#Camera view_from_bpy:
#return camera.matrix_world.normalized().inverted()
#shape = shape or Camera.shape_from_bpy()
#return camera.calc_matrix_camera(
#    bpy.context.evaluated_depsgraph_get(), x=shape[1], y=shape[0]
#)