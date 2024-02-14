import colorsys
import copy
import math
import os
import sys
from pathlib import Path, PurePosixPath, WindowsPath
from typing import Any, Callable, Optional, cast

import numpy as np
import simplepbr
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.Loader import Loader
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    Camera,
    CardMaker,
    DirectionalLight,
    FrameBufferProperties,
    GeomNode,
    GraphicsOutput,
    GraphicsPipe,
    Lens,
    LoaderOptions,
    NodePath,
    PandaNode,
    PointLight,
    TextNode,
    Texture,
    WindowProperties,
    loadPrcFileData,
)

from ._data import RGBWithZbuf, ThreeDPoint, ThreeDSize, TwoDSize
from ._errors import Panda3dAssumptionViolation
from ._lighting import configure_simplepbr_lighting_internal
from ._scene_config import SceneConfigDict, load_scene_config
from ._scene_util import (
    directly_contains_geometry_data,
    find_node,
    get_obj_size,
    get_properties_copy,
    get_vertex_count,
    get_vertex_positions_copy,
    load_model_from_local_file,
)


class Panda3dShowBase(ShowBase):
    """
    A Panda3d-based renderer.

    By default it contains an empty scene with the following properties:

    * objects:
      * directional light (`light_node`)
        * color: (3, 3, 3, 1)
        * direction: (0, 0, -2.5)
    * a top-left aligned textarea on the HUD (`help_text_object`)
      * default text: ""
    * shaders: auto shaders enabled on the `render` object
    * background color: sky-like blue

    The users of this class may access and modify the scene directly, however this class provides numerous convenience methods to simplify these operations.

    Parameters
    ----------
    offscreen
        Do the rendering offscreen or to a window.
    """

    STANDARD_TARGET_MESH_PATH_EXPR = "**/TargetMesh"
    STANDARD_VIEWPOINTS_MESH_PATH_EXPR = "**/ViewpointsMesh"

    def __init__(self, offscreen: bool, win_size: TwoDSize) -> None:
        windowType = "offscreen" if offscreen else None

        prc_data = f"""win-size {win_size.x_size} {win_size.y_size}
compressed-textures 1
color-bits 8 8 8
depth-bits 24
framebuffer-float 0
"""

        loadPrcFileData("", prc_data)

        super().__init__(windowType=windowType)

        self.cam.node().getLens().setNearFar(0.01, 1000)

        self.pipeline = cast(
            simplepbr.Pipeline,
            simplepbr.init(
                enable_shadows=True,
                use_330=True,
                use_normal_maps=True,
                use_occlusion_maps=True,
                max_lights=8,
                shadow_blur=0,
                sdr_lut_factor=10,
                exposure=1,
            ),
        )

        got_fbprops = self.win.getFbProperties()
        assert _has_unsigned_byte_rgba_format(
            got_fbprops
        ), "The buffer does not have the expected format (unsigned byte RGBA color; 24 bytes depth). See the console warnings for the got format."

        # see https://github.com/Moguri/panda3d-simplepbr/issues/45
        self.pipeline._filtermgr.buffers[0].setClearColor((117 / 255, 207 / 255, 1, 1))

        self.disableMouse()

        self.setBackgroundColor(187 / 255, 229 / 255, 250 / 255)

        self.help_text_object = OnscreenText(
            text="",
            pos=(-1, 0.9),
            scale=0.09,
            parent=self.render2d,
            align=TextNode.ALeft,
        )

        depth_tex, depth_buffer, depth_cam = _configure_zbuf_camera(base=self)

        self.zbuf_tex: Texture = depth_tex
        self.depth_buffer: GraphicsOutput = depth_buffer
        self.depth_cam: NodePath = depth_cam

        cm = CardMaker("card")
        self.zbuf_view_card = self.render2d.attachNewNode(cm.generate())
        self.zbuf_view_card.setPos(
            -1, 0, -1
        )  # bring it to center, put it in front of camera
        self.zbuf_view_card.setScale(2, 1, 2)
        self.zbuf_view_card.setTexture(self.zbuf_tex)
        self.zbuf_view_card.hide()

        self.world_model: Optional[NodePath] = None
        self.expected_world_size: ThreeDSize | None = None
        """
        The expected size of the world in metres.
        """

    def set_tonemapping_exposure_for_tests(self, exposure: float) -> None:
        self.pipeline.exposure = exposure

    def get_standard_scene_format_errors(self) -> list[str]:
        target_obj = find_node(self.render, self.STANDARD_TARGET_MESH_PATH_EXPR)
        viewpoints_obj = find_node(self.render, self.STANDARD_VIEWPOINTS_MESH_PATH_EXPR)

        if target_obj is None:
            return [
                f'The target object mesh ("{self.STANDARD_TARGET_MESH_PATH_EXPR[3:]}") was not found.'
            ]

        if viewpoints_obj is None:
            return [
                f'The viewpoints object mesh ("{self.STANDARD_VIEWPOINTS_MESH_PATH_EXPR[3:]}") was not found.'
            ]

        if self.world_model is None:
            return ["There is no loaded world."]
        if self.expected_world_size is None:
            return ["There is no specified expected world size."]

        return (
            _get_target_mesh_errors(target_obj)
            + _get_cam_viewpoint_errors(viewpoints_obj)
            + _get_world_size_errors(self.expected_world_size, self.world_model)
        )

    def get_target_obj_mesh_path(self) -> NodePath:
        obj = find_node(self.render, self.STANDARD_TARGET_MESH_PATH_EXPR)
        assert obj is not None
        return obj

    def get_viewpoints_obj_mesh_path(self) -> NodePath:
        obj = find_node(self.render, self.STANDARD_VIEWPOINTS_MESH_PATH_EXPR)
        assert obj is not None
        return obj

    def show_depth(self, show: bool) -> None:
        if show:
            self.zbuf_view_card.show()
        else:
            self.zbuf_view_card.hide()

    def set_help_text(self, new_text: str) -> None:
        """
        Set the text of `help_text_object`.

        Parameters
        ----------
        new_text
            The new text.
        """
        self.help_text_object.setText(new_text)

    def load_world_from_blender(
        self, scene_path: Path, scene_config_override: Optional[SceneConfigDict] = None
    ) -> list[str]:
        """
        Load a 3d object from the specified path.

        The function assumes that the model is created using Blender.

        The function preserves the internal object hierarchy.

        Parameters
        ----------
        scene_path
            The path of the scene to load. Unlike in Panda3d, this path is not case-sensitive under Windows.
        scene_config_override
            TBD

        Raises
        ------
        Exception
            If the world is not loadable for any reason.
        """

        assert self.world_model is None

        if scene_config_override is None:
            scene_config_or_errors = load_scene_config(scene_path.with_suffix(".json"))
            if isinstance(scene_config_or_errors, list):
                return scene_config_or_errors
        else:
            scene_config_or_errors = scene_config_override

        self.world_model = load_model_from_local_file(self, scene_path)
        assert self.world_model is not None, "Failed to load the world model."

        self.world_model.set_two_sided(False)
        self.world_model.reparentTo(self.render)

        configure_simplepbr_lighting_internal(
            render=self.render,
            model_root=self.world_model,
            scene_config=scene_config_or_errors,
        )

        # self.pipeline.shadow_blur = scene_config["shadow_blur"]
        self.world_model.set_shader_input(
            "shadow_blur", scene_config_or_errors["shadow_blur"]
        )

        target_obj = find_node(self.world_model, self.STANDARD_TARGET_MESH_PATH_EXPR)
        if target_obj is not None:
            target_obj.set_shader_input("force_shadow", 1)

        viewpoints_obj = find_node(
            self.world_model, self.STANDARD_VIEWPOINTS_MESH_PATH_EXPR
        )
        if viewpoints_obj is not None:
            viewpoints_obj.hide()

        for force_shadow_0_name in scene_config_or_errors["force_shadow_0"]:
            force_shadow_0_obj = find_node(self.render, "**/" + force_shadow_0_name)
            if force_shadow_0_obj is None:
                return [f'The PandaNode "{force_shadow_0_name}" was not found.']
            force_shadow_0_obj.set_shader_input("force_shadow", 0)

        self.expected_world_size = ThreeDSize(
            x_size=scene_config_or_errors["world_size_x"],
            y_size=scene_config_or_errors["world_size_y"],
            z_size=scene_config_or_errors["world_size_z"],
        )

        return []

    def save_world_bam(self, target_path: Path) -> None:
        """
        Save the current world to a ``.bam`` file. This function does not save the camera positions and the light sources that do not belong to the world.

        The function fails if the world does not exist.

        Parameters
        ----------
        target_path
            The path of the target ``.bam`` file.
        -----
        """
        assert self.world_model is not None, "There is no loaded world."
        assert target_path.name.endswith(
            ".bam"
        ), 'The path does not point to a ".bam" file.'

        self.world_model.writeBamFile(target_path)

    def render_single_RGBB_frame(self) -> RGBWithZbuf:
        """
        Render a single new frame to a numpy array.

        This function makes sure that the returned numpy array contains
        the rendered frame (and not the previous one).

        Returns
        -------
        v
            The captured image with the Z-buffer.

        Notes
        -----
        This function instructs Panda3d to render only one (or two for technical
        reasons) frame. This means that if something has 1-2 frame delay, then that
        may be broken in the rendered frame.
        """
        self._render_frame_sync()
        rgb_image = self._capture_last_RGB()
        zbuf_image = self._capture_last_zbuffer()

        return RGBWithZbuf(rgbs=rgb_image, zbufs=zbuf_image)

    def _render_frame_sync(self) -> None:
        """
        Instruct Panda3d to render a single frame.

        This function blocks until the rendering is actually done
        (`panda3d.core.GraphicsEngine.renderFrame` is asynchronous).
        -----
        """
        # initiate rendering
        self.graphicsEngine.renderFrame()
        # another rendering (Panda3d commonly calls
        # this function twice)
        # see https://discourse.panda3d.org/t/forcing-render-to-update/5750/3
        self.graphicsEngine.renderFrame()
        # wait for the render to complete
        self.graphicsEngine.syncFrame()

    def _capture_last_RGB(self) -> np.ndarray:
        """
        Get the last rendered RGB image as a Numpy array.

        It assumes without checking that the frame buffer
        uses unsigned bytes to represent colors.

        Returns
        -------
        v
            Format: ``Im::RGBs[Single]``
        """
        # This function is somewhat hacky, since the frame buffer properties
        # expected by this function are hardware-dependent.
        tex: Texture = self.win.getScreenshot()
        fmt = tex.getFormat()
        data = tex.getRamImage()
        v = memoryview(data).cast("B", (tex.getYSize(), tex.getXSize(), 4))  # type: ignore
        img = np.array(v).copy()
        img = img[::-1, :, [2, 1, 0]] / 255.0
        return img.transpose([2, 0, 1]).reshape((1, 3, tex.getYSize(), tex.getXSize()))

    def _capture_last_zbuffer(self) -> np.ndarray:
        """
        Get the last Z-buffer data as a Numpy array.

        Returns
        -------
        v
            Format: ``Im::ZBuffers[Single]``
        """
        data = self.zbuf_tex.getRamImage()
        ysize = self.zbuf_tex.getYSize()
        xsize = self.zbuf_tex.getXSize()
        buffer_raw = memoryview(data)  # type: ignore
        buffer_casted = buffer_raw.cast("f", (ysize, xsize))
        img = np.array(buffer_casted).reshape(1, 1, ysize, xsize)
        img = img[:, :, ::-1].copy()
        return img

    def set_cam_pos_free(
        self,
        point: ThreeDPoint,
        fluid: bool = False,
        local_coordinates: bool = False,
    ) -> None:
        """
        Freely set the camera to the (x, y, z) point in the specified coordinate system.

        Parameters
        ----------
        point
            The new position of the camera.
        fluid
            The function to use to change the position. If true, then `panda3d.core.NodePath.setFluidPos` will be used, otherwise `panda3d.core.NodePath.setPos`.
        local_coordinates
            Use the local coordinate system of the object or the global coordinate system to calculate the new position of the object.

        Returns
        -------

        See Also
        --------
        set_cam_hpr_free : Freely set the camera rotation.
        set_cam_around : Set the position and the rotation of the camera using (distance, azimuth, elevation) data.
        """
        cam = cast(NodePath, self.cam)

        if fluid:
            if local_coordinates:
                cam.setFluidPos(cam, point.x, point.y, point.z)
            else:
                cam.setFluidPos(point.x, point.y, point.z)
        else:
            if local_coordinates:
                cam.setPos(cam, point.x, point.y, point.z)
            else:
                cam.setPos(point.x, point.y, point.z)

    def get_cam_xyz(self) -> ThreeDPoint:
        """
        Get the camera position in the global coordinate system.

        Returns
        -------
        v
            The camera position.
        """
        cam = cast(NodePath, self.cam)
        pos = cam.getPos()
        return ThreeDPoint(pos.x, pos.y, pos.z)

    def show_mouse(self, show: bool) -> None:
        """
        Show or hide the mouse.

        Parameters
        ----------
        show
            If true, then the mouse will be visible. Otherwise, the function will hide the mouse.
        """
        props = WindowProperties()
        props.setCursorHidden(not show)
        cast(Any, self.win).requestProperties(props)

    def set_cam_hpr_free(
        self, h: float, p: float, r: float, local_coordinates: bool = False
    ) -> None:
        """
        Set the camera rotation in the specified coordinate system.

        Parameters
        ----------
        h
            The head parameter in degree.
        p
            The pitch parameter in degree.
        r
            The roll parameter in degree.
        local_coordinates
            If true, then the rotation will be calculated in the local coordinate system of the camera. Otherwise the function uses the global coordinate system.
        """
        cam = cast(NodePath, self.cam)
        if local_coordinates:
            curr_hpr = cam.getHpr()

            h += curr_hpr.x
            p += curr_hpr.y
            r += curr_hpr.z

        cam.setHpr(h, p, r)

    def set_cam_pos_and_look_at(
        self,
        new_cam_pos: ThreeDPoint,
        look_at: ThreeDPoint,
    ) -> None:
        r"""
        Set the camera position to the specified point and make the camera to point at the specified object.

        Returns
        -------
        new_cam_pos
            The new camera position in (x, y, z) format.
        object
            The point to look at in (x, y, z) format.
        """

        cam = cast(NodePath, self.cam)

        cam.setPos(new_cam_pos.x, new_cam_pos.y, new_cam_pos.z)
        cam.lookAt(look_at.x, look_at.y, look_at.z)

    def get_cam_lens(self) -> Lens:
        return cast(Camera, cast(NodePath, self.cam).node()).getLens()


def _has_unsigned_byte_rgba_format(got_buffer_props: FrameBufferProperties) -> bool:
    r_bits = got_buffer_props.getRedBits()
    g_bits = got_buffer_props.getGreenBits()
    b_bits = got_buffer_props.getBlueBits()
    depth_bits = got_buffer_props.getDepthBits()
    color_bits = got_buffer_props.getColorBits()
    float_color = got_buffer_props.getFloatColor()

    return (
        (r_bits == 8)
        and (g_bits == 8)
        and (b_bits == 8)
        and (depth_bits == 24)
        and (color_bits == 24)
        and (float_color == False)
    )


def _configure_zbuf_camera(base: ShowBase) -> tuple[Texture, GraphicsOutput, NodePath]:
    """
    Configure a texture that contains the Z-buffer for the specified `direct.showbase.ShowBase.ShowBase`.

    Parameters
    ----------
    base
        The `direct.showbase.ShowBase.ShowBase` to use.

    Returns
    -------
    texture
        The texture, to which the up to date Z-buffer is rendered.
    buffer
        A new buffer that has a single role: capture the depth map.
    cam_nodepath
        The node path of the newly added camera to capture the depth data.
    """
    # from <https://gist.github.com/alexlee-gk/b28fb962c9b2da586d1591bac8888f1f>
    # and <https://docs.panda3d.org/1.10/python/programming/render-to-texture/low-level-render-to-texture#the-advanced-api>

    win: GraphicsOutput = base.win

    # Create a new buffer to store the depth.
    # This is required, since the window is already
    # written by the main camera.
    # The depth buffer also contains color data too.
    # Only the texture will select the depth data.
    # The reason that we do not use the depth data of
    # the window is that the window buffer might be
    # modified by a lot of different systems in Panda3d.
    # Request 8 RGB bits, no alpha bits, and a depth buffer.
    fb_prop = FrameBufferProperties()
    fb_prop.setRgbColor(True)
    fb_prop.setRgbaBits(8, 8, 8, 0)
    fb_prop.setDepthBits(32)
    # Don't open a window - force it to be an offscreen buffer.
    flags = GraphicsPipe.BF_refuse_window
    depth_buffer = base.graphicsEngine.make_output(
        base.pipe,
        "Depth buffer",
        -1000,
        fb_prop,
        get_properties_copy(win),
        flags,
        win.getGsg(),
        base.win,
    )

    # Create texture to access the rendered depth.
    depthTex = Texture()
    depthTex.setFormat(Texture.FDepthComponent32)
    depth_buffer.addRenderTexture(
        depthTex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth
    )
    lens = cast("NodePath[Camera]", base.cam).node().getLens()

    # Create a depth camera.
    # This step is not necessary, since we could probably just use the
    # existing camera, however this camera is completely controlled by us.
    # The camera has a huge amount of configuration options that may
    # affect the depth data.
    depth_cam = base.makeCamera(
        depth_buffer, lens=lens, scene=base.render, camName="depth_cam"
    )
    depth_cam.reparentTo(base.cam)

    return depthTex, depth_buffer, depth_cam


def _get_cam_viewpoint_errors(viewp_obj_candidate: NodePath) -> list[str]:
    if not directly_contains_geometry_data(viewp_obj_candidate):
        return [
            "The viewpoints object does not have any directly associated geometry data."
        ]

    if not _has_at_least_one_vertex(viewp_obj_candidate):
        return [
            "The viewpoints object has directly associated geometry data, but it not have any vertex."
        ]

    curr_obj = viewp_obj_candidate
    while True:
        node_path_errors: list[str] = []

        if not _has_no_rotation(curr_obj):
            node_path_errors.append(
                "The viewpoint object or one of its ancestors has non-zero rotation."
            )

        if not _has_one_scale(curr_obj):
            node_path_errors.append(
                "The viewpoint object or one of its ancestors has non-one scale."
            )

        if len(node_path_errors) > 0:
            return node_path_errors

        if curr_obj.hasParent():
            curr_obj = curr_obj.getParent()
        else:
            return []


def _get_world_size_errors(
    expected_world_size: ThreeDSize, world_obj: NodePath
) -> list[str]:
    actual_world_size = None
    try:
        actual_world_size = get_obj_size(world_obj)
    except:
        return ["The world does not contain any vertex."]

    errors: list[str] = []

    actual_world_size = get_obj_size(world_obj)
    if abs(actual_world_size.x_size - expected_world_size.x_size) > 1e-1:
        errors.append(
            f"The specified size of the world ({expected_world_size.x_size}) along the X axis does not match to its actual size ({actual_world_size.x_size}) within 4 places."
        )
    if abs(actual_world_size.y_size - expected_world_size.y_size) > 1e-1:
        errors.append(
            f"The specified size of the world ({expected_world_size.y_size}) along the Y axis does not match to its actual size ({actual_world_size.y_size}) within 4 places."
        )
    if abs(actual_world_size.z_size - expected_world_size.z_size) > 1e-1:
        errors.append(
            f"The specified size of the world ({expected_world_size.z_size}) along the Z axis does not match to its actual size ({actual_world_size.z_size}) within 4 places."
        )

    return errors


def _get_target_mesh_errors(target_obj_mesh: NodePath) -> list[str]:
    errors: list[str] = []

    if not directly_contains_geometry_data(target_obj_mesh):
        errors.append(
            "The target object does not have any directly associated geometry data."
        )
        return errors

    if not _has_at_least_one_vertex(target_obj_mesh):
        errors.append(
            "The target object has directly associated geometry data, but it does not have any vertex."
        )

    if not _is_at_origin(target_obj_mesh):
        errors.append("The mesh of the target object has non-zero translation.")

    if not _has_no_rotation(target_obj_mesh):
        errors.append("The mesh of the target object has non-zero rotation.")

    if not _has_one_scale(target_obj_mesh):
        errors.append("The mesh of the target object has non-one scale.")

    if not target_obj_mesh.hasParent():
        errors.append("The mesh of the target object does not have any parent.")
        return errors

    target_obj = target_obj_mesh.getParent()

    if not _has_no_rotation(target_obj):
        errors.append("The target object has non-zero rotation.")

    if not _has_one_scale(target_obj):
        errors.append("The target object has non-one scale.")

    if not target_obj.hasParent():
        errors.append("The target object does not have any parent.")
        return errors

    last_obj = target_obj
    while True:
        curr_obj = last_obj.getParent()

        if not _has_no_rotation(curr_obj):
            errors.append(
                "One of the ancestors of the target object has non-zero rotation."
            )

        if not _has_one_scale(curr_obj):
            errors.append(
                "One of the ancestors of the target object has non-one scale."
            )

        if not _is_at_origin(curr_obj):
            errors.append(
                "One of the ancestors of the target object is not at the origin."
            )

        if len(errors) > 0:
            return errors

        if curr_obj.hasParent():
            last_obj = curr_obj
        else:
            return []


def _is_at_origin(obj: NodePath) -> bool:
    pos_x, pos_y, pos_z = obj.getPos()
    return pos_x == 0.0 and pos_y == 0.0 and pos_z == 0.0


def _has_no_rotation(obj: NodePath) -> bool:
    rot_h, rot_p, rot_r = obj.getHpr()
    return rot_h == 0.0 and rot_p == 0.0 and rot_r == 0.0


def _has_one_scale(obj: NodePath) -> bool:
    scale_x, scale_y, scale_z = obj.getScale()
    return scale_x == 1.0 and scale_y == 1.0 and scale_z == 1.0


def _has_at_least_one_vertex(obj: NodePath) -> bool:
    return get_vertex_count(obj) > 0
