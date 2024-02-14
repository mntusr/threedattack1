import datetime
import math
import os
import sys
import webbrowser
from dataclasses import dataclass
from email.mime import base
from pathlib import Path
from typing import Any, cast

import numpy as np
from direct.showbase.InputStateGlobal import inputState
from direct.showbase.ShowBase import taskMgr  # type: ignore
from direct.task.Task import TaskManager
from panda3d.core import (
    GraphicsOutput,
    GraphicsWindow,
    MouseWatcher,
    NodePath,
    PerspectiveLens,
    PStatClient,
)

from ._custom_show_base import Panda3dShowBase
from ._data import ThreeDPoint, TwoDSize
from ._point_cloud import CamProjSpec, show_depths_as_point_clouds_in_browser
from ._scene_util import get_near_far_planes_safe, zbuf_2_depth_and_mask
from ._viewpoints import SampleType, ViewpointBasedCamController, ViewpointSplit

taskMgr: TaskManager = cast(TaskManager, taskMgr)
from ._scene_util import get_cam_proj_spec_for_showbase
from ._viewpoints import get_viewpoints_path_for_world


class LivePreviewController:
    """
    This class adds interactive features to the specified `Panda3dShowBase`.

    This is useful for diagnosing a scene.

    Main features:

    * Interactive free camera mode.
    * Interactive (distance, azimuth, elevation) camera around (0, 0, 0).
    * Show the current coordinates during preview.
    * Show a help text to describe the controls.

    The constructor of this class is not intended for public use. The instances of the class should be created using the `attach` method instead.
    """

    FREE_CAM_HELP_TEXT = """Move camera: WASD
Connect pstats: M
Save depth to a Numpy array: N
Rotate camera: Mouse
Switch to constrained cam.: Tab
View depth as point cloud: C
Toggle rendering info: R
Toggle position info: P
Point cloud depth limit: {0}
Toggle raw (non-limited) zbuffer: Z
Quit: Escape{1}{2}
"""
    FREE_CAM_POS_INFO_TEXT = "(x={0:2.2f}, y={1:2.2f}, z={2:2.2f})"

    CONSTRAINED_CAM_HELP_TEXT = """Set viewp. idx.: {0}
Connect pstats: M
Save depth to a Numpy array: N
Switch to free cam.: Tab
View depth as point cloud: C
Toggle rendering info: R
Toggle position info: P
Point cloud depth limit: {1}
Toggle raw (non-limited) zbuffer: Z
Quit: Escape{2}{3}
"""
    SET_VIEW_INDEX_EMPTY_TEXT = "<type then ENTER> or AD"
    CONSTRAINED_CAM_POS_INFO_TEXT = "{0}/{1}\n(x={2:2.2f}, y={3:2.2f}, z={4:2.2f})"

    RENDERING_INFO_TEXT = """Near plane: {0:2.3f}
Film size (x): {1}
Film size (y): {2}"""

    def __init__(
        self,
        base: Panda3dShowBase,
        viewpoint_split: ViewpointSplit,
        initial_cam_pos: ThreeDPoint,
        cam_points_at: ThreeDPoint,
        max_depth: float,
    ) -> None:
        self.base: Panda3dShowBase = base
        self._cam = cast(NodePath, base.cam)

        inputState.watchWithModifiers("forward", "w")
        inputState.watchWithModifiers("left", "a")
        inputState.watchWithModifiers("reverse", "s")
        inputState.watchWithModifiers("right", "d")

        taskMgr.add(self._keyboard_controls_task, "keyboard_controls", delay=0.01)
        taskMgr.add(self._mouse_controls_task, "mouse_controls", delay=0.01)
        self._mw: MouseWatcher = self.base.mouseWatcherNode

        base.accept("escape", sys.exit)
        base.accept("tab", self._switch_camera_mode)
        base.accept("p", self._toggle_pos_info)
        base.accept("r", self._toggle_render_info)
        base.accept("z", self._toggle_depth)
        base.accept("c", self._view_depth_point_cloud)
        base.accept("n", self._save_depth_to_npy)
        base.accept("m", PStatClient.connect)
        for i in range(10):
            base.accept(str(i), self._type_number, [str(i)])
        base.accept("enter", self._type_number, ["enter"])
        base.accept("backspace", self._type_number, ["backspace"])
        base.accept("a", self._update_viewpoint_idx_if_fixed, ["a"])
        base.accept("d", self._update_viewpoint_idx_if_fixed, ["d"])

        self._show_pos = False
        self._show_depth = False
        self._show_rendering_info = False

        self._constrained_mode = None
        self._max_depth = max_depth

        self._update_on_screen_info_text()
        base.show_mouse(False)

        self._viewpoint_controller = ViewpointBasedCamController(
            self.base, viewpoint_split
        )
        self.base.set_cam_pos_and_look_at(
            new_cam_pos=initial_cam_pos, look_at=cam_points_at
        )

    @staticmethod
    def attach(
        base: Panda3dShowBase,
        initial_cam_pos: ThreeDPoint,
        viewpoint_split: ViewpointSplit,
        cam_points_at: ThreeDPoint,
        max_depth: float,
    ) -> "LivePreviewController":
        """
        Attach a new instance to a specified `Panda3dShowBase`.

        Keep in mind, that this class assumes exclusive control (with the exception of the `Panda3dShowBase.run` call) on the controlled `Panda3dShowBase` after the attach operation.

        Parameters
        ----------
        base
            The `Panda3dShowBase` to control.
        initial_cam_pos
            The initial position of the camera.
        cam_points_at
            The position to which the camera should initially look at.
        viewpoint_split
            The viewpoint split to use.

        Returns
        -------
        v
            The attached new instance of `DiagnosticsController`.
        """
        controls = LivePreviewController(
            base,
            initial_cam_pos=initial_cam_pos,
            cam_points_at=cam_points_at,
            viewpoint_split=viewpoint_split,
            max_depth=max_depth,
        )
        controls._recenterMouse()
        return controls

    def _toggle_render_info(self) -> None:
        self._show_rendering_info = not self._show_rendering_info
        self._update_on_screen_info_text()

    def _toggle_pos_info(self) -> None:
        """Show/hide the current position in the on-screen info text."""
        self._show_pos = not self._show_pos
        self._update_on_screen_info_text()

    def _toggle_depth(self) -> None:
        """Show/hide the depth map."""
        self._show_depth = not self._show_depth
        self.base.show_depth(self._show_depth)

    def _view_depth_point_cloud(self) -> None:
        """Show the current **depth map** as a point cloud."""
        frame = self.base.render_single_RGBB_frame()
        lens = self.base.get_cam_lens()
        assert isinstance(lens, PerspectiveLens)
        depth_and_mask = zbuf_2_depth_and_mask(
            zbuf_data=frame.zbufs, camera_lens=lens, max_depth=self._max_depth
        )
        cam_proj_spec = get_cam_proj_spec_for_showbase(self.base)
        show_depths_as_point_clouds_in_browser(
            cam_proj_spec=cam_proj_spec, depths={"Depth map": depth_and_mask}
        )

    def _save_depth_to_npy(self) -> None:
        """Save the current **depth map** to a numpy array."""
        frame = self.base.render_single_RGBB_frame()
        lens = self.base.get_cam_lens()
        assert isinstance(lens, PerspectiveLens)
        depth_and_mask = zbuf_2_depth_and_mask(
            zbuf_data=frame.zbufs, camera_lens=lens, max_depth=self._max_depth
        )
        date_at_saving = datetime.datetime.now()
        depth_name = f"depth-{date_at_saving.year}-{date_at_saving.month}-{date_at_saving.day}_{date_at_saving.hour}-{date_at_saving.minute}-{date_at_saving.second}.npy"
        mask_name = f"mask-{date_at_saving.year}-{date_at_saving.month}-{date_at_saving.day}_{date_at_saving.hour}-{date_at_saving.minute}-{date_at_saving.second}.npy"
        np.save(depth_name, depth_and_mask.depths)
        np.save(mask_name, depth_and_mask.masks)

    def _update_on_screen_info_text(self) -> None:
        """Update the on-screen info text based on the current state of the controlled object and this controller."""

        if self._constrained_mode is None:
            if self._show_rendering_info:
                rendering_info_text = "\n" + self._get_rendering_info_text()
            else:
                rendering_info_text = ""

            if self._show_pos:
                cam_pos = self.base.get_cam_xyz()
                pos_info_text = "\n" + self.FREE_CAM_POS_INFO_TEXT.format(
                    cam_pos.x, cam_pos.y, cam_pos.z
                )
            else:
                pos_info_text = ""

            self.base.set_help_text(
                self.FREE_CAM_HELP_TEXT.format(
                    self._max_depth, rendering_info_text, pos_info_text
                )
            )
        else:
            viewpoint_selector_text = self._get_viewpoint_selector_text(
                new_viewp_acc=self._constrained_mode.new_viewp_acc,
                n_viewpoints=self._get_n_viewpoints(self._constrained_mode.viewpt_type),
            )

            if self._show_rendering_info:
                rendering_info_text = "\n" + self._get_rendering_info_text()
            else:
                rendering_info_text = ""

            if self._show_pos:
                x, y, z = self.base.get_cam_xyz()
                n_viewpoints = self._get_n_viewpoints(
                    self._constrained_mode.viewpt_type
                )

                pos_info_text = "\n" + self.CONSTRAINED_CAM_POS_INFO_TEXT.format(
                    self._constrained_mode.viewpoint_idx,
                    n_viewpoints,
                    x,
                    y,
                    z,
                )
            else:
                pos_info_text = ""

            self.base.set_help_text(
                self.CONSTRAINED_CAM_HELP_TEXT.format(
                    viewpoint_selector_text,
                    self._max_depth,
                    rendering_info_text,
                    pos_info_text,
                )
            )

    @staticmethod
    def _get_viewpoint_selector_text(new_viewp_acc: str, n_viewpoints: int) -> str:
        if new_viewp_acc == "":
            return LivePreviewController.SET_VIEW_INDEX_EMPTY_TEXT
        else:
            return new_viewp_acc + f"_ / {n_viewpoints}"

    def _get_rendering_info_text(self) -> str:
        near_plane = self.base.cam.node().getLens().near
        film_size = self.base.cam.node().getLens().getFilmSize()

        return self.RENDERING_INFO_TEXT.format(near_plane, film_size.x, film_size.y)

    def _switch_camera_mode(self) -> None:
        """Switch between completely free camera and constrained camera."""
        if self._constrained_mode is None:
            self.base.set_help_text(self.CONSTRAINED_CAM_HELP_TEXT)
            self._constrained_mode: "_ConstrainedCamState | None" = (
                _ConstrainedCamState(
                    viewpoint_idx=0,
                    new_viewp_acc="",
                    viewpt_type=SampleType.Train,
                )
            )
            self._viewpoint_controller.select_viewpoint(0, SampleType.Train)
        else:
            self.base.set_help_text(self.FREE_CAM_HELP_TEXT)
            self._constrained_mode = None

    def _type_number(self, num_str: str) -> None:
        if self._constrained_mode is not None:
            if num_str == "enter":
                if len(self._constrained_mode.new_viewp_acc) > 0:
                    new_viewpoint_idx_candidate = int(
                        self._constrained_mode.new_viewp_acc
                    )
                else:
                    new_viewpoint_idx_candidate = 0
                self._constrained_mode.new_viewp_acc = ""
                n_viewpoints = self._get_n_viewpoints(
                    self._constrained_mode.viewpt_type
                )
                new_viewpoint_idx = min(n_viewpoints - 1, new_viewpoint_idx_candidate)

                self._viewpoint_controller.select_viewpoint(
                    new_viewpoint_idx, viewpt_type=self._constrained_mode.viewpt_type
                )
                self._constrained_mode.viewpoint_idx = new_viewpoint_idx
            elif num_str == "backspace":
                if len(self._constrained_mode.new_viewp_acc) > 0:
                    self._constrained_mode.new_viewp_acc = (
                        self._constrained_mode.new_viewp_acc[1:]
                    )
            else:
                self._constrained_mode.new_viewp_acc += num_str

            self._update_on_screen_info_text()

    def _keyboard_controls_task(self, task):
        """The task that implements the keyboard-based camera movement operations in both camera modes."""
        dt = task.time

        speed = 10
        dside = 0
        dforw = 0
        if inputState.isSet("forward"):
            dforw += 1
        if inputState.isSet("reverse"):
            dforw -= 1
        if inputState.isSet("left"):
            dside -= 1
        if inputState.isSet("right"):
            dside += 1

        if self._constrained_mode is None:
            self.base.set_cam_pos_free(
                point=ThreeDPoint(x=dside * speed * dt, y=dforw * speed * dt, z=0),
                fluid=True,
                local_coordinates=True,
            )
        self._update_on_screen_info_text()

        return task.again

    def _update_viewpoint_idx_if_fixed(self, key: str):
        if self._constrained_mode is not None:
            if key == "a":
                new_viewpoint_idx = max(
                    self._constrained_mode.viewpoint_idx - 1,
                    0,
                )
            else:
                new_viewpoint_idx = min(
                    self._constrained_mode.viewpoint_idx + 1,
                    self._viewpoint_controller.get_filtered_viewpoint_count(
                        self._constrained_mode.viewpt_type
                    )
                    - 1,
                )

            self._constrained_mode.new_viewp_acc = ""
            self._viewpoint_controller.select_viewpoint(
                new_viewpoint_idx, viewpt_type=self._constrained_mode.viewpt_type
            )
            self._constrained_mode.viewpoint_idx = new_viewpoint_idx
            self._update_on_screen_info_text()

    def _mouse_controls_task(self, task):
        """The task that implements the mouse-based movement/rotation operations in both camera modes."""
        dt = task.time

        if self._mw.hasMouse():
            dx = self._mw.getMouseX()
            dy = self._mw.getMouseY()

            self._recenterMouse()

            if self._constrained_mode is None:
                free_cam_sensitivity = 5000
                self.base.set_cam_hpr_free(
                    -dx * free_cam_sensitivity * dt,
                    dy * free_cam_sensitivity * dt,
                    0,
                    local_coordinates=True,
                )
            else:
                self._recenterMouse()

        self._update_on_screen_info_text()

        return task.again

    def _recenterMouse(self) -> None:
        """Move the mouse to the center of the screen."""
        win: GraphicsOutput = self.base.win
        assert isinstance(win, GraphicsWindow), "The window should not be offscreen."

        win.movePointer(
            0,
            int(win.getProperties().getXSize() / 2),
            int(win.getProperties().getYSize() / 2),
        )

    def _get_n_viewpoints(self, viewpt_type: SampleType) -> int:
        return self._viewpoint_controller.get_filtered_viewpoint_count(viewpt_type)


@dataclass
class _ConstrainedCamState:
    viewpoint_idx: int
    new_viewp_acc: str
    viewpt_type: SampleType
