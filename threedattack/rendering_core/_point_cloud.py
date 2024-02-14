import math
import os
import webbrowser
from dataclasses import dataclass
from enum import Enum, auto
from typing import Mapping, NamedTuple

import numpy as np
import plotly.graph_objects as go
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Lens, PerspectiveLens

from ..dataset_model import CamProjSpec, DepthsWithMasks
from ..tempfolder import new_temp_file
from ..tensor_types.idx import *
from ..tensor_types.npy import *
from ._custom_show_base import Panda3dShowBase
from ._errors import Panda3dAssumptionViolation
from ._im import scale_depth_maps_with_masks
from ._scene_util import get_projection_mat_col


def show_depths_as_point_clouds_in_browser(
    cam_proj_spec: "CamProjSpec", depths: Mapping[str, DepthsWithMasks]
) -> None:
    """
    Show the depth maps as point cloud in a new browser window.

    This function assumes a Z-up right-handed coordinate system. This is the default coordinate system in Panda3d.

    Parameters
    ----------
    cam_proj_spec
        The projection properties of the camera.
    depths
        A dictionary, where the keys are the names of the depth maps. The values are the depth maps (with the corresponding masks).

    Notes
    -----
    This function internally creates a Plotly figure, then saves it to a temporary HTML file, then opens it using the default browser.

    This means that this function needs the global temporary dictionary to be activated.

    This function might downlscale the depth maps to improve the usability of the exported Plotly figure in the browser.
    """
    fig = depthmaps_2_point_cloud_fig(depths=depths, cam_proj_spec=cam_proj_spec)
    _show_fig_in_browser(fig)


def _show_fig_in_browser(fig: go.Figure) -> None:
    filepath = new_temp_file(suffix=".html")
    fig.write_html(file=filepath)
    webbrowser.open(f"file://{os.path.realpath(filepath)}")


def depthmaps_2_point_cloud_fig(
    cam_proj_spec: "CamProjSpec", depths: Mapping[str, DepthsWithMasks]
) -> go.Figure:
    """
    Create a Plotly figure that contains the specified depth maps as 3D point clouds.

    This function assumes a Z-up right-handed coordinate system. This is the default coordinate system in Panda3d.

    The individual depth maps are added as traces to the figure.

    This function does not invoke the `plotly.graph_objects.Figure.show` function.

    Parameters
    ----------
    cam_proj_spec
        The projection properties of the camera.
    depths
        A dictionary, where the keys are the names of the depth maps. The values are the depth maps (with the corresponding masks).

    Returns
    -------
    v
        The created figure.

    Raises
    ------
    ValueError
        If there are more than one depth maps.

    Notes
    -----
    This function downscales the depth maps to improve the performance of the presented plotly figure if ``min(width, height)>100``, where ``width`` and ``height`` are respectively the width and height of the depth map. The function does not apply downscaling if the last condition does not hold.
    """

    scaled_data: dict[str, DepthsWithMasks] = dict()
    for depth_name, depth_val in depths.items():
        width = depth_val.depths.shape[DIM_IM_W]
        height = depth_val.depths.shape[DIM_IM_H]
        if min(width, height) > 100:
            scaled_data[depth_name] = scale_depth_maps_with_masks(depth_val, 0.1)
        else:
            scaled_data[depth_name] = depth_val

    fig = go.Figure()
    for depth_map_name, data in scaled_data.items():
        if data.depths.shape[DIM_IM_N] != 1:
            raise ValueError("Each trace should contain exactly one depth map.")
        if data.masks.shape[DIM_IM_N] != 1:
            raise ValueError("Each trace should contain exactly one mask.")

        point_cloud = depth_map_2_point_cloud(
            depths_with_masks=data, cam_proj_spec=cam_proj_spec
        )

        fig.add_scatter3d(
            x=point_cloud[:, CAT_POINTS_SPACE_DATA_X],
            y=point_cloud[:, CAT_POINTS_SPACE_DATA_Y],
            z=point_cloud[:, CAT_POINTS_SPACE_DATA_Z],
            name=depth_map_name,
            mode="markers",
        )

    fig = fig.update_traces(marker_size=2).update_scenes(aspectmode="data")
    fig = fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig


def depth_map_2_point_cloud(
    depths_with_masks: DepthsWithMasks, cam_proj_spec: "CamProjSpec"
) -> np.ndarray:
    """
    Convert the specified depth map to a point cloud.

    This function assumes a Z-up right-handed coordinate coordinate system. This is the default coordinate system in Panda3d if it is not configured otherwise.

    This function assumes that each point is projected to the center of its corresponding pixel originally.

    Parameters
    ----------
    depths_with_masks
        The depth map to convert with the corresponding mask. Only a single depth map is supported.
    cam_proj_spec
        The projection properties of the camera.

    Returns
    -------
    v
        The point cloud. Format: ``points::space``

    Raises
    ------
    ValueError
        If more than one depth map is given.
    """
    homog_im_points = depth_map_2_homog_points_in_im_space(
        depths_with_masks=depths_with_masks,
        im_left_x_val=cam_proj_spec.im_left_x_val,
        im_right_x_val=cam_proj_spec.im_right_x_val,
        im_top_y_val=cam_proj_spec.im_top_y_val,
        im_bottom_y_val=cam_proj_spec.im_bottom_y_val,
    )
    points_3d = invert_projection(
        homog_im_points=homog_im_points, invertable_proj_mat=cam_proj_spec.proj_mat
    )

    return points_3d


def depth_map_2_homog_points_in_im_space(
    depths_with_masks: DepthsWithMasks,
    im_left_x_val: float,
    im_right_x_val: float,
    im_top_y_val: float,
    im_bottom_y_val: float,
) -> np.ndarray:
    """
    Convert the specified depth map to a set of points in the image space using homogen coordinates. The ``w`` coordinate is the depth value.

    This function drops the points that belong to masked pixels in the depth map.

    Parameters
    ----------
    depths_with_masks
        The masked depth maps.
    im_left_x_val
        The image x coordinate corresponding to ``x=0`` in the depth map.
    im_right_x_val
        The image x coordinate corresponding to ``x=-1`` in the depth map.
    im_top_y_val
        The image y coordinate corresponding to ``y=0`` in the depth map.
    im_top_y_val
        The image y coordinate corresponding to ``y=-1`` in the depth map.

    Returns
    -------
    v
        The corresponding points.

    Raises
    ------
    ValueError
        If there is more than one depth map.
    """
    n_depth_maps = depths_with_masks.depths.shape[DIM_IM_N]
    if n_depth_maps != 1:
        raise ValueError(
            f"Exactly one depth map should be specified instead of {n_depth_maps}."
        )

    px_height = depths_with_masks.depths.shape[DIM_IM_H]
    px_width = depths_with_masks.depths.shape[DIM_IM_W]

    x_grid, y_grid = np.meshgrid(
        np.linspace(im_left_x_val, im_right_x_val, px_width),
        np.linspace(im_top_y_val, im_bottom_y_val, px_height),
    )

    flat_mask_data: np.ndarray = depths_with_masks.masks.flatten()

    flat_depth_data: np.ndarray = depths_with_masks.depths.flatten()
    homog_points_2d_x = x_grid.flatten() * flat_depth_data
    homog_points_2d_y = y_grid.flatten() * flat_depth_data

    homog_points_2d_x = homog_points_2d_x[flat_mask_data == 1]
    homog_points_2d_y = homog_points_2d_y[flat_mask_data == 1]
    flat_depth_data = flat_depth_data[flat_mask_data == 1]

    homog_points_2d = np.stack(
        [
            homog_points_2d_x,
            homog_points_2d_y,
            flat_depth_data,
        ]
    ).T
    return homog_points_2d


def invert_projection(
    homog_im_points: np.ndarray, invertable_proj_mat: np.ndarray
) -> np.ndarray:
    """
    Invert the specified projection. This function assumes without checking that this projection meets the conditions specified by `CamProjSpec`.

    Parameters
    ----------
    homog_im_points
        The points in the homogen coordinate system of the image. Format: ``Points::APlane``
    invertable_proj_mat
        The projection matrix. Format: ``Mat::Float[f3x4]``.

    Returns
    -------
    v
        The original points. It uses Cartesian coordinates, since the w coordinate is known 1. Format: ``Points::Space``

    Raises
    ------
    LinAlgError
        If B is not invertable.
    """
    B = idx_mat(invertable_proj_mat, col=[0, 1, 2])
    h = idx_mat(invertable_proj_mat, col=[3])

    B_inv = np.linalg.inv(B)

    points_3d = B_inv @ (homog_im_points.T - h)

    return points_3d.T


def get_matrix_to_coord_system_from_standard(
    dst_sys: "ThreeDCoordSysConvention",
) -> np.ndarray:
    """
    Get the matrix that transforms the points from a Z-up right handed coordinate system to the specified coordinate system.

    This function assumes 3d homogen coordinates.

    The points are represented by column vectors.

    Parameters
    ----------
    dst_sys
        The target coordinate system.

    Returns
    -------
    v
        Format: ``Mat::Float[f4x4]``
    """
    match dst_sys:
        case ThreeDCoordSysConvention.YupRightHanded:
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )
        case ThreeDCoordSysConvention.ZupRightHanded:
            return np.eye(4, dtype=np.float32)
        case ThreeDCoordSysConvention.YupLeftHanded:
            return np.array(
                [
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )
        case ThreeDCoordSysConvention.ZupLeftHanded:
            return np.array(
                [
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )


class ThreeDCoordSysConvention(Enum):
    YupRightHanded = auto()
    ZupRightHanded = auto()
    YupLeftHanded = auto()
    ZupLeftHanded = auto()
