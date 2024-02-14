"""
This module implements the following:

* The low-level rendering logic.
* The low-level scene loading and manipulation operations.
* An interactive previewer for the scenes.
* The low-level camera manipulation.
* Some utility functions to handle images and masked depth maps.
* Plotly-based visualization for the depth maps.
* The data structures (sizes, points) to control the low-level rendering classes.
"""

from ..dataset_model import DepthsWithMasks, RGBsWithDepthsAndMasks
from ._areas import ScaledStandingAreas, ThreeDArea
from ._custom_show_base import Panda3dShowBase
from ._data import RGBWithZbuf, ThreeDPoint, ThreeDSize, TwoDAreas, TwoDSize
from ._errors import Panda3dAssumptionViolation
from ._im import cv_resize_multiple_images, imshow, scale_depth_maps_with_masks
from ._live_preview import LivePreviewController
from ._object_creation import verts_and_faces_2_obj
from ._object_transform import (
    MIN_CONTROL_POINTS_FOR_WARP,
    MeshBasedObjectTransform,
    ObjectTransform,
    ObjectTransformResult,
    ObjectTransformType,
    PointBasedVectorFieldSpec,
    VolumeBasedObjectTransform,
    get_object_transform_by_type,
    get_object_transform_type_by_name,
    get_supported_object_transform_type_names,
    get_target_obj_field_cache_path_for_world,
)
from ._occupacy_fields import (
    InterpolatedOccupacyField,
    OccupacyField,
    OccupacyFieldSamples,
    VertsAndFaces,
    WarpField,
    get_d_coord_from_obj_size_along_the_shortest_axis,
    occupacy_field_2_occupacy_field_samples,
    occupacy_field_samples_2_interpolator,
    occupacy_field_samples_2_verts_and_faces,
    verts_and_faces_2_occupacy_field_samples,
)
from ._point_cloud import (
    ThreeDCoordSysConvention,
    depth_map_2_homog_points_in_im_space,
    depth_map_2_point_cloud,
    depthmaps_2_point_cloud_fig,
    get_matrix_to_coord_system_from_standard,
    invert_projection,
    show_depths_as_point_clouds_in_browser,
)
from ._scene_config import (
    SceneConfDictKey,
    SceneConfigDict,
    get_scene_config_errors,
    load_scene_config,
)
from ._scene_util import (
    directly_contains_geometry_data,
    find_node,
    get_all_vertex_arrays_copy_from_vertex_data,
    get_bounding_rectangle_2d,
    get_bounding_rectangle_on_screen,
    get_cam_proj_spec_for_lens,
    get_cam_proj_spec_for_showbase,
    get_col_copy_from_vertex_data,
    get_near_far_planes_safe,
    get_ob_size_from_vertices,
    get_obj_copy,
    get_obj_size,
    get_projection_inv_mat_col,
    get_sky_mask_limit,
    get_vertex_count,
    get_vertex_face_copy_most_common_errors,
    get_vertex_positions_copy,
    get_vertices_and_faces_copy,
    is_geom_node_obj,
    load_model_from_local_file,
    project_points_to_screen,
    put_obj,
    set_col_in_vertex_data,
    zbuf_2_depth_and_mask,
)
from ._twod_area_util import get_twod_area_masks
from ._verts_and_faces_presentation import verts_and_faces_2_plotly_figure
from ._viewpoints import (
    DesiredViewpointCounts,
    SplitFormatError,
    ViewpointBasedCamController,
    ViewpointSplit,
    get_viewpoints_path_for_world,
)

__all__ = [
    "Panda3dShowBase",
    "LivePreviewController",
    "RGBsWithDepthsAndMasks",
    "RGBWithZbuf",
    "depthmaps_2_point_cloud_fig",
    "show_depths_as_point_clouds_in_browser",
    "DepthsWithMasks",
    "Panda3dAssumptionViolation",
    "ViewpointBasedCamController",
    "get_ob_size_from_vertices",
    "get_sky_mask_limit",
    "get_vertex_positions_copy",
    "zbuf_2_depth_and_mask",
    "ThreeDSize",
    "ScaledStandingAreas",
    "get_vertex_count",
    "find_node",
    "get_near_far_planes_safe",
    "ThreeDPoint",
    "put_obj",
    "get_obj_copy",
    "directly_contains_geometry_data",
    "imshow",
    "TwoDSize",
    "SceneConfigDict",
    "load_scene_config",
    "cv_resize_multiple_images",
    "get_scene_config_errors",
    "SceneConfDictKey",
    "scale_depth_maps_with_masks",
    "get_projection_inv_mat_col",
    "depth_map_2_point_cloud",
    "ThreeDCoordSysConvention",
    "get_matrix_to_coord_system_from_standard",
    "invert_projection",
    "depth_map_2_homog_points_in_im_space",
    "ViewpointSplit",
    "get_viewpoints_path_for_world",
    "DesiredViewpointCounts",
    "project_points_to_screen",
    "get_bounding_rectangle_on_screen",
    "TwoDAreas",
    "get_twod_area_masks",
    "get_bounding_rectangle_2d",
    "SplitFormatError",
    "InterpolatedOccupacyField",
    "OccupacyField",
    "WarpField",
    "MeshBasedObjectTransform",
    "ObjectTransform",
    "VolumeBasedObjectTransform",
    "get_object_transform_by_type",
    "ObjectTransformType",
    "get_target_obj_field_cache_path_for_world",
    "PointBasedVectorFieldSpec",
    "get_object_transform_type_by_name",
    "get_supported_object_transform_type_names",
    "VertsAndFaces",
    "get_vertices_and_faces_copy",
    "is_geom_node_obj",
    "get_vertex_face_copy_most_common_errors",
    "verts_and_faces_2_plotly_figure",
    "occupacy_field_samples_2_interpolator",
    "load_model_from_local_file",
    "verts_and_faces_2_obj",
    "get_all_vertex_arrays_copy_from_vertex_data",
    "get_col_copy_from_vertex_data",
    "set_col_in_vertex_data",
    "ObjectTransformResult",
    "MIN_CONTROL_POINTS_FOR_WARP",
    "occupacy_field_2_occupacy_field_samples",
    "occupacy_field_samples_2_verts_and_faces",
    "verts_and_faces_2_occupacy_field_samples",
    "OccupacyFieldSamples",
    "get_d_coord_from_obj_size_along_the_shortest_axis",
    "ThreeDArea",
    "get_cam_proj_spec_for_lens",
    "get_cam_proj_spec_for_showbase",
]
