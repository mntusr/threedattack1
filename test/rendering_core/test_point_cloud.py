import math
import unittest
from unittest import mock

import numpy as np
from panda3d.core import PerspectiveLens

from threedattack.dataset_model import DepthsWithMasks
from threedattack.rendering_core import (
    depth_map_2_homog_points_in_im_space,
    depthmaps_2_point_cloud_fig,
    get_cam_proj_spec_for_lens,
    imshow,
    invert_projection,
)
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *


class TestShowPointCloud(unittest.TestCase):
    def test_depthmaps_2_point_cloud_fig_happy_path(self):
        depth_maps_with_masks: dict[str, DepthsWithMasks] = dict()

        for i in range(5):
            depth_map = np.full((1, 1, 12, 12), 3, dtype=np.float32)
            mask = np.ones_like(depth_map, dtype=np.bool_)
            depth_with_mask = DepthsWithMasks(depths=depth_map, masks=mask)

            depth_maps_with_masks[f"depth_{i}"] = depth_with_mask

        cam_proj_spec = _get_cam_proj_spec(infinite_near=False)

        fig = depthmaps_2_point_cloud_fig(
            cam_proj_spec=cam_proj_spec, depths=depth_maps_with_masks
        )

        n_traces = len(fig.data)  # type: ignore
        self.assertEqual(n_traces, len(depth_maps_with_masks.keys()))

    def test_depthmaps_2_point_cloud_fig_more_than_one_depths(self):
        cam_proj_spec = _get_cam_proj_spec(infinite_near=False)
        depth_map = np.full((10, 1, 12, 12), 3, dtype=np.float32)
        mask = np.ones_like(depth_map, dtype=np.bool_)
        depth_maps_with_masks: dict[str, DepthsWithMasks] = dict()
        depth_maps_with_masks["d1"] = DepthsWithMasks(depths=depth_map, masks=mask)

        with self.assertRaises(ValueError):
            depthmaps_2_point_cloud_fig(
                cam_proj_spec=cam_proj_spec, depths=depth_maps_with_masks
            )

    def test_imshow_happy_path(self):
        for call_show in [True, False]:
            for call_figure in [True, False]:
                with self.subTest(f"call_show={call_show},create_figure={call_figure}"):
                    image = np.full((1, 3, 12, 15), 0.5, dtype=np.float32)
                    ax_mock = mock.Mock()
                    show_fn_mock = mock.Mock()
                    figure_mock = mock.Mock()
                    figure_fn_mock = mock.Mock(return_value=figure_mock)
                    figure_mock.gca = mock.Mock(return_value=ax_mock)

                    if call_figure:
                        imshow_on = "newfig"
                    else:
                        imshow_on = ax_mock

                    with mock.patch("matplotlib.pyplot.figure", new=figure_fn_mock):
                        with mock.patch("matplotlib.pyplot.show", new=show_fn_mock):
                            imshow(im=image, show=call_show, on=imshow_on)

                    expected_show_call_count = _get_expected_call_count(call_show)
                    expected_figure_call_count = _get_expected_call_count(call_figure)

                    self.assertEqual(show_fn_mock.call_count, expected_show_call_count)
                    self.assertEqual(
                        figure_fn_mock.call_count, expected_figure_call_count
                    )
                    self.assertEqual(
                        figure_mock.gca.call_count, expected_figure_call_count
                    )

                    ax_mock.imshow.assert_called_once()
                    shown_im = ax_mock.imshow.call_args.args[0]

                    self.assertEqual(tuple(shown_im.shape), (12, 15, 3))

    def test_imshow_non_im_input(self):
        image = np.full((1, 3, 12), 0.5, dtype=np.float32)
        ax_mock = mock.Mock()
        with self.assertRaises(ValueError):
            imshow(im=image, show=False, on=ax_mock)

    def test_invert_projection(self):
        original_affine_points = scast_points_aspace(
            np.array(
                [
                    [3, 5, 2, 1],
                    [4, 5, 3, 1],
                    [7, 0, 2, 1],
                ],
                dtype=np.float32,
            )
        )
        proj_mat = scast_mat_float(
            np.array(
                [
                    [1, 0, 0, 1],
                    [1, 3, 0, 1],
                    [4, 3, 2, 1],
                ],
                dtype=np.float32,
            )
        )

        projected_points = (proj_mat @ original_affine_points.T).T

        restored_cart_points = invert_projection(
            projected_points, invertable_proj_mat=proj_mat
        )
        self.assertTrue(match_points_space(restored_cart_points))

        original_cart_points = idx_points_aspace(
            original_affine_points, data=["x", "y", "z"]
        )

        self.assertTrue(
            np.allclose(original_cart_points, restored_cart_points, atol=1e-4)
        )

    def test_depth_map_2_homog_points_in_im_space(self):
        depth_map = np.expand_dims(
            np.array(
                [
                    [3, 2, 2.5],
                    [1, 100, 7],
                    [3, 3, 1],
                    [2, 4, 100],
                    [9, 0.3, 1],
                ],
                dtype=np.float32,
            ),
            axis=(0, 1),
        )
        depth_map = scast_im_depthmaps(depth_map)
        point_cloud_in_im_planewithd = scast_points_planewithd(
            np.array(
                [
                    [-1, 1, 3],
                    [0, 1, 2],
                    [1, 1, 2.5],
                    [-1, 0.5, 1],
                    # skip [0, 0.5, -1],
                    [1, 0.5, 7],
                    [-1, 0, 3],
                    [0, 0, 3],
                    [1, 0, 1],
                    [-1, -0.5, 2],
                    [0, -0.5, 4],
                    # skip [1, -0.5, -1],
                    [-1, -1, 9],
                    [0, -1, 0.3],
                    [1, -1, 1],
                ],
                dtype=np.float32,
            )
        )

        expected_point_cloud_in_im_affine = np.zeros_like(
            point_cloud_in_im_planewithd, dtype=np.float32
        )
        upd_points(
            expected_point_cloud_in_im_affine,
            data=0,
            value_=idx_points_planewithd(point_cloud_in_im_planewithd, data="x")
            * idx_points_planewithd(point_cloud_in_im_planewithd, data="d"),
        )
        upd_points(
            expected_point_cloud_in_im_affine,
            data=1,
            value_=idx_points_planewithd(point_cloud_in_im_planewithd, data="y")
            * idx_points_planewithd(point_cloud_in_im_planewithd, data="d"),
        )
        upd_points(
            expected_point_cloud_in_im_affine,
            data=2,
            value_=idx_points_planewithd(point_cloud_in_im_planewithd, data="d"),
        )
        n_expected_points = expected_point_cloud_in_im_affine.shape[DIM_POINTS_N]

        actual_point_cloud_in_im_affine = depth_map_2_homog_points_in_im_space(
            depths_with_masks=DepthsWithMasks(depths=depth_map, masks=depth_map <= 20),
            im_left_x_val=np.min(
                idx_points_planewithd(point_cloud_in_im_planewithd, data="x")
            ),
            im_right_x_val=np.max(
                idx_points_planewithd(point_cloud_in_im_planewithd, data="x")
            ),
            im_top_y_val=np.max(
                idx_points_planewithd(point_cloud_in_im_planewithd, data="y")
            ),
            im_bottom_y_val=np.min(
                idx_points_planewithd(point_cloud_in_im_planewithd, data="y")
            ),
        )
        self.assertTrue(
            match_points_aplane(
                actual_point_cloud_in_im_affine, shape={"n": n_expected_points}
            )
        )

        self.assertTrue(
            np.allclose(
                expected_point_cloud_in_im_affine, actual_point_cloud_in_im_affine
            )
        )


def _get_expected_call_count(call_needed: bool) -> int:
    return 1 if call_needed else 0


def _get_cam_proj_spec(infinite_near: bool):
    lens = PerspectiveLens()

    if infinite_near:
        lens.setNear(math.inf)
    else:
        lens.setNear(0.1)

    return get_cam_proj_spec_for_lens(lens)
