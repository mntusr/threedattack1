import builtins
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import gltf
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import PerspectiveLens

from threedattack.rendering_core import (
    Panda3dShowBase,
    SceneConfigDict,
    ThreeDPoint,
    TwoDSize,
    get_scene_config_errors,
    imshow,
)
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *

from .util import load_xsortable_octagon_panda3d


class TestPanda3dShowBase(unittest.TestCase):
    def setUp(self):
        self.WINDOW_SIZE = TwoDSize(500, 500)
        self.show_base = Panda3dShowBase(offscreen=True, win_size=self.WINDOW_SIZE)
        self.TEST_SCENE_PATH = (
            Path(__file__).resolve().parent.parent.parent
            / "test_resources"
            / "test_scene.glb"
        )

    def tearDown(self):
        self.show_base.destroy()

    def test_happy_path(self):
        self.show_base.loader
        self.show_base.load_world_from_blender(self.TEST_SCENE_PATH)

        # camera movement
        self.show_base.set_cam_pos_and_look_at(
            new_cam_pos=ThreeDPoint(4, 3.5, 3.25), look_at=ThreeDPoint(0, 0, 0)
        )

        expected_cam_pos = ThreeDPoint(4, 3.5, 3.25)
        actual_cam_x, actual_cam_y, actual_cam_z = self.show_base.cam.getPos()
        actual_cam_h, actual_cam_p, actual_cam_r = self.show_base.cam.getHpr()
        self.show_base.cam.lookAt(0, 0, 0)
        expected_cam_h, expected_cam_p, expected_cam_r = self.show_base.cam.getHpr()
        actual_cam_pos = ThreeDPoint(actual_cam_x, actual_cam_y, actual_cam_z)

        self.assertTrue(expected_cam_pos.is_almost_equal(actual_cam_pos, epsilon=1e-6))
        self.assertAlmostEqual(actual_cam_h, expected_cam_h, 5)
        self.assertAlmostEqual(actual_cam_p, expected_cam_p, 5)
        self.assertAlmostEqual(actual_cam_r, expected_cam_r, 5)

        # capturing
        capture1 = self.show_base.render_single_RGBB_frame()
        self.assertTrue(
            match_im_rgbs(
                capture1.rgbs,
                shape={
                    "n": 1,
                    "w": self.WINDOW_SIZE.x_size,
                    "h": self.WINDOW_SIZE.y_size,
                },
            )
        )
        self.assertTrue(
            match_im_zbuffers(
                capture1.zbufs,
                shape={
                    "n": 1,
                    "h": capture1.rgbs.shape[DIM_IM_H],
                    "w": capture1.rgbs.shape[DIM_IM_W],
                },
            )
        )

        mean_r = float(np.mean(idx_im(capture1.rgbs, n=0, c=CAT_IM_RGBS_C_R)))
        mean_g = float(np.mean(idx_im(capture1.rgbs, n=0, c=CAT_IM_RGBS_C_G)))
        mean_b = float(np.mean(idx_im(capture1.rgbs, n=0, c=CAT_IM_RGBS_C_B)))

        self.assertLessEqual(mean_g, mean_b)
        self.assertLessEqual(mean_r, mean_g)

        zbuf_at_top_left = float(idx_im(capture1.zbufs, n=0, h=0, w=0))
        self.assertAlmostEqual(zbuf_at_top_left, 1)

        zbuf_at_center = float(
            idx_im(
                capture1.zbufs,
                n=0,
                h=capture1.zbufs.shape[DIM_IM_H] // 2,
                w=capture1.zbufs.shape[DIM_IM_W] // 2,
            )
        )

        self.assertLess(zbuf_at_center, zbuf_at_top_left)

        # capturing 2 (unchanged scene)
        capture2 = self.show_base.render_single_RGBB_frame()
        self.assertTrue(np.allclose(capture1.rgbs, capture2.rgbs, atol=1e-5))
        self.assertTrue(np.allclose(capture1.zbufs, capture2.zbufs, atol=1e-5))

        # capturing 3 (changed scene)
        self.show_base.cam.setPos(1000, 0, 0)
        capture3 = self.show_base.render_single_RGBB_frame()
        self.assertFalse(np.allclose(capture2.rgbs, capture3.rgbs, atol=1e-5))
        self.assertFalse(np.allclose(capture2.zbufs, capture3.zbufs, atol=1e-5))

        # capturing 4 (unchanged scene)
        capture4 = self.show_base.render_single_RGBB_frame()
        self.assertTrue(np.allclose(capture3.rgbs, capture4.rgbs, atol=1e-5))
        self.assertTrue(np.allclose(capture3.zbufs, capture4.zbufs, atol=1e-5))

        # important objects
        target_mesh_obj = self.show_base.get_target_obj_mesh_path()
        viewpoints_mesh_obj = self.show_base.get_viewpoints_obj_mesh_path()

        self.assertFalse(target_mesh_obj.isHidden())
        self.assertTrue(viewpoints_mesh_obj.isHidden())

    def test_loading_from_invalid_gltf_file(self):
        with self.assertRaises(Exception):
            self.show_base.load_world_from_blender(Path(__file__) / "j.glb")

    def test_get_cam_lens(self):
        lens = self.show_base.get_cam_lens()

        self.assertIsInstance(lens, PerspectiveLens)

    def test_tonemapping_application_on_captured_images(self):
        captured1 = self.show_base.render_single_RGBB_frame()
        self.show_base.set_tonemapping_exposure_for_tests(1000)
        captured2 = self.show_base.render_single_RGBB_frame()

        self.assertFalse(np.allclose(captured1.rgbs, captured2.rgbs, atol=1e-6))

    def test_conf_shadow_blur(self):
        rgb_a, rgb_b = self.get_rgbs_for_configs(
            config_a_patch={"shadow_blur": 0.0},
            config_b_patch={"shadow_blur": 1.0},
        )

        self.assertFalse(np.allclose(rgb_a, rgb_b, atol=1e-6))

    def test_conf_shadow_map_resolution(self):
        rgb_a, rgb_b = self.get_rgbs_for_configs(
            config_a_patch={"shadow_map_resolution": 25},
            config_b_patch={"shadow_map_resolution": 1024},
        )

        self.assertFalse(np.allclose(rgb_a, rgb_b, atol=1e-6))

    def test_conf_dir_ambient_hpr_diff(self):
        rgb_a, rgb_b = self.get_rgbs_for_configs(
            config_a_patch={"dir_ambient_hpr_diff": 0},
            config_b_patch={"dir_ambient_hpr_diff": 45},
        )

        self.assertFalse(np.allclose(rgb_a, rgb_b, atol=1e-6))

    def test_conf_dir_ambient_hpr_weight(self):
        rgb_a, rgb_b = self.get_rgbs_for_configs(
            config_a_patch={"dir_ambient_hpr_weight": 0.1},
            config_b_patch={"dir_ambient_hpr_weight": 0.7},
        )

        self.assertFalse(np.allclose(rgb_a, rgb_b, atol=1e-6))

    def test_conf_nondir_ambient_light_r(self):
        rgb_a, rgb_b = self.get_rgbs_for_configs(
            config_a_patch={"nondir_ambient_light_r": 0.1},
            config_b_patch={"nondir_ambient_light_r": 100.0},
        )

        mean_chan_a = float(idx_im_rgbs(rgb_a, c="r").mean())
        mean_chan_b = float(idx_im_rgbs(rgb_b, c="r").mean())

        self.assertGreater(mean_chan_b, mean_chan_a)

    def test_conf_nondir_ambient_light_g(self):
        rgb_a, rgb_b = self.get_rgbs_for_configs(
            config_a_patch={"nondir_ambient_light_g": 0.1},
            config_b_patch={"nondir_ambient_light_g": 100.0},
        )

        mean_chan_a = float(idx_im_rgbs(rgb_a, c="g").mean())
        mean_chan_b = float(idx_im_rgbs(rgb_b, c="g").mean())

        self.assertGreater(mean_chan_b, mean_chan_a)

    def test_conf_nondir_ambient_light_b(self):
        rgb_a, rgb_b = self.get_rgbs_for_configs(
            config_a_patch={"nondir_ambient_light_b": 0.1},
            config_b_patch={"nondir_ambient_light_b": 100.0},
        )

        mean_chan_a = float(idx_im_rgbs(rgb_a, c="b").mean())
        mean_chan_b = float(idx_im_rgbs(rgb_b, c="b").mean())

        self.assertGreater(mean_chan_b, mean_chan_a)

    def test_conf_shadow_area_root(self):
        rgb_a, rgb_b = self.get_rgbs_for_configs(
            config_a_patch={"shadow_area_root": "BoundingObj"},
            config_b_patch={"shadow_area_root": "FarOutObj"},
        )

        rgb_a_mean = float(rgb_a.mean())
        rgb_b_mean = float(rgb_b.mean())

        self.assertLess(rgb_a_mean, rgb_b_mean)

    def test_conf_force_shadow_0(self):
        rgb_a, rgb_b = self.get_rgbs_for_configs(
            config_a_patch={"force_shadow_0": ["BoundingObj"]},
            config_b_patch={"force_shadow_0": []},
        )

        rgb_a_mean = float(rgb_a.mean())
        rgb_b_mean = float(rgb_b.mean())

        self.assertLess(rgb_a_mean, rgb_b_mean)

    def get_rgbs_for_configs(
        self,
        config_a_patch: dict[str, Any],
        config_b_patch: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Render the test scene with two different set of changes on the scene configuration.

        Parameters
        ----------
        config_a_patch
            The first configuration patch.
        config_a_patch
            The second configuration patch.

        Returns
        -------
        rgb_a
            Format: ``Im::RGBs``
        rgb_b
            Format: ``Im::RGBs``
        """
        self.show_base.destroy()
        return _get_rgbs_for_configs(
            config_a_patch=config_a_patch,
            config_b_patch=config_b_patch,
            test_scene_path=self.TEST_SCENE_PATH,
        )

    def test_get_standard_scene_format_errors_happy_path(self):
        loading_errors = self.show_base.load_world_from_blender(self.TEST_SCENE_PATH)
        self.assertEqual(len(loading_errors), 0)

        standard_scene_format_errors = self.show_base.get_standard_scene_format_errors()
        self.assertEqual(len(standard_scene_format_errors), 0)

    def test_get_standard_scene_format_errors_target_obj_mesh_not_found(self):
        errors = self.show_base.load_world_from_blender(self.TEST_SCENE_PATH)
        self.assertEqual(len(errors), 0)

        self.show_base.get_target_obj_mesh_path().name = "mesh2"
        standard_scene_format_errors = self.show_base.get_standard_scene_format_errors()
        self.assertGreater(len(standard_scene_format_errors), 0)

    def test_get_standard_scene_format_errors_viewpoints_mesh_not_found(self):
        errors = self.show_base.load_world_from_blender(self.TEST_SCENE_PATH)
        self.assertEqual(len(errors), 0)

        self.show_base.get_viewpoints_obj_mesh_path().name = "mesh2"
        standard_scene_format_errors = self.show_base.get_standard_scene_format_errors()
        self.assertGreater(len(standard_scene_format_errors), 0)

    def test_get_standard_scene_format_errors_viewpoints_moved(self):
        cases = [
            ("x", [1000, 0, 0]),
            ("y", [0, 1000, 0]),
            ("z", [0, 0, 1000]),
        ]
        self.show_base.destroy()
        for case_name, new_obj_pos in cases:
            with self.subTest(case_name):
                base = Panda3dShowBase(offscreen=True, win_size=TwoDSize(800, 600))
                try:
                    errors = base.load_world_from_blender(self.TEST_SCENE_PATH)
                    self.assertEqual(len(errors), 0)

                    base.get_viewpoints_obj_mesh_path().setPos(*new_obj_pos)
                    standard_scene_format_errors = (
                        base.get_standard_scene_format_errors()
                    )
                    self.assertGreater(len(standard_scene_format_errors), 0)
                finally:
                    base.destroy()

    def test_get_standard_scene_format_errors_viewpoints_obj_mesh_moved(self):
        cases = [
            ("x", [1000, 0, 0]),
            ("y", [0, 1000, 0]),
            ("z", [0, 0, 1000]),
        ]
        self.show_base.destroy()
        for case_name, new_obj_pos in cases:
            with self.subTest(case_name):
                base = Panda3dShowBase(offscreen=True, win_size=TwoDSize(800, 600))
                try:
                    errors = base.load_world_from_blender(self.TEST_SCENE_PATH)
                    self.assertEqual(len(errors), 0)

                    base.get_viewpoints_obj_mesh_path().setPos(*new_obj_pos)
                    standard_scene_format_errors = (
                        base.get_standard_scene_format_errors()
                    )
                    self.assertGreater(len(standard_scene_format_errors), 0)
                finally:
                    base.destroy()

    def test_get_standard_scene_format_errors_target_obj_mesh_moved(self):
        cases = [
            ("x", [1000, 0, 0]),
            ("y", [0, 1000, 0]),
            ("z", [0, 0, 1000]),
        ]
        self.show_base.destroy()
        for case_name, new_obj_pos in cases:
            with self.subTest(case_name):
                base = Panda3dShowBase(offscreen=True, win_size=TwoDSize(800, 600))
                try:
                    errors = base.load_world_from_blender(self.TEST_SCENE_PATH)
                    self.assertEqual(len(errors), 0)

                    base.get_target_obj_mesh_path().setPos(*new_obj_pos)
                    standard_scene_format_errors = (
                        base.get_standard_scene_format_errors()
                    )
                    self.assertGreater(len(standard_scene_format_errors), 0)
                finally:
                    base.destroy()

    def test_get_standard_scene_format_errors_target_obj_parent_moved(self):
        errors = self.show_base.load_world_from_blender(self.TEST_SCENE_PATH)
        self.assertEqual(len(errors), 0)

        target_obj = self.show_base.get_target_obj_mesh_path().parent
        assert target_obj is not None
        target_obj_parent = target_obj.parent
        assert target_obj_parent is not None

        target_obj_parent.setPos(5, 0, 0)
        standard_scene_format_errors = self.show_base.get_standard_scene_format_errors()
        self.assertGreater(len(standard_scene_format_errors), 0)

    def test_get_standard_scene_too_big(self):
        cases = [
            ("x", [1000, 0, 0]),
            ("y", [0, 1000, 0]),
            ("z", [0, 0, 1000]),
        ]
        self.show_base.destroy()
        for case_name, new_obj_pos in cases:
            with self.subTest(case_name):
                base = Panda3dShowBase(offscreen=True, win_size=TwoDSize(800, 600))
                try:
                    loading_errors = base.load_world_from_blender(self.TEST_SCENE_PATH)
                    self.assertEqual(len(loading_errors), 0)
                    extra_obj = load_xsortable_octagon_panda3d(base)
                    assert base.world_model is not None
                    extra_obj.reparentTo(base.world_model)

                    extra_obj.setPos(*new_obj_pos)

                    standard_scene_format_errors = (
                        base.get_standard_scene_format_errors()
                    )
                    self.assertGreater(len(standard_scene_format_errors), 0)
                finally:
                    base.destroy()


def _get_rgbs_for_configs(
    config_a_patch: dict[str, Any],
    config_b_patch: dict[str, Any],
    test_scene_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Render the test scene with two different set of changes on the scene configuration.

    Parameters
    ----------
    config_a_patch
        The first configuration patch.
    config_a_patch
        The second configuration patch.

    Returns
    -------
    rgb_a
        Format: ``Im::RGBs``
    rgb_b
        Format: ``Im::RGBs``
    """

    rgb_a = _render_rgb_with_customized_config(
        test_scene_path=test_scene_path, config_patch=config_a_patch
    )
    rgb_b = _render_rgb_with_customized_config(
        test_scene_path=test_scene_path, config_patch=config_b_patch
    )

    return rgb_a, rgb_b


def _render_rgb_with_customized_config(
    test_scene_path: Path, config_patch: dict[str, Any]
):
    """
    Render the test scene with the specified changes on the scene configuration.

    Parameters
    ----------
    config_patch
        The configuration patch.

    Returns
    -------
    v
        Format: ``Im::RGBs``
    """
    test_conf_path = test_scene_path.with_suffix(".json")
    orig_conf = json.loads(test_conf_path.read_text())

    modified_conf = cast(dict[str, Any], orig_conf.copy())
    for key, value in config_patch.items():
        modified_conf[key] = value
    assert len(get_scene_config_errors(modified_conf)) == 0
    modified_conf = cast(SceneConfigDict, modified_conf.copy())

    showbase = Panda3dShowBase(offscreen=True, win_size=TwoDSize(500, 500))
    try:
        showbase.load_world_from_blender(
            test_scene_path, scene_config_override=modified_conf
        )
        showbase.set_cam_pos_and_look_at(
            new_cam_pos=ThreeDPoint(4, 4, 4), look_at=ThreeDPoint(0, 0, 0)
        )

        cam_pos = showbase.cam.getPos()
        cam_hpr = showbase.cam.getHpr()

        rgbs = showbase.render_single_RGBB_frame().rgbs
    finally:
        showbase.destroy()
    return rgbs
