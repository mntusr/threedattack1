import itertools
import math
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from turtle import fd
from typing import TYPE_CHECKING, Any, Callable, Sequence
from unittest import mock

import numpy as np

from threedattack import (
    Scene,
    SceneConfig,
    SceneSamples,
    calc_aggr_delta_loss_dict_from_losses_on_scene,
    calc_raw_loss_values_on_scene,
)
from threedattack._typing import type_instance
from threedattack.dataset_model import (
    DepthsWithMasks,
    ExactSampleCounts,
    RGBsWithDepthsAndMasks,
    SampleType,
    SampleTypeError,
)
from threedattack.losses import (
    LossDerivationMethod,
    LossPrecision,
    RawLossFn,
    cropped_d1_loss,
    cropped_log10_loss,
    cropped_rmse_loss,
    d1_loss,
    log10_loss,
    rmse_loss,
)
from threedattack.rendering_core import (
    DesiredViewpointCounts,
    MeshBasedObjectTransform,
    ObjectTransformResult,
    ObjectTransformType,
    Panda3dShowBase,
    PointBasedVectorFieldSpec,
    ThreeDPoint,
    TwoDAreas,
    TwoDSize,
    ViewpointSplit,
    VolumeBasedObjectTransform,
    find_node,
    get_bounding_rectangle_on_screen,
    get_vertex_positions_copy,
    get_viewpoints_path_for_world,
    imshow,
    load_model_from_local_file,
)
from threedattack.target_model import AlignmentFunction, AsyncDepthPredictor
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *


class TestScene(unittest.TestCase):
    def setUp(self):
        self.TEST_SCENE_PATH = (
            Path(__file__).resolve().parent.parent / "test_resources" / "test_scene.glb"
        )
        self.WINDOW_SIZE = TwoDSize(500, 500)
        self.depth_cap = 13

        self.show_base = Panda3dShowBase(offscreen=True, win_size=self.WINDOW_SIZE)

        self.show_base.load_world_from_blender(self.TEST_SCENE_PATH)
        split_path = get_viewpoints_path_for_world(self.TEST_SCENE_PATH)
        self.split_for_scene = ViewpointSplit.load_npz(split_path)
        self.actual_viewpoint_counts_for_scene = DesiredViewpointCounts(
            n_train_samples=4,
            n_test_samples=5,
            n_val_samples=6,
        )

        target_obj_pos_panda3d = (
            self.show_base.get_target_obj_mesh_path().getParent().getPos()
        )
        self.target_obj_pos = ThreeDPoint(
            x=target_obj_pos_panda3d.x,
            y=target_obj_pos_panda3d.y,
            z=target_obj_pos_panda3d.z,
        )
        self.OBJECT_TRANSFORM_TYPE = ObjectTransformType.MeshBased

        self.scene = Scene(
            self.show_base,
            viewpoint_split=self.split_for_scene,
            viewpt_counts=self.actual_viewpoint_counts_for_scene,
            rendering_resolution=self.WINDOW_SIZE,
            world_path=self.TEST_SCENE_PATH,
            n_volume_sampling_steps_along_shortest_axis=40,
            object_transform_type=self.OBJECT_TRANSFORM_TYPE,
            target_obj_field_cache=mock.Mock(),
            target_size_multiplier=1.01,
            depth_cap=self.depth_cap,
        )
        a = 2

    def tearDown(self) -> None:
        self.show_base.destroy()

    def test_get_transform_type(self):
        self.assertEqual(self.OBJECT_TRANSFORM_TYPE, self.scene.get_transform_type())

    def test_viewpt_counts_kept(self):
        expected_n_viewpoints_for_types = {
            SampleType.Train: self.actual_viewpoint_counts_for_scene.n_train_samples,
            SampleType.Test: self.actual_viewpoint_counts_for_scene.n_test_samples,
            SampleType.Val: self.actual_viewpoint_counts_for_scene.n_val_samples,
        }
        for viewpt_type in SampleType:
            with self.subTest(f"{viewpt_type=}"):
                actual_n_viewpoints = self.scene.get_n_samples_for_type(viewpt_type)
                expected_n_viewpoints = expected_n_viewpoints_for_types[viewpt_type]

                self.assertEqual(actual_n_viewpoints, expected_n_viewpoints)

    def test_get_n_available_viewpoints(self):
        self.assertEqual(
            self.scene.get_n_samples_for_type(SampleType.Train),
            self.actual_viewpoint_counts_for_scene.n_train_samples,
        )
        self.assertEqual(
            self.scene.get_n_samples_for_type(SampleType.Test),
            self.actual_viewpoint_counts_for_scene.n_test_samples,
        )
        self.assertEqual(
            self.scene.get_n_samples_for_type(SampleType.Val),
            self.actual_viewpoint_counts_for_scene.n_val_samples,
        )

    def test_get_samples_array_masking_correctness(self):
        sample = self.scene.get_samples([0], SampleType.Train)
        mask_at_top_left = bool(idx_im(sample.rgbds.masks, w=0, h=0))
        mask_at_center = bool(
            idx_im(
                sample.rgbds.masks,
                w=sample.rgbds.masks.shape[DIM_IM_W] // 2,
                h=sample.rgbds.masks.shape[DIM_IM_H] // 2,
            )
        )

        self.assertFalse(mask_at_top_left)
        self.assertTrue(mask_at_center)

    def test_get_samples_array_format_correctness(self):
        sample = self.scene.get_samples([0], SampleType.Train)
        self.assertTrue(match_im_rgbs(sample.rgbds.rgbs, shape={"n": 1}))
        self.assertTrue(match_im_depthmasks(sample.rgbds.masks, shape={"n": 1}))
        self.assertTrue(match_im_depthmaps(sample.rgbds.depths, shape={"n": 1}))

    def test_get_samples_slice_indexing_equivalence(self):
        samples1 = self.scene.get_samples([0, 2], SampleType.Train)
        samples2 = self.scene.get_samples(slice(0, 3, 2), SampleType.Train)

        self.assertTrue(
            np.allclose(samples1.rgbds.rgbs, samples2.rgbds.rgbs, atol=1e-5)
        )
        self.assertTrue(
            np.allclose(samples1.rgbds.depths, samples2.rgbds.depths, atol=1e-5)
        )
        self.assertTrue(np.array_equal(samples1.rgbds.masks, samples2.rgbds.masks))
        self.assertTrue(
            np.array_equal(
                samples1.target_obj_areas_on_screen.x_mins_including,
                samples2.target_obj_areas_on_screen.x_mins_including,
            )
        )
        self.assertTrue(
            np.array_equal(
                samples1.target_obj_areas_on_screen.x_maxes_excluding,
                samples2.target_obj_areas_on_screen.x_maxes_excluding,
            )
        )
        self.assertTrue(
            np.array_equal(
                samples1.target_obj_areas_on_screen.y_mins_including,
                samples2.target_obj_areas_on_screen.y_mins_including,
            )
        )
        self.assertTrue(
            np.array_equal(
                samples1.target_obj_areas_on_screen.y_maxes_excluding,
                samples2.target_obj_areas_on_screen.y_maxes_excluding,
            )
        )

    def test_get_samples_depth_correctness(self):
        self.scene.destroy_showbase()
        resolution = TwoDSize(800, 600)
        max_depth = 15
        distances = [5, 8, 2, max_depth + 1]
        for expected_distance in distances:
            with self.subTest(f"{expected_distance=}"):
                base = Panda3dShowBase(offscreen=True, win_size=resolution)
                try:
                    # a series of hacks to create a fake scene
                    # this only fakes the parts of a scene that are actually
                    # needed by the tested class
                    cube_obj_world_like = load_model_from_local_file(
                        base, Path("test_resources/cube_o0_w2.obj")
                    )
                    assert cube_obj_world_like is not None
                    cube_obj_world_like.getChild(0).getChild(0).setName("TargetMesh")
                    base.world_model = cube_obj_world_like
                    base.world_model.reparent_to(base.render)
                    base.get_standard_scene_format_errors = (
                        lambda: []
                    )  # patch out the scene format checks

                    # create the tested object
                    scene = Scene(
                        show_base=base,
                        n_volume_sampling_steps_along_shortest_axis=20,
                        object_transform_type=ObjectTransformType.MeshBased,
                        rendering_resolution=resolution,
                        target_obj_field_cache=mock.Mock(),
                        target_size_multiplier=1.01,
                        viewpoint_split=ViewpointSplit(
                            train_viewpoints=np.array(
                                [
                                    [expected_distance + 1, 0, 0],
                                ]
                            ),
                            test_viewpoints=np.array(
                                [
                                    [6, 0, 0],
                                ]
                            ),
                            val_viewpoints=np.array(
                                [
                                    [7, 0, 0],
                                ]
                            ),
                        ),
                        viewpt_counts=DesiredViewpointCounts(None, None, None),
                        world_path=mock.Mock(),
                        depth_cap=max_depth,
                    )

                    # render and check results
                    sample = scene.get_samples([0], SampleType.Train)
                    if expected_distance < max_depth:
                        masked_min = float(
                            sample.rgbds.depths[sample.rgbds.masks].min()
                        )
                        masked_max = float(
                            sample.rgbds.depths[sample.rgbds.masks].max()
                        )

                        self.assertAlmostEqual(masked_min, expected_distance, places=3)
                        self.assertAlmostEqual(masked_max, expected_distance, places=3)
                    else:
                        self.assertEqual(sample.rgbds.masks.astype(np.int32).sum(), 0)
                finally:
                    base.destroy()

    def test_get_samples_index_dependence(self):
        VIEWPT_TYPE = SampleType.Train
        samples1 = self.scene.get_samples([0, 1], VIEWPT_TYPE)
        samples2 = self.scene.get_samples([1, 0], VIEWPT_TYPE)
        self.assertFalse(
            np.allclose(samples1.rgbds.rgbs, samples2.rgbds.rgbs, atol=1e-4)
        )

    def test_get_samples_type_dependence(self):
        VIEWPT_INDICES = [0, 1]
        samples1 = self.scene.get_samples(VIEWPT_INDICES, SampleType.Train)
        samples2 = self.scene.get_samples(VIEWPT_INDICES, SampleType.Test)
        samples3 = self.scene.get_samples(VIEWPT_INDICES, SampleType.Val)
        self.assertFalse(
            np.allclose(samples1.rgbds.rgbs, samples2.rgbds.rgbs, atol=1e-4)
        )
        self.assertFalse(
            np.allclose(samples1.rgbds.rgbs, samples3.rgbds.rgbs, atol=1e-4)
        )
        self.assertFalse(
            np.allclose(samples2.rgbds.rgbs, samples3.rgbds.rgbs, atol=1e-4)
        )

    def test_set_cam_pos_happy_path_index_dependence(self):
        cam_xyz_before = self.show_base.get_cam_xyz()
        self.scene._select_viewpoint(1, SampleType.Train)
        cam_xyz_after = self.show_base.get_cam_xyz()

        self.assertFalse(cam_xyz_before.is_almost_equal(cam_xyz_after, 1e-5))

    def test_set_cam_pos_happy_path_type_dependence(self):
        VIEWPOINT_IDX = 1
        self.scene._select_viewpoint(VIEWPOINT_IDX, SampleType.Train)
        cam_xyz_before = self.show_base.get_cam_xyz()
        self.scene._select_viewpoint(VIEWPOINT_IDX, SampleType.Val)
        cam_xyz_after = self.show_base.get_cam_xyz()

        self.assertFalse(cam_xyz_before.is_almost_equal(cam_xyz_after, 1e-5))

    def test_get_samples_invalid_index(self):
        with self.assertRaises(IndexError):
            self.scene.get_samples([10000], SampleType.Train)

    def test_get_samples_invalid_sample_type(self):
        self.scene.destroy_showbase()
        scene = Scene(
            self.show_base,
            viewpoint_split=self.split_for_scene,
            viewpt_counts=DesiredViewpointCounts(
                n_train_samples=None,
                n_test_samples=0,
                n_val_samples=None,
            ),
            rendering_resolution=self.WINDOW_SIZE,
            world_path=self.TEST_SCENE_PATH,
            n_volume_sampling_steps_along_shortest_axis=40,
            object_transform_type=self.OBJECT_TRANSFORM_TYPE,
            target_obj_field_cache=mock.Mock(),
            target_size_multiplier=1.01,
            depth_cap=self.depth_cap,
        )
        try:
            with self.assertRaises(SampleTypeError):
                scene.get_samples([0], SampleType.Test)
        finally:
            scene.destroy_showbase()

    def test_get_sample_get_samples_equivalence(self):
        sample_indices = [0, 1, 2]
        sample_type = SampleType.Train

        samples1 = self.scene.get_samples(sample_indices, sample_type)
        sample_list = [
            self.scene.get_sample(sample_idx, sample_type)
            for sample_idx in sample_indices
        ]

        self.assertEqual(len(sample_list), samples1.rgbds.rgbs.shape[DIM_IM_N])

        for i in range(len(sample_list)):
            rgbd1 = samples1.rgbds[[i]]
            rgbd2 = sample_list[i].rgbds

            self.assertTrue(np.allclose(rgbd1.rgbs, rgbd2.rgbs, atol=1e-5))
            self.assertTrue(np.allclose(rgbd1.depths, rgbd2.depths, atol=1e-5))
            self.assertTrue(np.array_equal(rgbd1.masks, rgbd2.masks))

            area1 = samples1.target_obj_areas_on_screen.idx_areas(i)
            area2 = sample_list[i].target_obj_areas_on_screen

            self.assertEqual(int(area1.x_mins_including), int(area2.x_mins_including))
            self.assertEqual(int(area1.x_maxes_excluding), int(area2.x_maxes_excluding))
            self.assertEqual(int(area1.y_mins_including), int(area2.y_mins_including))
            self.assertEqual(int(area1.y_maxes_excluding), int(area2.y_maxes_excluding))

    def test_temporary_target_transform(self):
        set_target_transform_mock = mock.Mock(name="Scene.set_target_transform")
        vector_field_spec_mock = mock.Mock(name="vector_field_spec_mock")
        # a simple hack to simplify testing
        self.scene.set_target_transform = set_target_transform_mock
        with self.scene.temporary_target_transform(vector_field_spec_mock):
            self.assertEqual(len(set_target_transform_mock.mock_calls), 1)
        self.assertEqual(len(set_target_transform_mock.mock_calls), 2)
        self.assertIs(
            set_target_transform_mock.mock_calls[0].args[0], vector_field_spec_mock
        )
        self.assertIs(set_target_transform_mock.mock_calls[1].args[0], None)

    def test_get_depth_cap(self):
        actual_depth_cap = self.scene.get_depth_cap()
        self.assertAlmostEqual(actual_depth_cap, self.depth_cap)

    def test_get_target_obj_area_on_screen(self):
        actual_area = self.scene._get_target_obj_area_on_screen()

        corners = (
            self.scene.get_target_areas()
            .get_full_area(
                include_outside_offset=False, origin_of_obj=self.target_obj_pos
            )
            .get_corners()
        )

        expected_area = get_bounding_rectangle_on_screen(
            base=self.show_base,
            points=corners,
            rendering_resolution=self.WINDOW_SIZE,
        )

        self.assertTrue(
            np.allclose(
                actual_area.x_maxes_excluding,
                expected_area.x_maxes_excluding,
                atol=1e-4,
            )
        )

        self.assertTrue(
            np.allclose(
                actual_area.y_maxes_excluding,
                expected_area.y_maxes_excluding,
                atol=1e-4,
            )
        )

        self.assertTrue(
            np.allclose(
                actual_area.y_mins_including,
                expected_area.y_mins_including,
                atol=1e-4,
            )
        )

        self.assertTrue(
            np.allclose(
                actual_area.x_mins_including,
                expected_area.x_mins_including,
                atol=1e-4,
            )
        )

    def test_get_samples_target_obj_areas_correctness(self):
        viewpt_idx_list = [1, 3, 0]
        for viewpt_type in SampleType:
            start_cam_pos = self.show_base.get_cam_xyz()
            samples = self.scene.get_samples(viewpt_idx_list, viewpt_type)
            end_cam_pos = self.show_base.get_cam_xyz()

            self.assertAlmostEqual(start_cam_pos.x, end_cam_pos.x)
            self.assertAlmostEqual(start_cam_pos.y, end_cam_pos.y)
            self.assertAlmostEqual(start_cam_pos.z, end_cam_pos.z)

            for i in range(len(viewpt_idx_list)):
                actual_area = samples.target_obj_areas_on_screen.idx_areas([i])

                self.scene._select_viewpoint(viewpt_idx_list[i], viewpt_type)
                expected_area = self.scene._get_target_obj_area_on_screen()

                self.assertTrue(
                    np.array_equal(
                        actual_area.x_maxes_excluding, expected_area.x_maxes_excluding
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        actual_area.x_mins_including, expected_area.x_mins_including
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        actual_area.y_maxes_excluding, expected_area.y_maxes_excluding
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        actual_area.y_mins_including, expected_area.y_mins_including
                    )
                )

    def test_live_preview_then_quit(self):
        new_showbase_mock = mock.Mock(name="new_showbase_instance")
        panda3d_showbase_class_mock = mock.Mock(
            name="panda3d_showbase_class", return_value=new_showbase_mock
        )
        live_preview_controller_class_mock = mock.Mock(
            name="live_preview_controller_class"
        )
        vector_field_spec_mock = mock.Mock(name="vector_field_spec")
        self.scene._target_object_transform = mock.Mock(name="target_object_transform")
        put_obj_mock = mock.Mock(name="put_obj")
        with mock.patch(
            "threedattack._scene.Panda3dShowBase", panda3d_showbase_class_mock
        ):
            with mock.patch(
                "threedattack._scene.LivePreviewController",
                live_preview_controller_class_mock,
            ):
                with mock.patch("threedattack._scene.put_obj", put_obj_mock):
                    with mock.patch("sys.exit", mock.Mock()):
                        self.scene.set_target_transform(vector_field_spec_mock)
                        self.scene.live_preview_then_quit()
                        self.assertIsNone(self.scene._show_base)
                        panda3d_showbase_class_mock.assert_called_once()
                        self.assertFalse(
                            panda3d_showbase_class_mock.call_args.kwargs["offscreen"]
                        )
                        live_preview_controller_class_mock.attach.assert_called()
                        self.assertIs(
                            live_preview_controller_class_mock.attach.call_args.kwargs[
                                "base"
                            ],
                            new_showbase_mock,
                        )
                        self.scene._target_object_transform.transform_obj_new.assert_called()
                        new_showbase_mock.run.assert_called()
                        self.assertEqual(put_obj_mock.call_count, 2)

    def test_from_config_or_error_happy_path(self):
        self.scene.destroy_showbase()
        depth_cap = 7.31
        n_volume_sampling_steps_along_shortest_axis = 13
        object_transform_type = ObjectTransformType.MeshBased
        resolution = TwoDSize(x_size=63, y_size=18)
        target_size_multiplier = 3.16
        desired_viewpt_counts = DesiredViewpointCounts(
            n_train_samples=9, n_test_samples=0, n_val_samples=21
        )
        world_path = self.scene.get_world_path()

        transform_control_points = (
            self.scene.get_target_areas()
            .get_full_area(origin_of_obj=None, include_outside_offset=False)
            .get_corners()
        )
        transform_vectors = np.full_like(transform_control_points, 0.3)

        transform_specs: list[PointBasedVectorFieldSpec | None] = [
            PointBasedVectorFieldSpec(
                control_points=transform_control_points,
                vectors=transform_vectors,
            ),
            None,
        ]

        for expected_applied_transform in transform_specs:
            scene = Scene.from_config_or_error(
                SceneConfig(
                    applied_transform=expected_applied_transform,
                    depth_cap=depth_cap,
                    n_volume_sampling_steps_along_shortest_axis=n_volume_sampling_steps_along_shortest_axis,
                    object_transform_type=object_transform_type,
                    resolution=resolution,
                    target_size_multiplier=target_size_multiplier,
                    viewpt_counts=desired_viewpt_counts,
                    world_path=world_path,
                )
            )
            self.assertIsInstance(scene, Scene)
            assert isinstance(scene, Scene)
            try:
                self.assertAlmostEqual(scene.get_depth_cap(), depth_cap)
                if isinstance(
                    scene._target_object_transform, VolumeBasedObjectTransform
                ):
                    self.assertEqual(
                        scene._target_object_transform.get_n_steps_along_shortest_axis(),
                        n_volume_sampling_steps_along_shortest_axis,
                    )
                self.assertEqual(
                    scene._target_object_transform.get_transform_type(),
                    object_transform_type,
                )
                self.assertEqual(scene.get_rendering_resolution(), resolution)
                self.assertEqual(
                    scene.get_target_areas().get_size_multiplier(),
                    target_size_multiplier,
                )
                self.assertEqual(scene.get_world_path(), world_path)

                if expected_applied_transform is not None:
                    actual_applied_transform = (
                        scene.get_applied_transform_vector_field_deepcopy()
                    )
                    self.assertIsNotNone(actual_applied_transform)
                    assert actual_applied_transform is not None
                    self.assertTrue(
                        np.allclose(
                            actual_applied_transform.control_points,
                            expected_applied_transform.control_points,
                            atol=1e-4,
                        )
                    )
                    self.assertGreater(scene.get_transform_change_amount_score(), 0.01)
                else:
                    actual_applied_transform = (
                        scene.get_applied_transform_vector_field_deepcopy()
                    )
                    self.assertIsNone(actual_applied_transform)
                    self.assertLess(scene.get_transform_change_amount_score(), 0.01)
            finally:
                scene.destroy_showbase()

    def test_reload_path(self):
        viewpt_idx = 1
        sample_type = SampleType.Train
        with TemporaryDirectory() as td:
            td_path = Path(td)
            config_path = td_path / "testscene.scene"
            target_obj_area_corners = (
                self.scene.get_target_areas()
                .get_full_area(origin_of_obj=None, include_outside_offset=False)
                .get_corners()
            )
            transform_spec_vectors = np.full_like(target_obj_area_corners, 0.1)
            transform_spec = PointBasedVectorFieldSpec(
                control_points=target_obj_area_corners, vectors=transform_spec_vectors
            )
            self.scene.set_target_transform(transform_spec)
            self.scene.save(config_path)
            rgbds_before = self.scene.get_sample(viewpt_idx, sample_type)
            self.scene.destroy_showbase()

            reloaded_scene = Scene.load(config_path)
            try:
                rgbds_after = reloaded_scene.get_sample(viewpt_idx, sample_type)

                self.assertTrue(
                    np.allclose(
                        rgbds_before.rgbds.rgbs, rgbds_after.rgbds.rgbs, atol=1e-6
                    )
                )
                self.assertTrue(
                    np.allclose(
                        rgbds_before.rgbds.depths, rgbds_after.rgbds.depths, atol=1e-6
                    )
                )
                self.assertTrue(
                    np.array_equal(rgbds_before.rgbds.masks, rgbds_after.rgbds.masks)
                )
            finally:
                reloaded_scene.destroy_showbase()

    def test_reload_textio(self):
        viewpt_idx = 1
        sample_type = SampleType.Train
        scene_config_target_io = StringIO()

        target_obj_area_corners = (
            self.scene.get_target_areas()
            .get_full_area(origin_of_obj=None, include_outside_offset=False)
            .get_corners()
        )
        transform_spec_vectors = np.full_like(target_obj_area_corners, 0.1)
        transform_spec = PointBasedVectorFieldSpec(
            control_points=target_obj_area_corners, vectors=transform_spec_vectors
        )
        self.scene.set_target_transform(transform_spec)
        self.scene.save(scene_config_target_io)
        rgbds_before = self.scene.get_sample(viewpt_idx, sample_type)
        self.scene.destroy_showbase()

        scene_config_str = scene_config_target_io.getvalue()
        scene_config_source_io = StringIO(scene_config_str)
        reloaded_scene = Scene.load(scene_config_source_io)
        try:
            rgbds_after = reloaded_scene.get_sample(viewpt_idx, sample_type)

            self.assertTrue(
                np.allclose(rgbds_before.rgbds.rgbs, rgbds_after.rgbds.rgbs, atol=1e-6)
            )
            self.assertTrue(
                np.allclose(
                    rgbds_before.rgbds.depths, rgbds_after.rgbds.depths, atol=1e-6
                )
            )
            self.assertTrue(
                np.array_equal(rgbds_before.rgbds.masks, rgbds_after.rgbds.masks)
            )
        finally:
            reloaded_scene.destroy_showbase()

    def test_set_target_transform_happy_path(self):
        transformed_obj_mock = mock.Mock()
        transformed_obj_node_mock = mock.Mock()
        transformed_obj_mock.node = mock.Mock(return_value=transformed_obj_node_mock)
        TRANSFORMED_OBJ_N_BODIES = 5
        CHANGE_AMOUNT_SCORE = 0.4

        def transform_obj_new(*args, **kwargs):
            return ObjectTransformResult(
                new_obj=transformed_obj_mock,
                n_bodies=TRANSFORMED_OBJ_N_BODIES,
                change_amount_score=CHANGE_AMOUNT_SCORE,
            )

        transform_mock = mock.Mock()
        transform_mock.transform_obj_new = mock.Mock(side_effect=transform_obj_new)
        self.scene._target_object_transform = transform_mock

        put_obj_mock = mock.Mock()

        self.assertFalse(self.scene.is_transformed())

        with mock.patch("threedattack._scene.put_obj", put_obj_mock):
            self.scene.set_target_transform(mock.Mock())
            put_obj_mock.assert_called_once()
            self.assertIs(
                put_obj_mock.call_args.kwargs["new_obj"], transformed_obj_node_mock
            )
            self.assertEqual(self.scene.get_target_n_bodies(), TRANSFORMED_OBJ_N_BODIES)
            self.assertAlmostEqual(
                self.scene.get_transform_change_amount_score(),
                CHANGE_AMOUNT_SCORE,
                places=4,
            )
            self.assertTrue(self.scene.is_transformed())

            self.scene.set_target_transform(None)
            self.assertFalse(self.scene.is_transformed())
            self.assertEqual(put_obj_mock.call_count, 2)

    def test_set_target_transform_already_transformed(self):
        control_points = (
            self.scene.get_target_areas()
            .get_full_area(origin_of_obj=None, include_outside_offset=False)
            .get_corners()
        )
        vectors = np.full_like(control_points, 0.3)
        transform_spec = PointBasedVectorFieldSpec(
            control_points=control_points, vectors=vectors
        )
        self.scene.set_target_transform(transform_spec)
        with self.assertRaises(RuntimeError):
            self.scene.set_target_transform(transform_spec)

    def test_get_config_deepcopy(self):
        transform_specs: list[PointBasedVectorFieldSpec | None] = [
            PointBasedVectorFieldSpec(
                control_points=np.arange(0, 30)
                .reshape(newshape_points_space(n=10))
                .astype(np.float32)
                / 10,
                vectors=np.arange(0, 30).reshape(newshape_points_space(n=10)) / 100.0,
            ),
            None,
        ]
        for transform_spec in transform_specs:
            self.scene.set_target_transform(transform_spec)
            config = self.scene.get_config_deepcopy()
            self.assertAlmostEqual(config.depth_cap, self.scene.get_depth_cap())
            self.assertAlmostEqual(
                config.target_size_multiplier,
                self.scene.get_target_areas().get_size_multiplier(),
            )
            self.assertEqual(
                config.viewpt_counts.n_train_samples,
                self.scene.get_n_samples_for_type(SampleType.Train),
            )
            self.assertEqual(
                config.viewpt_counts.n_test_samples,
                self.scene.get_n_samples_for_type(SampleType.Test),
            )
            self.assertEqual(
                config.viewpt_counts.n_val_samples,
                self.scene.get_n_samples_for_type(SampleType.Val),
            )
            self.assertEqual(
                config.object_transform_type, self.scene.get_transform_type()
            )
            self.assertEqual(config.world_path, self.scene.get_world_path())
            self.assertEqual(
                config.resolution.x_size, self.scene.get_rendering_resolution().x_size
            )
            self.assertEqual(
                config.resolution.y_size, self.scene.get_rendering_resolution().y_size
            )
            if transform_spec is None:
                self.assertIsNone(config.applied_transform)
            else:
                actual_applied_transform = config.applied_transform
                self.assertIsNotNone(actual_applied_transform)
                assert actual_applied_transform is not None
                self.assertTrue(
                    np.allclose(
                        actual_applied_transform.control_points,
                        transform_spec.control_points,
                        atol=1e-6,
                    )
                )
                self.assertTrue(
                    np.allclose(
                        actual_applied_transform.vectors,
                        transform_spec.vectors,
                        atol=1e-6,
                    )
                )


class TestSceneConfig(unittest.TestCase):
    def test_init_invalid_params(self):
        cases: list[tuple[str, float, float, int]] = [
            ("depth_cap_negative", -0.1, 1.2, 3),
            ("depth_cap_zero", 0, 1.2, 3),
            ("target_size_multiplier_less_than_1", 3.1, 0.9, 3),
            ("target_size_multiplier_equal_1", 3.1, 1, 3),
            ("n_volume_sampling_steps_along_shortest_axis_zero", 3.1, 1.3, 0),
            ("n_volume_sampling_steps_along_shortest_axis_negative", 3.1, 1.3, -3),
        ]

        for (
            case_name,
            depth_cap,
            target_size_multiplier,
            n_volume_sampling_steps_along_shortest_axis,
        ) in cases:
            with self.subTest(case_name):
                with self.assertRaises(ValueError):
                    SceneConfig(
                        applied_transform=None,
                        depth_cap=depth_cap,
                        n_volume_sampling_steps_along_shortest_axis=n_volume_sampling_steps_along_shortest_axis,
                        target_size_multiplier=target_size_multiplier,
                        object_transform_type=ObjectTransformType.VolumeBased,
                        resolution=TwoDSize(76, 152),
                        viewpt_counts=DesiredViewpointCounts(None, None, None),
                        world_path=Path("scenes/world1.glb"),
                    )

    def test_reload_path(self):
        cases: list[tuple[str, PointBasedVectorFieldSpec | None]] = [
            ("no_transform", None),
            (
                "transform_present",
                PointBasedVectorFieldSpec(
                    control_points=np.arange(0, 30).reshape(
                        newshape_points_space(n=10)
                    ),
                    vectors=np.arange(0, 30).reshape(newshape_points_space(n=10)) / 79,
                ),
            ),
        ]
        for subtest_name, initial_applied_transform in cases:
            with self.subTest(subtest_name):
                with TemporaryDirectory() as td:
                    td_path = Path(td)
                    scene_config_path = td_path / "scene35.scene"

                    scene_config1 = SceneConfig(
                        applied_transform=initial_applied_transform,
                        depth_cap=5,
                        n_volume_sampling_steps_along_shortest_axis=37,
                        target_size_multiplier=3.1,
                        object_transform_type=ObjectTransformType.VolumeBased,
                        resolution=TwoDSize(76, 152),
                        viewpt_counts=DesiredViewpointCounts(None, None, None),
                        world_path=Path("scenes/world1.glb"),
                    )
                    scene_config1.save_json(scene_config_path)
                    scene_config2 = SceneConfig.from_json(scene_config_path)

                    if initial_applied_transform is None:
                        self.assertIsNone(scene_config2.applied_transform)
                    else:
                        reloaded_applied_transform = scene_config2.applied_transform
                        self.assertIsNotNone(reloaded_applied_transform)
                        assert reloaded_applied_transform is not None
                        self.assertTrue(
                            np.allclose(
                                reloaded_applied_transform.control_points,
                                initial_applied_transform.control_points,
                            )
                        )
                        self.assertTrue(
                            np.allclose(
                                reloaded_applied_transform.vectors,
                                initial_applied_transform.vectors,
                            )
                        )

                    self.assertAlmostEqual(
                        scene_config1.depth_cap, scene_config2.depth_cap
                    )
                    self.assertAlmostEqual(
                        scene_config1.target_size_multiplier,
                        scene_config2.target_size_multiplier,
                    )
                    self.assertEqual(
                        scene_config1.resolution.x_size, scene_config2.resolution.x_size
                    )
                    self.assertEqual(
                        scene_config1.resolution.y_size, scene_config2.resolution.y_size
                    )
                    self.assertEqual(scene_config1.world_path, scene_config2.world_path)
                    self.assertEqual(
                        scene_config1.viewpt_counts.n_train_samples,
                        scene_config2.viewpt_counts.n_train_samples,
                    )
                    self.assertEqual(
                        scene_config1.viewpt_counts.n_test_samples,
                        scene_config2.viewpt_counts.n_test_samples,
                    )
                    self.assertEqual(
                        scene_config1.viewpt_counts.n_val_samples,
                        scene_config2.viewpt_counts.n_val_samples,
                    )

    def test_reload_textio(self):
        cases: list[tuple[str, PointBasedVectorFieldSpec | None]] = [
            ("no_transform", None),
            (
                "transform_present",
                PointBasedVectorFieldSpec(
                    control_points=np.arange(0, 30).reshape(
                        newshape_points_space(n=10)
                    ),
                    vectors=np.arange(0, 30).reshape(newshape_points_space(n=10)) / 79,
                ),
            ),
        ]
        for subtest_name, initial_applied_transform in cases:
            with self.subTest(subtest_name):
                scene_config_target_io = StringIO()

                scene_config1 = SceneConfig(
                    applied_transform=initial_applied_transform,
                    depth_cap=5,
                    n_volume_sampling_steps_along_shortest_axis=37,
                    target_size_multiplier=3.1,
                    object_transform_type=ObjectTransformType.VolumeBased,
                    resolution=TwoDSize(76, 152),
                    viewpt_counts=DesiredViewpointCounts(None, None, None),
                    world_path=Path("scenes/world1.glb"),
                )
                scene_config1.save_json(scene_config_target_io)

                saved_scene_config_str = scene_config_target_io.getvalue()
                scene_config_source_io = StringIO(saved_scene_config_str)
                scene_config2 = SceneConfig.from_json(scene_config_source_io)

                if initial_applied_transform is None:
                    self.assertIsNone(scene_config2.applied_transform)
                else:
                    reloaded_applied_transform = scene_config2.applied_transform
                    self.assertIsNotNone(reloaded_applied_transform)
                    assert reloaded_applied_transform is not None
                    self.assertTrue(
                        np.allclose(
                            reloaded_applied_transform.control_points,
                            initial_applied_transform.control_points,
                        )
                    )
                    self.assertTrue(
                        np.allclose(
                            reloaded_applied_transform.vectors,
                            initial_applied_transform.vectors,
                        )
                    )

                self.assertAlmostEqual(scene_config1.depth_cap, scene_config2.depth_cap)
                self.assertAlmostEqual(
                    scene_config1.target_size_multiplier,
                    scene_config2.target_size_multiplier,
                )
                self.assertEqual(
                    scene_config1.resolution.x_size, scene_config2.resolution.x_size
                )
                self.assertEqual(
                    scene_config1.resolution.y_size, scene_config2.resolution.y_size
                )
                self.assertEqual(scene_config1.world_path, scene_config2.world_path)
                self.assertEqual(
                    scene_config1.viewpt_counts.n_train_samples,
                    scene_config2.viewpt_counts.n_train_samples,
                )
                self.assertEqual(
                    scene_config1.viewpt_counts.n_test_samples,
                    scene_config2.viewpt_counts.n_test_samples,
                )
                self.assertEqual(
                    scene_config1.viewpt_counts.n_val_samples,
                    scene_config2.viewpt_counts.n_val_samples,
                )


class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.n_train_viewpoints = 3
        self.n_val_viewpoints = 4
        self.n_test_viewpoints = 5

        rng = np.random.default_rng(1000)

        self.train_rgbs = rng.uniform(
            0, 1, size=newshape_im_rgbs(n=self.n_train_viewpoints, h=20, w=23)
        )
        self.train_gt_depths = rng.uniform(
            0.1, 2, size=newshape_im_depthmaps(n=self.n_train_viewpoints, h=20, w=23)
        )
        self.train_masks = rng.choice(
            [0, 1], newshape_im_depthmasks(n=self.n_train_viewpoints, h=20, w=23)
        ).astype(np.bool_)
        self.val_rgbs = rng.uniform(
            0, 1, size=newshape_im_rgbs(n=self.n_val_viewpoints, h=20, w=23)
        )
        self.val_gt_depths = rng.uniform(
            3, 5, size=newshape_im_depthmaps(n=self.n_val_viewpoints, h=20, w=23)
        )
        self.val_masks = rng.choice(
            [0, 1], newshape_im_depthmasks(n=self.n_val_viewpoints, h=20, w=23)
        ).astype(np.bool_)
        self.test_rgbs = rng.uniform(
            0, 1, size=newshape_im_rgbs(n=self.n_test_viewpoints, h=20, w=23)
        )
        self.test_gt_depths = rng.uniform(
            0.7, 3.1, size=newshape_im_depthmaps(n=self.n_test_viewpoints, h=20, w=23)
        )
        self.test_masks = rng.choice(
            [0, 1], newshape_im_depthmasks(n=self.n_test_viewpoints, h=20, w=23)
        ).astype(np.bool_)

        self.rgbds_for_viewpt_types = {
            SampleType.Train: RGBsWithDepthsAndMasks(
                rgbs=self.train_rgbs,
                depths=self.train_gt_depths,
                masks=self.train_masks,
            ),
            SampleType.Val: RGBsWithDepthsAndMasks(
                rgbs=self.val_rgbs, depths=self.val_gt_depths, masks=self.val_masks
            ),
            SampleType.Test: RGBsWithDepthsAndMasks(
                rgbs=self.test_rgbs, depths=self.test_gt_depths, masks=self.test_masks
            ),
        }

        self.target_obj_areas_for_viewpt_types = {
            SampleType.Train: TwoDAreas(
                x_maxes_excluding=np.arange(0, self.n_train_viewpoints) + 5,
                x_mins_including=np.arange(0, self.n_train_viewpoints),
                y_maxes_excluding=np.arange(0, self.n_train_viewpoints) + 5,
                y_mins_including=np.arange(0, self.n_train_viewpoints),
            ),
            SampleType.Val: TwoDAreas(
                x_maxes_excluding=np.arange(0, self.n_val_viewpoints) + 5,
                x_mins_including=np.arange(0, self.n_val_viewpoints),
                y_maxes_excluding=np.arange(0, self.n_val_viewpoints) + 2,
                y_mins_including=np.arange(0, self.n_val_viewpoints),
            ),
            SampleType.Test: TwoDAreas(
                x_maxes_excluding=np.arange(0, self.n_test_viewpoints) + 5,
                x_mins_including=np.arange(0, self.n_test_viewpoints),
                y_maxes_excluding=np.arange(0, self.n_test_viewpoints) + 2,
                y_mins_including=np.arange(0, self.n_test_viewpoints),
            ),
        }

        self.predictor = _LambdaDepthPredictor(
            native_transform_fn=lambda a: np.expand_dims(
                np.mean(a, axis=(DIM_IM_C,)), axis=1
            ),
            alignment_function=lambda native_preds, gt_depths, masks, depth_cap: native_preds
            * gt_depths
            * depth_cap,
        )

        self.scene_mock = _new_scene_mock_for_rendering(
            rgbds_for_viewpt_types=self.rgbds_for_viewpt_types,
            target_obj_areas_for_viewpt_types=self.target_obj_areas_for_viewpt_types,
            depth_cap=100,
        )

        self.logging_freq_fn_mock = mock.Mock()

        self.progress_logger_mock = mock.Mock()

        self.logging_freq_fn_mock.needs_logging = mock.Mock(return_value=True)

        self.n_viewpts_for_viewpt_types = {
            SampleType.Train: self.n_train_viewpoints,
            SampleType.Val: self.n_val_viewpoints,
            SampleType.Test: self.n_test_viewpoints,
        }

    def test_calc_raw_loss_values_on_scene(self):
        aligned_train_preds = self.predictor.alignment_function(
            native_preds=self.predictor.predict_native(self.train_rgbs).get(),
            gt_depths=self.train_gt_depths,
            depth_cap=100,
            masks=self.train_masks,
        )
        aligned_val_preds = self.predictor.alignment_function(
            native_preds=self.predictor.predict_native(self.val_rgbs).get(),
            gt_depths=self.val_gt_depths,
            depth_cap=100,
            masks=self.val_masks,
        )
        aligned_test_preds = self.predictor.alignment_function(
            native_preds=self.predictor.predict_native(self.test_rgbs).get(),
            gt_depths=self.test_gt_depths,
            depth_cap=100,
            masks=self.test_masks,
        )

        expected_losses_base: dict[tuple[SampleType, RawLossFn], np.ndarray] = {
            (SampleType.Train, RawLossFn.RMSE): rmse_loss(
                gt=DepthsWithMasks(depths=self.train_gt_depths, masks=self.train_masks),
                pred_depths=aligned_train_preds,
            ),
            (SampleType.Val, RawLossFn.RMSE): rmse_loss(
                gt=DepthsWithMasks(depths=self.val_gt_depths, masks=self.val_masks),
                pred_depths=aligned_val_preds,
            ),
            (SampleType.Train, RawLossFn.D1): d1_loss(
                gt=DepthsWithMasks(depths=self.train_gt_depths, masks=self.train_masks),
                pred_depths=aligned_train_preds,
            ),
            (SampleType.Val, RawLossFn.D1): d1_loss(
                gt=DepthsWithMasks(depths=self.val_gt_depths, masks=self.val_masks),
                pred_depths=aligned_val_preds,
            ),
            (SampleType.Train, RawLossFn.Log10): log10_loss(
                gt=DepthsWithMasks(depths=self.train_gt_depths, masks=self.train_masks),
                pred_depths=aligned_train_preds,
            ),
            (SampleType.Val, RawLossFn.Log10): log10_loss(
                gt=DepthsWithMasks(depths=self.val_gt_depths, masks=self.val_masks),
                pred_depths=aligned_val_preds,
            ),
            (SampleType.Train, RawLossFn.CroppedRMSE): cropped_rmse_loss(
                gt=DepthsWithMasks(depths=self.train_gt_depths, masks=self.train_masks),
                pred_depths=aligned_train_preds,
                target_obj_areas=self.target_obj_areas_for_viewpt_types[
                    SampleType.Train
                ],
            ),
            (SampleType.Val, RawLossFn.CroppedRMSE): cropped_rmse_loss(
                gt=DepthsWithMasks(depths=self.val_gt_depths, masks=self.val_masks),
                pred_depths=aligned_val_preds,
                target_obj_areas=self.target_obj_areas_for_viewpt_types[SampleType.Val],
            ),
            (SampleType.Train, RawLossFn.CroppedD1): cropped_d1_loss(
                gt=DepthsWithMasks(depths=self.train_gt_depths, masks=self.train_masks),
                pred_depths=aligned_train_preds,
                target_obj_areas=self.target_obj_areas_for_viewpt_types[
                    SampleType.Train
                ],
            ),
            (SampleType.Val, RawLossFn.CroppedD1): cropped_d1_loss(
                gt=DepthsWithMasks(depths=self.val_gt_depths, masks=self.val_masks),
                pred_depths=aligned_val_preds,
                target_obj_areas=self.target_obj_areas_for_viewpt_types[SampleType.Val],
            ),
            (SampleType.Train, RawLossFn.CroppedLog10): cropped_log10_loss(
                gt=DepthsWithMasks(depths=self.train_gt_depths, masks=self.train_masks),
                pred_depths=aligned_train_preds,
                target_obj_areas=self.target_obj_areas_for_viewpt_types[
                    SampleType.Train
                ],
            ),
            (SampleType.Val, RawLossFn.CroppedLog10): cropped_log10_loss(
                gt=DepthsWithMasks(depths=self.val_gt_depths, masks=self.val_masks),
                pred_depths=aligned_val_preds,
                target_obj_areas=self.target_obj_areas_for_viewpt_types[SampleType.Val],
            ),
        }
        expected_losses_with_test: dict[
            tuple[SampleType, RawLossFn], np.ndarray
        ] = expected_losses_base | {
            (SampleType.Test, RawLossFn.RMSE): rmse_loss(
                gt=DepthsWithMasks(depths=self.test_gt_depths, masks=self.test_masks),
                pred_depths=aligned_test_preds,
            ),
            (SampleType.Test, RawLossFn.CroppedRMSE): cropped_rmse_loss(
                gt=DepthsWithMasks(depths=self.test_gt_depths, masks=self.test_masks),
                pred_depths=aligned_test_preds,
                target_obj_areas=self.target_obj_areas_for_viewpt_types[
                    SampleType.Test
                ],
            ),
            (SampleType.Test, RawLossFn.Log10): log10_loss(
                gt=DepthsWithMasks(depths=self.test_gt_depths, masks=self.test_masks),
                pred_depths=aligned_test_preds,
            ),
            (SampleType.Test, RawLossFn.CroppedLog10): cropped_log10_loss(
                gt=DepthsWithMasks(depths=self.test_gt_depths, masks=self.test_masks),
                pred_depths=aligned_test_preds,
                target_obj_areas=self.target_obj_areas_for_viewpt_types[
                    SampleType.Test
                ],
            ),
            (SampleType.Test, RawLossFn.D1): d1_loss(
                gt=DepthsWithMasks(depths=self.test_gt_depths, masks=self.test_masks),
                pred_depths=aligned_test_preds,
            ),
            (SampleType.Test, RawLossFn.CroppedD1): cropped_d1_loss(
                gt=DepthsWithMasks(depths=self.test_gt_depths, masks=self.test_masks),
                pred_depths=aligned_test_preds,
                target_obj_areas=self.target_obj_areas_for_viewpt_types[
                    SampleType.Test
                ],
            ),
        }

        cases: list[
            tuple[bool, dict[tuple[SampleType, RawLossFn], np.ndarray], int]
        ] = [
            (
                False,
                expected_losses_base,
                self.n_train_viewpoints + self.n_val_viewpoints,
            ),
            (
                True,
                expected_losses_with_test,
                self.n_train_viewpoints
                + self.n_val_viewpoints
                + self.n_test_viewpoints,
            ),
        ]
        for eval_on_test, expected_losses, expected_needs_logging_call_count in cases:
            with self.subTest(f"{eval_on_test=}"):
                self.logging_freq_fn_mock.needs_logging.reset_mock()
                self.progress_logger_mock.reset_mock()
                actual_losses = calc_raw_loss_values_on_scene(
                    scene=self.scene_mock,
                    logging_freq_fn=self.logging_freq_fn_mock,
                    predictor=self.predictor,
                    progress_logger=self.progress_logger_mock,
                    eval_on_test=eval_on_test,
                )

                self.assertEqual(
                    self.logging_freq_fn_mock.needs_logging.call_count,
                    expected_needs_logging_call_count,
                )
                self.assertEqual(
                    self.progress_logger_mock.call_count,
                    expected_needs_logging_call_count,
                )

                self.assertEqual(actual_losses.keys(), expected_losses.keys())

                for key in actual_losses.keys():
                    actual_losses_for_key = actual_losses[key]
                    expected_losses_for_key = actual_losses[key]

                    self.assertTrue(
                        np.allclose(actual_losses_for_key, expected_losses_for_key)
                    )

    def test_calc_aggr_delta_loss_dict_from_losses_on_scene(self):
        for eval_on_test in [True, False]:
            with self.subTest(f"{eval_on_test=}"):
                initial_raw_losses: dict[
                    tuple[SampleType, RawLossFn], np.ndarray
                ] = dict()

                if eval_on_test:
                    relevant_sample_types = SampleType
                else:
                    relevant_sample_types = [SampleType.Train, SampleType.Val]

                for viewpt_type, loss_fn in itertools.product(
                    relevant_sample_types, RawLossFn
                ):
                    initial_raw_losses[viewpt_type, loss_fn] = (
                        np.arange(self.n_viewpts_for_viewpt_types[viewpt_type]).astype(
                            np.float32
                        )
                        + 0.1
                    )

                raw_losses_new = calc_raw_loss_values_on_scene(
                    scene=self.scene_mock,
                    logging_freq_fn=self.logging_freq_fn_mock,
                    predictor=self.predictor,
                    progress_logger=self.progress_logger_mock,
                    eval_on_test=eval_on_test,
                )

                actual_delta_loss_dict = calc_aggr_delta_loss_dict_from_losses_on_scene(
                    scene=self.scene_mock,
                    initial_raw_losses=initial_raw_losses,
                    logging_freq_fn=self.logging_freq_fn_mock,
                    progress_logger=self.progress_logger_mock,
                    predictor=self.predictor,
                    eval_on_test=eval_on_test,
                )

                expected_train_mean_delta_rmse = float(
                    np.mean(
                        raw_losses_new[SampleType.Train, RawLossFn.RMSE]
                        - initial_raw_losses[SampleType.Train, RawLossFn.RMSE]
                    )
                )
                expected_val_min_delta_log10 = float(
                    np.min(
                        raw_losses_new[SampleType.Val, RawLossFn.Log10]
                        - initial_raw_losses[SampleType.Val, RawLossFn.Log10]
                    )
                )

                expected_agg_delta_count = (
                    len(LossDerivationMethod)
                    * 1
                    * len(relevant_sample_types)
                    * len(RawLossFn)
                )

                self.assertEqual(
                    len(actual_delta_loss_dict.keys()), expected_agg_delta_count
                )

                self.assertAlmostEqual(
                    expected_train_mean_delta_rmse,
                    actual_delta_loss_dict[
                        LossDerivationMethod.MeanDelta,
                        LossPrecision.Exact,
                        SampleType.Train,
                        RawLossFn.RMSE,
                    ],
                )
                self.assertAlmostEqual(
                    expected_val_min_delta_log10,
                    actual_delta_loss_dict[
                        LossDerivationMethod.MinDelta,
                        LossPrecision.Exact,
                        SampleType.Val,
                        RawLossFn.Log10,
                    ],
                )
                if eval_on_test:
                    expected_test_median_delta_log10 = float(
                        np.median(
                            raw_losses_new[SampleType.Test, RawLossFn.Log10]
                            - initial_raw_losses[SampleType.Test, RawLossFn.Log10]
                        )
                    )
                    self.assertAlmostEqual(
                        expected_test_median_delta_log10,
                        actual_delta_loss_dict[
                            LossDerivationMethod.MedianDelta,
                            LossPrecision.Exact,
                            SampleType.Test,
                            RawLossFn.Log10,
                        ],
                    )
                else:
                    self.assertFalse(
                        any(
                            SampleType.Test in set(key)
                            for key in actual_delta_loss_dict.keys()
                        )
                    )


class _LambdaDepthPredictor:
    """
    A class that implements depth prediction using lambda functions for testing purposes.

    Parameters
    ----------
    native_transform_fn
        The function that produces the native depth predictions from the RGBD images. Format: ``Im::RGBs-> ArbSamples::*``
    alignment_function
        The alignment function.
    """

    def __init__(
        self,
        native_transform_fn: Callable[[np.ndarray], np.ndarray],
        alignment_function: AlignmentFunction,
    ) -> None:
        self.transform_fn = native_transform_fn
        self.alignment_function = alignment_function

    def get_name(self) -> str:
        raise NotImplementedError()

    def predict_native(self, rgbs: np.ndarray) -> "LambdaDepthPredictorFuture":
        unaligned_preds = self.transform_fn(rgbs)
        return LambdaDepthPredictorFuture(unaligned_preds)


def _new_scene_mock_for_rendering(
    rgbds_for_viewpt_types: dict[SampleType, RGBsWithDepthsAndMasks],
    target_obj_areas_for_viewpt_types: dict[SampleType, TwoDAreas],
    depth_cap: float,
) -> mock.Mock:
    viewpt_counts_for_types = {
        viewpt_type: rgbds.rgbs.shape[DIM_POINTS_N]
        for viewpt_type, rgbds in rgbds_for_viewpt_types.items()
    }

    def get_sample(idx: int, sample_type: SampleType) -> SceneSamples:
        all_rgbds = rgbds_for_viewpt_types[sample_type]

        rgbds = all_rgbds[[idx]]
        target_obj_areas = target_obj_areas_for_viewpt_types[sample_type].idx_areas(
            [idx]
        )

        result = SceneSamples(rgbds=rgbds, target_obj_areas_on_screen=target_obj_areas)
        return result

    def get_n_samples() -> ExactSampleCounts:
        result = ExactSampleCounts(
            n_train_samples=viewpt_counts_for_types.get(SampleType.Train, 0),
            n_test_samples=viewpt_counts_for_types.get(SampleType.Test, 0),
            n_val_samples=viewpt_counts_for_types.get(SampleType.Val, 0),
        )
        return result

    def get_n_samples_for_type(viewpt_type: SampleType):
        return viewpt_counts_for_types.get(viewpt_type, 0)

    scene_mock = mock.Mock()

    scene_mock.get_sample = get_sample
    scene_mock.get_n_samples = get_n_samples
    scene_mock.get_n_samples_for_type = get_n_samples_for_type
    scene_mock.get_depth_cap = mock.Mock(return_value=depth_cap)

    return scene_mock


class LambdaDepthPredictorFuture:
    def __init__(self, unaligned_preds: np.ndarray):
        self.unaligned_preds = unaligned_preds

    def get(self) -> np.ndarray:
        return self.unaligned_preds


if TYPE_CHECKING:
    v: AsyncDepthPredictor = type_instance(_LambdaDepthPredictor)
