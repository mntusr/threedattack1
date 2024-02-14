import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Callable, Optional
from unittest import mock

import numpy as np
from panda3d.core import GeomNode, NodePath, PerspectiveLens

from threedattack.dataset_model import ExactSampleCounts, SampleType
from threedattack.rendering_core import (
    DesiredViewpointCounts,
    Panda3dAssumptionViolation,
    SplitFormatError,
    ThreeDPoint,
    ThreeDSize,
    ViewpointBasedCamController,
    ViewpointSplit,
    get_col_copy_from_vertex_data,
    set_col_in_vertex_data,
)
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *

from .util import new_vertex_only_object


class TestViewpointBasedCamController(unittest.TestCase):
    def setUp(self) -> None:
        self.TARGET_POS = ThreeDPoint(1, 0, 0)
        self.TARGET_OBJ = NodePath("Target")
        self.TARGET_MESH_OBJ = new_vertex_only_object(
            np.array(
                [
                    [1, 1, 1],
                    [1, 1, -1],
                    [1, -1, 1],
                    [1, -1, -1],
                    [-1, 1, 1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [-1, -1, -1],
                ]
            ),
            "TargetMesh",
        )
        self.TARGET_MESH_OBJ.reparentTo(self.TARGET_OBJ)

        self.TARGET_OBJ.setPos(self.TARGET_POS.x, self.TARGET_POS.y, self.TARGET_POS.z)
        self.TARGET_OBJ_ORIG_SIZE = ThreeDSize(2, 2, 2)
        self.VIEWPT_SPLIT = ViewpointSplit(
            train_viewpoints=np.array(
                [
                    [0, 0, 0],
                    [5, 0, 0],
                    [100, 0, 0],
                ]
            ),
            test_viewpoints=np.array(
                [
                    [0, 0, 0],
                    [0, 5, 0],
                    [0, 100, 0],
                    [0, 101, 0],
                ]
            ),
            val_viewpoints=np.array(
                [
                    [0, 0, 0],
                    [0, 0, 5],
                    [0, 0, 100],
                    [0, 0, 102],
                    [0, 0, 103],
                ]
            ),
        )
        self.POINT1_FOR_TYPES = {
            SampleType.Train: ThreeDPoint(5, 0, 0),
            SampleType.Test: ThreeDPoint(0, 5, 0),
            SampleType.Val: ThreeDPoint(0, 0, 5),
        }
        self.NEAR_PLANE_DISTANCE = 0.1

        self.SHOW_BASE_MOCK = _new_showbase_mock(
            target_mesh_obj=self.TARGET_MESH_OBJ,
            scene_errors=[],
            near_plane_distance=self.NEAR_PLANE_DISTANCE,
        )

    def test_init_happy_path(self):
        wp_controller = ViewpointBasedCamController(
            base=self.SHOW_BASE_MOCK, viewpt_split=self.VIEWPT_SPLIT
        )

        self.SHOW_BASE_MOCK.get_standard_scene_format_errors.assert_called_with()

        self.assertEqual(wp_controller.get_target_position(), self.TARGET_POS)
        self.assertEqual(
            wp_controller.get_filtered_viewpoint_count(SampleType.Train), 2
        )
        self.assertEqual(wp_controller.get_filtered_viewpoint_count(SampleType.Test), 3)
        self.assertEqual(wp_controller.get_filtered_viewpoint_count(SampleType.Val), 4)
        self.assertEqual(
            wp_controller.get_original_target_obj_size(), ThreeDSize(2, 2, 2)
        )
        self.assertGreater(
            wp_controller.get_extra_offset_due_to_near_plane(), self.NEAR_PLANE_DISTANCE
        )

    def test_get_filtered_viewpt_counts(self):
        wp_controller = ViewpointBasedCamController(
            base=self.SHOW_BASE_MOCK, viewpt_split=self.VIEWPT_SPLIT
        )
        self.assertEqual(
            wp_controller.get_filtered_viewpoint_counts(),
            ExactSampleCounts(n_train_samples=2, n_test_samples=3, n_val_samples=4),
        )

    def test_negative_near_plane_distance(self):
        showbase = _new_showbase_mock(
            target_mesh_obj=self.TARGET_MESH_OBJ,
            scene_errors=[],
            near_plane_distance=-1,
        )
        with self.assertRaises(Panda3dAssumptionViolation):
            ViewpointBasedCamController(showbase, viewpt_split=self.VIEWPT_SPLIT)

    def test_init_nonstandard_scene(self):
        show_base_mock_with_nonstandard_scene = _new_showbase_mock(
            target_mesh_obj=self.TARGET_MESH_OBJ,
            scene_errors=["Some error here"],
            near_plane_distance=self.NEAR_PLANE_DISTANCE,
        )
        with self.assertRaises(ValueError):
            ViewpointBasedCamController(
                base=show_base_mock_with_nonstandard_scene,
                viewpt_split=self.VIEWPT_SPLIT,
            )

    def test_update_target_area_happy_path(self):
        NEW_SIZE_MULTIPLIER = 5
        wp_controller = ViewpointBasedCamController(
            base=self.SHOW_BASE_MOCK, viewpt_split=self.VIEWPT_SPLIT
        )

        wiewpt_counts_before = wp_controller.get_filtered_viewpoint_counts()

        new_scaled_standing_area = wp_controller.update_target_area(NEW_SIZE_MULTIPLIER)
        wiewpt_counts_after = wp_controller.get_filtered_viewpoint_counts()

        self.assertEqual(
            new_scaled_standing_area.get_size_multiplier(), NEW_SIZE_MULTIPLIER
        )
        self.assertEqual(
            new_scaled_standing_area.get_original_size(), self.TARGET_OBJ_ORIG_SIZE
        )

        self.assertLess(
            wiewpt_counts_after.n_train_samples,
            wiewpt_counts_before.n_train_samples,
        )
        self.assertLess(
            wiewpt_counts_after.n_test_samples,
            wiewpt_counts_before.n_test_samples,
        )
        self.assertLess(
            wiewpt_counts_after.n_val_samples, wiewpt_counts_before.n_val_samples
        )

    def test_update_target_area_too_small_size_multiplier(self):
        wp_controller = ViewpointBasedCamController(
            base=self.SHOW_BASE_MOCK, viewpt_split=self.VIEWPT_SPLIT
        )
        with self.assertRaises(ValueError):
            wp_controller.update_target_area(0.9)

    def test_select_viewpoint_no_selectable_viewpoint(self):
        wp_controller = ViewpointBasedCamController(
            base=self.SHOW_BASE_MOCK, viewpt_split=self.VIEWPT_SPLIT
        )
        wp_controller.update_target_area(10000)
        with self.assertRaises(ValueError):
            wp_controller.select_viewpoint(0, SampleType.Train)

    def test_select_viewpoint_happy_path(self):
        for viewpt_type in SampleType:
            with self.subTest(f"{viewpt_type=}"):
                wp_controller = ViewpointBasedCamController(
                    base=self.SHOW_BASE_MOCK, viewpt_split=self.VIEWPT_SPLIT
                )

                class SetCamPosAndLookAtMock:
                    def __init__(self, test_class: unittest.TestCase) -> None:
                        self.data: Optional[tuple[ThreeDPoint, ThreeDPoint]] = None
                        self.test_class = test_class

                    def __call__(
                        self, new_cam_pos: ThreeDPoint, look_at: ThreeDPoint
                    ) -> None:
                        self.data = (new_cam_pos, look_at)

                    def get_data(self) -> tuple[ThreeDPoint, ThreeDPoint]:
                        if self.data is None:
                            self.test_class.fail("The function was not invoked.")

                        return self.data

                matcher = SetCamPosAndLookAtMock(self)
                self.SHOW_BASE_MOCK.set_cam_pos_and_look_at = matcher

                wp_controller.select_viewpoint(0, viewpt_type)

                new_cam_pos, assumed_target_origin = matcher.get_data()

                self.assertTrue(
                    self.TARGET_POS.is_almost_equal(assumed_target_origin, epsilon=1e-5)
                )

                self.assertTrue(
                    self.POINT1_FOR_TYPES[viewpt_type].is_almost_equal(
                        new_cam_pos, epsilon=1e-5
                    )
                )

    def test_update_invariance(self) -> None:
        target_mesh_obj = NodePath("Target").attachNewNode(
            copy.deepcopy(self.TARGET_MESH_OBJ.node())
        )

        showbase = _new_showbase_mock(
            target_mesh_obj=target_mesh_obj,
            scene_errors=[],
            near_plane_distance=self.NEAR_PLANE_DISTANCE,
        )

        wp_controller = ViewpointBasedCamController(
            base=showbase, viewpt_split=self.VIEWPT_SPLIT
        )

        target_obj_size_before = wp_controller.get_original_target_obj_size()
        _ugly_hack_to_change_vertexes_without_normal_recalculation(
            target_mesh_obj, lambda x: x * 2
        )
        target_obj_size_after = wp_controller.get_original_target_obj_size()

        self.assertEqual(target_obj_size_before, target_obj_size_after)


def _ugly_hack_to_change_vertexes_without_normal_recalculation(
    obj: "NodePath[GeomNode]", transform_fn: Callable[[np.ndarray], np.ndarray]
) -> None:
    """
    This function transforms the vertexes of the specified object inplace, without the recalculation of the normal vectors. This is generally a bad practice and should not be used for anything but testing purposes.

    Parameters
    ----------
    obj
        The object to modify.
    transform_fn
        The function to transform the vertexes of the object. Signature: ``Points::Space->Points::Space``
    """
    VERTEX_COL = "vertex"
    vertex_data = obj.node().modifyGeom(0).modifyVertexData()
    verts = get_col_copy_from_vertex_data(VERTEX_COL, vertex_data)
    verts = transform_fn(verts)
    set_col_in_vertex_data(vertex_data, VERTEX_COL, verts)


class TestDesiredViewpointCounts(unittest.TestCase):
    def test_to_exact(self):
        total_viewpt_counts = ExactSampleCounts(
            n_train_samples=10, n_test_samples=11, n_val_samples=12
        )
        desired_viewpt_counts_and_expected_versions = [
            (
                DesiredViewpointCounts(
                    n_train_samples=5, n_test_samples=None, n_val_samples=None
                ),
                ExactSampleCounts(
                    n_train_samples=5, n_test_samples=11, n_val_samples=12
                ),
            ),
            (
                DesiredViewpointCounts(
                    n_train_samples=None, n_test_samples=5, n_val_samples=None
                ),
                ExactSampleCounts(
                    n_train_samples=10, n_test_samples=5, n_val_samples=12
                ),
            ),
            (
                DesiredViewpointCounts(
                    n_train_samples=None, n_test_samples=None, n_val_samples=5
                ),
                ExactSampleCounts(
                    n_train_samples=10, n_test_samples=11, n_val_samples=5
                ),
            ),
        ]

        for desired, expected_result in desired_viewpt_counts_and_expected_versions:
            desired_repr = repr(desired).replace(" ", "")
            with self.subTest(desired_repr):
                actual_result = desired.to_exact(total_viewpt_counts)

                self.assertEqual(expected_result, actual_result)


class TestViewpointSplit(unittest.TestCase):
    def setUp(self):
        self.split = _new_viewpt_split_with_single_elem_and_equal_coord(
            train_coord=3, test_coord=4, val_coord=5
        )

        showbase_mock = _new_showbase_mock(
            target_mesh_obj=mock.Mock(), near_plane_distance=0.01, scene_errors=[]
        )

        self.N_VIEPWOINTS_IN_SHOWBASE = 40
        total_viewpoints = np.linspace(0, 1000, 120).reshape(
            newshape_points_space(n=self.N_VIEPWOINTS_IN_SHOWBASE)
        )

        viewpoints_obj = new_vertex_only_object(
            total_viewpoints, obj_name="ViewpointsMesh"
        )

        showbase_mock.get_viewpoints_obj_mesh_path = mock.Mock(
            return_value=viewpoints_obj
        )

        self.SHOWBASE_MOCK_WITH_VIEWPOINTS = showbase_mock
        self.CORRECT_VIEWPOINT_COUNTS_FOR_SHOWBASE_MOCK = ExactSampleCounts(
            n_train_samples=20, n_val_samples=9, n_test_samples=11
        )

    def test_load_and_save(self):
        with TemporaryDirectory() as td:
            td = Path(td)
            npz_path = td / "file1.npz"
            self.split.save_npz(npz_path)

            reloaded_split = ViewpointSplit.load_npz(npz_path)

            self._assert_splits_allclose(self.split, reloaded_split)

    def test_save_npz_format_correctness(self):
        with TemporaryDirectory() as td:
            td = Path(td)
            npz_path = td / "file1.npz"
            self.split.save_npz(npz_path)

            self.split.train_viewpoints

            with np.load(npz_path) as data:
                self.assertIsInstance(data["train_viewpoints"], np.ndarray)
                self.assertTrue(match_points_space(data["train_viewpoints"]))
                self.assertIsInstance(data["test_viewpoints"], np.ndarray)
                self.assertTrue(match_points_space(data["test_viewpoints"]))
                self.assertIsInstance(data["val_viewpoints"], np.ndarray)
                self.assertTrue(match_points_space(data["val_viewpoints"]))

    def test_load_npz_invalid_file(self):
        with TemporaryDirectory() as td:
            td = Path(td)
            original_npz_path = td / "file1.npz"
            modified_npz_path = td / "modified.npz"
            self.split.save_npz(original_npz_path)

            with np.load(original_npz_path) as original_data:
                for key in original_data.keys():
                    with self.subTest(f"invalid_{key}"):
                        with np.load(original_npz_path) as data_to_modify:
                            data_dict = dict(data_to_modify)
                            data_dict[key] = np.array([1, 2, 3])
                            np.savez(modified_npz_path, **data_dict)

                            with self.assertRaises(SplitFormatError):
                                ViewpointSplit.load_npz(modified_npz_path)

    def test_transform_or_filter(self):
        def transform_fn(x: np.ndarray) -> np.ndarray:
            self.assertTrue(match_points_space(x))
            x = x**2
            x = np.concatenate(
                [x, np.zeros(newshape_points_space(n=1))], axis=DIM_POINTS_N
            )
            return x

        actual_transformed_split = self.split.transform_or_filter(transform_fn)
        expected_transformed_split = ViewpointSplit(
            train_viewpoints=transform_fn(self.split.train_viewpoints),
            test_viewpoints=transform_fn(self.split.test_viewpoints),
            val_viewpoints=transform_fn(self.split.val_viewpoints),
        )

        self._assert_splits_allclose(
            expected_transformed_split, actual_transformed_split
        )

    def test_get_viewpt_counts(self):
        expected_counts = ExactSampleCounts(
            n_train_samples=5, n_test_samples=4, n_val_samples=3
        )
        split_with_different_viewpt_counts = _new_viewpt_split_with_counts(
            viewpt_counts=expected_counts,
            fill_value=0,
        )

        actual_counts = split_with_different_viewpt_counts.get_viewpt_counts()

        self.assertEqual(expected_counts, actual_counts)

    def test_get_viewpoint_count(self):
        expected_counts = ExactSampleCounts(
            n_train_samples=5, n_test_samples=4, n_val_samples=3
        )
        split_with_different_viewpt_counts = _new_viewpt_split_with_counts(
            viewpt_counts=expected_counts,
            fill_value=0,
        )

        self.assertEqual(
            expected_counts.n_train_samples,
            split_with_different_viewpt_counts.get_viewpt_count(SampleType.Train),
        )
        self.assertEqual(
            expected_counts.n_test_samples,
            split_with_different_viewpt_counts.get_viewpt_count(SampleType.Test),
        )
        self.assertEqual(
            expected_counts.n_val_samples,
            split_with_different_viewpt_counts.get_viewpt_count(SampleType.Val),
        )

    def test_select_n_happy_path_all_specified(self):
        split = ViewpointSplit(
            train_viewpoints=np.linspace(0, 1, 15).reshape(newshape_points_space(n=5)),
            test_viewpoints=np.linspace(0, 2, 18).reshape(newshape_points_space(n=6)),
            val_viewpoints=np.linspace(0, 3, 21).reshape(newshape_points_space(n=7)),
        )

        expected_selection = ViewpointSplit(
            train_viewpoints=idx_points_space(split.train_viewpoints, n=[0, 1]),
            test_viewpoints=idx_points_space(split.test_viewpoints, n=[0, 1, 2]),
            val_viewpoints=idx_points_space(split.val_viewpoints, n=[0, 1, 2, 3]),
        )

        desired_viewpt_counts = DesiredViewpointCounts(
            n_train_samples=2, n_test_samples=3, n_val_samples=4
        )
        actual_selection = split.select_n(desired_viewpt_counts)
        self.assertEqual(len(split.get_select_n_errors(desired_viewpt_counts)), 0)

        self._assert_splits_allclose(expected_selection, actual_selection)

    def test_selection_happy_path_some_not_specified(self):
        initial_viewpt_counts = ExactSampleCounts(
            n_train_samples=5, n_test_samples=6, n_val_samples=7
        )

        cases: list[tuple[DesiredViewpointCounts, ExactSampleCounts]] = [
            (
                DesiredViewpointCounts(
                    n_train_samples=None, n_test_samples=5, n_val_samples=1
                ),
                ExactSampleCounts(
                    n_train_samples=initial_viewpt_counts.n_train_samples,
                    n_test_samples=5,
                    n_val_samples=1,
                ),
            ),
            (
                DesiredViewpointCounts(
                    n_train_samples=2, n_test_samples=None, n_val_samples=1
                ),
                ExactSampleCounts(
                    n_train_samples=2,
                    n_test_samples=initial_viewpt_counts.n_test_samples,
                    n_val_samples=1,
                ),
            ),
            (
                DesiredViewpointCounts(
                    n_train_samples=2,
                    n_test_samples=5,
                    n_val_samples=initial_viewpt_counts.n_val_samples,
                ),
                ExactSampleCounts(
                    n_train_samples=2,
                    n_test_samples=5,
                    n_val_samples=initial_viewpt_counts.n_val_samples,
                ),
            ),
        ]

        for desired_viewpt_counts_in, expected_viewpt_counts in cases:
            subtest_name = repr(desired_viewpt_counts_in).replace(" ", "")
            with self.subTest(subtest_name):
                split = ViewpointSplit(
                    train_viewpoints=np.linspace(0, 1, 15).reshape(
                        newshape_points_space(n=5)
                    ),
                    test_viewpoints=np.linspace(0, 2, 18).reshape(
                        newshape_points_space(n=6)
                    ),
                    val_viewpoints=np.linspace(0, 3, 21).reshape(
                        newshape_points_space(n=7)
                    ),
                )

                expected_selection = ViewpointSplit(
                    train_viewpoints=idx_points_space(
                        split.train_viewpoints,
                        n=slice(expected_viewpt_counts.n_train_samples),
                    ),
                    test_viewpoints=idx_points_space(
                        split.test_viewpoints,
                        n=slice(expected_viewpt_counts.n_test_samples),
                    ),
                    val_viewpoints=idx_points_space(
                        split.val_viewpoints,
                        n=slice(expected_viewpt_counts.n_val_samples),
                    ),
                )

                actual_selection = split.select_n(desired_viewpt_counts_in)
                self.assertEqual(
                    len(split.get_select_n_errors(desired_viewpt_counts_in)), 0
                )
                self._assert_splits_allclose(expected_selection, actual_selection)

    def test_selection_invalid_selection(self):
        split = _new_viewpt_split_with_counts(
            viewpt_counts=ExactSampleCounts(
                n_train_samples=5, n_test_samples=6, n_val_samples=7
            ),
            fill_value=0,
        )

        invalid_counts = [
            DesiredViewpointCounts(
                n_train_samples=100, n_test_samples=None, n_val_samples=None
            ),
            DesiredViewpointCounts(
                n_train_samples=None, n_test_samples=100, n_val_samples=None
            ),
            DesiredViewpointCounts(
                n_train_samples=None, n_test_samples=None, n_val_samples=100
            ),
        ]

        for invalid_count in invalid_counts:
            count_repr = repr(invalid_count).replace(" ", "")
            with self.subTest(count_repr):
                with self.assertRaises(ValueError):
                    split.select_n(invalid_count)
                self.assertGreater(len(split.get_select_n_errors(invalid_count)), 0)

    def test_generation_happy_path(self):
        generate_errors = ViewpointSplit.get_generate_errors(
            base=self.SHOWBASE_MOCK_WITH_VIEWPOINTS,
            viewpt_counts=self.CORRECT_VIEWPOINT_COUNTS_FOR_SHOWBASE_MOCK,
        )
        self.assertEqual(len(generate_errors), 0)

        generated_split = ViewpointSplit.generate(
            base=self.SHOWBASE_MOCK_WITH_VIEWPOINTS,
            viewpt_counts=self.CORRECT_VIEWPOINT_COUNTS_FOR_SHOWBASE_MOCK,
            seed=5,
        )

        self.assertTrue(match_points_space(generated_split.train_viewpoints))
        self.assertTrue(match_points_space(generated_split.test_viewpoints))
        self.assertTrue(match_points_space(generated_split.val_viewpoints))
        self.assertEqual(
            generated_split.get_viewpt_counts(),
            self.CORRECT_VIEWPOINT_COUNTS_FOR_SHOWBASE_MOCK,
        )

        self.assertTrue(
            self._no_point_in(
                generated_split.train_viewpoints,
                [generated_split.test_viewpoints, generated_split.val_viewpoints],
                atol=1e-4,
            )
        )

        self.assertTrue(
            self._no_point_in(
                generated_split.test_viewpoints,
                [generated_split.train_viewpoints, generated_split.val_viewpoints],
                atol=1e-4,
            )
        )
        self.assertTrue(
            self._no_point_in(
                generated_split.val_viewpoints,
                [generated_split.train_viewpoints, generated_split.test_viewpoints],
                atol=1e-4,
            )
        )

    def test_generation_invalid_counts(self):
        showbase_mock = _new_showbase_mock(
            target_mesh_obj=mock.Mock(), near_plane_distance=0.01, scene_errors=[]
        )

        total_viewpoints = np.linspace(0, 1000, 120).reshape(
            newshape_points_space(n=40)
        )

        viewpoints_obj = new_vertex_only_object(
            total_viewpoints, obj_name="ViewpointsMesh"
        )

        showbase_mock.get_viewpoints_obj_mesh_path = mock.Mock(
            return_value=viewpoints_obj
        )

        invalid_counts = ExactSampleCounts(
            n_train_samples=self.N_VIEPWOINTS_IN_SHOWBASE - 1,
            n_test_samples=1,
            n_val_samples=1,
        )

        with self.assertRaises(ValueError):
            ViewpointSplit.generate(
                base=showbase_mock, viewpt_counts=invalid_counts, seed=5
            )
        generation_errors = ViewpointSplit.get_generate_errors(
            base=showbase_mock, viewpt_counts=invalid_counts
        )
        self.assertGreater(len(generation_errors), 0)

    def test_copy(self):
        INITIAL_VALUE = 0
        NEW_VALUE = 9
        initial_split = _new_viewpt_split_with_counts(
            ExactSampleCounts(n_train_samples=5, n_test_samples=3, n_val_samples=6),
            fill_value=INITIAL_VALUE,
        )
        split_copy = initial_split.copy()

        initial_split.train_viewpoints[:] = NEW_VALUE
        initial_split.test_viewpoints[:] = NEW_VALUE
        initial_split.val_viewpoints[:] = NEW_VALUE

        self.assertAlmostEqual(float(split_copy.train_viewpoints.max()), INITIAL_VALUE)
        self.assertAlmostEqual(float(split_copy.test_viewpoints.max()), INITIAL_VALUE)
        self.assertAlmostEqual(float(split_copy.val_viewpoints.max()), INITIAL_VALUE)

        initial_counts = initial_split.get_viewpt_counts()
        copy_counts = split_copy.get_viewpt_counts()

        self.assertEqual(initial_counts, copy_counts)

    def test_selftest_no_point_in(self):
        cases: list[tuple[bool, np.ndarray, list[np.ndarray]]] = [
            (
                True,
                np.array([[1, 2, 3], [0, 0, 0]], dtype=np.float32),
                [
                    np.array([[4, 5, 6]], dtype=np.float32),
                    np.array([[7, 8, 9]], dtype=np.float32),
                ],
            ),
            (
                False,
                np.array([[1, 2, 3], [0, 0, 0]], dtype=np.float32),
                [
                    np.array([[1, 2, 3], [10, 11, 12]], dtype=np.float32),
                    np.array([[7, 8, 9]], dtype=np.float32),
                ],
            ),
            (
                False,
                np.array([[1, 2, 3], [0, 0, 0]], dtype=np.float32),
                [
                    np.array([[7, 8, 9]], dtype=np.float32),
                    np.array([[1, 2, 3], [10, 11, 12]], dtype=np.float32),
                ],
            ),
        ]

        for i, (expected_containment, checked_points, other_point_sets) in enumerate(
            cases
        ):
            with self.subTest(i):
                actual_containment = self._no_point_in(
                    checked_points=checked_points,
                    other_point_sets=other_point_sets,
                    atol=1e-4,
                )

                self.assertEqual(expected_containment, actual_containment)

    def _no_point_in(
        self,
        checked_points: np.ndarray,
        other_point_sets: list[np.ndarray],
        atol: float,
    ):
        all_other_points = np.concatenate(other_point_sets, axis=DIM_POINTS_N)

        for i_checked in range(checked_points.shape[DIM_POINTS_N]):
            for i_other in range(all_other_points.shape[DIM_POINTS_N]):
                checked_point = idx_points_space(checked_points, n=[i_checked])
                other_point = idx_points_space(all_other_points, n=[i_other])

                if np.allclose(checked_point, other_point, atol=atol):
                    return False

        return True

    def _assert_splits_allclose(
        self, split1: ViewpointSplit, split2: ViewpointSplit
    ) -> None:
        self.assertTrue(
            np.allclose(
                split1.train_viewpoints,
                split2.train_viewpoints,
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                split1.test_viewpoints,
                split2.test_viewpoints,
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(split1.val_viewpoints, split2.val_viewpoints, atol=1e-4)
        )


def _new_viewpt_split_with_single_elem_and_equal_coord(
    train_coord: float, test_coord: float, val_coord: float
):
    return ViewpointSplit(
        train_viewpoints=np.array(
            [[train_coord, train_coord, train_coord]], dtype=np.float32
        ),
        test_viewpoints=np.array(
            [[test_coord, test_coord, test_coord]], dtype=np.float32
        ),
        val_viewpoints=np.array([[val_coord, val_coord, val_coord]], dtype=np.float32),
    )


def _new_viewpt_split_with_counts(
    viewpt_counts: ExactSampleCounts, fill_value: float
) -> ViewpointSplit:
    return ViewpointSplit(
        train_viewpoints=np.full(
            fill_value=fill_value,
            shape=newshape_points_space(n=viewpt_counts.n_train_samples),
        ),
        test_viewpoints=np.full(
            fill_value=fill_value,
            shape=newshape_points_space(n=viewpt_counts.n_test_samples),
        ),
        val_viewpoints=np.full(
            fill_value=fill_value,
            shape=newshape_points_space(n=viewpt_counts.n_val_samples),
        ),
    )


def _new_showbase_mock(
    target_mesh_obj: NodePath,
    scene_errors: list[str],
    near_plane_distance: float,
) -> mock.Mock:
    showbase = mock.Mock()

    cam_lens = PerspectiveLens()
    cam_lens.near = near_plane_distance

    cam_node = mock.Mock()
    cam_node.getLens = mock.Mock(return_value=cam_lens)

    showbase.get_standard_scene_format_errors = mock.Mock(return_value=scene_errors)
    showbase.get_target_obj_mesh_path = mock.Mock(return_value=target_mesh_obj)
    showbase.cam = mock.Mock()
    showbase.cam.node = mock.Mock(return_value=cam_node)

    return showbase
