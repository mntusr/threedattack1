import sys
import unittest
from typing import Self

import numpy as np

v = sys.path
a = 2

from threedattack.rendering_core import (
    ScaledStandingAreas,
    ThreeDArea,
    ThreeDPoint,
    ThreeDSize,
)
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *


class TestScaledStandingAreas(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ORIGINAL_SIZE = ThreeDSize(x_size=1, y_size=2, z_size=3)
        self.SIZE_MULTIPLIER = 1.5
        self.EXTRA_OFFSET = 0.3
        self.areas = ScaledStandingAreas(
            original_size=self.ORIGINAL_SIZE,
            size_multiplier=self.SIZE_MULTIPLIER,
            extra_offset_after_size_mult=self.EXTRA_OFFSET,
        )
        self.ORIGINS: list[ThreeDPoint] = [
            ThreeDPoint(0, 0, 0),
            ThreeDPoint(5, 4, 3),
        ]

    def test_get_original_size(self):
        self.assertEqual(self.ORIGINAL_SIZE, self.areas.get_original_size())

    def test_get_size_multiplier(self):
        self.assertEqual(self.SIZE_MULTIPLIER, self.areas.get_size_multiplier())

    def test_extra_offset_relations(self) -> None:
        for origin in self.ORIGINS:
            for include_outside_offset in [True, False]:
                with self.subTest(f"origin={origin}"):
                    with self.subTest(
                        f"include_outside_offset={include_outside_offset}"
                    ):
                        areas1 = ScaledStandingAreas(
                            original_size=self.ORIGINAL_SIZE,
                            size_multiplier=self.SIZE_MULTIPLIER,
                            extra_offset_after_size_mult=0.4,
                        )
                        areas2 = ScaledStandingAreas(
                            original_size=self.ORIGINAL_SIZE,
                            size_multiplier=self.SIZE_MULTIPLIER,
                            extra_offset_after_size_mult=2,
                        )

                        area1 = areas1.get_full_area(
                            include_outside_offset=True, origin_of_obj=origin
                        )
                        area2 = areas2.get_full_area(
                            include_outside_offset=True, origin_of_obj=origin
                        )

                        x_min1, x_max1 = area1.get_x_bounds()
                        y_min1, y_max1 = area1.get_y_bounds()
                        z_min1, z_max1 = area1.get_z_bounds()
                        x_min2, x_max2 = area2.get_x_bounds()
                        y_min2, y_max2 = area2.get_y_bounds()
                        z_min2, z_max2 = area2.get_z_bounds()

                        self.assertLess(x_min2, x_min1)
                        self.assertLess(y_min2, y_min1)
                        self.assertLess(z_min2, z_min1)

                        self.assertGreater(x_max2, x_max1)
                        self.assertGreater(y_max2, y_max1)
                        self.assertGreater(z_max2, z_max1)

    def test_origin_change_effect(self) -> None:
        for include_outside_offset in [True, False]:
            with self.subTest(f"include_outside_offset={include_outside_offset}"):
                origin1 = self.ORIGINS[0]
                origin2 = self.ORIGINS[1]

                area1 = self.areas.get_full_area(
                    origin_of_obj=origin1, include_outside_offset=include_outside_offset
                )

                x_min1, x_max1 = area1.get_x_bounds()
                y_min1, y_max1 = area1.get_y_bounds()
                z_min1, z_max1 = area1.get_z_bounds()

                area2 = self.areas.get_full_area(
                    origin_of_obj=origin2, include_outside_offset=include_outside_offset
                )

                x_min2, x_max2 = area2.get_x_bounds()
                y_min2, y_max2 = area2.get_y_bounds()
                z_min2, z_max2 = area2.get_z_bounds()

                self.assertAlmostEqual(x_min1 - x_min2, origin1.x - origin2.x)
                self.assertAlmostEqual(y_min1 - y_min2, origin1.y - origin2.y)
                self.assertAlmostEqual(z_min1 - z_min2, origin1.z - origin2.z)
                self.assertAlmostEqual(x_max1 - x_max2, origin1.x - origin2.x)
                self.assertAlmostEqual(y_max1 - y_max2, origin1.y - origin2.y)
                self.assertAlmostEqual(z_max1 - z_max2, origin1.z - origin2.z)


class TestThreeDArea(unittest.TestCase):
    def setUp(self):
        self.X_BOUNDS = (-3, 6)
        self.Y_BOUNDS = (-4, 15)
        self.Z_BOUNDS = (-1, 9)
        self.area = ThreeDArea(
            x_bounds=self.X_BOUNDS,
            y_bounds=self.Y_BOUNDS,
            z_bounds=self.Z_BOUNDS,
        )

    def test_select_points_outside(self):
        x_min, x_max = self.X_BOUNDS
        y_min, y_max = self.Y_BOUNDS
        z_min, z_max = self.Z_BOUNDS
        points = np.array(
            [
                [x_min * 100, 0, 0],
                [x_max * 100, 0, 0],
                [0, y_min * 100, 0],
                [0, y_max * 100, 0],
                [0, 0, z_min * 100],
                [0, 0, z_max * 100],
                [x_min * 0.95, 0, 0],
                [x_max * 0.95, 0, 0],
                [0, y_min * 0.95, 0],
                [0, y_max * 0.95, 0],
                [0, 0, z_min * 0.95],
                [0, 0, z_max * 0.95],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )

        target_selected_points = idx_points(points, n=slice(6))
        actual_selected_points = self.area.select_points_outside(points)
        self.assertTrue(match_points_space(actual_selected_points))
        self.assertTrue(np.array_equal(target_selected_points, actual_selected_points))

    def test_clip_points_to_inside(self):
        x_min, x_max = self.X_BOUNDS
        y_min, y_max = self.Y_BOUNDS
        z_min, z_max = self.Z_BOUNDS

        non_clipped_points = np.array(
            [
                [x_max * 10, 0, 0],
                [x_min * 10, 0, 0],
                [0, y_max * 10, 0],
                [0, y_min * 10, 0],
                [0, 0, z_max * 10],
                [0, 0, z_min * 10],
                [0, 0, 0.1],
            ]
        )

        expected_clipped_points = np.array(
            [
                [x_max, 0, 0],
                [x_min, 0, 0],
                [0, y_max, 0],
                [0, y_min, 0],
                [0, 0, z_max],
                [0, 0, z_min],
                [0, 0, 0.1],
            ]
        )

        actual_clipped_points = self.area.clip_points_to_inside(non_clipped_points)
        self.assertTrue(match_points_space(actual_clipped_points))

        self.assertTrue(
            np.allclose(actual_clipped_points, expected_clipped_points, atol=1e-5)
        )

    def test_make_rel_points_absolute(self):
        rel_points = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.5, 1.0],
            ]
        )
        x_min, x_max = self.X_BOUNDS
        miny, maxy = self.Y_BOUNDS
        minz, maxz = self.Z_BOUNDS
        expected_abs_points = np.array(
            [[x_min, maxy, minz], [x_max, miny + (maxy - miny) / 2, maxz]]
        )

        actual_abs_points = self.area.make_rel_points_absolute(rel_points=rel_points)
        self.assertTrue(match_points_space(actual_abs_points))

        self.assertTrue(np.allclose(expected_abs_points, actual_abs_points, atol=1e-5))

    def test_get_corners(self):
        actual_corners = self.area.get_corners()

        self.assertTrue(match_points_space(actual_corners))

        x_min, x_max = self.X_BOUNDS
        y_min, y_max = self.Y_BOUNDS
        z_min, z_max = self.Z_BOUNDS

        expected_corners = np.array(
            [
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max],
            ]
        )
        self.assertTrue(
            self._are_point_sets_equal(actual_corners, expected_corners, atol=1e-4)
        )

    def _are_point_sets_equal(
        self, points1: np.ndarray, points2: np.ndarray, atol: float
    ) -> bool:
        """
        Check whether the two point clouds contain the same points regardless of their orders.

        Parameters
        ----------
        points1
            The first point cloud. Format: ``Points::Space``
        points2
            The second point cloud. Fromat: ``Points::Space``
        atol
            The absolute tolerance when comparing the points.

        Returns
        -------
        v
            True if the two point sets contain the same points. Otherwise False.
        """
        self.assertEqual(points1.shape, points2.shape)

        for point1_idx in range(points1.shape[DIM_POINTS_N]):
            point_found = False
            for point2_idx in range(points2.shape[DIM_POINTS_N]):
                if np.allclose(
                    idx_points_space(points1, n=point1_idx),
                    idx_points_space(points2, n=point2_idx),
                    atol=atol,
                ):
                    point_found = True

            if not point_found:
                return False

        return True

    def test_selftest_assert_point_set_equal_false(self):
        points1 = np.array(
            [
                [5, 2, 9],
                [5, 5, 9],
                [5, 5, 6],
                [3, 5, 6],
            ],
            dtype=np.float32,
        )
        points2 = points1.copy()
        upd_points_space(points1, n=3, value_=0)

        self.assertFalse(self._are_point_sets_equal(points1, points2, atol=1e-4))

    def test_get_x_bounds(self):
        self.assertEqual(self.area.get_x_bounds(), self.X_BOUNDS)

    def test_get_y_bounds(self):
        self.assertEqual(self.area.get_y_bounds(), self.Y_BOUNDS)

    def test_get_z_bounds(self):
        self.assertEqual(self.area.get_z_bounds(), self.Z_BOUNDS)

    def test_init(self):
        cases: list[
            tuple[str, tuple[float, float], tuple[float, float], tuple[float, float]]
        ] = [
            ("x_bounds_invalid", (10, 1), self.Y_BOUNDS, self.Z_BOUNDS),
            ("y_bounds_invalid", self.X_BOUNDS, (10, 1), self.Z_BOUNDS),
            ("z_bounds_invalid", self.X_BOUNDS, self.Y_BOUNDS, (10, 1)),
        ]
        for case_name, x_bounds, y_bounds, z_bounds in cases:
            with self.subTest(case_name):
                with self.assertRaises(ValueError):
                    ThreeDArea(
                        x_bounds=x_bounds,
                        y_bounds=y_bounds,
                        z_bounds=z_bounds,
                    )

    def test_get_min_size(self):
        greater_val1 = 9
        greater_val2 = 6.3
        smaller_val = 3

        greater_dim1_min = -6.3
        greater_dim2_min = 0
        smaller_dim_min = -1

        cases: list[tuple[str, ThreeDArea]] = [
            (
                "x_greatest",
                ThreeDArea(
                    x_bounds=(smaller_dim_min, smaller_dim_min + smaller_val),
                    y_bounds=(greater_dim1_min, greater_val1 + greater_dim1_min),
                    z_bounds=(greater_dim2_min, greater_val2 + greater_dim2_min),
                ),
            ),
            (
                "y_greatest",
                ThreeDArea(
                    x_bounds=(greater_dim1_min, greater_val1 + greater_dim1_min),
                    y_bounds=(smaller_dim_min, smaller_dim_min + smaller_val),
                    z_bounds=(greater_dim2_min, greater_val2 + greater_dim2_min),
                ),
            ),
            (
                "z_greatest",
                ThreeDArea(
                    x_bounds=(greater_dim1_min, greater_val1 + greater_dim1_min),
                    y_bounds=(greater_dim2_min, greater_val2 + greater_dim2_min),
                    z_bounds=(smaller_dim_min, smaller_dim_min + smaller_val),
                ),
            ),
        ]

        for subtest_name, area in cases:
            with self.subTest(subtest_name):
                actual_min_size = area.get_min_size()
                self.assertAlmostEqual(actual_min_size, smaller_val)
