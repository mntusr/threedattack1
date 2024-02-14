import itertools
import math
import sys
import unittest
from ctypes import Union
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Literal, cast, get_args
from unittest import mock

import numpy as np
import trimesh
from skimage import measure

from threedattack.rendering_core import (
    InterpolatedOccupacyField,
    OccupacyFieldSamples,
    ThreeDArea,
    VertsAndFaces,
    WarpField,
    get_d_coord_from_obj_size_along_the_shortest_axis,
    occupacy_field_samples_2_interpolator,
    verts_and_faces_2_occupacy_field_samples,
)
from threedattack.rendering_core._occupacy_fields import (
    _fail_if_pyembree_instaled,
    occupacy_field_samples_2_verts_and_faces,
)
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *

from .util import (
    load_xsortable_octagon_as_verts_and_faces,
    load_xsortable_octagon_panda3d,
)


class TestFunctions(unittest.TestCase):
    def test_fail_if_pyembree_instaled_happy_path(self):
        _fail_if_pyembree_instaled()

    def test_fail_if_pyembree_instaled_pyembree_installed(self):
        PYEMBREE_NAME = "pyembree"
        self.assertNotIn(PYEMBREE_NAME, sys.modules.keys())
        sys.modules[PYEMBREE_NAME] = mock.Mock()
        with self.assertRaises(Exception):
            _fail_if_pyembree_instaled()
        del sys.modules[PYEMBREE_NAME]

    def test_occupacy_field_samples_2_interpolator(self):
        interp_grid = np.full(
            shape=newshape_fieldgrid_occupacyfieldgrid(x=20, y=40, z=40),
            fill_value=-1.0,
        )
        upd_fieldgrid_occupacyfieldgrid(
            interp_grid, x=slice(2, 10), y=slice(3, 7), z=slice(1, 18), value_=1
        )

        D_COORD = 0.7

        X_MIN = 3
        Y_MIN = -1
        Z_MIN = 11

        samples = OccupacyFieldSamples(
            grid=interp_grid,
            d_coord=D_COORD,
            x_min=X_MIN,
            y_min=Y_MIN,
            z_min=Z_MIN,
        )

        interpolator = occupacy_field_samples_2_interpolator(samples)

        points = _grid_positions_2_points(
            occupacy_field_samples=samples,
            grid_positions=[
                (-1, -1, -1),
                (5, 5, 5),
                (5, 5, -1),
                (5, -1, 5),
                (-1, 5, 5),
                (8, 5, 15),
            ],
            extra_points=np.array([[X_MIN + 1000, Y_MIN + 1000, Z_MIN + 1000]]),
        )

        expected_values = np.array([-1, 1, -1, -1, -1, 1, -1], dtype=np.float32)
        actual_values = interpolator(points)

        self.assertTrue(np.allclose(expected_values, actual_values))
        self.assertEqual(interpolator.method, "linear")

    def test_occupacy_field_samples_2_verts_and_faces_correspondence_in_case_of_closed_objects(
        self,
    ):
        volume = np.full(shape=newshape_fieldgrid(x=10, y=13, z=16), fill_value=-1)
        upd_fieldgrid_occupacyfieldgrid(
            volume, x=slice(4, 9), y=slice(3, 8), z=slice(4, 13), value_=1
        )

        D_COORD = 1.3
        X_MIN = -3.5
        Y_MIN = -6.1
        Z_MIN = 1.3

        occupacy_field_samples = OccupacyFieldSamples(
            d_coord=D_COORD, grid=volume, x_min=X_MIN, y_min=Y_MIN, z_min=Z_MIN
        )

        actual_verts_and_faces = occupacy_field_samples_2_verts_and_faces(
            occupacy_field_samples
        )

        mc_verts, mc_faces, _, _ = measure.marching_cubes(
            level=0,
            spacing=(D_COORD, D_COORD, D_COORD),
            volume=volume.transpose(
                [DIM_FIELDGRID_Y, DIM_FIELDGRID_X, DIM_FIELDGRID_Z]
            ),
        )

        expected_verts = mc_verts.copy()
        upd_points_space(expected_verts, data="x", value_=lambda a: a + X_MIN)
        upd_points_space(expected_verts, data="y", value_=lambda a: a + Y_MIN)
        upd_points_space(expected_verts, data="z", value_=lambda a: a + Z_MIN)

        expected_faces = idx_faces_faces(mc_faces.copy(), corner=[0, 2, 1])

        self.assertTrue(np.allclose(actual_verts_and_faces.vertices, expected_verts))
        self.assertTrue(np.allclose(actual_verts_and_faces.faces, expected_faces))

    def test_occupacy_field_samples_2_verts_and_faces_closedness(self):
        volume1 = np.full(newshape_fieldgrid(x=10, y=10, z=10), -1)
        upd_fieldgrid(
            volume1, x=slice(2, None), y=slice(2, None), z=slice(2, None), value_=1
        )

        volume2 = np.full(newshape_fieldgrid(x=10, y=10, z=10), -1)
        upd_fieldgrid(volume2, x=slice(1, -1), y=slice(1, -1), z=slice(1, -1), value_=1)
        X_MIN = -3
        Y_MIN = -3
        Z_MIN = -3
        D_COORD = 1

        occupacy_field_samples1 = OccupacyFieldSamples(
            d_coord=D_COORD, grid=volume1, x_min=X_MIN, y_min=Y_MIN, z_min=Z_MIN
        )

        occupacy_field_samples2 = OccupacyFieldSamples(
            d_coord=D_COORD, grid=volume2, x_min=X_MIN, y_min=Y_MIN, z_min=Z_MIN
        )

        verts_and_faces1 = occupacy_field_samples_2_verts_and_faces(
            occupacy_field_samples1
        )
        verts_and_faces2 = occupacy_field_samples_2_verts_and_faces(
            occupacy_field_samples2
        )

        n_faces1 = verts_and_faces1.faces.shape[DIM_FACES_FACE]
        n_faces2 = verts_and_faces2.faces.shape[DIM_FACES_FACE]

        self.assertEqual(n_faces1, n_faces2)

    def test_occupacy_field_samples_2_verts_and_faces_empty_area(self):
        volume1 = np.full(newshape_fieldgrid(x=10, y=10, z=10), -1)

        verts_and_faces = occupacy_field_samples_2_verts_and_faces(
            OccupacyFieldSamples(
                d_coord=1,
                grid=volume1,
                x_min=-3,
                y_min=-3,
                z_min=-3,
            )
        )
        n_verts = verts_and_faces.vertices.shape[DIM_POINTS_N]
        n_faces = verts_and_faces.faces.shape[DIM_FACES_FACE]

        self.assertEqual(n_verts, 0)
        self.assertEqual(n_faces, 0)

    def test_occupacy_field_samples_2_verts_and_faces_1_body(self):
        volume1 = np.full(newshape_fieldgrid(x=10, y=10, z=10), -1)
        upd_fieldgrid_occupacyfieldgrid(
            volume1, x=slice(1, 8), y=slice(1, 8), z=slice(1, 8), value_=1
        )
        verts_and_faces = occupacy_field_samples_2_verts_and_faces(
            OccupacyFieldSamples(d_coord=1, grid=volume1, x_min=-3, y_min=-8, z_min=-6)
        )
        self.assertEqual(verts_and_faces.get_n_bodies(), 1)

    def test_occupacy_field_samples_2_verts_and_faces_2_bodies(self):
        volume2 = np.full(newshape_fieldgrid(x=10, y=30, z=10), -1)
        upd_fieldgrid_occupacyfieldgrid(
            volume2, x=slice(1, 8), y=slice(1, 8), z=slice(1, 8), value_=1
        )
        upd_fieldgrid_occupacyfieldgrid(
            volume2, x=slice(1, 8), y=slice(12, 19), z=slice(1, 8), value_=1
        )

        verts_and_faces = occupacy_field_samples_2_verts_and_faces(
            OccupacyFieldSamples(d_coord=1, grid=volume2, x_min=-3, y_min=-8, z_min=-6)
        )
        self.assertEqual(verts_and_faces.get_n_bodies(), 2)

    def test_occupacy_field_samples_2_verts_and_faces_2_bodies_alt(self):
        volume2 = np.full(newshape_fieldgrid(x=10, y=30, z=10), -1)
        upd_fieldgrid_occupacyfieldgrid(
            volume2, x=slice(1, 8), y=slice(1, 8), z=slice(1, 8), value_=1
        )
        upd_fieldgrid_occupacyfieldgrid(volume2, x=5, y=19, z=3, value_=1)

        verts_and_faces = occupacy_field_samples_2_verts_and_faces(
            OccupacyFieldSamples(d_coord=1, grid=volume2, x_min=-3, y_min=-8, z_min=-6)
        )
        self.assertEqual(verts_and_faces.get_n_bodies(), 2)

    def test_verts_and_faces_2_occupacy_field_samples_happy_path(self):
        verts_and_faces = load_xsortable_octagon_as_verts_and_faces()

        vert_x_coordinates = idx_points_space(verts_and_faces.vertices, data="x")
        vert_y_coordinates = idx_points_space(verts_and_faces.vertices, data="y")
        vert_z_coordinates = idx_points_space(verts_and_faces.vertices, data="z")

        N_STEPS_ALONG_SHORTEST_AXIS = 10

        samples = verts_and_faces_2_occupacy_field_samples(
            verts_and_faces=verts_and_faces,
            n_steps_along_shortest_axis=N_STEPS_ALONG_SHORTEST_AXIS,
        )

        x_coord_min = vert_x_coordinates.min()
        x_coord_max = vert_x_coordinates.max()
        y_coord_min = vert_y_coordinates.min()
        y_coord_max = vert_y_coordinates.max()
        z_coord_min = vert_z_coordinates.min()
        z_coord_max = vert_z_coordinates.max()

        x_coord_range = x_coord_max - x_coord_min
        y_coord_range = y_coord_max - y_coord_min
        z_coord_range = z_coord_max - z_coord_min

        shortest_axis_size = min(x_coord_range, y_coord_range, z_coord_range)

        d_coord = shortest_axis_size / (N_STEPS_ALONG_SHORTEST_AXIS - 1)
        self.assertAlmostEqual(samples.d_coord, d_coord)

        self.assertAlmostEqual(x_coord_min, samples.x_min)
        self.assertAlmostEqual(y_coord_min, samples.y_min)
        self.assertAlmostEqual(z_coord_min, samples.z_min)

        n_x_samples = samples.get_n_x_steps()
        n_y_samples = samples.get_n_y_steps()
        n_z_samples = samples.get_n_z_steps()

        self.assertTrue(
            (n_x_samples == N_STEPS_ALONG_SHORTEST_AXIS)
            or (n_y_samples == N_STEPS_ALONG_SHORTEST_AXIS)
            or (n_z_samples == N_STEPS_ALONG_SHORTEST_AXIS)
        )

        max_sample_x = samples.x_min + (n_x_samples - 1) * samples.d_coord
        max_sample_y = samples.y_min + (n_y_samples - 1) * samples.d_coord
        max_sample_z = samples.z_min + (n_z_samples - 1) * samples.d_coord

        d_coord_epsilon = d_coord / 10000  # to avoid numeric problems
        self.assertLess(abs(max_sample_x - x_coord_max), d_coord - d_coord_epsilon)
        self.assertLess(abs(max_sample_y - y_coord_max), d_coord - d_coord_epsilon)
        self.assertLess(abs(max_sample_z - z_coord_max), d_coord - d_coord_epsilon)

        self.assertAlmostEqual(samples.grid.min(), -1)
        self.assertAlmostEqual(samples.grid.max(), 1)

    def test_verts_and_faces_2_occupacy_field_samples_too_few_steps(self):
        with self.assertRaises(ValueError):
            verts_and_faces_2_occupacy_field_samples(
                mock.Mock(), n_steps_along_shortest_axis=1
            )

    def test_verts_and_faces_2_occupacy_field_too_no_vertex(self):
        verts_and_faces = load_xsortable_octagon_as_verts_and_faces()
        verts_and_faces.vertices = np.zeros(
            newshape_points_space(n=0), dtype=np.float32
        )
        with self.assertRaises(ValueError):
            verts_and_faces_2_occupacy_field_samples(
                verts_and_faces, n_steps_along_shortest_axis=10
            )

    def test_verts_and_faces_2_occupacy_field_samples_no_faces(self):
        verts_and_faces = load_xsortable_octagon_as_verts_and_faces()
        verts_and_faces.faces = np.zeros(newshape_faces_faces(face=0, corner=3))
        with self.assertRaises(ValueError):
            verts_and_faces_2_occupacy_field_samples(
                verts_and_faces, n_steps_along_shortest_axis=10
            )


class TestInterpolatedOccupacyField(unittest.TestCase):
    def setUp(self):
        interp_grid = np.full(
            shape=newshape_fieldgrid_occupacyfieldgrid(x=20, y=40, z=40),
            fill_value=-1.0,
        )
        upd_fieldgrid_occupacyfieldgrid(
            interp_grid, x=slice(2, 10), y=slice(3, 7), z=slice(1, 18), value_=1
        )

        self.OCCUPACY_FIELD_SAMPLES = OccupacyFieldSamples(
            grid=interp_grid, x_min=-3, y_min=-4, z_min=9, d_coord=0.6
        )
        self.OCCUPIED_POINTS = _grid_positions_2_points(
            occupacy_field_samples=self.OCCUPACY_FIELD_SAMPLES,
            grid_positions=[
                (5, 5, 5),
                (8, 5, 15),
            ],
        )
        self.NON_OCCUPIED_POINTS = _grid_positions_2_points(
            occupacy_field_samples=self.OCCUPACY_FIELD_SAMPLES,
            grid_positions=[
                (-1, -1, -1),
                (5, 5, -1),
                (5, -1, 5),
                (-1, 5, 5),
            ],
            extra_points=np.array(
                [
                    [
                        self.OCCUPACY_FIELD_SAMPLES.x_min + 1000,
                        self.OCCUPACY_FIELD_SAMPLES.y_min + 1000,
                        self.OCCUPACY_FIELD_SAMPLES.z_min + 1000,
                    ]
                ]
            ),
        )

        self.object_field = InterpolatedOccupacyField(self.OCCUPACY_FIELD_SAMPLES)

    def test_eval_at_occupied(self):
        actual_vals = self.object_field.eval_at(
            idx_points_space(self.OCCUPIED_POINTS, data="x"),
            idx_points_space(self.OCCUPIED_POINTS, data="y"),
            idx_points_space(self.OCCUPIED_POINTS, data="z"),
        )
        expected_vals = np.full(
            fill_value=1,
            shape=self.OCCUPIED_POINTS.shape[DIM_POINTS_N],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(expected_vals, actual_vals))

    def test_eval_at_non_occupied(self):
        actual_vals = self.object_field.eval_at(
            idx_points_space(self.NON_OCCUPIED_POINTS, data="x"),
            idx_points_space(self.NON_OCCUPIED_POINTS, data="y"),
            idx_points_space(self.NON_OCCUPIED_POINTS, data="z"),
        )
        expected_vals = np.full(
            fill_value=-1,
            shape=self.NON_OCCUPIED_POINTS.shape[DIM_POINTS_N],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(expected_vals, actual_vals))

    def test_get_d_coord_from_obj_size_along_the_shortest_axis_happy_path(self):
        UNMODIF_AX_BOUNDS: dict[str, tuple[float, float]] = {
            "x": (-6, 8),
            "y": (3, 13),
            "z": (0, 6),
        }
        N_STEPS_ALONG_SHORTEST_AXIS = 13

        for smallest_ax in ["x", "y", "z"]:
            with self.subTest(f"{smallest_ax=}"):
                modif_ax_bounds = UNMODIF_AX_BOUNDS.copy()
                modif_ax_bounds[smallest_ax] = (
                    modif_ax_bounds[smallest_ax][0],
                    modif_ax_bounds[smallest_ax][0] + 3.14,
                )
                area = ThreeDArea(
                    x_bounds=modif_ax_bounds["x"],
                    y_bounds=modif_ax_bounds["y"],
                    z_bounds=modif_ax_bounds["z"],
                )
                d_coord = get_d_coord_from_obj_size_along_the_shortest_axis(
                    n_steps_along_shortest_axis=N_STEPS_ALONG_SHORTEST_AXIS, area=area
                )

                self.assertAlmostEqual(
                    d_coord, 3.14 / (N_STEPS_ALONG_SHORTEST_AXIS - 1)
                )

    def test_get_d_coord_from_obj_size_along_the_shortest_axis_too_few_steps(self):
        for invalid_val in [-1, 0, 1]:
            with self.assertRaises(ValueError):
                get_d_coord_from_obj_size_along_the_shortest_axis(
                    n_steps_along_shortest_axis=invalid_val, area=mock.Mock()
                )


class TestWarpField(unittest.TestCase):
    def setUp(self):
        self.warped_field = _TestOccupacyField(
            x_bounds=(-3, 5),
            y_bounds=(-2, 7),
            z_bounds=(-6, 10),
        )
        self.WARP_DX = -1
        self.WARP_DY = -2
        self.WARP_DZ = -3

        vectors = np.zeros(newshape_points_space(n=8))
        upd_points_space(vectors, data="x", value_=self.WARP_DX)
        upd_points_space(vectors, data="y", value_=self.WARP_DY)
        upd_points_space(vectors, data="z", value_=self.WARP_DZ)

        self.warp_field = WarpField(
            control_points=self.warped_field.get_corner_points(),
            field_fn=self.warped_field,
            vectors=vectors,
        )

    def test_eval_at(self):
        points = self.warped_field.rel_points_to_absolute(
            np.array(
                [
                    [0.5, 0.3, 0.6],
                    [0.2, 0.9, 0.1],
                    [1.1, 0.9, 0.1],
                    [0.1, 1.1, 0.1],
                    [0.1, 0.1, 1.1],
                ]
            )
        )
        expected_vals = self.warped_field.eval_at(
            x=idx_points_space(points, data="x"),
            y=idx_points_space(points, data="y"),
            z=idx_points_space(points, data="z"),
        )
        actual_vals = self.warp_field.eval_at(
            x=idx_points_space(points, data="x") - self.WARP_DX,
            y=idx_points_space(points, data="y") - self.WARP_DY,
            z=idx_points_space(points, data="z") - self.WARP_DZ,
        )
        self.assertTrue(np.allclose(expected_vals, actual_vals, atol=1e-4))


_Signing = Literal["++", "+-", "--"]
SUPPORTED_SIGNINGS: list[_Signing] = ["++", "+-", "--"]


class TestOccupacyFieldSamples(unittest.TestCase):
    def test_save_and_load(self):
        grid = np.full(
            shape=newshape_fieldgrid_occupacyfieldgrid(x=10, y=8, z=23),
            fill_value=-1,
            dtype=np.float32,
        )
        upd_fieldgrid_occupacyfieldgrid(
            grid, x=slice(1, 5), y=slice(2, 6), z=slice(3, 19), value_=0.7
        )

        samples1 = OccupacyFieldSamples(
            grid=grid, d_coord=0.7, x_min=-3, y_min=49, z_min=-5.1
        )
        with TemporaryDirectory() as td:
            td_path = Path(td)
            npz_path = td_path / "saved.npz"

            samples1.save_npz(npz_path)
            samples2 = OccupacyFieldSamples.load_npz(npz_path)

        self.assertTrue(np.allclose(samples1.grid, samples2.grid))
        self.assertEqual(samples1.d_coord, samples2.d_coord)
        self.assertEqual(samples1.x_min, samples2.x_min)
        self.assertEqual(samples1.y_min, samples2.y_min)
        self.assertEqual(samples1.z_min, samples2.z_min)

    def test_get_n_x_steps(self):
        N_X_STEPS = 25
        samples = _new_empty_occupacy_field_samples(
            n_x_steps=N_X_STEPS,
            n_y_steps=13,
            n_z_steps=26,
            d_coord=0.9,
            x_min=-1,
            y_min=-2,
            z_min=-3,
        )
        self.assertEqual(N_X_STEPS, samples.get_n_x_steps())

    def test_get_n_y_steps(self):
        N_Y_STEPS = 11
        samples = _new_empty_occupacy_field_samples(
            n_x_steps=14,
            n_y_steps=N_Y_STEPS,
            n_z_steps=26,
            d_coord=0.9,
            x_min=-1,
            y_min=-2,
            z_min=-3,
        )
        self.assertEqual(N_Y_STEPS, samples.get_n_y_steps())

    def test_get_n_z_steps(self):
        N_Z_STEPS = 15
        samples = _new_empty_occupacy_field_samples(
            n_x_steps=14,
            n_y_steps=30,
            n_z_steps=N_Z_STEPS,
            d_coord=0.9,
            x_min=-1,
            y_min=-2,
            z_min=-3,
        )
        self.assertEqual(N_Z_STEPS, samples.get_n_z_steps())

    def test_get_x_coordinates(self):
        samples = _new_empty_occupacy_field_samples(
            n_x_steps=5,
            n_y_steps=30,
            n_z_steps=19,
            d_coord=0.9,
            x_min=-1.1,
            y_min=-2.2,
            z_min=-3.6,
        )
        expected_x_coordinates = np.array([-1.1, -0.2, 0.7, 1.6, 2.5])
        self.assertTrue(
            np.allclose(expected_x_coordinates, samples.get_x_coordinates(), atol=1e-4)
        )

    def test_get_y_coordinates(self):
        samples = _new_empty_occupacy_field_samples(
            n_x_steps=5,
            n_y_steps=4,
            n_z_steps=19,
            d_coord=0.7,
            x_min=-1.1,
            y_min=-2.2,
            z_min=-3.6,
        )
        expected_y_coordinates = np.array([-2.2, -1.5, -0.8, -0.1])
        self.assertTrue(
            np.allclose(expected_y_coordinates, samples.get_y_coordinates(), atol=1e-4)
        )

    def test_get_z_coordinates(self):
        samples = _new_empty_occupacy_field_samples(
            n_x_steps=5,
            n_y_steps=30,
            n_z_steps=6,
            d_coord=0.5,
            x_min=-1.1,
            y_min=-2.2,
            z_min=-3.6,
        )
        expected_z_coordinates = np.array([-3.6, -3.1, -2.6, -2.1, -1.6, -1.1])
        self.assertTrue(
            np.allclose(expected_z_coordinates, samples.get_z_coordinates(), atol=1e-4)
        )

    def test_contains_no_object_no_vals(self):
        n_steps_for_axes: list[tuple[int, int, int]] = [
            (0, 5, 4),
            (5, 0, 4),
            (5, 4, 0),
        ]
        for n_x_steps, n_y_steps, n_z_steps in n_steps_for_axes:
            with self.subTest():
                samples = _new_empty_occupacy_field_samples(
                    n_x_steps=n_x_steps,
                    n_y_steps=n_y_steps,
                    d_coord=0.6,
                    n_z_steps=n_z_steps,
                    x_min=6,
                    y_min=3,
                    z_min=8,
                )

                self.assertTrue(samples.contains_no_object())

    def test_contains_no_object_empty_field(self):
        samples = _new_empty_occupacy_field_samples(
            n_x_steps=10,
            n_y_steps=10,
            d_coord=0.6,
            n_z_steps=10,
            x_min=6,
            y_min=3,
            z_min=8,
        )
        self.assertTrue(samples.contains_no_object())

    def test_contains_no_object_object_present(self):
        samples = _new_empty_occupacy_field_samples(
            n_x_steps=10,
            n_y_steps=10,
            d_coord=0.6,
            n_z_steps=10,
            x_min=6,
            y_min=3,
            z_min=8,
        )
        upd_fieldgrid_occupacyfieldgrid(samples.grid, x=3, value_=1)
        self.assertFalse(samples.contains_no_object())


def _new_empty_occupacy_field_samples(
    n_x_steps: int,
    n_y_steps: int,
    n_z_steps: int,
    x_min: float,
    y_min: float,
    z_min: float,
    d_coord: float,
) -> OccupacyFieldSamples:
    grid = np.full(
        shape=newshape_fieldgrid_occupacyfieldgrid(
            x=n_x_steps, y=n_y_steps, z=n_z_steps
        ),
        fill_value=-1,
        dtype=np.float32,
    )
    return OccupacyFieldSamples(
        grid=grid, d_coord=d_coord, x_min=x_min, y_min=y_min, z_min=z_min
    )


class _TestOccupacyField:
    """
    This occupacy field returns with different values for each point inside of its bounds.

    The occupacy field returns with -1 for the points outside of its bounds.
    """

    def __init__(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
    ):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds

    def rel_points_to_absolute(self, rel_points: np.ndarray):
        exact_points = np.zeros_like(rel_points)

        x_range = self.x_bounds[1] - self.x_bounds[0]
        upd_points_space(
            exact_points,
            data="x",
            value_=idx_points_space(rel_points, data="x") * x_range - self.x_bounds[0],
        )

        y_range = self.y_bounds[1] - self.y_bounds[0]
        upd_points_space(
            exact_points,
            data="y",
            value_=idx_points_space(rel_points, data="y") * y_range - self.y_bounds[0],
        )

        z_range = self.z_bounds[1] - self.z_bounds[0]
        upd_points_space(
            exact_points,
            data="z",
            value_=idx_points_space(rel_points, data="z") * z_range - self.z_bounds[0],
        )

        return exact_points

    def get_corner_points(self) -> np.ndarray:
        corner_point_list = list(
            itertools.product(self.x_bounds, self.y_bounds, self.z_bounds)
        )
        corner_point_array = np.array(corner_point_list)
        return corner_point_array

    def eval_at(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        z_min, z_max = self.z_bounds

        rel_x = (x - x_min) / (x_max - x_min)
        rel_y = (y - y_min) / (y_max - y_min)
        rel_z = (z - z_min) / (z_max - z_min)

        unscaled_val = rel_x * (rel_y**2) * (rel_z ** (1.5))
        scaled_val = unscaled_val * 2 - 1

        inside_mask = (
            (x >= self.x_bounds[0])
            & (x <= self.x_bounds[1])
            & (y >= self.y_bounds[0])
            & (y <= self.y_bounds[1])
            & (z >= self.z_bounds[0])
            & (z <= self.z_bounds[1])
        )

        scaled_val[~inside_mask] = -1

        return scaled_val


def _grid_positions_2_points(
    occupacy_field_samples: OccupacyFieldSamples,
    grid_positions: list[tuple[int, int, int]],
    extra_points: np.ndarray | None = None,
):
    """
    Convert the grid indices in the grid of the occupacy field samples to actual points.

    Parameters
    ----------
    occupacy_field_samples
        The occupacy field samples with the relevant grid.
    positions
        The positions in the grid of occupacy field samples.
    extra_points
        The additional points that are specified explicitly and not via grid indices. Format: ``Points::Space``

    Returns
    -------
    v
        The generated points. Format: ``Points::Space``
    """
    point_list: list[list[float]] = []

    for position in grid_positions:
        x = occupacy_field_samples.x_min + position[0] * occupacy_field_samples.d_coord
        y = occupacy_field_samples.y_min + position[1] * occupacy_field_samples.d_coord
        z = occupacy_field_samples.z_min + position[2] * occupacy_field_samples.d_coord

        point_list.append([x, y, z])

    all_points = np.array(point_list, dtype=np.float32)
    if extra_points is not None:
        all_points = np.concatenate(
            [all_points, extra_points],
            axis=DIM_POINTS_N,
        )
    return all_points
