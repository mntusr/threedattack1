import unittest

import numpy as np
import trimesh

from threedattack.rendering_core import TwoDAreas, TwoDSize, VertsAndFaces
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *

from .util import load_xsortable_octagon_as_verts_and_faces


class TestTwoDAreas(unittest.TestCase):
    def test_idx(self):
        n_areas = 5
        areas = TwoDAreas(
            x_mins_including=np.arange(0, n_areas),
            x_maxes_excluding=np.arange(0, n_areas) + 9,
            y_mins_including=np.arange(0, n_areas) + 1,
            y_maxes_excluding=np.arange(0, n_areas) + 13,
        )

        actual_selected_areas = areas.idx_areas([2, 3])
        expected_selected_areas = TwoDAreas(
            x_mins_including=np.array([2, 3]),
            x_maxes_excluding=np.array([11, 12]),
            y_mins_including=np.array([3, 4]),
            y_maxes_excluding=np.array([15, 16]),
        )

        self.assertTrue(
            np.array_equal(
                actual_selected_areas.x_mins_including,
                expected_selected_areas.x_mins_including,
            )
        )
        self.assertTrue(
            np.array_equal(
                actual_selected_areas.x_maxes_excluding,
                expected_selected_areas.x_maxes_excluding,
            )
        )
        self.assertTrue(
            np.array_equal(
                actual_selected_areas.y_mins_including,
                expected_selected_areas.y_mins_including,
            )
        )
        self.assertTrue(
            np.array_equal(
                actual_selected_areas.y_maxes_excluding,
                expected_selected_areas.y_maxes_excluding,
            )
        )


class TestTwoDSize(unittest.TestCase):
    def test_is_positive(self):
        sizes_and_positivities: list[tuple[TwoDSize, bool]] = [
            (TwoDSize(0, 5), False),
            (TwoDSize(5, 0), False),
            (TwoDSize(5, 5), True),
        ]

        for size, positivity in sizes_and_positivities:
            with self.subTest(repr(size).replace(" ", "")):
                self.assertEqual(size.is_positive(), positivity)

    def test_init_happy_path(self):
        expected_x_size = 701
        expected_y_size = 560

        sizes = TwoDSize(x_size=expected_x_size, y_size=expected_y_size)
        self.assertEqual(expected_x_size, sizes.x_size)
        self.assertEqual(expected_y_size, sizes.y_size)

    def test_init_happy_path2(self):
        expected_x_size = 701
        expected_y_size = 560

        sizes = TwoDSize(expected_x_size, expected_y_size)
        self.assertEqual(expected_x_size, sizes.x_size)
        self.assertEqual(expected_y_size, sizes.y_size)

    def test_init_invalid_size(self):
        cases: list[tuple[str, int, int]] = [
            ("invalid_x_size", -1, 130),
            ("invalid_y_size", 130, -1),
        ]
        for case_name, x_size, y_size in cases:
            with self.subTest(case_name):
                with self.assertRaises(ValueError):
                    TwoDSize(x_size=x_size, y_size=y_size)


class TestVertsAndFaces(unittest.TestCase):
    def test_get_n_bodies_1_body(self):
        verts_and_faces = load_xsortable_octagon_as_verts_and_faces()
        self.assertEqual(verts_and_faces.get_n_bodies(), 1)

    def test_get_n_bodies_5_bodies(self):
        verts_and_faces_original = load_xsortable_octagon_as_verts_and_faces()

        trimesh_mesh = trimesh.Trimesh(
            faces=verts_and_faces_original.faces,
            vertices=verts_and_faces_original.vertices,
        )
        N_BODIES = 5

        total_mesh = trimesh_mesh
        for i in range(N_BODIES - 1):
            new_mesh = trimesh_mesh.copy()
            new_mesh.apply_translation((0, 10 * (i + 1), 0))
            total_mesh = total_mesh + new_mesh

        total_mesh.merge_vertices(digits_norm=4)  # type: ignore

        total_verts_and_faces = VertsAndFaces(
            faces=total_mesh.faces, vertices=total_mesh.vertices
        )

        self.assertEqual(total_verts_and_faces.get_n_bodies(), N_BODIES)

    def test_get_n_bodies_empty_mesh(self):
        verts_and_faces = VertsAndFaces(
            vertices=np.zeros(newshape_points_space(n=0), dtype=np.float32),
            faces=np.zeros(newshape_faces_faces(face=0, corner=3), dtype=np.float32),
        )
        n_bodies = verts_and_faces.get_n_bodies()
        self.assertEqual(n_bodies, 1)
