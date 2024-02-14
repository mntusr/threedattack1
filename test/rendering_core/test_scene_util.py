import math
import unittest
from email.errors import NonPrintableDefect
from pathlib import Path
from typing import cast
from unittest import mock

import gltf
import numpy as np
import trimesh
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Camera,
    GeomNode,
    GeomPrimitive,
    GeomTrifans,
    Lens,
    LensNode,
    LPoint3f,
    NodePath,
    PandaNode,
    PerspectiveLens,
    Point2,
)
from panda3d.fx import FisheyeLens, OSphereLens

from threedattack.dataset_model import CamProjSpec, DepthsWithMasks
from threedattack.rendering_core import (
    Panda3dAssumptionViolation,
    TwoDSize,
    depth_map_2_point_cloud,
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
    get_sky_mask_limit,
    get_vertex_count,
    get_vertex_face_copy_most_common_errors,
    get_vertex_positions_copy,
    get_vertices_and_faces_copy,
    invert_projection,
    is_geom_node_obj,
    project_points_to_screen,
    put_obj,
    set_col_in_vertex_data,
    zbuf_2_depth_and_mask,
)
from threedattack.rendering_core._scene_util import get_projection_mat_col
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *

from .util import (
    load_xsortable_octagon_as_verts_and_faces,
    load_xsortable_octagon_panda3d,
    new_vertex_only_object,
)


class TestFunctions(unittest.TestCase):
    def setUp(self) -> None:
        x, y = np.meshgrid(np.linspace(1, 3, 4), np.linspace(0.5, 2, 4))
        depth_map = np.expand_dims(x * y, axis=(0, 1))
        masks = np.ones_like(depth_map, dtype=np.bool_)
        upd_im(masks, n=0, c=CAT_IM_DEPTHMASKS_C_MASK, h=0, w=0, value_=0)
        self.depth_data = DepthsWithMasks(depths=depth_map, masks=masks)

    def test_get_obj_size_happy_path(self):
        obj_mock = mock.Mock()
        min_point = LPoint3f()
        min_point.x = -5
        min_point.y = 3
        min_point.z = 8

        expected_size_x = 9
        expected_size_y = 13
        expected_size_z = 7

        max_point = LPoint3f()
        max_point.x = min_point.x + expected_size_x
        max_point.y = min_point.y + expected_size_y
        max_point.z = min_point.z + expected_size_z

        obj_mock.getTightBounds = mock.Mock(return_value=(min_point, max_point))

        actual_size = get_obj_size(obj_mock)

        self.assertAlmostEqual(actual_size.x_size, expected_size_x)
        self.assertAlmostEqual(actual_size.y_size, expected_size_y)
        self.assertAlmostEqual(actual_size.z_size, expected_size_z)

    def test_get_obj_size_empty_obj(self):
        obj_mock = mock.Mock()
        obj_mock.getTightBounds = mock.Mock(return_value=None)

        with self.assertRaises(ValueError):
            get_obj_size(obj_mock)

    def test_depth_map_2_point_cloud_invalid_depth_map_count(self):
        lens = PerspectiveLens()
        lens.setNear(0.1)
        lens.setFar(10)

        invalid_depth_datas: list[tuple[str, DepthsWithMasks]] = [
            (
                "no_item",
                DepthsWithMasks(
                    depths=np.zeros((0, 0, 0, 0)), masks=np.zeros((0, 0, 0, 0))
                ),
            ),
            (
                "too_many_items",
                DepthsWithMasks(
                    depths=np.zeros((10, 1, 50, 50)), masks=np.zeros((10, 1, 50, 50))
                ),
            ),
        ]

        for name, data in invalid_depth_datas:
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    depth_map_2_point_cloud(data, get_cam_proj_spec_for_lens(lens))

    def test_depth_map_2_point_cloud_invalid_lenses(self) -> None:
        inf_near_plane_lens = PerspectiveLens()
        inf_near_plane_lens.setNearFar(math.inf, 100)

        inf_far_plane_lens = PerspectiveLens()
        inf_far_plane_lens.setNearFar(0.1, math.inf)

        swapped_planes_lens = PerspectiveLens()
        swapped_planes_lens.setNearFar(100, 0.1)

        non_perspective_lens = OSphereLens()

        invalid_lenses: list[tuple[str, Lens]] = [
            ("inf_near_plane_lens", inf_near_plane_lens),
            ("inf_far_plane_lens", inf_far_plane_lens),
            ("non_perspective_lens", non_perspective_lens),
            ("swapped_planes_lens", swapped_planes_lens),
        ]

        for lens_name, lens in invalid_lenses:
            with self.subTest(lens_name):
                with self.assertRaises(Panda3dAssumptionViolation):
                    depth_map_2_point_cloud(
                        self.depth_data, get_cam_proj_spec_for_lens(lens)
                    )

    def test_zbuf_2_depth_and_mask_happy_path(self) -> None:
        original_depth_map = np.linspace(2, 10, 30).reshape(
            newshape_im_depthmaps(n=1, h=5, w=6)
        )
        max_depths = [
            original_depth_map.max() + 0.1,
            original_depth_map.min()
            + (original_depth_map.max() - original_depth_map.min()) * 0.6,
        ]
        for max_depth in max_depths:
            with self.subTest(f"{max_depth=}"):
                lens = PerspectiveLens()
                lens.setNearFar(1, original_depth_map.max())

                flat_depth_map = original_depth_map.flatten()

                depth_vecs = np.stack(
                    [
                        np.zeros_like(flat_depth_map),
                        flat_depth_map,
                        np.zeros_like(flat_depth_map),
                        np.ones_like(flat_depth_map),
                    ]
                )

                proj_mat = get_projection_mat_col(lens)

                zw_part = idx_mat(proj_mat, row=[2, 3])

                zbuf_flat_homog = zw_part @ depth_vecs

                zbuf = (zbuf_flat_homog[0] / zbuf_flat_homog[1]) / 2 + 0.5
                zbuf = zbuf.reshape(original_depth_map.shape)

                calc_depth_with_mask = zbuf_2_depth_and_mask(
                    zbuf_data=zbuf, camera_lens=lens, max_depth=max_depth
                )

                self.assertGreater(calc_depth_with_mask.masks.astype(np.int32).sum(), 0)

                self.assertTrue(
                    np.allclose(
                        original_depth_map[calc_depth_with_mask.masks],
                        calc_depth_with_mask.depths[calc_depth_with_mask.masks],
                        atol=1e-5,
                    )
                )
                self.assertLess(
                    original_depth_map[calc_depth_with_mask.masks].max(), max_depth
                )
                if calc_depth_with_mask.masks.astype(np.int32).sum() > 0:
                    self.assertAlmostEqual(
                        float(
                            calc_depth_with_mask.depths[
                                ~calc_depth_with_mask.masks
                            ].min()
                        ),
                        0,
                        places=4,
                    )
                else:
                    if original_depth_map.max() < max_depth:
                        self.fail(
                            "At least one masked pixel should exist if the max_depth is less than maximal value of the input depth map."
                        )

    def test_zbuf_2_depth_and_mask_invalid_input(self):
        inv1 = np.full(shape=(5, 1, 10, 10), fill_value=0.5)
        inv2 = np.full(shape=(0, 0, 0, 0), fill_value=0.5)
        valid_zbuf = np.full(shape=(1, 1, 10, 10), fill_value=0.5)
        lens = PerspectiveLens()
        lens.setNearFar(1, 500)

        invalid_arrays: list[tuple[str, np.ndarray, float]] = [
            ("too_many_arrays", inv1, 100),
            ("no_array", inv2, 100),
            ("invalid_max_depth", valid_zbuf, -1),
        ]

        for arr_name, arr, max_depth in invalid_arrays:
            with self.subTest(arr_name):
                with self.assertRaises(ValueError):
                    zbuf_2_depth_and_mask(
                        zbuf_data=arr, camera_lens=lens, max_depth=max_depth
                    )

    def test_get_sky_mask_limit_happy_path(self):
        FAR_PLANE = 500
        lens = PerspectiveLens()
        lens.setNear(0.1)
        lens.setFar(FAR_PLANE)

        sky_mask_limit = get_sky_mask_limit(lens)

        self.assertFalse(math.isinf(sky_mask_limit))
        self.assertGreaterEqual(sky_mask_limit, FAR_PLANE * 0.95)
        self.assertLessEqual(sky_mask_limit, FAR_PLANE * 1.1)

    def test_get_sky_mask_limit_infinite_planes(self):
        infinite_far_plane_lens = PerspectiveLens()
        infinite_far_plane_lens.setNearFar(0.1, math.inf)

        infinite_near_plane_lens = PerspectiveLens()
        infinite_near_plane_lens.setNearFar(math.inf, 500)

        swapped_planes_lens = PerspectiveLens()
        swapped_planes_lens.setNearFar(500, 0.1)

        lenses: list[tuple[str, PerspectiveLens]] = [
            ("infinite_near_plane_lens", infinite_near_plane_lens),
            ("infinite_far_plane_lens", infinite_far_plane_lens),
            ("swapped_planes_lens", swapped_planes_lens),
        ]

        for lens_name, lens in lenses:
            with self.subTest(lens_name):
                with self.assertRaises(Panda3dAssumptionViolation):
                    get_sky_mask_limit(lens)

    def test_get_true_near_far_planes_happy_path(self):
        NEAR = 0.1
        FAR = 1000
        lens = PerspectiveLens()
        lens.setNearFar(NEAR, FAR)

        actual_near, actual_far = get_near_far_planes_safe(lens)

        self.assertAlmostEqual(actual_near, NEAR, places=5)
        self.assertAlmostEqual(actual_far, FAR, places=5)

    def test_get_true_near_far_planes_invalid_lens(self):
        infinite_far_plane_lens = PerspectiveLens()
        infinite_far_plane_lens.setNearFar(0.1, math.inf)

        infinite_near_plane_lens = PerspectiveLens()
        infinite_near_plane_lens.setNearFar(math.inf, 500)

        swapped_planes_lens = PerspectiveLens()
        swapped_planes_lens.setNearFar(500, 0.1)

        lenses: list[tuple[str, PerspectiveLens]] = [
            ("infinite_near_plane_lens", infinite_near_plane_lens),
            ("infinite_far_plane_lens", infinite_far_plane_lens),
            ("swapped_planes_lens", swapped_planes_lens),
        ]

        for lens_name, lens in lenses:
            with self.subTest(lens_name):
                with self.assertRaises(Panda3dAssumptionViolation):
                    get_sky_mask_limit(lens)

    def test_find_node_node_found(self) -> None:
        root_node = NodePath("root")
        child = NodePath("child")
        child.reparentTo(root_node)

        found_node = find_node(root_node, "**/child")
        self.assertEqual(child, found_node)

    def test_find_node_node_not_found(self) -> None:
        root_node = NodePath("root")

        _ = NodePath("nonChild")

        found_node = find_node(root_node, "**/nonChild")
        self.assertIsNone(found_node)

    def test_get_object_size_happy_path1(self) -> None:
        vertex_positions = np.array(
            [
                [0.5, 5, 9],
                [0.4, 2, 9],
            ]
        )
        calc_size = get_ob_size_from_vertices(vertex_positions)

        self.assertAlmostEqual(calc_size.x_size, 0.1, places=5)
        self.assertAlmostEqual(calc_size.y_size, 3, places=5)
        self.assertAlmostEqual(calc_size.z_size, 0, places=5)

    def test_get_object_size_happy_path2(self) -> None:
        vertex_positions = np.zeros((0, 0))
        calc_size = get_ob_size_from_vertices(vertex_positions)

        self.assertAlmostEqual(calc_size.x_size, 0, places=5)
        self.assertAlmostEqual(calc_size.y_size, 0, places=5)
        self.assertAlmostEqual(calc_size.y_size, 0, places=5)

    def test_get_object_size_invalid_data(self) -> None:
        with self.assertRaises(ValueError):
            get_ob_size_from_vertices(np.zeros((1, 2, 4)))

    def test_get_vertex_positions_copy_happy_path(self):
        expected_verts_pos_copy = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ]
        )

        obj = new_vertex_only_object(expected_verts_pos_copy, "my_obj")

        actual_vert_pos_copy = get_vertex_positions_copy(obj)
        self.assertTrue(match_points_space(actual_vert_pos_copy))
        actual_vert_pos_copy = _sort_by_x(actual_vert_pos_copy)

        self.assertTrue(
            np.allclose(expected_verts_pos_copy, actual_vert_pos_copy, atol=1e-5)
        )

    def test_get_vertex_positions_copy_no_vertex(self):
        expected_verts_pos_copy = np.zeros((0, 0))

        obj = new_vertex_only_object(expected_verts_pos_copy, "my_obj")
        obj2 = new_vertex_only_object(expected_verts_pos_copy, "my_obj")
        obj2.node().removeGeom(0)

        obj_with_reason_of_no_verts: list[tuple[str, NodePath]] = [
            ("no_vertex_in_vertex_data", obj),
            ("no_geom", obj2),
        ]

        for problem_desc, obj in obj_with_reason_of_no_verts:
            with self.subTest(problem_desc):
                with self.assertRaises(Panda3dAssumptionViolation):
                    get_vertex_positions_copy(obj)

    def test_get_vertex_count_happy_path(self):
        obj1 = new_vertex_only_object(np.array([[1, 2, 3], [4, 5, 6]]), "obj1")
        obj2 = new_vertex_only_object(np.zeros((0, 0)), "obj2")

        objects: list[tuple[str, NodePath, int]] = [
            ("simplest_case", obj1, 2),
            ("zero_vertex_with_geom_node", obj2, 0),
        ]

        for obj_name, obj, expected_vert_count in objects:
            with self.subTest(obj_name):
                actual_vertex_count = get_vertex_count(obj)

                self.assertEqual(expected_vert_count, actual_vertex_count)

    def test_get_vertex_count_no_geom_node(self):
        with self.assertRaises(Panda3dAssumptionViolation):
            get_vertex_count(NodePath())

    def test_put_obj_happy_path(self):
        obj1 = new_vertex_only_object(
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
            "obj1",
        )

        expected_new_obj1_verts = np.array(
            [
                [9, 2, 3],
                [3, 5, 6],
            ]
        )
        obj2 = new_vertex_only_object(
            expected_new_obj1_verts,
            "obj1",
        )

        parent = NodePath("top")
        obj1.reparentTo(parent)

        put_obj(obj1, get_obj_copy(obj2))
        obj1_new = find_node(parent, "**/obj1")

        self.assertIsNotNone(obj1_new)
        assert obj1_new is not None

        actual_new_obj1_verts = get_vertex_positions_copy(obj1_new)

        self.assertTrue(np.allclose(expected_new_obj1_verts, actual_new_obj1_verts))

    def test_put_obj_no_parent(self) -> None:
        obj1 = new_vertex_only_object(
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
            "obj1",
        )
        obj2 = new_vertex_only_object(
            np.array(
                [
                    [9, 2, 3],
                    [3, 5, 6],
                ]
            ),
            "obj1",
        )
        obj1.detachNode()
        with self.assertRaises(Panda3dAssumptionViolation):
            put_obj(obj1, get_obj_copy(obj2))

    def test_project_points_to_screen_happy_path(self) -> None:
        # set projected points
        rendering_resolution = TwoDSize(x_size=600, y_size=200)
        expected_projected_points = np.array(
            [
                [-10, 200],
                [50, 100],
                [50, 205],
                [50, -10],
            ],
            dtype=np.float32,
        )
        z_values = np.array([5, 9, 3, 2], dtype=np.float32)
        n_points = 4

        # 1. create scene
        base = ShowBase(windowType="offscreen")

        # everything else should be in a try: ... block to
        # make sure `base` is destroyed at the end
        try:
            # 2. configure base
            rendering_resolution = TwoDSize(800, 600)  # fake resolution

            base.cam.node().getLens().setFilmSize(1, 600 / 800)
            base.cam.setPos((0, 0, 5))

            # 3. invert_projection
            homog_projected_points = np.zeros(
                newshape_points_aplane(n=n_points), dtype=np.float32
            )
            upd_points_aplane(
                homog_projected_points,
                data="x",
                value_=(
                    idx_points_plane(expected_projected_points, data="x")
                    / rendering_resolution.x_size
                    * 2
                    - 1
                )
                * z_values,
            )
            upd_points_aplane(
                homog_projected_points,
                data="y",
                value_=(
                    idx_points_plane(expected_projected_points, data="y")
                    / rendering_resolution.y_size
                    * 2
                    - 1
                )
                * z_values,
            )
            upd_points_aplane(homog_projected_points, data="w", value_=z_values)

            cam_proj_spec = get_cam_proj_spec_for_showbase(base)
            backprojected_points = invert_projection(
                homog_im_points=homog_projected_points,
                invertable_proj_mat=cam_proj_spec.proj_mat,
            )
            upd_points_space(backprojected_points, data="z", value_=lambda a: a + 5)

            # 2. use project_points_to_screen
            actual_projected_points = project_points_to_screen(
                points=backprojected_points,
                base=base,
                rendering_resolution=rendering_resolution,
            )

            self.assertTrue(
                match_points_plane(actual_projected_points, {"n": n_points})
            )
            self.assertTrue(
                np.allclose(expected_projected_points, actual_projected_points)
            )
        finally:
            base.destroy()

    def test_project_points_to_screen_invalid_size(self):
        with self.assertRaises(ValueError):
            project_points_to_screen(
                points=mock.Mock(),
                base=mock.Mock(),
                rendering_resolution=TwoDSize(0, 9),
            )

    def test_get_bounding_rectangles_2d_happy_path(self):
        points = np.array([[5, 3.9], [-10.1, 2], [6, 8]])

        rect = get_bounding_rectangle_2d(points)

        self.assertTrue(np.array_equal(rect.x_maxes_excluding, np.array([7])))
        self.assertTrue(np.array_equal(rect.x_mins_including, np.array([-10])))
        self.assertTrue(np.array_equal(rect.y_maxes_excluding, np.array([9])))
        self.assertTrue(np.array_equal(rect.y_mins_including, np.array([2])))

    def test_get_bounding_rectangles_2d_no_points(self):
        with self.assertRaises(ValueError):
            get_bounding_rectangle_2d(
                np.zeros(newshape_points_plane(n=0), dtype=np.float32)
            )

    def test_is_geom_node_obj(self):
        OBJ_NAME = "obj1"
        cases: list[tuple[PandaNode, bool]] = [
            (GeomNode(OBJ_NAME), True),
            (PandaNode(OBJ_NAME), False),
            (Camera(OBJ_NAME), False),
        ]

        for obj, expected_result in cases:
            with self.subTest(type(obj).name):
                self.assertEqual(is_geom_node_obj(NodePath(obj)), expected_result)

    def test_get_vertices_and_faces_copy_happy_path(self) -> None:
        base = ShowBase(windowType="offscreen")
        try:
            loaded_model = load_xsortable_octagon_panda3d(base)

            vertices_and_faces = get_vertices_and_faces_copy(loaded_model)

            self.assertTrue(match_points_space(vertices_and_faces.vertices))
            self.assertTrue(
                match_faces_faces(vertices_and_faces.faces, kinds={"triangles"})
            )

            expected_verts_and_faces = load_xsortable_octagon_as_verts_and_faces()

            actual_verts_sorted = _sort_points(vertices_and_faces.vertices)
            expected_verts_sorted = _sort_points(expected_verts_and_faces.vertices)

            diff = abs(actual_verts_sorted - expected_verts_sorted).max(
                axis=DIM_POINTS_DATA
            )

            self.assertTrue(
                np.allclose(actual_verts_sorted, expected_verts_sorted, atol=1e-4)
            )

            self.assertEqual(
                vertices_and_faces.faces.shape, expected_verts_and_faces.faces.shape
            )
        finally:
            base.destroy()

    def test_get_vertices_and_faces_copy_non_geom_node(self) -> None:
        with self.assertRaises(Panda3dAssumptionViolation):
            get_vertices_and_faces_copy(NodePath(PandaNode("obj1")))

    def test_get_vertices_and_faces_copy_no_primitive(self) -> None:
        base = ShowBase(windowType="offscreen")
        try:
            loaded_model = load_xsortable_octagon_panda3d(base)

            model_node = loaded_model.node()

            assert isinstance(model_node, GeomNode)

            while model_node.getGeom(0).getNumPrimitives() > 0:
                model_node.modifyGeom(0).removePrimitive(0)

            with self.assertRaises(Panda3dAssumptionViolation):
                get_vertices_and_faces_copy(loaded_model)
        finally:
            base.destroy()

    def test_get_vertices_and_faces_copy_incorrect_primitive(self) -> None:
        base = ShowBase(windowType="offscreen")
        try:
            loaded_model = load_xsortable_octagon_panda3d(base)

            model_node = loaded_model.node()

            prim = GeomTrifans(GeomPrimitive.UH_dynamic)
            prim.add_vertex(0)
            prim.add_vertex(1)
            prim.add_vertex(2)
            prim.close_primitive()
            model_node.modifyGeom(0).addPrimitive(prim)

            with self.assertRaises(Panda3dAssumptionViolation):
                get_vertices_and_faces_copy(loaded_model)
        finally:
            base.destroy()

    def test_get_vertices_and_faces_copy_have_children(self):
        base = ShowBase(windowType="offscreen")
        try:
            loaded_model = load_xsortable_octagon_panda3d(base)

            model_node = loaded_model.node()

            assert isinstance(model_node, GeomNode)

            model_node.add_child(PandaNode("mynode"))

            with self.assertRaises(Panda3dAssumptionViolation):
                get_vertices_and_faces_copy(loaded_model)
        finally:
            base.destroy()

    def test_get_all_vertex_arrays_copy_from_vertex_data(self):
        obj = new_vertex_only_object(
            verts=np.array(
                [
                    [3, 2, 5],
                    [3, 2, 6],
                ],
                dtype=np.float32,
            ),
            obj_name="myobj",
        )
        arrays = get_all_vertex_arrays_copy_from_vertex_data(
            obj.node().getGeom(0).getVertexData()
        )
        self.assertEqual(len(arrays), 1)
        self.assertEqual(arrays[0].shape[DIM_TABLE_ROW], 2)
        self.assertEqual(arrays[0].shape[DIM_TABLE_COL], 3)

    def test_get_col_copy_from_vertex_data_get_values(self):
        initial_verts = np.array(
            [
                [3, 2, 5],
                [3, 2, 6],
            ],
            dtype=np.float32,
        )
        initial_normals = np.array(
            [
                [0.1, 0.4, 0.2],
                [0.4, 0.3, 0.8],
            ]
        )

        obj = new_vertex_only_object(
            verts=initial_verts,
            normals=initial_normals,
            obj_name="myobj",
        )
        vertices_copy = get_col_copy_from_vertex_data(
            col_name="vertex", vertex_data=obj.node().getGeom(0).getVertexData()
        )
        normals_copy = get_col_copy_from_vertex_data(
            col_name="normal", vertex_data=obj.node().getGeom(0).getVertexData()
        )

        self.assertTrue(np.allclose(initial_verts, vertices_copy))
        self.assertTrue(np.allclose(initial_normals, normals_copy))

    def test_get_cam_proj_spec_for_lens_happy_path(self):
        lens = PerspectiveLens()
        lens.setNearFar(5, 10)
        spec = get_cam_proj_spec_for_lens(lens)

        original_vector = np.array([[3], [5], [9], [1]])
        original_vector_as_point = (
            float(original_vector[0]),
            float(original_vector[1]),
            float(original_vector[2]),
        )
        coord2d = Point2()

        lens.project(original_vector_as_point, coord2d)

        expected_projected_point_cart = np.array(
            [[coord2d.x], [coord2d.y]], dtype=np.float32
        )

        actual_projected_point_homog = spec.proj_mat @ original_vector
        actual_projected_point_cart = np.zeros(
            newshape_mat(col=1, row=2), dtype=np.float32
        )
        upd_mat_float(
            actual_projected_point_cart,
            row=0,
            value_=idx_mat_float(actual_projected_point_homog, row=0)
            / idx_mat_float(actual_projected_point_homog, row=2),
        )
        upd_mat_float(
            actual_projected_point_cart,
            row=1,
            value_=idx_mat_float(actual_projected_point_homog, row=1)
            / idx_mat_float(actual_projected_point_homog, row=2),
        )

        self.assertTrue(
            np.allclose(expected_projected_point_cart, actual_projected_point_cart)
        )

    def test_get_cam_proj_spec_for_lens_invalid_cam(self):
        infinite_near_lens = PerspectiveLens()
        infinite_near_lens.setNearFar(math.inf, 100)
        infinite_far_lens = PerspectiveLens()
        infinite_far_lens.setNearFar(math.inf, 100)
        fisheye_lens = FisheyeLens()
        lenses: list[tuple[str, Lens]] = [
            ("infinite_near", infinite_near_lens),
            ("infinite_far", infinite_far_lens),
            ("fisheye", fisheye_lens),
        ]

        for lens_name, lens in lenses:
            with self.subTest(lens_name):
                with self.assertRaises(Panda3dAssumptionViolation):
                    get_cam_proj_spec_for_lens(lens)

    def test_get_cam_proj_spec_for_showbase(self):
        lens = PerspectiveLens()
        lens.setNearFar(3, 10)
        cam = NodePath(Camera("cam"))
        cam.node().setLens(lens)

        base = mock.Mock()
        base.cam = cam

        expected_spec = get_cam_proj_spec_for_lens(lens)
        actual_spec = get_cam_proj_spec_for_showbase(base)

        self.assertTrue(
            np.allclose(expected_spec.proj_mat, actual_spec.proj_mat, atol=1e-4)
        )
        self.assertAlmostEqual(expected_spec.im_left_x_val, actual_spec.im_left_x_val)
        self.assertAlmostEqual(expected_spec.im_right_x_val, actual_spec.im_right_x_val)
        self.assertAlmostEqual(expected_spec.im_top_y_val, actual_spec.im_top_y_val)
        self.assertAlmostEqual(
            expected_spec.im_bottom_y_val, actual_spec.im_bottom_y_val
        )


def _sort_points(points: np.ndarray) -> np.ndarray:
    """
    Sort the points by their coordinates lexicographically.

    Parameters
    ----------
    points
        The points to sort. Format: ``Points::Space``

    Returns
    -------
    v
        The sorted points. Format: ``Points::Space``
    """
    return points[np.lexsort(np.rot90(points))]


def _sort_by_x(points: np.ndarray) -> np.ndarray:
    """
    Sort the specified points by the x coordinate.

    Parameters
    ----------
    points
        The points to sort. Format: ``Points::Space``

    Returns
    -------
    v
        The same points sorted. Format: ``Points::Space``
    """
    return idx_points(
        points,
        n=np.argsort(idx_points(points, data=CAT_POINTS_SPACE_DATA_X)),
    )
