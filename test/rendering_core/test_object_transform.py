import copy
import itertools
import unittest
from sys import orig_argv
from typing import cast

import numpy as np
from direct.showbase.ShowBase import ShowBase
from numpy.lib.function_base import vectorize
from panda3d.core import GeomNode, NodePath
from scipy.interpolate import LinearNDInterpolator

from threedattack.rendering_core import (
    InterpolatedOccupacyField,
    MeshBasedObjectTransform,
    ObjectTransformType,
    PointBasedVectorFieldSpec,
    ScaledStandingAreas,
    ThreeDSize,
    VolumeBasedObjectTransform,
    WarpField,
    get_col_copy_from_vertex_data,
    get_d_coord_from_obj_size_along_the_shortest_axis,
    get_object_transform_type_by_name,
    get_supported_object_transform_type_names,
    get_vertex_positions_copy,
    get_vertices_and_faces_copy,
    occupacy_field_2_occupacy_field_samples,
    occupacy_field_samples_2_verts_and_faces,
    verts_and_faces_2_obj,
    verts_and_faces_2_occupacy_field_samples,
)
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *

from .util import (
    load_xsortable_octagon_as_verts_and_faces,
    load_xsortable_octagon_panda3d,
)


class TestTransformClasses(unittest.TestCase):
    def setUp(self):
        control_points = np.array(
            [
                [1, 0, 0],
                [10, 10, 10],
                [10, -10, 10],
                [10, -10, -10],
            ]
        )
        n_control_points = control_points.shape[DIM_POINTS_N]
        vectors = np.stack(
            [
                np.full(shape=n_control_points, fill_value=1),
                np.full(shape=n_control_points, fill_value=2),
                np.full(shape=n_control_points, fill_value=3),
            ],
            axis=1,
        )

        nonzero_vector_field_spec = PointBasedVectorFieldSpec(
            control_points=control_points, vectors=vectors
        )

        self.nonzero_vector_field_spec = nonzero_vector_field_spec
        """
        A vector field spec that describes the transformation to test.

        This vector field does not affect the number of bodies in case of convex objects.
        """

        self.zero_vector_field_spec = PointBasedVectorFieldSpec(
            control_points=control_points, vectors=np.zeros_like(vectors)
        )

        mesh_nobounds = ScaledStandingAreas(
            original_size=ThreeDSize(
                x_size=100, y_size=100, z_size=100
            ),  # simple hack to get some bounds (you should use exact bounds in production code)
            size_multiplier=1.1,
            extra_offset_after_size_mult=0.03,
        )

        self.nobounds = mesh_nobounds
        """
        The bounds are large enough to make the transformation behave as if there were no bounds at all.
        """

        self.mesh_relevant_bounds_by_axis = {
            "x": ScaledStandingAreas(
                original_size=ThreeDSize(x_size=1, y_size=100, z_size=100),
                size_multiplier=1.1,
                extra_offset_after_size_mult=0.03,
            ),
            "y": ScaledStandingAreas(
                original_size=ThreeDSize(x_size=100, y_size=1, z_size=100),
                size_multiplier=1.1,
                extra_offset_after_size_mult=0.03,
            ),
            "z": ScaledStandingAreas(
                original_size=ThreeDSize(x_size=100, y_size=100, z_size=1),
                size_multiplier=1.1,
                extra_offset_after_size_mult=0.03,
            ),
        }

        self.volume_nobounds = ScaledStandingAreas(
            original_size=ThreeDSize(x_size=3, y_size=3, z_size=3),
            size_multiplier=5,
            extra_offset_after_size_mult=0.03,
        )

        self.volume_bounds_present = ScaledStandingAreas(
            original_size=ThreeDSize(x_size=3, y_size=3, z_size=3),
            size_multiplier=1.2,
            extra_offset_after_size_mult=0.03,
        )

    def test_meshtransform_nobounds(self):
        base = ShowBase(windowType="offscreen")
        try:
            initial_obj = load_xsortable_octagon_panda3d(base)
            nonindexed_init_obj = copy.deepcopy(initial_obj)
            nonindexed_init_obj.node().modifyGeom(0).makeNonindexed(False)

            transform = MeshBasedObjectTransform(
                obj=initial_obj, transformed_obj_areas=self.nobounds
            )

            actual_result = transform.transform_obj_new(
                vector_field_spec=self.nonzero_vector_field_spec
            )

            interpolator = LinearNDInterpolator(
                self.nonzero_vector_field_spec.control_points,
                self.nonzero_vector_field_spec.vectors,
                fill_value=0,
            )

            def transform_fn(a):
                d_positions = interpolator(a)
                return a + d_positions

            # the vertexes should be updated
            actual_verts = get_vertex_positions_copy(actual_result.new_obj)
            expected_verts = transform_fn(
                get_vertex_positions_copy(nonindexed_init_obj)
            )
            self.assertTrue(np.allclose(expected_verts, actual_verts))

            # the normal vectors should be updated too
            actual_normals = get_col_copy_from_vertex_data(
                col_name="normal",
                vertex_data=cast(GeomNode, actual_result.new_obj.node())
                .getGeom(0)
                .getVertexData(),
            )
            expected_normals = get_col_copy_from_vertex_data(
                col_name="normal",
                vertex_data=cast(GeomNode, nonindexed_init_obj.node())
                .getGeom(0)
                .getVertexData(),
            )
            self.assertFalse(np.allclose(expected_normals, actual_normals, atol=1e-4))
            self.assertEqual(actual_result.n_bodies, 1)
        finally:
            base.destroy()

    def test_meshtransform_bounds_present(self):
        base = ShowBase(windowType="offscreen")
        try:
            initial_obj = load_xsortable_octagon_panda3d(base)

            for bound_axis in ["x", "y", "z"]:
                with self.subTest(bound_axis + "_bounds"):
                    bounds = self.mesh_relevant_bounds_by_axis[bound_axis]
                    transform = MeshBasedObjectTransform(
                        obj=initial_obj, transformed_obj_areas=bounds
                    )

                    nozero_transform_result = transform.transform_obj_new(
                        vector_field_spec=self.nonzero_vector_field_spec
                    )
                    zero_transform_result = transform.transform_obj_new(
                        vector_field_spec=self.zero_vector_field_spec
                    )

                    actual_verts_with_changed_pos = get_vertex_positions_copy(
                        nozero_transform_result.new_obj
                    )
                    actual_verts_with_nonchanged_pos = get_vertex_positions_copy(
                        zero_transform_result.new_obj
                    )

                    points_outside = bounds.get_full_area(
                        origin_of_obj=None, include_outside_offset=False
                    ).select_points_outside(actual_verts_with_changed_pos)
                    self.assertEqual(points_outside.shape[DIM_POINTS_N], 0)
                    self.assertEqual(nozero_transform_result.n_bodies, 1)

                    pos_changes = (
                        actual_verts_with_changed_pos - actual_verts_with_nonchanged_pos
                    )
                    max_pos_change_along_axis = pos_changes.max()
                    self.assertLessEqual(
                        max_pos_change_along_axis,
                        self.nonzero_vector_field_spec.vectors.max(),
                    )
                    self.assertGreater(max_pos_change_along_axis, 0)

                    self.assertFalse(
                        np.allclose(
                            idx_points_space(pos_changes, data="x"),
                            idx_points_space(pos_changes, data="y"),
                        ),
                    )
                    self.assertFalse(
                        np.allclose(
                            idx_points_space(pos_changes, data="x"),
                            idx_points_space(pos_changes, data="z"),
                        ),
                    )
                    self.assertFalse(
                        np.allclose(
                            idx_points_space(pos_changes, data="x"),
                            idx_points_space(pos_changes, data="z"),
                        ),
                    )
        finally:
            base.destroy()

    def test_volumentransform_general(self):
        base = ShowBase(windowType="offscreen")
        try:
            initial_obj = load_xsortable_octagon_panda3d(base)
            subtest_names = ["nobounds", "bounds_present"]
            bound_types = [
                self.volume_nobounds,
                self.volume_bounds_present,
            ]
            for bounds, subtest_name in zip(bound_types, subtest_names):
                with self.subTest(subtest_name):
                    verts_and_faces_copy = get_vertices_and_faces_copy(initial_obj)
                    obj_field_spec = verts_and_faces_2_occupacy_field_samples(
                        verts_and_faces_copy, n_steps_along_shortest_axis=5
                    )
                    N_STEPS_ALONG_SHORTEST_AXIS = 6

                    transform = VolumeBasedObjectTransform(
                        obj=initial_obj,
                        field_cache=obj_field_spec,
                        n_steps_along_shortest_axis=N_STEPS_ALONG_SHORTEST_AXIS,
                        target_obj_areas=bounds,
                    )

                    actual_result = transform.transform_obj_new(
                        vector_field_spec=self.nonzero_vector_field_spec
                    )

                    expected_fn = WarpField(
                        control_points=self.nonzero_vector_field_spec.control_points,
                        field_fn=InterpolatedOccupacyField(
                            samples=obj_field_spec,
                        ),
                        vectors=self.nonzero_vector_field_spec.vectors,
                    )

                    expected_d_coord = (
                        get_d_coord_from_obj_size_along_the_shortest_axis(
                            area=bounds.get_original_area(origin_of_obj=None),
                            n_steps_along_shortest_axis=N_STEPS_ALONG_SHORTEST_AXIS,
                        )
                    )
                    self.assertEqual(expected_d_coord, transform.get_d_coord())

                    expected_occupacy_field_samples = (
                        occupacy_field_2_occupacy_field_samples(
                            d_coord=expected_d_coord,
                            occupacy_field=expected_fn,
                            relevant_area=bounds.get_full_area(
                                origin_of_obj=None, include_outside_offset=False
                            ),
                        )
                    )
                    expected_verts_and_faces = occupacy_field_samples_2_verts_and_faces(
                        expected_occupacy_field_samples
                    )
                    expected_obj = verts_and_faces_2_obj(
                        expected_verts_and_faces,
                        render_state_source=initial_obj,
                        name=initial_obj.name,
                    )

                    expected_verts = get_vertex_positions_copy(expected_obj)
                    actual_verts = get_vertex_positions_copy(actual_result.new_obj)

                    self.assertTrue(np.allclose(expected_verts, actual_verts))

                    self.assertEqual(actual_result.n_bodies, 1)
                    self.assertGreater(actual_result.change_amount_score, 0.01)
        finally:
            base.destroy()

    def test_volumentransform_stay_in_place(self):
        base = ShowBase(windowType="offscreen")
        try:
            initial_obj = load_xsortable_octagon_panda3d(base)
            verts_and_faces_copy = get_vertices_and_faces_copy(initial_obj)
            obj_field_spec = verts_and_faces_2_occupacy_field_samples(
                verts_and_faces_copy, n_steps_along_shortest_axis=5
            )
            N_STEPS_ALONG_SHORTEST_AXIS = 6

            transform = VolumeBasedObjectTransform(
                obj=initial_obj,
                field_cache=obj_field_spec,
                n_steps_along_shortest_axis=N_STEPS_ALONG_SHORTEST_AXIS,
                target_obj_areas=self.volume_nobounds,
            )

            actual_result = transform.transform_obj_new(
                vector_field_spec=self.zero_vector_field_spec
            )

            self.assertEqual(actual_result.n_bodies, 1)
            self.assertAlmostEqual(actual_result.change_amount_score, 0, delta=1e-4)
        finally:
            base.destroy()

    def test_volumentransform_change_amount_score(self):
        base = ShowBase(windowType="offscreen")
        try:
            initial_obj = load_xsortable_octagon_panda3d(base)
            subtest_names = ["nobounds", "bounds_present"]
            bound_types = [
                self.volume_nobounds,
                self.volume_bounds_present,
            ]
            for bounds, subtest_name in zip(bound_types, subtest_names):
                with self.subTest(subtest_name):
                    verts_and_faces_copy = get_vertices_and_faces_copy(initial_obj)
                    obj_field_spec = verts_and_faces_2_occupacy_field_samples(
                        verts_and_faces_copy, n_steps_along_shortest_axis=5
                    )
                    N_STEPS_ALONG_SHORTEST_AXIS = 6

                    transform = VolumeBasedObjectTransform(
                        obj=initial_obj,
                        field_cache=obj_field_spec,
                        n_steps_along_shortest_axis=N_STEPS_ALONG_SHORTEST_AXIS,
                        target_obj_areas=bounds,
                    )
                    original_samples = transform.get_orig_field_samples_deepcopy()

                    new_samples = transform.get_orig_field_samples_deepcopy()
                    self._swap_n_points_inplace(
                        grid=new_samples.grid, in_points_to_swap=2, out_points_to_swap=3
                    )

                    actual_change_amount_score = transform._change_amount_score_fn(
                        new_samples.grid
                    )

                    expected_change_amount_score = 5 / self._get_n_in_points(
                        original_samples.grid
                    )

                    self.assertEqual(
                        actual_change_amount_score, expected_change_amount_score
                    )
        finally:
            base.destroy()

    def test_selftest_swap_n_points_inplace(self):
        initial_grid = np.full(
            shape=newshape_fieldgrid_occupacyfieldgrid(x=10, y=12, z=13),
            fill_value=-0.3,
            dtype=np.float32,
        )
        upd_fieldgrid_occupacyfieldgrid(initial_grid, x=3, y=slice(1, 6), value_=0.5)

        N_IN_POINTS_TO_SWAP = 2
        N_OUT_POINTS_TO_SWAP = 3

        modified_grid = initial_grid.copy()
        self._swap_n_points_inplace(
            modified_grid,
            in_points_to_swap=N_IN_POINTS_TO_SWAP,
            out_points_to_swap=N_OUT_POINTS_TO_SWAP,
        )

        actual_modified_point_count = (
            np.logical_xor((modified_grid > 0), (initial_grid > 0))
            .astype(np.int32)
            .sum()
        )
        expected_modified_point_count = N_IN_POINTS_TO_SWAP + N_OUT_POINTS_TO_SWAP
        self.assertEqual(actual_modified_point_count, expected_modified_point_count)

    def _get_n_in_points(self, grid: np.ndarray) -> int:
        """Get the number of inside points in the specified ``FieldGrid::OccupacyFieldGrid``."""
        return (grid > 0).astype(np.int32).sum().item()

    def _swap_n_points_inplace(
        self, grid: np.ndarray, in_points_to_swap: int, out_points_to_swap: int
    ) -> None:
        """
        Make ``in_points_to_swap`` originally inside points outside points and make ``out_points_to_swap`` originally outside points inside points.

        Parameters
        ----------
        grid
            The ``FieldGrid::OccupacyFieldGrid`` to modify. Format: ``FieldGrid::OccupacyFieldGrid``
        in_points_to_swap
            How many originally inside points should be swapped.
        out_points_to_swap
            How many originally outside pionts should be swapped.
        """
        x_size = grid.shape[DIM_FIELDGRID_X]
        y_size = grid.shape[DIM_FIELDGRID_Y]
        z_size = grid.shape[DIM_FIELDGRID_Z]

        in_points_counter = 0
        out_points_counter = 0
        for x, y, z in itertools.product(range(x_size), range(y_size), range(z_size)):
            val = idx_fieldgrid_occupacyfieldgrid(grid, x=x, y=y, z=z).item()
            if val > 0:
                if in_points_counter < in_points_to_swap:
                    upd_fieldgrid_occupacyfieldgrid(grid, x=x, y=y, z=z, value_=-0.7)
                in_points_counter += 1
            elif val < 0:
                if out_points_counter < out_points_to_swap:
                    upd_fieldgrid_occupacyfieldgrid(grid, x=x, y=y, z=z, value_=0.7)
                out_points_counter += 1

            if (in_points_counter >= in_points_to_swap) and (
                out_points_counter >= out_points_to_swap
            ):
                break


class TestFunctions(unittest.TestCase):
    def test_get_object_transform_type_by_name_happy_path(self):
        for expected_transform_type in ObjectTransformType:
            with self.subTest(expected_transform_type.public_name):
                actual_transform_type = get_object_transform_type_by_name(
                    expected_transform_type.public_name
                )

                self.assertEqual(expected_transform_type, actual_transform_type)

    def test_get_object_transform_type_by_name_invalid_name(self):
        with self.assertRaises(ValueError):
            get_object_transform_type_by_name("invalid_type")

    def test_get_supported_object_transform_type_names(self):
        names = get_supported_object_transform_type_names()
        for name in names:
            get_object_transform_type_by_name(name)
