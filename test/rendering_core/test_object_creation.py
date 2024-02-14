import unittest
from unittest import mock

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomNode, LVecBase3f, PandaNode

from threedattack.rendering_core import (
    VertsAndFaces,
    get_all_vertex_arrays_copy_from_vertex_data,
    get_col_copy_from_vertex_data,
    get_vertex_count,
    set_col_in_vertex_data,
    verts_and_faces_2_obj,
)
from threedattack.rendering_core._object_creation import _calculate_normals_flat
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *

from .util import (
    load_xsortable_octagon_as_verts_and_faces,
    load_xsortable_octagon_panda3d,
)


class TestObjectCreation(unittest.TestCase):
    def test_verts_and_faces_2_obj_non_empty_obj(self):
        base = ShowBase(windowType="offscreen")
        try:
            mesh = load_xsortable_octagon_as_verts_and_faces()
            render_state_source = load_xsortable_octagon_panda3d(base)

            set_col_in_vertex_data(
                render_state_source.node().modifyGeom(0).modifyVertexData(),
                col_name="vertex",
                new_values=get_col_copy_from_vertex_data(
                    col_name="vertex",
                    vertex_data=render_state_source.node().getGeom(0).getVertexData(),
                )
                * 0.0,
            )
            set_col_in_vertex_data(
                render_state_source.node().modifyGeom(0).modifyVertexData(),
                col_name="normal",
                new_values=get_col_copy_from_vertex_data(
                    col_name="normal",
                    vertex_data=render_state_source.node().getGeom(0).getVertexData(),
                )
                * 0.0,
            )

            EXPECTED_NEW_OBJ_NAME = "obj1"
            new_obj = verts_and_faces_2_obj(
                verts_and_faces=mesh,
                name=EXPECTED_NEW_OBJ_NAME,
                render_state_source=render_state_source,
            )

            new_obj_node = new_obj.node()
            self.assertIsInstance(new_obj_node, GeomNode)
            assert isinstance(new_obj_node, GeomNode)

            self.assertEqual(get_vertex_count(new_obj), 6 * 2 * 3)
            all_vert_arrays_of_new = get_all_vertex_arrays_copy_from_vertex_data(
                new_obj_node.getGeom(0).getVertexData()
            )
            self.assertEqual(len(all_vert_arrays_of_new), 1)

            for col_name in ["vertex", "normal"]:
                new_col_vals = get_col_copy_from_vertex_data(
                    col_name=col_name,
                    vertex_data=new_obj_node.getGeom(0).getVertexData(),
                )
                self.assertGreater(new_col_vals.max(), 0.2)
            self.assertEqual(EXPECTED_NEW_OBJ_NAME, new_obj.name)
        finally:
            base.destroy()

    def test_verts_and_faces_2_obj_empty_obj(self):
        mesh = VertsAndFaces(
            vertices=np.zeros(newshape_points_space(n=0), dtype=np.float32),
            faces=np.zeros(newshape_faces_faces(face=0, corner=3), dtype=np.ushort),
        )
        new_obj = verts_and_faces_2_obj(
            name="myobj", render_state_source=mock.Mock(), verts_and_faces=mesh
        )

        new_obj_node = new_obj.node()

        self.assertIsInstance(new_obj_node, PandaNode)
        self.assertNotIsInstance(new_obj_node, GeomNode)

    def test_normal_calculation_panda3d_gltf_general_equivalence(self):
        rng = np.random.default_rng(6000)
        n_points = 51

        verts = rng.uniform(-3, 3, size=newshape_points_space(n=n_points))

        actual_normals = _calculate_normals_flat(verts)
        expected_normals = _panda3d_gltf_normal_calc_fn(verts)
        v = abs((expected_normals - actual_normals)).max()
        self.assertTrue(np.allclose(actual_normals, expected_normals, atol=1e-4))

    def test_normal_calculation_small_face_handling(self):
        verts = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [0, 2, 0], [0, 2 + 1e-14, 0]],
            dtype=np.float32,
        )

        actual_normals = _calculate_normals_flat(verts)
        expected_normals = np.concatenate(
            [
                _panda3d_gltf_normal_calc_fn(verts[0:3]),
                np.sqrt(np.array([[1 / 3, 1 / 3, 1 / 3]])),
                np.sqrt(np.array([[1 / 3, 1 / 3, 1 / 3]])),
                np.sqrt(np.array([[1 / 3, 1 / 3, 1 / 3]])),
            ],
            axis=DIM_POINTS_N,
        )
        self.assertTrue(np.allclose(actual_normals, expected_normals, atol=1e-4))


def _panda3d_gltf_normal_calc_fn(verts: np.ndarray) -> np.ndarray:
    """
    Calculate the flat normals from the vertexes of a non-indexed Geom as if panda3d-gltf would do (except the readers and writers).

    Parameters
    ----------
    verts
        The assumed vertexes. Format: ``Points::Space``

    Returns
    -------
    v
        The generated normal vectors. Format: ``Points::Space``
    """
    vertex_vectors = [LVecBase3f(x, y, z) for x, y, z in verts]

    normals: list[list[float]] = []

    for i in range(len(vertex_vectors) // 3):
        vtx1 = vertex_vectors[i * 3]
        vtx2 = vertex_vectors[i * 3 + 1]
        vtx3 = vertex_vectors[i * 3 + 2]
        normal = (vtx2 - vtx1).cross(vtx3 - vtx1)
        normal.normalize()
        normals.append([float(normal.x), float(normal.y), float(normal.z)])
        normals.append([float(normal.x), float(normal.y), float(normal.z)])
        normals.append([float(normal.x), float(normal.y), float(normal.z)])

    return np.array(normals)
