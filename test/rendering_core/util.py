from pathlib import Path
from typing import cast

import numpy as np
import trimesh
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Geom,
    GeomNode,
    GeomPoints,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
)

from threedattack.rendering_core import VertsAndFaces, load_model_from_local_file
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *


def new_vertex_only_object(
    verts: np.ndarray, obj_name: str, normals: np.ndarray | None = None
) -> "NodePath[GeomNode]":
    """
    Creates a new vertex-only object with the specified vertexes.

    If the normals are not specified, then the vertex data of the created object will have ``v3`` format. If they are specified, then the vertex data of the created object will have ``v3n3`` format.

    Parameters
    ----------
    verts
        Format: ``Points::Space``
    obj_name
        The name of the object.
    normals
        The normal vectors of the vertices. Format: ``Points::Space``

    Returns
    -------
    v
        The object with the specified vertexes.
    """
    if normals is not None:
        vdata_format = GeomVertexFormat.getV3n3()
    else:
        vdata_format = GeomVertexFormat.getV3()
    vdata = GeomVertexData(obj_name, vdata_format, Geom.UHStatic)
    vdata.setNumRows(verts.shape[DIM_POINTS_N])

    writer = GeomVertexWriter(vdata, "vertex")
    for idx in range(verts.shape[DIM_POINTS_N]):
        x = float(idx_points(verts, n=idx, data=CAT_POINTS_SPACE_DATA_X))
        y = float(idx_points(verts, n=idx, data=CAT_POINTS_SPACE_DATA_Y))
        z = float(idx_points(verts, n=idx, data=CAT_POINTS_SPACE_DATA_Z))

        writer.addData3(x, y, z)

    if normals is not None:
        writer = GeomVertexWriter(vdata, "normal")
        for idx in range(normals.shape[DIM_POINTS_N]):
            x = float(idx_points(normals, n=idx, data=CAT_POINTS_SPACE_DATA_X))
            y = float(idx_points(normals, n=idx, data=CAT_POINTS_SPACE_DATA_Y))
            z = float(idx_points(normals, n=idx, data=CAT_POINTS_SPACE_DATA_Z))

            writer.addData3(x, y, z)

    geom_primitive = GeomPoints(Geom.UHStatic)
    for i in range(verts.shape[DIM_POINTS_N]):
        geom_primitive.addVertex(i)
    geom_primitive.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(geom_primitive)

    geom_node = GeomNode(obj_name)
    geom_node.addGeom(geom)

    parent_node_path = NodePath()
    node_path = parent_node_path.attachNewNode(geom_node)

    return node_path


X_SORTABLE_OCTAGON_PATH = Path("test_resources/x_sortable_octagon.obj")


def load_xsortable_octagon_panda3d(base: ShowBase) -> "NodePath[GeomNode]":
    loaded_model_root = load_model_from_local_file(base, X_SORTABLE_OCTAGON_PATH)
    assert loaded_model_root is not None
    geom_obj = loaded_model_root.getChild(0).getChild(0)
    assert isinstance(geom_obj.node(), GeomNode)
    return geom_obj  # type: ignore


def load_xsortable_octagon_as_verts_and_faces() -> VertsAndFaces:
    trimesh_mesh = cast(
        trimesh.Trimesh, trimesh.load_mesh(str(X_SORTABLE_OCTAGON_PATH))
    )

    return VertsAndFaces(vertices=trimesh_mesh.vertices, faces=trimesh_mesh.faces)
