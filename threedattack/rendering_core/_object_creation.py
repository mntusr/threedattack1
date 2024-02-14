import math

import numpy as np
from panda3d.core import (
    AlphaTestAttrib,
    AntialiasAttrib,
    AudioVolumeAttrib,
    AuxBitplaneAttrib,
    ClipPlaneAttrib,
    ColorAttrib,
    ColorBlendAttrib,
    ColorScaleAttrib,
    ColorWriteAttrib,
    CullBinAttrib,
    CullFaceAttrib,
    DepthOffsetAttrib,
    DepthTestAttrib,
    DepthWriteAttrib,
    FogAttrib,
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayData,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexReader,
    GeomVertexWriter,
    InternalName,
    LightAttrib,
    LightRampAttrib,
    MaterialAttrib,
    NodePath,
    RenderModeAttrib,
    RenderState,
    RescaleNormalAttrib,
    ShadeModelAttrib,
    ShaderAttrib,
    StencilAttrib,
    TexGenAttrib,
    TexMatrixAttrib,
    TextureAttrib,
    TransparencyAttrib,
)

from ..tensor_types.idx import *
from ..tensor_types.npy import *
from ._data import VertsAndFaces
from ._errors import Panda3dAssumptionViolation
from ._scene_util import get_col_copy_from_vertex_data, set_col_in_vertex_data


def verts_and_faces_2_obj(
    verts_and_faces: VertsAndFaces, render_state_source: NodePath, name: str
) -> NodePath:
    """
    Create a Panda3d object from a mesh.

    This function takes care about:

    * The creation of the faces and vertexes
    * The calculation of the normal vectors
    * The appearance of the object

    If the mesh is empty, then the function returns with a PandaNode. If the mesh is not empty, then the function returns with an object that has an empty PandaNode. If the mesh is not empty, then the function returns with an object that contains the calculated GeomNode.

    Parameters
    ----------
    verts_and_faces
        The mesh to use.

    Returns
    -------
    v
        The created GeomNode.

    Raises
    ------
    Panda3dAssumptionViolation
        If the mesh is not empty, but the number of points inside of it after making the corners of the different faces different points is not divisible by 3.
        If

    Notes
    -----
    The geom of the created GeomNode is not indexed. See `this description <https://docs.panda3d.org/1.10/python/reference/panda3d.core.GeomPrimitive#panda3d.core.GeomPrimitive.isIndexed>` about the difference between indexed and not indexed geoms in Panda3d.
    """
    n_points = verts_and_faces.vertices.shape[DIM_POINTS_N]
    if n_points == 0:
        return NodePath(name)

    # indexed vs non-indexed primitives in Panda3d: https://docs.panda3d.org/1.10/python/reference/panda3d.core.GeomPrimitive#panda3d.core.GeomPrimitive.isIndexed

    geom = _generate_indexed_geom_from_verts_and_faces(verts_and_faces)
    _add_normals_to_geom_and_make_non_indexed(geom)

    new_obj_node = GeomNode(name)
    new_obj_node.addGeom(geom)

    new_obj = NodePath(new_obj_node)

    _copy_render_state(source_obj=render_state_source, target_obj=new_obj)

    return new_obj


def _generate_indexed_geom_from_verts_and_faces(verts_and_faces: VertsAndFaces) -> Geom:
    """
    Generate an indexed Geom that contains the vertexes and faces of the specified mesh.

    This Geom has specified float normal coordinates as column, but this column is not actually initialized.

    Parameters
    ----------
    verts_and_faces
        The mesh to use.

    Returns
    -------
    v
        The created geom.

    Raises
    ------
    ValueError
        If the mesh does not contain any point.
    """
    UNSIGNED_SHORT_MAX = 65535

    if verts_and_faces.vertices.shape[DIM_POINTS_N] < 1:
        raise ValueError(
            "The mesh should contain at least one point to enable the cration of the Geom."
        )

    # panda3d index handling: https://discourse.panda3d.org/t/procedurally-efficient-way-to-create-vertices-faces-etc/24610/13

    n_points = verts_and_faces.vertices.shape[DIM_POINTS_N]
    n_faces = verts_and_faces.faces.shape[DIM_POINTS_N]

    format = GeomVertexFormat.getV3n3()
    geom_vertex_data = GeomVertexData("vertices", format, usage_hint=Geom.UHDynamic)
    geom_vertex_data.setNumRows(n_points)

    set_col_in_vertex_data(
        col_name="vertex",
        new_values=verts_and_faces.vertices,
        vertex_data=geom_vertex_data,
    )

    prim = GeomTriangles(Geom.UHDynamic)
    faces_array = prim.modifyVertices()
    faces_array.unclean_set_num_rows(n_faces * 3)
    raw_view = memoryview(faces_array)  # type: ignore

    format_str = "H"
    view_arr_dtype = np.uint16
    casted_view = raw_view.cast("B").cast(format_str)
    faces_array_np = np.ndarray(
        buffer=casted_view, shape=n_faces * 3, dtype=view_arr_dtype
    )
    faces_array_np[:] = verts_and_faces.faces.flatten()

    geom = Geom(geom_vertex_data)
    geom.addPrimitive(prim)
    return geom


def _add_normals_to_geom_and_make_non_indexed(geom: Geom) -> None:
    """
    Calculate the normals for the specified Geom, add them, and make the geom nonindexed.

    This function assumes without checking that the object contains at least one vertex, has an existing ``normal`` column of its vertex data and all columns in its vertex data have float32 type.

    Parameters
    ----------
    geom
        The Geom to modify.
    """
    # based on https://github.com/Moguri/panda3d-gltf/blob/4dab08e323d38a19d06be6b30433f943ffda9dde/gltf/_converter.py#L1108

    geom.decompose_in_place()
    geom.make_nonindexed(False)

    gvd = geom.modifyVertexData()

    vertex_positions = get_col_copy_from_vertex_data(col_name="vertex", vertex_data=gvd)
    normals = _calculate_normals_flat(vertex_positions)
    set_col_in_vertex_data(col_name="normal", new_values=normals, vertex_data=gvd)


def _calculate_normals_flat(vertices_noindexed: np.ndarray) -> np.ndarray:
    """
    Calculate the normal vectors for the specified vertices, assuming that the vertices are the vertices of a non-indexed geom.

    The normal vectors are normalized to have length 1. If the cross product of the two vectors of the three vertexes of the faces is smaller than 1e-13, then the following failback value will be used for those normal vectors: ``vector(sqrt(1/3), sqrt(1/3), sqrt(1/3))``.

    Parameters
    ----------
    vertices_noindexed
        The vertices. Format: ``Points::Space``

    Returns
    -------
    v
        The calculated normal vectors. Format: ``Points::Space``

    Raises
    ------
    Panda3dAssumptionViolation
        If the number of vertexes is not divisible by 3.
    """
    n_points = vertices_noindexed.shape[DIM_POINTS_N]
    if n_points % 3 != 0:
        raise Panda3dAssumptionViolation(
            f"The number of vertexes ({n_points}) in the Geom is not divisible by 3, although it should be a non-indexed Geom."
        )

    vtxs1 = idx_points_space(vertices_noindexed, n=slice(0, None, 3))
    vtxs2 = idx_points_space(vertices_noindexed, n=slice(1, None, 3))
    vtxs3 = idx_points_space(vertices_noindexed, n=slice(2, None, 3))

    normals = np.cross((vtxs2 - vtxs1), (vtxs3 - vtxs1), axis=DIM_POINTS_DATA)
    lens = np.linalg.norm(normals, axis=DIM_POINTS_DATA, ord=2)
    len_ok_mask = lens > 1e-13  # select too small lengths
    normals[len_ok_mask] = normals[len_ok_mask] / np.expand_dims(
        lens[len_ok_mask], axis=1
    )
    normals[~len_ok_mask] = math.sqrt(
        1 / 3
    )  # failback normal value for faces that are too small to have meaningful normals
    normals = np.repeat(normals, repeats=3, axis=DIM_POINTS_N)
    return normals


def _copy_render_state(source_obj: NodePath, target_obj: NodePath) -> None:
    """
    Copy all object-level shading preferences from the original object to the new object.

    This function assumes without checking the following:

    * All geoms of the original object use the same geom state.
    * The potentially applied textures of the original object consist of a single color (i. e. the UV coordinates do not matter).
    * The source object does not have any child.
    * The target object does not have any child.

    Parameters
    ----------
    source_obj
        The source object.
    target_obj
        The target object.
    """
    copied_attribs = [
        AlphaTestAttrib,
        AntialiasAttrib,
        AudioVolumeAttrib,
        AuxBitplaneAttrib,
        ClipPlaneAttrib,
        ColorAttrib,
        ColorBlendAttrib,
        ColorScaleAttrib,
        ColorWriteAttrib,
        CullBinAttrib,
        CullFaceAttrib,
        DepthOffsetAttrib,
        DepthTestAttrib,
        DepthWriteAttrib,
        FogAttrib,
        LightAttrib,
        LightRampAttrib,
        MaterialAttrib,
        RenderModeAttrib,
        RescaleNormalAttrib,
        ShadeModelAttrib,
        ShaderAttrib,
        StencilAttrib,
        TexGenAttrib,
        TexMatrixAttrib,
        TextureAttrib,
        TransparencyAttrib,
        AudioVolumeAttrib,
        CullBinAttrib,
        DepthOffsetAttrib,
        RenderModeAttrib,
        RescaleNormalAttrib,
        ShadeModelAttrib,
    ]

    new_geom_state = RenderState.makeEmpty()

    source_node = source_obj.node()
    target_node = target_obj.node()

    assert isinstance(source_node, GeomNode)
    assert isinstance(target_node, GeomNode)

    state = source_obj.getState()
    geom_state = source_node.getGeomState(0)

    for attrib in copied_attribs:
        obj_attrib = state.getAttrib(attrib)
        if obj_attrib is not None:
            target_obj.setAttrib(obj_attrib)

        geom_attrib = geom_state.getAttrib(attrib)
        if geom_attrib is not None:
            new_geom_state = new_geom_state.setAttrib(geom_attrib)

    for geom_idx in range(target_node.getNumGeoms()):
        target_node.setGeomState(geom_idx, new_geom_state)
