import plotly.graph_objects as go

from ..tensor_types.npy import *
from ._data import VertsAndFaces


def verts_and_faces_2_plotly_figure(verts_and_faces: VertsAndFaces) -> go.Figure:
    """
    Convert the vertexes and faces to a Plotly figure.
    """
    return go.Figure(
        data=go.Mesh3d(
            x=idx_points_space(verts_and_faces.vertices, data="x"),
            y=idx_points_space(verts_and_faces.vertices, data="y"),
            z=idx_points_space(verts_and_faces.vertices, data="z"),
            i=idx_faces_faces(verts_and_faces.faces, corner=0),
            j=idx_faces_faces(verts_and_faces.faces, corner=1),
            k=idx_faces_faces(verts_and_faces.faces, corner=2),
        )
    ).update_layout(scene=dict(aspectmode="data"))
