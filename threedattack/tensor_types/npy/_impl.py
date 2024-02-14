import numpy as np
from typing import Union, Any, Literal, Optional



















def idx_im(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice]=slice(None) , c: Union[int, list[int],np.ndarray, slice]=slice(None) , h: Union[int, list[int],np.ndarray, slice]=slice(None) , w: Union[int, list[int],np.ndarray, slice]=slice(None) ) -> np.ndarray:
    """
    Select items from ``Im::*`` alognside its named dimensions.

    Same as ``array[n, c, h, w]``
    """
    return array[n, c, h, w]

def idx_points(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice]=slice(None) , data: Union[int, list[int],np.ndarray, slice]=slice(None) ) -> np.ndarray:
    """
    Select items from ``Points::*`` alognside its named dimensions.

    Same as ``array[n, data]``
    """
    return array[n, data]

def idx_mat(array: np.ndarray, /, row: Union[int, list[int],np.ndarray, slice]=slice(None) , col: Union[int, list[int],np.ndarray, slice]=slice(None) ) -> np.ndarray:
    """
    Select items from ``Mat::*`` alognside its named dimensions.

    Same as ``array[row, col]``
    """
    return array[row, col]

def idx_scalars(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice]=slice(None) ) -> np.ndarray:
    """
    Select items from ``Scalars::*`` alognside its named dimensions.

    Same as ``array[n]``
    """
    return array[n]

def idx_svals(array: np.ndarray, /, v: Union[int, list[int],np.ndarray, slice]=slice(None) ) -> np.ndarray:
    """
    Select items from ``SVals::*`` alognside its named dimensions.

    Same as ``array[v]``
    """
    return array[v]

def idx_arbsamples(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice]=slice(None) ) -> np.ndarray:
    """
    Select items from ``ArbSamples::*`` alognside its named dimensions.

    Same as ``array[n]``
    """
    return array[n]

def idx_fieldgrid(array: np.ndarray, /, x: Union[int, list[int],np.ndarray, slice]=slice(None) , y: Union[int, list[int],np.ndarray, slice]=slice(None) , z: Union[int, list[int],np.ndarray, slice]=slice(None) ) -> np.ndarray:
    """
    Select items from ``FieldGrid::*`` alognside its named dimensions.

    Same as ``array[x, y, z]``
    """
    return array[x, y, z]

def idx_faces(array: np.ndarray, /, face: Union[int, list[int],np.ndarray, slice]=slice(None) , corner: Union[int, list[int],np.ndarray, slice]=slice(None) ) -> np.ndarray:
    """
    Select items from ``Faces::*`` alognside its named dimensions.

    Same as ``array[face, corner]``
    """
    return array[face, corner]

def idx_table(array: np.ndarray, /, row: Union[int, list[int],np.ndarray, slice]=slice(None) , col: Union[int, list[int],np.ndarray, slice]=slice(None) ) -> np.ndarray:
    """
    Select items from ``Table::*`` alognside its named dimensions.

    Same as ``array[row, col]``
    """
    return array[row, col]

def idx_coords(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice]=slice(None) ) -> np.ndarray:
    """
    Select items from ``Coords::*`` alognside its named dimensions.

    Same as ``array[n]``
    """
    return array[n]


def idx_im_rgbs(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[Literal["r", "g", "b"], list[Literal["r", "g", "b"]], None] = None, h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Im::RGBs`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, c, h, w]``
    """


    v0 = n

    if c is None:
        v1 = slice(None)
    else:
        idx_map = { "r": 0,"g": 1,"b": 2,}
        if isinstance(c, list):
            v1 = [idx_map[v] for v in c]
        else:
            v1 = idx_map[c]


    v2 = h

    v3 = w
    return array[v0, v1, v2, v3]
def idx_im_floatmap(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[int, list[int],np.ndarray, slice] = slice(None), h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Im::FloatMap`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, c, h, w]``
    """


    v0 = n

    v1 = c

    v2 = h

    v3 = w
    return array[v0, v1, v2, v3]
def idx_im_intmap(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[int, list[int],np.ndarray, slice] = slice(None), h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Im::IntMap`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, c, h, w]``
    """


    v0 = n

    v1 = c

    v2 = h

    v3 = w
    return array[v0, v1, v2, v3]
def idx_im_depthmaps(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[Literal["depth"], list[Literal["depth"]], None] = None, h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Im::DepthMaps`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, c, h, w]``
    """


    v0 = n

    if c is None:
        v1 = slice(None)
    else:
        idx_map = { "depth": 0,}
        if isinstance(c, list):
            v1 = [idx_map[v] for v in c]
        else:
            v1 = idx_map[c]


    v2 = h

    v3 = w
    return array[v0, v1, v2, v3]
def idx_im_dispmaps(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[Literal["depth"], list[Literal["depth"]], None] = None, h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Im::DispMaps`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, c, h, w]``
    """


    v0 = n

    if c is None:
        v1 = slice(None)
    else:
        idx_map = { "depth": 0,}
        if isinstance(c, list):
            v1 = [idx_map[v] for v in c]
        else:
            v1 = idx_map[c]


    v2 = h

    v3 = w
    return array[v0, v1, v2, v3]
def idx_im_zbuffers(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[Literal["zdata"], list[Literal["zdata"]], None] = None, h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Im::ZBuffers`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, c, h, w]``
    """


    v0 = n

    if c is None:
        v1 = slice(None)
    else:
        idx_map = { "zdata": 0,}
        if isinstance(c, list):
            v1 = [idx_map[v] for v in c]
        else:
            v1 = idx_map[c]


    v2 = h

    v3 = w
    return array[v0, v1, v2, v3]
def idx_im_depthmasks(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[Literal["mask"], list[Literal["mask"]], None] = None, h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Im::DepthMasks`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, c, h, w]``
    """


    v0 = n

    if c is None:
        v1 = slice(None)
    else:
        idx_map = { "mask": 0,}
        if isinstance(c, list):
            v1 = [idx_map[v] for v in c]
        else:
            v1 = idx_map[c]


    v2 = h

    v3 = w
    return array[v0, v1, v2, v3]
def idx_points_space(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), data: Union[Literal["x", "y", "z"], list[Literal["x", "y", "z"]], None] = None) -> np.ndarray:
    """
    Select items from ``Points::Space`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, data]``
    """


    v0 = n

    if data is None:
        v1 = slice(None)
    else:
        idx_map = { "x": 0,"y": 1,"z": 2,}
        if isinstance(data, list):
            v1 = [idx_map[v] for v in data]
        else:
            v1 = idx_map[data]

    return array[v0, v1]
def idx_points_aspace(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), data: Union[Literal["x", "y", "z", "w"], list[Literal["x", "y", "z", "w"]], None] = None) -> np.ndarray:
    """
    Select items from ``Points::ASpace`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, data]``
    """


    v0 = n

    if data is None:
        v1 = slice(None)
    else:
        idx_map = { "x": 0,"y": 1,"z": 2,"w": 3,}
        if isinstance(data, list):
            v1 = [idx_map[v] for v in data]
        else:
            v1 = idx_map[data]

    return array[v0, v1]
def idx_points_aplane(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), data: Union[Literal["x", "y", "w"], list[Literal["x", "y", "w"]], None] = None) -> np.ndarray:
    """
    Select items from ``Points::APlane`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, data]``
    """


    v0 = n

    if data is None:
        v1 = slice(None)
    else:
        idx_map = { "x": 0,"y": 1,"w": 2,}
        if isinstance(data, list):
            v1 = [idx_map[v] for v in data]
        else:
            v1 = idx_map[data]

    return array[v0, v1]
def idx_points_plane(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), data: Union[Literal["x", "y"], list[Literal["x", "y"]], None] = None) -> np.ndarray:
    """
    Select items from ``Points::Plane`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, data]``
    """


    v0 = n

    if data is None:
        v1 = slice(None)
    else:
        idx_map = { "x": 0,"y": 1,}
        if isinstance(data, list):
            v1 = [idx_map[v] for v in data]
        else:
            v1 = idx_map[data]

    return array[v0, v1]
def idx_points_planewithd(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), data: Union[Literal["x", "y", "d"], list[Literal["x", "y", "d"]], None] = None) -> np.ndarray:
    """
    Select items from ``Points::PlaneWithD`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, data]``
    """


    v0 = n

    if data is None:
        v1 = slice(None)
    else:
        idx_map = { "x": 0,"y": 1,"d": 2,}
        if isinstance(data, list):
            v1 = [idx_map[v] for v in data]
        else:
            v1 = idx_map[data]

    return array[v0, v1]
def idx_points_arbdata(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None), data: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Points::ArbData`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n, data]``
    """


    v0 = n

    v1 = data
    return array[v0, v1]
def idx_mat_float(array: np.ndarray, /, row: Union[int, list[int],np.ndarray, slice] = slice(None), col: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Mat::Float`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[row, col]``
    """


    v0 = row

    v1 = col
    return array[v0, v1]
def idx_scalars_float(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Scalars::Float`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n]``
    """


    v0 = n
    return array[v0]
def idx_scalars_int(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Scalars::Int`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n]``
    """


    v0 = n
    return array[v0]
def idx_svals_float(array: np.ndarray, /, v: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``SVals::Float`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[v]``
    """


    v0 = v
    return array[v0]
def idx_svals_int(array: np.ndarray, /, v: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``SVals::Int`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[v]``
    """


    v0 = v
    return array[v0]
def idx_fieldgrid_scalarfieldgrid(array: np.ndarray, /, x: Union[int, list[int],np.ndarray, slice] = slice(None), y: Union[int, list[int],np.ndarray, slice] = slice(None), z: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``FieldGrid::ScalarFieldGrid`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[x, y, z]``
    """


    v0 = x

    v1 = y

    v2 = z
    return array[v0, v1, v2]
def idx_fieldgrid_occupacyfieldgrid(array: np.ndarray, /, x: Union[int, list[int],np.ndarray, slice] = slice(None), y: Union[int, list[int],np.ndarray, slice] = slice(None), z: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``FieldGrid::OccupacyFieldGrid`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[x, y, z]``
    """


    v0 = x

    v1 = y

    v2 = z
    return array[v0, v1, v2]
def idx_faces_faces(array: np.ndarray, /, face: Union[int, list[int],np.ndarray, slice] = slice(None), corner: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Faces::Faces`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[face, corner]``
    """


    v0 = face

    v1 = corner
    return array[v0, v1]
def idx_table_float(array: np.ndarray, /, row: Union[int, list[int],np.ndarray, slice] = slice(None), col: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Table::Float`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[row, col]``
    """


    v0 = row

    v1 = col
    return array[v0, v1]
def idx_coords_float(array: np.ndarray, /, n: Union[int, list[int],np.ndarray, slice] = slice(None)) -> np.ndarray:
    """
    Select items from ``Coords::Float`` alognside its named dimensions.

    This function only enables categorical filters for categorical axes.

    Aside of the categorical value index resolution, this is the same as ``array[n]``
    """


    v0 = n
    return array[v0]

def upd_im(array: np.ndarray, /, value_: Any, n: Union[int, list[int], np.ndarray, slice]=slice(None) , c: Union[int, list[int], np.ndarray, slice]=slice(None) , h: Union[int, list[int], np.ndarray, slice]=slice(None) , w: Union[int, list[int], np.ndarray, slice]=slice(None) ) -> None:
    """
    Update items from ``Im::*`` alognside its named dimensions.

    Meaning:

    * If value is not callable: ``array[n, c, h, w]=value_``
    * If value is callable: ``array[n, c, h, w]=value_(array[n, c, h, w])``
    """

    if callable(value_):
        value_ = value_(array[n, c, h, w])

    array[n, c, h, w] = value_

def upd_points(array: np.ndarray, /, value_: Any, n: Union[int, list[int], np.ndarray, slice]=slice(None) , data: Union[int, list[int], np.ndarray, slice]=slice(None) ) -> None:
    """
    Update items from ``Points::*`` alognside its named dimensions.

    Meaning:

    * If value is not callable: ``array[n, data]=value_``
    * If value is callable: ``array[n, data]=value_(array[n, data])``
    """

    if callable(value_):
        value_ = value_(array[n, data])

    array[n, data] = value_

def upd_mat(array: np.ndarray, /, value_: Any, row: Union[int, list[int], np.ndarray, slice]=slice(None) , col: Union[int, list[int], np.ndarray, slice]=slice(None) ) -> None:
    """
    Update items from ``Mat::*`` alognside its named dimensions.

    Meaning:

    * If value is not callable: ``array[row, col]=value_``
    * If value is callable: ``array[row, col]=value_(array[row, col])``
    """

    if callable(value_):
        value_ = value_(array[row, col])

    array[row, col] = value_

def upd_scalars(array: np.ndarray, /, value_: Any, n: Union[int, list[int], np.ndarray, slice]=slice(None) ) -> None:
    """
    Update items from ``Scalars::*`` alognside its named dimensions.

    Meaning:

    * If value is not callable: ``array[n]=value_``
    * If value is callable: ``array[n]=value_(array[n])``
    """

    if callable(value_):
        value_ = value_(array[n])

    array[n] = value_

def upd_svals(array: np.ndarray, /, value_: Any, v: Union[int, list[int], np.ndarray, slice]=slice(None) ) -> None:
    """
    Update items from ``SVals::*`` alognside its named dimensions.

    Meaning:

    * If value is not callable: ``array[v]=value_``
    * If value is callable: ``array[v]=value_(array[v])``
    """

    if callable(value_):
        value_ = value_(array[v])

    array[v] = value_

def upd_arbsamples(array: np.ndarray, /, value_: Any, n: Union[int, list[int], np.ndarray, slice]=slice(None) ) -> None:
    """
    Update items from ``ArbSamples::*`` alognside its named dimensions.

    Meaning:

    * If value is not callable: ``array[n]=value_``
    * If value is callable: ``array[n]=value_(array[n])``
    """

    if callable(value_):
        value_ = value_(array[n])

    array[n] = value_

def upd_fieldgrid(array: np.ndarray, /, value_: Any, x: Union[int, list[int], np.ndarray, slice]=slice(None) , y: Union[int, list[int], np.ndarray, slice]=slice(None) , z: Union[int, list[int], np.ndarray, slice]=slice(None) ) -> None:
    """
    Update items from ``FieldGrid::*`` alognside its named dimensions.

    Meaning:

    * If value is not callable: ``array[x, y, z]=value_``
    * If value is callable: ``array[x, y, z]=value_(array[x, y, z])``
    """

    if callable(value_):
        value_ = value_(array[x, y, z])

    array[x, y, z] = value_

def upd_faces(array: np.ndarray, /, value_: Any, face: Union[int, list[int], np.ndarray, slice]=slice(None) , corner: Union[int, list[int], np.ndarray, slice]=slice(None) ) -> None:
    """
    Update items from ``Faces::*`` alognside its named dimensions.

    Meaning:

    * If value is not callable: ``array[face, corner]=value_``
    * If value is callable: ``array[face, corner]=value_(array[face, corner])``
    """

    if callable(value_):
        value_ = value_(array[face, corner])

    array[face, corner] = value_

def upd_table(array: np.ndarray, /, value_: Any, row: Union[int, list[int], np.ndarray, slice]=slice(None) , col: Union[int, list[int], np.ndarray, slice]=slice(None) ) -> None:
    """
    Update items from ``Table::*`` alognside its named dimensions.

    Meaning:

    * If value is not callable: ``array[row, col]=value_``
    * If value is callable: ``array[row, col]=value_(array[row, col])``
    """

    if callable(value_):
        value_ = value_(array[row, col])

    array[row, col] = value_

def upd_coords(array: np.ndarray, /, value_: Any, n: Union[int, list[int], np.ndarray, slice]=slice(None) ) -> None:
    """
    Update items from ``Coords::*`` alognside its named dimensions.

    Meaning:

    * If value is not callable: ``array[n]=value_``
    * If value is callable: ``array[n]=value_(array[n])``
    """

    if callable(value_):
        value_ = value_(array[n])

    array[n] = value_


def upd_im_rgbs(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[Literal["r", "g", "b"], list[Literal["r", "g", "b"]], None] = None, h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Im::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, c, h, w]=value_``
    * If value is callable: ``array[n, c, h, w]=value_(array[n, c, h, w])``
    """


    v0 = n

    if c is None:
        v1 = slice(None)
    else:
        idx_map = { "r": 0,"g": 1,"b": 2,}
        if isinstance(c, list):
            v1 = [idx_map[v] for v in c]
        else:
            v1 = idx_map[c]


    v2 = h

    v3 = w
    if callable(value_):
        value_ = value_(array[v0, v1, v2, v3])

    array[v0, v1, v2, v3] = value_
def upd_im_floatmap(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[int, list[int],np.ndarray, slice] = slice(None), h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Im::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, c, h, w]=value_``
    * If value is callable: ``array[n, c, h, w]=value_(array[n, c, h, w])``
    """


    v0 = n

    v1 = c

    v2 = h

    v3 = w
    if callable(value_):
        value_ = value_(array[v0, v1, v2, v3])

    array[v0, v1, v2, v3] = value_
def upd_im_intmap(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[int, list[int],np.ndarray, slice] = slice(None), h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Im::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, c, h, w]=value_``
    * If value is callable: ``array[n, c, h, w]=value_(array[n, c, h, w])``
    """


    v0 = n

    v1 = c

    v2 = h

    v3 = w
    if callable(value_):
        value_ = value_(array[v0, v1, v2, v3])

    array[v0, v1, v2, v3] = value_
def upd_im_depthmaps(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[Literal["depth"], list[Literal["depth"]], None] = None, h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Im::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, c, h, w]=value_``
    * If value is callable: ``array[n, c, h, w]=value_(array[n, c, h, w])``
    """


    v0 = n

    if c is None:
        v1 = slice(None)
    else:
        idx_map = { "depth": 0,}
        if isinstance(c, list):
            v1 = [idx_map[v] for v in c]
        else:
            v1 = idx_map[c]


    v2 = h

    v3 = w
    if callable(value_):
        value_ = value_(array[v0, v1, v2, v3])

    array[v0, v1, v2, v3] = value_
def upd_im_dispmaps(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[Literal["depth"], list[Literal["depth"]], None] = None, h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Im::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, c, h, w]=value_``
    * If value is callable: ``array[n, c, h, w]=value_(array[n, c, h, w])``
    """


    v0 = n

    if c is None:
        v1 = slice(None)
    else:
        idx_map = { "depth": 0,}
        if isinstance(c, list):
            v1 = [idx_map[v] for v in c]
        else:
            v1 = idx_map[c]


    v2 = h

    v3 = w
    if callable(value_):
        value_ = value_(array[v0, v1, v2, v3])

    array[v0, v1, v2, v3] = value_
def upd_im_zbuffers(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[Literal["zdata"], list[Literal["zdata"]], None] = None, h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Im::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, c, h, w]=value_``
    * If value is callable: ``array[n, c, h, w]=value_(array[n, c, h, w])``
    """


    v0 = n

    if c is None:
        v1 = slice(None)
    else:
        idx_map = { "zdata": 0,}
        if isinstance(c, list):
            v1 = [idx_map[v] for v in c]
        else:
            v1 = idx_map[c]


    v2 = h

    v3 = w
    if callable(value_):
        value_ = value_(array[v0, v1, v2, v3])

    array[v0, v1, v2, v3] = value_
def upd_im_depthmasks(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), c: Union[Literal["mask"], list[Literal["mask"]], None] = None, h: Union[int, list[int],np.ndarray, slice] = slice(None), w: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Im::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, c, h, w]=value_``
    * If value is callable: ``array[n, c, h, w]=value_(array[n, c, h, w])``
    """


    v0 = n

    if c is None:
        v1 = slice(None)
    else:
        idx_map = { "mask": 0,}
        if isinstance(c, list):
            v1 = [idx_map[v] for v in c]
        else:
            v1 = idx_map[c]


    v2 = h

    v3 = w
    if callable(value_):
        value_ = value_(array[v0, v1, v2, v3])

    array[v0, v1, v2, v3] = value_
def upd_points_space(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), data: Union[Literal["x", "y", "z"], list[Literal["x", "y", "z"]], None] = None) -> None:
    """
    Update items from ``Points::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, data]=value_``
    * If value is callable: ``array[n, data]=value_(array[n, data])``
    """


    v0 = n

    if data is None:
        v1 = slice(None)
    else:
        idx_map = { "x": 0,"y": 1,"z": 2,}
        if isinstance(data, list):
            v1 = [idx_map[v] for v in data]
        else:
            v1 = idx_map[data]

    if callable(value_):
        value_ = value_(array[v0, v1])

    array[v0, v1] = value_
def upd_points_aspace(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), data: Union[Literal["x", "y", "z", "w"], list[Literal["x", "y", "z", "w"]], None] = None) -> None:
    """
    Update items from ``Points::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, data]=value_``
    * If value is callable: ``array[n, data]=value_(array[n, data])``
    """


    v0 = n

    if data is None:
        v1 = slice(None)
    else:
        idx_map = { "x": 0,"y": 1,"z": 2,"w": 3,}
        if isinstance(data, list):
            v1 = [idx_map[v] for v in data]
        else:
            v1 = idx_map[data]

    if callable(value_):
        value_ = value_(array[v0, v1])

    array[v0, v1] = value_
def upd_points_aplane(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), data: Union[Literal["x", "y", "w"], list[Literal["x", "y", "w"]], None] = None) -> None:
    """
    Update items from ``Points::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, data]=value_``
    * If value is callable: ``array[n, data]=value_(array[n, data])``
    """


    v0 = n

    if data is None:
        v1 = slice(None)
    else:
        idx_map = { "x": 0,"y": 1,"w": 2,}
        if isinstance(data, list):
            v1 = [idx_map[v] for v in data]
        else:
            v1 = idx_map[data]

    if callable(value_):
        value_ = value_(array[v0, v1])

    array[v0, v1] = value_
def upd_points_plane(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), data: Union[Literal["x", "y"], list[Literal["x", "y"]], None] = None) -> None:
    """
    Update items from ``Points::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, data]=value_``
    * If value is callable: ``array[n, data]=value_(array[n, data])``
    """


    v0 = n

    if data is None:
        v1 = slice(None)
    else:
        idx_map = { "x": 0,"y": 1,}
        if isinstance(data, list):
            v1 = [idx_map[v] for v in data]
        else:
            v1 = idx_map[data]

    if callable(value_):
        value_ = value_(array[v0, v1])

    array[v0, v1] = value_
def upd_points_planewithd(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), data: Union[Literal["x", "y", "d"], list[Literal["x", "y", "d"]], None] = None) -> None:
    """
    Update items from ``Points::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, data]=value_``
    * If value is callable: ``array[n, data]=value_(array[n, data])``
    """


    v0 = n

    if data is None:
        v1 = slice(None)
    else:
        idx_map = { "x": 0,"y": 1,"d": 2,}
        if isinstance(data, list):
            v1 = [idx_map[v] for v in data]
        else:
            v1 = idx_map[data]

    if callable(value_):
        value_ = value_(array[v0, v1])

    array[v0, v1] = value_
def upd_points_arbdata(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None), data: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Points::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n, data]=value_``
    * If value is callable: ``array[n, data]=value_(array[n, data])``
    """


    v0 = n

    v1 = data
    if callable(value_):
        value_ = value_(array[v0, v1])

    array[v0, v1] = value_
def upd_mat_float(array: np.ndarray, /, value_: Any, row: Union[int, list[int],np.ndarray, slice] = slice(None), col: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Mat::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[row, col]=value_``
    * If value is callable: ``array[row, col]=value_(array[row, col])``
    """


    v0 = row

    v1 = col
    if callable(value_):
        value_ = value_(array[v0, v1])

    array[v0, v1] = value_
def upd_scalars_float(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Scalars::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n]=value_``
    * If value is callable: ``array[n]=value_(array[n])``
    """


    v0 = n
    if callable(value_):
        value_ = value_(array[v0])

    array[v0] = value_
def upd_scalars_int(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Scalars::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n]=value_``
    * If value is callable: ``array[n]=value_(array[n])``
    """


    v0 = n
    if callable(value_):
        value_ = value_(array[v0])

    array[v0] = value_
def upd_svals_float(array: np.ndarray, /, value_: Any, v: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``SVals::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[v]=value_``
    * If value is callable: ``array[v]=value_(array[v])``
    """


    v0 = v
    if callable(value_):
        value_ = value_(array[v0])

    array[v0] = value_
def upd_svals_int(array: np.ndarray, /, value_: Any, v: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``SVals::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[v]=value_``
    * If value is callable: ``array[v]=value_(array[v])``
    """


    v0 = v
    if callable(value_):
        value_ = value_(array[v0])

    array[v0] = value_
def upd_fieldgrid_scalarfieldgrid(array: np.ndarray, /, value_: Any, x: Union[int, list[int],np.ndarray, slice] = slice(None), y: Union[int, list[int],np.ndarray, slice] = slice(None), z: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``FieldGrid::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[x, y, z]=value_``
    * If value is callable: ``array[x, y, z]=value_(array[x, y, z])``
    """


    v0 = x

    v1 = y

    v2 = z
    if callable(value_):
        value_ = value_(array[v0, v1, v2])

    array[v0, v1, v2] = value_
def upd_fieldgrid_occupacyfieldgrid(array: np.ndarray, /, value_: Any, x: Union[int, list[int],np.ndarray, slice] = slice(None), y: Union[int, list[int],np.ndarray, slice] = slice(None), z: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``FieldGrid::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[x, y, z]=value_``
    * If value is callable: ``array[x, y, z]=value_(array[x, y, z])``
    """


    v0 = x

    v1 = y

    v2 = z
    if callable(value_):
        value_ = value_(array[v0, v1, v2])

    array[v0, v1, v2] = value_
def upd_faces_faces(array: np.ndarray, /, value_: Any, face: Union[int, list[int],np.ndarray, slice] = slice(None), corner: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Faces::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[face, corner]=value_``
    * If value is callable: ``array[face, corner]=value_(array[face, corner])``
    """


    v0 = face

    v1 = corner
    if callable(value_):
        value_ = value_(array[v0, v1])

    array[v0, v1] = value_
def upd_table_float(array: np.ndarray, /, value_: Any, row: Union[int, list[int],np.ndarray, slice] = slice(None), col: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Table::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[row, col]=value_``
    * If value is callable: ``array[row, col]=value_(array[row, col])``
    """


    v0 = row

    v1 = col
    if callable(value_):
        value_ = value_(array[v0, v1])

    array[v0, v1] = value_
def upd_coords_float(array: np.ndarray, /, value_: Any, n: Union[int, list[int],np.ndarray, slice] = slice(None)) -> None:
    """
    Update items from ``Coords::*`` alognside its named dimensions.

    This function only enables categorical values for categorical axes.

    Aside of the categorical value index resolution, this is the same as:

    * If value is not callable: ``array[n]=value_``
    * If value is callable: ``array[n]=value_(array[n])``
    """


    v0 = n
    if callable(value_):
        value_ = value_(array[v0])

    array[v0] = value_

def dmatch_im(array: np.ndarray, /, shape: Optional[dict[Literal["n", "c", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the specified array may belong to ``Im::*`` solely based on its dimension.

    Additionally check whether the shape matches to further expectations and the array has the specified kinds.

    Parameters
    ----------
    array
        The array to check.
    shape
        The exact shape values.
    kinds
        The kinds to check.

    Returns
    -------
    v
        True if both checks succeed. Otherwise False.
    """


    if len(array.shape) != 4:
        return False

    if shape is not None:

        if "n" in shape.keys():
            if array.shape[0] != shape["n"]:
                return False

        if "c" in shape.keys():
            if array.shape[1] != shape["c"]:
                return False

        if "h" in shape.keys():
            if array.shape[2] != shape["h"]:
                return False

        if "w" in shape.keys():
            if array.shape[3] != shape["w"]:
                return False

    if kinds is not None:
        if "single" in kinds:

            if not (array.shape[0]==1):
                return False

    if dtype is not None:
        if array.dtype != dtype:
            return False
    return True

def scast_im(array: np.ndarray, /, shape: Optional[dict[Literal["n", "c", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None ) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Im::*`` with the specified shape and kinds.
    """
    if not dmatch_im(array, shape=shape, kinds=kinds):
        raise ValueError('The array does not belong to \"Im::*\"".')
    return array

def dmatch_points(array: np.ndarray, /, shape: Optional[dict[Literal["n", "data"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the specified array may belong to ``Points::*`` solely based on its dimension.

    Additionally check whether the shape matches to further expectations and the array has the specified kinds.

    Parameters
    ----------
    array
        The array to check.
    shape
        The exact shape values.
    kinds
        The kinds to check.

    Returns
    -------
    v
        True if both checks succeed. Otherwise False.
    """


    if len(array.shape) != 2:
        return False

    if shape is not None:

        if "n" in shape.keys():
            if array.shape[0] != shape["n"]:
                return False

        if "data" in shape.keys():
            if array.shape[1] != shape["data"]:
                return False

    if kinds is not None:
        if "nonempty" in kinds:

            if not (array.shape[0]>0):
                return False

    if dtype is not None:
        if array.dtype != dtype:
            return False
    return True

def scast_points(array: np.ndarray, /, shape: Optional[dict[Literal["n", "data"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None ) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Points::*`` with the specified shape and kinds.
    """
    if not dmatch_points(array, shape=shape, kinds=kinds):
        raise ValueError('The array does not belong to \"Points::*\"".')
    return array

def dmatch_mat(array: np.ndarray, /, shape: Optional[dict[Literal["row", "col"], int]] = None, kinds: Optional[set[Literal["f3x4"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the specified array may belong to ``Mat::*`` solely based on its dimension.

    Additionally check whether the shape matches to further expectations and the array has the specified kinds.

    Parameters
    ----------
    array
        The array to check.
    shape
        The exact shape values.
    kinds
        The kinds to check.

    Returns
    -------
    v
        True if both checks succeed. Otherwise False.
    """


    if len(array.shape) != 2:
        return False

    if shape is not None:

        if "row" in shape.keys():
            if array.shape[0] != shape["row"]:
                return False

        if "col" in shape.keys():
            if array.shape[1] != shape["col"]:
                return False

    if kinds is not None:
        if "f3x4" in kinds:

            if not (array.shape[0]==3):
                return False

            if not (array.shape[1]==4):
                return False

    if dtype is not None:
        if array.dtype != dtype:
            return False
    return True

def scast_mat(array: np.ndarray, /, shape: Optional[dict[Literal["row", "col"], int]] = None, kinds: Optional[set[Literal["f3x4"]]]=None ) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Mat::*`` with the specified shape and kinds.
    """
    if not dmatch_mat(array, shape=shape, kinds=kinds):
        raise ValueError('The array does not belong to \"Mat::*\"".')
    return array

def dmatch_scalars(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None, dtype: Optional[Any] = None) -> bool:
    """
    Check whether the specified array may belong to ``Scalars::*`` solely based on its dimension.

    Additionally check whether the shape matches to further expectations.

    Parameters
    ----------
    array
        The array to check.
    shape
        The exact shape values.

    Returns
    -------
    v
        True if both checks succeed. Otherwise False.
    """


    if len(array.shape) != 1:
        return False

    if shape is not None:

        if "n" in shape.keys():
            if array.shape[0] != shape["n"]:
                return False

    if dtype is not None:
        if array.dtype != dtype:
            return False
    return True

def scast_scalars(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Scalars::*`` with the specified shape and kinds.
    """
    if not dmatch_scalars(array, shape=shape):
        raise ValueError('The array does not belong to \"Scalars::*\"".')
    return array

def dmatch_svals(array: np.ndarray, /, shape: Optional[dict[Literal["v"], int]] = None, dtype: Optional[Any] = None) -> bool:
    """
    Check whether the specified array may belong to ``SVals::*`` solely based on its dimension.

    Additionally check whether the shape matches to further expectations.

    Parameters
    ----------
    array
        The array to check.
    shape
        The exact shape values.

    Returns
    -------
    v
        True if both checks succeed. Otherwise False.
    """


    if len(array.shape) != 1:
        return False

    if shape is not None:

        if "v" in shape.keys():
            if array.shape[0] != shape["v"]:
                return False

    if dtype is not None:
        if array.dtype != dtype:
            return False
    return True

def scast_svals(array: np.ndarray, /, shape: Optional[dict[Literal["v"], int]] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``SVals::*`` with the specified shape and kinds.
    """
    if not dmatch_svals(array, shape=shape):
        raise ValueError('The array does not belong to \"SVals::*\"".')
    return array

def dmatch_arbsamples(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None, dtype: Optional[Any] = None) -> bool:
    """
    Check whether the specified array may belong to ``ArbSamples::*`` solely based on its dimension.

    Additionally check whether the shape matches to further expectations.

    Parameters
    ----------
    array
        The array to check.
    shape
        The exact shape values.

    Returns
    -------
    v
        True if both checks succeed. Otherwise False.
    """


    if len(array.shape) > 1:
        return False

    if shape is not None:

        if "n" in shape.keys():
            if array.shape[0] != shape["n"]:
                return False

    if dtype is not None:
        if array.dtype != dtype:
            return False
    return True

def scast_arbsamples(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``ArbSamples::*`` with the specified shape and kinds.
    """
    if not dmatch_arbsamples(array, shape=shape):
        raise ValueError('The array does not belong to \"ArbSamples::*\"".')
    return array

def dmatch_fieldgrid(array: np.ndarray, /, shape: Optional[dict[Literal["x", "y", "z"], int]] = None, kinds: Optional[set[Literal["validfield"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the specified array may belong to ``FieldGrid::*`` solely based on its dimension.

    Additionally check whether the shape matches to further expectations and the array has the specified kinds.

    Parameters
    ----------
    array
        The array to check.
    shape
        The exact shape values.
    kinds
        The kinds to check.

    Returns
    -------
    v
        True if both checks succeed. Otherwise False.
    """


    if len(array.shape) != 3:
        return False

    if shape is not None:

        if "x" in shape.keys():
            if array.shape[0] != shape["x"]:
                return False

        if "y" in shape.keys():
            if array.shape[1] != shape["y"]:
                return False

        if "z" in shape.keys():
            if array.shape[2] != shape["z"]:
                return False

    if kinds is not None:
        if "validfield" in kinds:

            if not (array.shape[0]>2):
                return False

            if not (array.shape[1]>2):
                return False

            if not (array.shape[2]>2):
                return False

    if dtype is not None:
        if array.dtype != dtype:
            return False
    return True

def scast_fieldgrid(array: np.ndarray, /, shape: Optional[dict[Literal["x", "y", "z"], int]] = None, kinds: Optional[set[Literal["validfield"]]]=None ) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``FieldGrid::*`` with the specified shape and kinds.
    """
    if not dmatch_fieldgrid(array, shape=shape, kinds=kinds):
        raise ValueError('The array does not belong to \"FieldGrid::*\"".')
    return array

def dmatch_faces(array: np.ndarray, /, shape: Optional[dict[Literal["face", "corner"], int]] = None, kinds: Optional[set[Literal["triangles"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the specified array may belong to ``Faces::*`` solely based on its dimension.

    Additionally check whether the shape matches to further expectations and the array has the specified kinds.

    Parameters
    ----------
    array
        The array to check.
    shape
        The exact shape values.
    kinds
        The kinds to check.

    Returns
    -------
    v
        True if both checks succeed. Otherwise False.
    """


    if len(array.shape) != 2:
        return False

    if shape is not None:

        if "face" in shape.keys():
            if array.shape[0] != shape["face"]:
                return False

        if "corner" in shape.keys():
            if array.shape[1] != shape["corner"]:
                return False

    if kinds is not None:
        if "triangles" in kinds:

            if not (array.shape[1]==3):
                return False

    if dtype is not None:
        if array.dtype != dtype:
            return False
    return True

def scast_faces(array: np.ndarray, /, shape: Optional[dict[Literal["face", "corner"], int]] = None, kinds: Optional[set[Literal["triangles"]]]=None ) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Faces::*`` with the specified shape and kinds.
    """
    if not dmatch_faces(array, shape=shape, kinds=kinds):
        raise ValueError('The array does not belong to \"Faces::*\"".')
    return array

def dmatch_table(array: np.ndarray, /, shape: Optional[dict[Literal["row", "col"], int]] = None, dtype: Optional[Any] = None) -> bool:
    """
    Check whether the specified array may belong to ``Table::*`` solely based on its dimension.

    Additionally check whether the shape matches to further expectations.

    Parameters
    ----------
    array
        The array to check.
    shape
        The exact shape values.

    Returns
    -------
    v
        True if both checks succeed. Otherwise False.
    """


    if len(array.shape) != 2:
        return False

    if shape is not None:

        if "row" in shape.keys():
            if array.shape[0] != shape["row"]:
                return False

        if "col" in shape.keys():
            if array.shape[1] != shape["col"]:
                return False

    if dtype is not None:
        if array.dtype != dtype:
            return False
    return True

def scast_table(array: np.ndarray, /, shape: Optional[dict[Literal["row", "col"], int]] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Table::*`` with the specified shape and kinds.
    """
    if not dmatch_table(array, shape=shape):
        raise ValueError('The array does not belong to \"Table::*\"".')
    return array

def dmatch_coords(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None, dtype: Optional[Any] = None) -> bool:
    """
    Check whether the specified array may belong to ``Coords::*`` solely based on its dimension.

    Additionally check whether the shape matches to further expectations.

    Parameters
    ----------
    array
        The array to check.
    shape
        The exact shape values.

    Returns
    -------
    v
        True if both checks succeed. Otherwise False.
    """


    if len(array.shape) != 1:
        return False

    if shape is not None:

        if "n" in shape.keys():
            if array.shape[0] != shape["n"]:
                return False

    if dtype is not None:
        if array.dtype != dtype:
            return False
    return True

def scast_coords(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Coords::*`` with the specified shape and kinds.
    """
    if not dmatch_coords(array, shape=shape):
        raise ValueError('The array does not belong to \"Coords::*\"".')
    return array




def scast_im_rgbs(array: np.ndarray, /, shape: Optional[dict[Literal["n", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Im::RGBs`` with the specified shape and kinds.
    """
    if not match_im_rgbs(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Im::RGBs\"".')
    return array


def match_im_rgbs(array: np.ndarray, /, shape:  Optional[dict[Literal["n", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Im::RGBs``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_im(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    if len(array) > 0:

        if array.shape[1] != 3:
            return False
    

        
            
        if not (np.all(0.0 <= np.take(array, 0, axis=1)) and np.all(np.take(array, 0, axis=1) <= 1.0)):
            return False
        
            
        if not (np.all(0.0 <= np.take(array, 1, axis=1)) and np.all(np.take(array, 1, axis=1) <= 1.0)):
            return False
        
            
        if not (np.all(0.0 <= np.take(array, 2, axis=1)) and np.all(np.take(array, 2, axis=1) <= 1.0)):
            return False
    return True

def scast_im_floatmap(array: np.ndarray, /, shape: Optional[dict[Literal["n", "c", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Im::FloatMap`` with the specified shape and kinds.
    """
    if not match_im_floatmap(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Im::FloatMap\"".')
    return array


def match_im_floatmap(array: np.ndarray, /, shape:  Optional[dict[Literal["n", "c", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Im::FloatMap``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_im(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    return True

def scast_im_intmap(array: np.ndarray, /, shape: Optional[dict[Literal["n", "c", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Im::IntMap`` with the specified shape and kinds.
    """
    if not match_im_intmap(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Im::IntMap\"".')
    return array


def match_im_intmap(array: np.ndarray, /, shape:  Optional[dict[Literal["n", "c", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Im::IntMap``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.integer):
        return False
    
    if not dmatch_im(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    return True

def scast_im_depthmaps(array: np.ndarray, /, shape: Optional[dict[Literal["n", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Im::DepthMaps`` with the specified shape and kinds.
    """
    if not match_im_depthmaps(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Im::DepthMaps\"".')
    return array


def match_im_depthmaps(array: np.ndarray, /, shape:  Optional[dict[Literal["n", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Im::DepthMaps``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_im(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    if len(array) > 0:

        if array.shape[1] != 1:
            return False
    

        
            
        if not np.all(0.0 <= np.take(array, 0, axis=1)):
            return False
    return True

def scast_im_dispmaps(array: np.ndarray, /, shape: Optional[dict[Literal["n", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Im::DispMaps`` with the specified shape and kinds.
    """
    if not match_im_dispmaps(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Im::DispMaps\"".')
    return array


def match_im_dispmaps(array: np.ndarray, /, shape:  Optional[dict[Literal["n", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Im::DispMaps``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_im(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    if len(array) > 0:

        if array.shape[1] != 1:
            return False
    

        
            
        if not np.all(0.0 <= np.take(array, 0, axis=1)):
            return False
    return True

def scast_im_zbuffers(array: np.ndarray, /, shape: Optional[dict[Literal["n", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Im::ZBuffers`` with the specified shape and kinds.
    """
    if not match_im_zbuffers(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Im::ZBuffers\"".')
    return array


def match_im_zbuffers(array: np.ndarray, /, shape:  Optional[dict[Literal["n", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Im::ZBuffers``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_im(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    if len(array) > 0:

        if array.shape[1] != 1:
            return False
    

        
            
        if not (np.all(0.0 <= np.take(array, 0, axis=1)) and np.all(np.take(array, 0, axis=1) <= 1.0)):
            return False
    return True

def scast_im_depthmasks(array: np.ndarray, /, shape: Optional[dict[Literal["n", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Im::DepthMasks`` with the specified shape and kinds.
    """
    if not match_im_depthmasks(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Im::DepthMasks\"".')
    return array


def match_im_depthmasks(array: np.ndarray, /, shape:  Optional[dict[Literal["n", "h", "w"], int]] = None, kinds: Optional[set[Literal["single"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Im::DepthMasks``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.bool_):
        return False
    
    if not dmatch_im(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    if len(array) > 0:

        if array.shape[1] != 1:
            return False
    
            


    return True


def scast_points_space(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Points::Space`` with the specified shape and kinds.
    """
    if not match_points_space(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Points::Space\"".')
    return array


def match_points_space(array: np.ndarray, /, shape:  Optional[dict[Literal["n"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Points::Space``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_points(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    if len(array) > 0:

        if array.shape[1] != 3:
            return False
    
            


    return True

def scast_points_aspace(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Points::ASpace`` with the specified shape and kinds.
    """
    if not match_points_aspace(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Points::ASpace\"".')
    return array


def match_points_aspace(array: np.ndarray, /, shape:  Optional[dict[Literal["n"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Points::ASpace``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_points(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    if len(array) > 0:

        if array.shape[1] != 4:
            return False
    
            


    return True

def scast_points_aplane(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Points::APlane`` with the specified shape and kinds.
    """
    if not match_points_aplane(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Points::APlane\"".')
    return array


def match_points_aplane(array: np.ndarray, /, shape:  Optional[dict[Literal["n"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Points::APlane``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_points(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    if len(array) > 0:

        if array.shape[1] != 3:
            return False
    
            


    return True

def scast_points_plane(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Points::Plane`` with the specified shape and kinds.
    """
    if not match_points_plane(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Points::Plane\"".')
    return array


def match_points_plane(array: np.ndarray, /, shape:  Optional[dict[Literal["n"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Points::Plane``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_points(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    if len(array) > 0:

        if array.shape[1] != 2:
            return False
    
            


    return True

def scast_points_planewithd(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Points::PlaneWithD`` with the specified shape and kinds.
    """
    if not match_points_planewithd(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Points::PlaneWithD\"".')
    return array


def match_points_planewithd(array: np.ndarray, /, shape:  Optional[dict[Literal["n"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Points::PlaneWithD``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_points(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    if len(array) > 0:

        if array.shape[1] != 3:
            return False
    
            


    return True

def scast_points_arbdata(array: np.ndarray, /, shape: Optional[dict[Literal["n", "data"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Points::ArbData`` with the specified shape and kinds.
    """
    if not match_points_arbdata(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Points::ArbData\"".')
    return array


def match_points_arbdata(array: np.ndarray, /, shape:  Optional[dict[Literal["n", "data"], int]] = None, kinds: Optional[set[Literal["nonempty"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Points::ArbData``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_points(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    return True


def scast_mat_float(array: np.ndarray, /, shape: Optional[dict[Literal["row", "col"], int]] = None, kinds: Optional[set[Literal["f3x4"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Mat::Float`` with the specified shape and kinds.
    """
    if not match_mat_float(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Mat::Float\"".')
    return array


def match_mat_float(array: np.ndarray, /, shape:  Optional[dict[Literal["row", "col"], int]] = None, kinds: Optional[set[Literal["f3x4"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Mat::Float``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_mat(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    return True


def scast_scalars_float(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None, dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Scalars::Float`` with the specified shape and kinds.
    """
    if not match_scalars_float(array, shape=shape, dtype=dtype):
        raise ValueError('The array does not belong to \"Scalars::Float\"".')
    return array


def match_scalars_float(array: np.ndarray, /, shape:  Optional[dict[Literal["n"], int]] = None, dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Scalars::Float``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_scalars(array, shape=shape, dtype=dtype): # type: ignore
        return False

    return True

def scast_scalars_int(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None, dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Scalars::Int`` with the specified shape and kinds.
    """
    if not match_scalars_int(array, shape=shape, dtype=dtype):
        raise ValueError('The array does not belong to \"Scalars::Int\"".')
    return array


def match_scalars_int(array: np.ndarray, /, shape:  Optional[dict[Literal["n"], int]] = None, dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Scalars::Int``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.integer):
        return False
    
    if not dmatch_scalars(array, shape=shape, dtype=dtype): # type: ignore
        return False

    return True


def scast_svals_float(array: np.ndarray, /, shape: Optional[dict[Literal["v"], int]] = None, dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``SVals::Float`` with the specified shape and kinds.
    """
    if not match_svals_float(array, shape=shape, dtype=dtype):
        raise ValueError('The array does not belong to \"SVals::Float\"".')
    return array


def match_svals_float(array: np.ndarray, /, shape:  Optional[dict[Literal["v"], int]] = None, dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``SVals::Float``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_svals(array, shape=shape, dtype=dtype): # type: ignore
        return False

    return True

def scast_svals_int(array: np.ndarray, /, shape: Optional[dict[Literal["v"], int]] = None, dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``SVals::Int`` with the specified shape and kinds.
    """
    if not match_svals_int(array, shape=shape, dtype=dtype):
        raise ValueError('The array does not belong to \"SVals::Int\"".')
    return array


def match_svals_int(array: np.ndarray, /, shape:  Optional[dict[Literal["v"], int]] = None, dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``SVals::Int``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.integer):
        return False
    
    if not dmatch_svals(array, shape=shape, dtype=dtype): # type: ignore
        return False

    return True



def scast_fieldgrid_scalarfieldgrid(array: np.ndarray, /, shape: Optional[dict[Literal["x", "y", "z"], int]] = None, kinds: Optional[set[Literal["validfield"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``FieldGrid::ScalarFieldGrid`` with the specified shape and kinds.
    """
    if not match_fieldgrid_scalarfieldgrid(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"FieldGrid::ScalarFieldGrid\"".')
    return array


def match_fieldgrid_scalarfieldgrid(array: np.ndarray, /, shape:  Optional[dict[Literal["x", "y", "z"], int]] = None, kinds: Optional[set[Literal["validfield"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``FieldGrid::ScalarFieldGrid``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_fieldgrid(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    return True

def scast_fieldgrid_occupacyfieldgrid(array: np.ndarray, /, shape: Optional[dict[Literal["x", "y", "z"], int]] = None, kinds: Optional[set[Literal["validfield"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``FieldGrid::OccupacyFieldGrid`` with the specified shape and kinds.
    """
    if not match_fieldgrid_occupacyfieldgrid(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"FieldGrid::OccupacyFieldGrid\"".')
    return array


def match_fieldgrid_occupacyfieldgrid(array: np.ndarray, /, shape:  Optional[dict[Literal["x", "y", "z"], int]] = None, kinds: Optional[set[Literal["validfield"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``FieldGrid::OccupacyFieldGrid``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_fieldgrid(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    if len(array) > 0:
    
            
        if not (np.all(-1.0 <= array) and np.all(array <= 1.0)):
            return False

    return True


def scast_faces_faces(array: np.ndarray, /, shape: Optional[dict[Literal["face", "corner"], int]] = None, kinds: Optional[set[Literal["triangles"]]]=None , dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Faces::Faces`` with the specified shape and kinds.
    """
    if not match_faces_faces(array, shape=shape, kinds=kinds, dtype=dtype):
        raise ValueError('The array does not belong to \"Faces::Faces\"".')
    return array


def match_faces_faces(array: np.ndarray, /, shape:  Optional[dict[Literal["face", "corner"], int]] = None, kinds: Optional[set[Literal["triangles"]]]=None , dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Faces::Faces``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.integer):
        return False
    
    if not dmatch_faces(array, shape=shape, kinds=kinds, dtype=dtype): # type: ignore
        return False

    return True


def scast_table_float(array: np.ndarray, /, shape: Optional[dict[Literal["row", "col"], int]] = None, dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Table::Float`` with the specified shape and kinds.
    """
    if not match_table_float(array, shape=shape, dtype=dtype):
        raise ValueError('The array does not belong to \"Table::Float\"".')
    return array


def match_table_float(array: np.ndarray, /, shape:  Optional[dict[Literal["row", "col"], int]] = None, dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Table::Float``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_table(array, shape=shape, dtype=dtype): # type: ignore
        return False

    return True


def scast_coords_float(array: np.ndarray, /, shape: Optional[dict[Literal["n"], int]] = None, dtype: Optional[Any] = None) -> np.ndarray:
    """
    Returns
    -------
    v
        The same array.

    Raises
    ------
    ValueError
        If the array does not belong to ``Coords::Float`` with the specified shape and kinds.
    """
    if not match_coords_float(array, shape=shape, dtype=dtype):
        raise ValueError('The array does not belong to \"Coords::Float\"".')
    return array


def match_coords_float(array: np.ndarray, /, shape:  Optional[dict[Literal["n"], int]] = None, dtype: Optional[Any] = None) -> bool:
    """
    Check whether the array might belong to ``Coords::Float``

    This function checks the size of the array, the data type of the array and the value set of the array.
    """
    
    if not np.issubdtype(array.dtype, np.floating):
        return False
    
    if not dmatch_coords(array, shape=shape, dtype=dtype): # type: ignore
        return False

    return True



def newshape_im(*, n: int, c: int, h: int, w: int):
    """
    Get the shape of the arrays in ``Im::*`` with the specified dimensions. 
    """
    return (n,c,h,w,)

def newshape_im_rgbs(*, n: int, h: int, w: int):
    """
    Get the shape of the arrays in ``Im::RGBs`` with the specified dimensions. 
    """
    return (n,3,h,w,)

def newshape_im_floatmap(*, n: int, c: int, h: int, w: int):
    """
    Get the shape of the arrays in ``Im::FloatMap`` with the specified dimensions. 
    """
    return (n,c,h,w,)

def newshape_im_intmap(*, n: int, c: int, h: int, w: int):
    """
    Get the shape of the arrays in ``Im::IntMap`` with the specified dimensions. 
    """
    return (n,c,h,w,)

def newshape_im_depthmaps(*, n: int, h: int, w: int):
    """
    Get the shape of the arrays in ``Im::DepthMaps`` with the specified dimensions. 
    """
    return (n,1,h,w,)

def newshape_im_dispmaps(*, n: int, h: int, w: int):
    """
    Get the shape of the arrays in ``Im::DispMaps`` with the specified dimensions. 
    """
    return (n,1,h,w,)

def newshape_im_zbuffers(*, n: int, h: int, w: int):
    """
    Get the shape of the arrays in ``Im::ZBuffers`` with the specified dimensions. 
    """
    return (n,1,h,w,)

def newshape_im_depthmasks(*, n: int, h: int, w: int):
    """
    Get the shape of the arrays in ``Im::DepthMasks`` with the specified dimensions. 
    """
    return (n,1,h,w,)

def newshape_points(*, n: int, data: int):
    """
    Get the shape of the arrays in ``Points::*`` with the specified dimensions. 
    """
    return (n,data,)

def newshape_points_space(*, n: int):
    """
    Get the shape of the arrays in ``Points::Space`` with the specified dimensions. 
    """
    return (n,3,)

def newshape_points_aspace(*, n: int):
    """
    Get the shape of the arrays in ``Points::ASpace`` with the specified dimensions. 
    """
    return (n,4,)

def newshape_points_aplane(*, n: int):
    """
    Get the shape of the arrays in ``Points::APlane`` with the specified dimensions. 
    """
    return (n,3,)

def newshape_points_plane(*, n: int):
    """
    Get the shape of the arrays in ``Points::Plane`` with the specified dimensions. 
    """
    return (n,2,)

def newshape_points_planewithd(*, n: int):
    """
    Get the shape of the arrays in ``Points::PlaneWithD`` with the specified dimensions. 
    """
    return (n,3,)

def newshape_points_arbdata(*, n: int, data: int):
    """
    Get the shape of the arrays in ``Points::ArbData`` with the specified dimensions. 
    """
    return (n,data,)

def newshape_mat(*, row: int, col: int):
    """
    Get the shape of the arrays in ``Mat::*`` with the specified dimensions. 
    """
    return (row,col,)

def newshape_mat_float(*, row: int, col: int):
    """
    Get the shape of the arrays in ``Mat::Float`` with the specified dimensions. 
    """
    return (row,col,)

def newshape_scalars(*, n: int):
    """
    Get the shape of the arrays in ``Scalars::*`` with the specified dimensions. 
    """
    return (n,)

def newshape_scalars_float(*, n: int):
    """
    Get the shape of the arrays in ``Scalars::Float`` with the specified dimensions. 
    """
    return (n,)

def newshape_scalars_int(*, n: int):
    """
    Get the shape of the arrays in ``Scalars::Int`` with the specified dimensions. 
    """
    return (n,)

def newshape_svals(*, v: int):
    """
    Get the shape of the arrays in ``SVals::*`` with the specified dimensions. 
    """
    return (v,)

def newshape_svals_float(*, v: int):
    """
    Get the shape of the arrays in ``SVals::Float`` with the specified dimensions. 
    """
    return (v,)

def newshape_svals_int(*, v: int):
    """
    Get the shape of the arrays in ``SVals::Int`` with the specified dimensions. 
    """
    return (v,)

def newshape_arbsamples(*, n: int, wildcard_: list[int]):
    """
    Get the shape of the arrays in ``ArbSamples::*`` with the specified dimensions. 
    """
    return (n,*wildcard_,)

def newshape_fieldgrid(*, x: int, y: int, z: int):
    """
    Get the shape of the arrays in ``FieldGrid::*`` with the specified dimensions. 
    """
    return (x,y,z,)

def newshape_fieldgrid_scalarfieldgrid(*, x: int, y: int, z: int):
    """
    Get the shape of the arrays in ``FieldGrid::ScalarFieldGrid`` with the specified dimensions. 
    """
    return (x,y,z,)

def newshape_fieldgrid_occupacyfieldgrid(*, x: int, y: int, z: int):
    """
    Get the shape of the arrays in ``FieldGrid::OccupacyFieldGrid`` with the specified dimensions. 
    """
    return (x,y,z,)

def newshape_faces(*, face: int, corner: int):
    """
    Get the shape of the arrays in ``Faces::*`` with the specified dimensions. 
    """
    return (face,corner,)

def newshape_faces_faces(*, face: int, corner: int):
    """
    Get the shape of the arrays in ``Faces::Faces`` with the specified dimensions. 
    """
    return (face,corner,)

def newshape_table(*, row: int, col: int):
    """
    Get the shape of the arrays in ``Table::*`` with the specified dimensions. 
    """
    return (row,col,)

def newshape_table_float(*, row: int, col: int):
    """
    Get the shape of the arrays in ``Table::Float`` with the specified dimensions. 
    """
    return (row,col,)

def newshape_coords(*, n: int):
    """
    Get the shape of the arrays in ``Coords::*`` with the specified dimensions. 
    """
    return (n,)

def newshape_coords_float(*, n: int):
    """
    Get the shape of the arrays in ``Coords::Float`` with the specified dimensions. 
    """
    return (n,)

