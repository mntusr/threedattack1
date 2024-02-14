# Array groups


## Im

The group of arrays that store pixel-wise data for images. The point at `(x=0, y=0)` is the top left corner and the point at `(x=-1, y=-1)` is the bottom right corner.


**Dimensions**

| Idx | Name | Description |
|-----|------|-------------|
| 0 | `N` | The sample. |
| 1 | `C` | The channel. |
| 2 | `H` | The Y coordinate of the pixel. |
| 3 | `W` | The X coordinate of the pixel. |




**Array kind** Single

* Description: There is only one image.
* Fully qualified name: `Im::@Single`
* Rules: `N==1`


**Array** RGBs
* Description: The RGB images.
* Fully qualified name: `Im::RGBs`
* Set of elements: See the categorical dimensions.
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions:

  * Categories of `C`:
    | Idx | Name | Description | Element set |
    |-----|------|-------------|-------------| 
    | 0 | `R` | The red channel of the images. | `[0.0, 1.0]` |
    | 1 | `G` | The green channel of the images. | `[0.0, 1.0]` |
    | 2 | `B` | The blue channel of the images. | `[0.0, 1.0]` |

**Array** DepthMaps
* Description: The array that contains dense depth maps. Since the depth maps do not necessarily contain values for all pixels, you should denote the present values using separate masks (`Images::DepthMasks`). The depth maps are not necessarily exact. They might be affine invariant or scale invariant too.
* Fully qualified name: `Im::DepthMaps`
* Set of elements: See the categorical dimensions.
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions:

  * Categories of `C`:
    | Idx | Name | Description | Element set |
    |-----|------|-------------|-------------| 
    | 0 | `Depth` | The depth values for individual pixels.  | `[0.0, ∞)` |

**Array** DispMaps
* Description: The array that contains dense disparity maps. Since the depth maps do not necessarily contain values for all pixels, you should denote the present values using separate masks (`Images::DepthMasks`). The depth maps are not necessarily exact. They might be affine invariant or scale invariant too.
* Fully qualified name: `Im::DispMaps`
* Set of elements: See the categorical dimensions.
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions:

  * Categories of `C`:
    | Idx | Name | Description | Element set |
    |-----|------|-------------|-------------| 
    | 0 | `Depth` | The depth values for individual pixels.  | `[0.0, ∞)` |

**Array** ZBuffers
* Description: The Panda3d Z-buffer depth data. See the Panda3d docs for more details about the semantics of the individual values.
* Fully qualified name: `Im::ZBuffers`
* Set of elements: See the categorical dimensions.
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions:

  * Categories of `C`:
    | Idx | Name | Description | Element set |
    |-----|------|-------------|-------------| 
    | 0 | `Zdata` | The Z-buffer data for individual pixels.  | `[0.0, 1.0]` |

**Array** DepthMasks
* Description: The depth masks. Value True means that the pixel should be excluded. Value False means that the corresponding pixels should be included.
* Fully qualified name: `Im::DepthMasks`
* Set of elements: `{True, False}`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `BoolLike`
* Categorical dimensions:

  * Categories of `C`:
    | Idx | Name | Description |
    |-----|------|-------------|
    | 0 | `Mask` | The pixel-wise mask for depth maps. |




## Points

The group of arrays that store pointwise data.


**Dimensions**

| Idx | Name | Description |
|-----|------|-------------|
| 0 | `N` | Sample |
| 1 | `Data` | The data for the points. |




**Array kind** NonEmpty

* Description: The array contains at least one point.
* Fully qualified name: `Points::@NonEmpty`
* Rules: `N > 0`


**Array** Space
* Description: The Cartesian coordinates of points in the 3d space.
* Fully qualified name: `Points::Space`
* Set of elements: `ℝ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions:

  * Categories of `Data`:
    | Idx | Name | Description |
    |-----|------|-------------|
    | 0 | `X` | The x coordinate. |
    | 1 | `Y` | The y coordinate. |
    | 2 | `Z` | The z coordinate. |

**Array** ASpace
* Description: The coordinates of points in the affine 3d space.
* Fully qualified name: `Points::ASpace`
* Set of elements: `ℝ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions:

  * Categories of `Data`:
    | Idx | Name | Description |
    |-----|------|-------------|
    | 0 | `X` | The x coordinate. |
    | 1 | `Y` | The y coordinate. |
    | 2 | `Z` | The z coordinate. |
    | 3 | `W` | The w coordinate. |

**Array** APlane
* Description: The coordinates of points in the affine 2d space.
* Fully qualified name: `Points::APlane`
* Set of elements: `ℝ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions:

  * Categories of `Data`:
    | Idx | Name | Description |
    |-----|------|-------------|
    | 0 | `X` | The x coordinate. |
    | 1 | `Y` | The y coordinate. |
    | 2 | `W` | The w coordinate. |

**Array** Plane
* Description: The coordinates of points in the Cartesian 2d space.
* Fully qualified name: `Points::Plane`
* Set of elements: `ℝ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions:

  * Categories of `Data`:
    | Idx | Name | Description |
    |-----|------|-------------|
    | 0 | `X` | The x coordinate. |
    | 1 | `Y` | The y coordinate. |

**Array** PlaneWithD
* Description: The Cartesian coordinates of points in the image space and their true distance from the camera.
* Fully qualified name: `Points::PlaneWithD`
* Set of elements: `ℝ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions:

  * Categories of `Data`:
    | Idx | Name | Description |
    |-----|------|-------------|
    | 0 | `X` | The x coordinate. |
    | 1 | `Y` | The y coordinate. |
    | 2 | `D` | The true distance from the camera. |

**Array** ArbData
* Description: Arbitrary real data for the points.
* Fully qualified name: `Points::ArbData`
* Set of elements: `ℝ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions: -




## Mat

The matrices. These arrays contains only a single matrix or vector.


**Dimensions**

| Idx | Name | Description |
|-----|------|-------------|
| 0 | `Row` | The row. |
| 1 | `Col` | The column. |




**Array kind** F3x4

* Description: The matrix is a 3x4 matrix.
* Fully qualified name: `Mat::@F3x4`
* Rules: `Row==3,Col==4`


**Array** Float
* Description: The matrix containing floating point numbers.
* Fully qualified name: `Mat::Float`
* Set of elements: `ℝ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions: -




## Scalars

An array that contains exactly one value for each sample.


**Dimensions**

| Idx | Name | Description |
|-----|------|-------------|
| 0 | `N` | The sample. |





**Array** Float
* Description: The values are floating point numbers for each sample.
* Fully qualified name: `Scalars::Float`
* Set of elements: `ℝ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions: -

**Array** Int
* Description: The values are integer numbers for a each sample.
* Fully qualified name: `Scalars::Int`
* Set of elements: `ℕ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `IntLike`
* Categorical dimensions: -




## SVals

The array group contains an 1d tensor that belongs to a single sample.


**Dimensions**

| Idx | Name | Description |
|-----|------|-------------|
| 0 | `V` | The values. |





**Array** Float
* Description: The 1d tensor that contains floating point numbers for a single sample.
* Fully qualified name: `SVals::Float`
* Set of elements: `ℝ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions: -

**Array** Int
* Description: The 1d tensor that contains integer numbers for a single sample.
* Fully qualified name: `SVals::Int`
* Set of elements: `ℕ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `IntLike`
* Categorical dimensions: -




## ArbSamples

The arrays that contain non-defined values for each sample.


**Dimensions**

| Idx | Name | Description |
|-----|------|-------------|
| 0 | `N` | The index of the samples. |
| - | \<wildcard dimensions\>(min=0) |  |








## FieldGrid

The arrays that contain a sampled 3d field.


**Dimensions**

| Idx | Name | Description |
|-----|------|-------------|
| 0 | `X` | The X coordinate. |
| 1 | `Y` | The Y coordinate. |
| 2 | `Z` | The Z coordinate. |




**Array kind** ValidField

* Description: None
* Fully qualified name: `FieldGrid::@ValidField`
* Rules: `X > 2, Y > 2, Z > 2`


**Array** ScalarFieldGrid
* Description: The arrays that contain a sampled 3d field that has floating point values.
* Fully qualified name: `FieldGrid::ScalarFieldGrid`
* Set of elements: `ℝ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions: -

**Array** OccupacyFieldGrid
* Description: The arrays that contain a sampled occupacy function. Values greater than or equal 0 mean that the specified area is inside of the object. Values smaller than 0 mean that the area is outside of the object.
* Fully qualified name: `FieldGrid::OccupacyFieldGrid`
* Set of elements: `[-1.0, 1.0]`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions: -




## Faces

The arrays that contain the indices of faces.


**Dimensions**

| Idx | Name | Description |
|-----|------|-------------|
| 0 | `Face` | The index of the face. |
| 1 | `Corner` | The index of the corner. |




**Array kind** Triangles

* Description: The faces are specified by triangles.
* Fully qualified name: `Faces::@Triangles`
* Rules: `Corner==3`


**Array** Faces
* Description: The only meaningful array in this group.
* Fully qualified name: `Faces::Faces`
* Set of elements: `ℕ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `IntLike`
* Categorical dimensions: -




## Table

The arrays that contain a tabular data.


**Dimensions**

| Idx | Name | Description |
|-----|------|-------------|
| 0 | `Row` | The index of the row in the table. |
| 1 | `Col` | The index of the column in the table. |





**Array** Float
* Description: The array contains floating point data.
* Fully qualified name: `Table::Float`
* Set of elements: `ℝ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions: -




## Coords

The arrays that contain a sequence of coordinates alongside a single axis. This array group does not require the coordinates to be monothonically increasing or evenly spaced.


**Dimensions**

| Idx | Name | Description |
|-----|------|-------------|
| 0 | `N` | The index of the coordinate. |





**Array** Float
* Description: The array contains floating point data.
* Fully qualified name: `Coords::Float`
* Set of elements: `ℝ`
* Can the array contain 0 elements (without any array kind specified): True
* Data type: `FloatLike`
* Categorical dimensions: -



