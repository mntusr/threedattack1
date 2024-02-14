# Coordinate system

The program always uses Z-up **right** handed coordinate system for 3d representation.

# Scene format

The scenes are GLTF files with the following structure.

* **Target object.** The MESH of the target object is called: `TargetMesh`. The origin of the target object should be at the center of its bounding box. It should not have any rotation and scale, but it might have translation. Its ancestors should have neither rotation nor translation nor scale.
* **Viewpoints object.** The vertexes of this object object specify the fixed camera viewpoints. The MESH of the viewpoints object should be called `ViewpointsMesh`. Both the viewpoint object and its ancestors should not have any translation, rotation or scale.

The non-warped target object in the GLTF file should consist of a single body and its mesh should be closed. Unlike most other requirements, these are not checked at runtime, since this would make the code complicated too much.

The size of its bounding box matches an externaly specified value.

The previously mentioned scene format is called "standard scene format" in the code.

The program somewhat transforms the glb files on load.

1. Hides the viewpoints object.
2. Disables the appearance of the casted shadows on the target object. This does not prevent the target object to cast shadows, but no casted shadows appear on it.

# Authoring scenes

## Create the model in Blender

Generally speaking, you can achieve the necessary format in Blender in multiple ways, but these steps provide a mostly good checklist to do so.

Make sure that the names of the MESHES of the viewpoints object and the target object are correctly named.

Convert all objects to mesh (otherwise the exporter may ignore or not correctly export them). Steps:

1. Add "Realize instances" to all geometry node-based properties
3. Unlink the linked duplicates (`Object > Relations > Make single user > Object & data & materials`) with all objects selected
4. Do the actual conversion (`Object > Convert > Object to mesh`) with all objects selected

Make sure that the viewpoints object and the target object has the proper settings:

1. Make sure that the viewpoints object and its parents have no transforms.
   * If the viewpoints object has no parents, then you can just apply all transforms on it. (`Object > Apply > All transforms`)
2. Make sure that the target object has the proper settings.
   * Main things to consider.
     * The origin of the target object should be at the center of the boundaries of the target object. (`Object > Set Origin > Origin to Geometry` and set `Center` to `Bounds center`)
     * The target object should not be scaled or rotated.
     * The ancestors of the target object should not be scaled or rotated neither. Their location should also be at the world origin, however this **does not** apply to the target object itself.
   * If the target object has no parents, then you can do these transforms to make sure that everything is correct:
     * Apply the scale and rotation on the target object (`Object > Apply > Rotation & Scale`).
     * Place the origin of the target object to its bounding box center.  (`Object > Set Origin > Origin to Geometry` and set `Center` to `Bounds center`)

Make sure that the material can be exported using the GLTF exporter

* See the documentation of the GLTF exporter [here](https://docs.blender.org/manual/en/latest/addons/import_export/scene_gltf2.html#exported-materials).
* Some things to consider
  * Don't forget to bake the procedural materials before exporting.
  * You need explicit UV mapping instead of relying on the box projection. Generally, exporter ignores the `Texture Coordinate` shader node. Use the `UV Map` shader node instead.
  * The explicit UV mapping exporter also cannot export everything. See [this documentation](https://docs.blender.org/manual/en/latest/addons/import_export/scene_gltf2.html#uv-mapping) about the constraints on the exported nodes.

Make sure that the materials will be correctly processed:

* Diffuse color: sRGB
* Normal map, ...: non-Color

Do the GLTF export.

* Important properties:
  * Enable `Transform > +Y up`
  * Enable `Data > Mesh > Loose Points`
  * Enable: `Include > Limit to > Visible Objects`
  * Enable: `Include > Data > Punctual Lights`
  * Set: `Data > Lighting > Lighting Mode` to `Raw (deprecated)`
* Tips and tricks
  * Make sure that you do not accidentally export the hidden objects only used for boolean operations.
  * Do not forget to double check the texturing of the exported objects. Free online viewer: <https://github.khronos.org/glTF-Sample-Viewer-Release/>

## Configure the lighting and the scene size

Create a file, called `<scene name>.json` at the same directory as the gltf file of the scene. This json file specifies the lighting-related additional properties of the scene and the size of the scene.

The initial content of the file should look like this:

```json
{
    "shadow_blur": 0.0,
    "shadow_map_resolution": 1024,
    "dir_ambient_hpr_diff": 12,
    "dir_ambient_hpr_weight": 0.0,
    "nondir_ambient_light_r": 0.0,
    "nondir_ambient_light_g": 0.0,
    "nondir_ambient_light_b": 0.0,
    "shadow_area_root": "<the name of the root object of the created world>",
    "force_shadow_0": [],
    "world_size_x": <world size X>,
    "world_size_y": <world size Y>,
    "world_size_z": <world size Z>
}
```

The specified size of the scene is specified by three properties, `world_size_x`, `world_size_y` and `world_size_z`. You should specify the size of the tightest bounding box of all **non-point-cloud meshes** of the scene in metres. This bounding box does not include the lights, cameras and camera viewpoints. The role of this additional bounding box specification is to prevent incorrect scene scaling on export or due to a bug in the program itself.

The steps to configure the lighting.

1. Make sure that the shadows correctly appear. This is mostly affected by two things. 1) The UV map of the object. If some parts of the objects are black without any reason, then the UV map of the object is probably not correct. 2) Whether the mesh of the object is double sided or not. If the object is double sided, then the object is more suspectible to [shadow acne](https://computergraphics.stackexchange.com/questions/2192/cause-of-shadow-acne). If the object is not double sided, then it is more suspectible to [peter panning](https://learn.microsoft.com/en-us/windows/win32/dxtecharts/common-techniques-to-improve-shadow-depth-maps). You can control this double sided behavior materialwise in Blender using the `Backface culling` option.
2. Configure the directional ambient lights. The lighting of the scene is somewhat modified during import. The imporer function replaces the original directional lights with a shadow casting directional light and four non-shadow.casting directional lights. These non-shadow casting directional lights work like the regular ambient light, but they do not make the objects flat. The rotation of the shadow-casting directional light is the same as the rotation of the non-shadow-casting directional lights. The H and P parameter of the rotation of the non-shadow-casting directional lights is different from the direction of the shadow-casting directional light. The difference is controlled by the `dir_ambient_hpr_diff`. The intensity of the shadow-casting directional light is `(1-dir_ambient_hpr_weight)*<original intensity>`, the total intensity of the non-shadow-casting directional lights is `dir_ambient_hpr_weight*<original intensity>`
3. Configure the ambient light color. This is specified by `nondir_ambient_light_r`, `nondir_ambient_light_g`, `nondir_ambient_light_b`. These values might be greater than 1 too, since they also specify the intensity of the light. However, keep in mind that this non-directional ambient light makes the objects more flat, so use it sparingly.
4. Configure the shadows. The main properties:
   * **The resolution of the shadow map.** If this value is greater, then the shadows look less jagged, but the rendering needs more memory. Configuration option: `shadow_map_resolution`
   * **Shadow blur.** In the real world, the shadows are not sharp. This option adds an artificial blur of the shadows. This might make the shadows more realistic and hide the jaggedness of the low resolution shadow map. However, the the increased shadow blur makes the peter panning worse. Configuration option: `shadow_blur`.
   * The name of the root of the objects that have shadows and cast shadows. This option is useful, since it enables you to reduce the area that is covered by the shadow map (i. e. the shadow map might have smaller resolution). Configuration option: `shadow_area_root`.
   * **Forced shadows.** You can mark some objects as shadowed, regardless of the lighting. This is useful, since this makes it possible to fake shadows on the background objects. The list of the objects to which this fake shadow should be added: `force_shadow_0`

## Generate the auxillary files

When you exported the GLB file of the scene, then you should generate some additional files too.

These are:

* The viewpoint split. This splits the viewpoints to training, validation and test sets.
* The cache for the occupacy field of the target object. This is a cache that makes the occupacy field-based operations on the target object way faster. In order to simplify the implementation, the usage of this cache is mandatory.

You can generate the viewpoint splits using `generate_viewpoint_split.py`. The number of viewpoints in each set should sum up to the total number of viewpoints. This script does not use CLI arguments. You can set the size of the sets and the scene on a GUI at the start of the script instead.

You can generate the occupacy field cache of the target object using the `generate_target_occup_fn_cache.py` command. This script does not use CLI arguments. You can select the scene on a GUI at the start of the script instead.