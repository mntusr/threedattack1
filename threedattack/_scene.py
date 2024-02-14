import base64
import copy
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence, TextIO

import numpy as np
from panda3d.core import NodePath, PerspectiveLens

from ._logging import LoggingFreqFunction
from ._typing import type_instance
from .dataset_model import (
    CamProjSpec,
    DatasetLike,
    DepthsWithMasks,
    ExactSampleCounts,
    RGBsWithDepthsAndMasks,
    SamplesBase,
    SampleType,
    SampleTypeError,
)
from .losses import (
    LossDerivationMethod,
    LossPrecision,
    RawLossFn,
    calculate_loss_values,
    collect_raw_losses_with_viewpt_type,
    concat_losses,
    get_aggr_delta_loss_dict_from_losses,
)
from .rendering_core import (
    DesiredViewpointCounts,
    LivePreviewController,
    ObjectTransform,
    ObjectTransformResult,
    ObjectTransformType,
    OccupacyFieldSamples,
    Panda3dAssumptionViolation,
    Panda3dShowBase,
    PointBasedVectorFieldSpec,
    ScaledStandingAreas,
    ThreeDPoint,
    TwoDAreas,
    TwoDSize,
    ViewpointBasedCamController,
    ViewpointSplit,
    get_bounding_rectangle_on_screen,
    get_cam_proj_spec_for_showbase,
    get_object_transform_by_type,
    get_object_transform_type_by_name,
    get_sky_mask_limit,
    get_target_obj_field_cache_path_for_world,
    get_viewpoints_path_for_world,
    put_obj,
    show_depths_as_point_clouds_in_browser,
    zbuf_2_depth_and_mask,
)
from .target_model import AsyncDepthPredictor, predict_aligned
from .tensor_types.idx import *


def create_scene_or_quit_wit_error(scene_config: "SceneConfig") -> "Scene":
    """
    Load the scene from the specified config.

    If any problem is found, then this function writes them to the console, then finally quits. Otherwise returns with the loaded scene.

    The parameters of this function match the parameters of `Scene.from_path`.
    """
    scene = Scene.from_config_or_error(scene_config)

    if isinstance(scene, list):
        print("The scene is not valid. Errors:")

        for error in scene:
            print(error)
        sys.exit(1)
    else:
        return scene


class Scene:
    """
    A component that renders or modifies a particular scene.

    Parameters
    ----------
    show_base
        The controlled show base. This component asssumes exclusive control over the show base.
    viewpoint_split
        The viewpoint split to use as the source of the viewpoints.
    viewpt_counts
        The actually used number of viewpoints for the different viewpoint types.
    rendering_resolution
        The resolution of the rendered images.
    world_path
        The path of the ``.glb`` file of the loaded world.
    object_transform_type
        The type of the warping transformation applied on the target object.
    target_obj_field_cache
        The cache of the occupacy field of the target object. Ignored if the mesh of the target object is transformed.
    n_volume_sampling_steps_along_shortest_axis
        If the volume of the object is transformed, then this parameter controls the number of sampling steps alongside the shortest axis. The shortest axis is the axis alongside which the transformed object in the relevant area the shortest is. Ignored if the mesh of the target object is directly transformed instead.
    target_size_multiplier
        A value that controls the connection between the original bounding box of the target object and the area in which the transformed target object should stay (the relevant area of the transformation).
    max_depth
        The maximal depth of the scene. The greater depth values are masked out. This emulates the maximal depth of some real depth sensors.

    Raises
    ------
    ValueError
        If the specified scene is not a valid scene.

    See also
    --------
    Scene.from_path
        A more convenient method to load scenes from gltf files.

    Developer notes
    ---------------
    This class tries to pretend statelessness at rendering, but the rendering is actually stateful behind the scenes. When a sample at index `i` is rendered, the following things happen:

    1. Select the viewpoint at index `i` with the proper viewpoint type
    2. Render a single frame using Panda3d, then capture it (this might initiate more than rendering operations in Panda3d)
    3. Calculate the area of the target object on the screen
    4. Restore the previously selected viewpoint

    The documentation refers to the functions that implement these operations as the internal stateful rendering API of this class.
    """

    def __init__(
        self,
        show_base: Panda3dShowBase,
        viewpoint_split: ViewpointSplit,
        viewpt_counts: DesiredViewpointCounts,
        rendering_resolution: TwoDSize,
        world_path: Path,
        object_transform_type: ObjectTransformType,
        target_obj_field_cache: OccupacyFieldSamples,
        n_volume_sampling_steps_along_shortest_axis: int,
        target_size_multiplier: float,
        depth_cap: float,
    ) -> None:
        viewpoint_split = viewpoint_split.select_n(viewpt_counts)

        scene_format_errors = show_base.get_standard_scene_format_errors()
        if len(show_base.get_standard_scene_format_errors()) > 0:
            raise ValueError(
                f"The specified show base contains a non-standard scene. Errors: {str(scene_format_errors)}"
            )

        self._show_base = show_base
        self._viewpoints = ViewpointBasedCamController(
            base=self._show_base, viewpt_split=viewpoint_split
        )
        self.__target_area = self._viewpoints.update_target_area(target_size_multiplier)
        self._last_set_viewpoint_idx = 0
        self._select_viewpoint(self._last_set_viewpoint_idx, SampleType.Train)
        self._rendering_resolution = rendering_resolution
        self._scene_path = world_path
        self._viewpoint_split = viewpoint_split
        self._target_object_transform = get_object_transform_by_type(
            transform_type=object_transform_type,
            field_cache=target_obj_field_cache,
            n_volume_sampling_steps_along_shortest_axis=n_volume_sampling_steps_along_shortest_axis,
            original_obj=show_base.get_target_obj_mesh_path(),
            transform_bounds=self.__target_area,
        )
        self.__target_obj_copy = copy.deepcopy(
            self._show_base.get_target_obj_mesh_path()
        )
        self._target_n_bodies: int = 1
        self._transform_change_amount_score: float = 0
        self.__depth_cap = depth_cap
        self.__n_volume_sampling_steps_along_shortest_axis = (
            n_volume_sampling_steps_along_shortest_axis
        )
        self.__applied_transform: PointBasedVectorFieldSpec | None = None
        self.__is_transformed: bool = False

    def get_rendering_resolution(self) -> TwoDSize:
        """
        Get the resolution of the rendered images of this scene.
        """
        return self._rendering_resolution

    def set_target_transform(
        self, vector_field: PointBasedVectorFieldSpec | None
    ) -> None:
        """
        Set the transformation of the target object or restore the original target ojbect.

        Parameters
        ----------
        vector_field
            The vector field that controls the transformation of the target object if it is not None. If it is None, then the non-transformed target object should be restored instead. The control points of the vector field are in the local coordinate system of the mesh of the target object (but they are not relative).

        Raises
        ------
        RuntimeError
            If a new transformation is applied, when the target object is already transformed.
        ShowBaseDestroyedError
            If the internal Panda3d show base of the scene is already destroyed.
        """
        if self._show_base is None:
            raise self._show_base_destroyed_error()

        if vector_field is not None:
            if self.__is_transformed:
                raise RuntimeError("The target object is already transformed.")

            transform_result = self._target_object_transform.transform_obj_new(
                vector_field,
            )
            put_obj(
                path=self._show_base.get_target_obj_mesh_path(),
                new_obj=transform_result.new_obj.node(),
            )
            self._target_n_bodies = transform_result.n_bodies
            self._transform_change_amount_score = transform_result.change_amount_score
            self.__applied_transform = vector_field
            self.__is_transformed = True
        else:
            put_obj(
                path=self._show_base.get_target_obj_mesh_path(),
                new_obj=self.__target_obj_copy.node(),
            )
            self._target_n_bodies = 1
            self._transform_change_amount_score = 0.0
            self.__applied_transform = None
            self.__is_transformed = False

    def is_transformed(self) -> bool:
        """
        Return true if the target object is already transformed.
        """
        return self.__is_transformed

    def get_target_n_bodies(self) -> int:
        """
        Get the number of separate bodies in the mesh of the target object.
        """
        return self._target_n_bodies

    def get_transform_type(self) -> ObjectTransformType:
        """
        Get the type of the transformation operation that the scene might apply to the target object.

        This function returns with the transform type configured for this scene regardless of the transform state of the target object itself.
        """
        return self._target_object_transform.get_transform_type()

    def get_transform_change_amount_score(self) -> float:
        """
        Get the change amount score for the for the currently applied transformation. Returns 0 if there is no currently applied transformation.
        """
        return self._transform_change_amount_score

    def get_world_path(self) -> Path:
        """Get the path of the world loaded by this scene."""
        return self._scene_path

    def get_target_areas(self) -> ScaledStandingAreas:
        """
        Get the different bounding boxes of the target object.
        """
        return self.__target_area

    def _select_viewpoint(self, viewpoint_index: int, viewpt_type: SampleType) -> None:
        """
        Set the camera position to the specified viewpoint. The rotation of the camera will be updated to point to the target object during the position update.

        This function is part pf the internal stateful rendering API of this class.

        Parameters
        ----------
        viewpoint_index
            The index of the viewpoint to use. Negative indexes are supported.
        viewpt_type
            The type of the viewpoint to select.

        Raises
        ------
        IndexError
            If the viewpoint index is out of range.
        ShowBaseDestroyedError
            If the internal Panda3d show base of the scene is already destroyed.
        """
        if self._show_base is None:
            raise self._show_base_destroyed_error()

        self._last_set_viewpoint_idx = viewpoint_index
        self._last_set_viewpt_type = viewpt_type
        self._viewpoints.select_viewpoint(viewpoint_index, viewpt_type)

    def get_n_samples_for_type(self, viewpt_type: SampleType) -> int:
        return self._viewpoints.get_filtered_viewpoint_count(viewpt_type)

    def get_n_samples(self) -> ExactSampleCounts:
        n_train_viewpints = self._viewpoints.get_filtered_viewpoint_count(
            SampleType.Train
        )
        n_test_viewpints = self._viewpoints.get_filtered_viewpoint_count(
            SampleType.Test
        )
        n_val_viewpints = self._viewpoints.get_filtered_viewpoint_count(SampleType.Val)
        return ExactSampleCounts(
            n_train_samples=n_train_viewpints,
            n_test_samples=n_test_viewpints,
            n_val_samples=n_val_viewpints,
        )

    def get_applied_transform_vector_field_deepcopy(
        self,
    ) -> "PointBasedVectorFieldSpec | None":
        """
        Get a deep copy of the specification of the vector field of the transformation currently applied.

        This function returns None if there is currently no transformation applied on the target object.
        """
        if self.__applied_transform is None:
            return None
        else:
            return PointBasedVectorFieldSpec(
                control_points=self.__applied_transform.control_points.copy(),
                vectors=self.__applied_transform.vectors.copy(),
            )

    def get_config_deepcopy(self) -> "SceneConfig":
        """
        Get a copy of the scene configuration that enables you to fully recreate this scene, including the transformation currently applied on the target object.

        You can safely call this function when the internal show base of this scene is already destroyed.
        """
        viewpt_counts = self._viewpoint_split.get_viewpt_counts()

        return SceneConfig(
            applied_transform=self.get_applied_transform_vector_field_deepcopy(),
            depth_cap=self.__depth_cap,
            n_volume_sampling_steps_along_shortest_axis=self.__n_volume_sampling_steps_along_shortest_axis,
            object_transform_type=self._target_object_transform.get_transform_type(),
            resolution=self._rendering_resolution,
            target_size_multiplier=self.__target_area.get_size_multiplier(),
            viewpt_counts=DesiredViewpointCounts(
                n_train_samples=viewpt_counts.n_train_samples,
                n_test_samples=viewpt_counts.n_test_samples,
                n_val_samples=viewpt_counts.n_val_samples,
            ),
            world_path=self._scene_path,
        )

    def save(self, target: Path | TextIO) -> None:
        """
        Same as ``self.get_config_deepcopy().save_json(config_path)``
        """
        config = self.get_config_deepcopy()
        config.save_json(target)

    @staticmethod
    def load(source: Path | TextIO) -> "Scene":
        """
        Same as ``Scene.from_config(SceneConfig.from_json(config_path))``
        """
        config = SceneConfig.from_json(source)
        return Scene.from_config(config)

    @staticmethod
    def from_config_or_error(scene_config: "SceneConfig") -> "Scene | list[str]":
        """
        Create a scene from the specified scene config or return with the possible errors.

        Parameters
        ----------
        scene_config
            The scene config to use.

        Returns
        -------
        v
            The loaded scene if the world is loadable and matches the standard scene format. Otherwise the list of the errors.
        """
        show_base = Panda3dShowBase(offscreen=True, win_size=scene_config.resolution)
        errors = show_base.load_world_from_blender(scene_config.world_path)

        if len(errors) > 0:
            return errors

        errors = show_base.get_standard_scene_format_errors()
        if len(errors) > 0:
            return errors

        viewpoints_path = get_viewpoints_path_for_world(scene_config.world_path)
        if not viewpoints_path.exists():
            return [f'The viewpoints split "{viewpoints_path}" does not exist.']

        viewpoint_split = ViewpointSplit.load_npz(viewpoints_path)
        errors = viewpoint_split.get_select_n_errors(scene_config.viewpt_counts)
        if len(errors) > 0:
            return errors

        target_obj_field_cache_path = get_target_obj_field_cache_path_for_world(
            scene_config.world_path
        )
        target_obj_field_cache = OccupacyFieldSamples.load_npz(
            target_obj_field_cache_path
        )

        scene = Scene(
            show_base=show_base,
            viewpoint_split=viewpoint_split,
            viewpt_counts=scene_config.viewpt_counts,
            rendering_resolution=scene_config.resolution,
            world_path=scene_config.world_path,
            target_obj_field_cache=target_obj_field_cache,
            object_transform_type=scene_config.object_transform_type,
            n_volume_sampling_steps_along_shortest_axis=scene_config.n_volume_sampling_steps_along_shortest_axis,
            target_size_multiplier=scene_config.target_size_multiplier,
            depth_cap=scene_config.depth_cap,
        )

        if scene_config.applied_transform is not None:
            scene.set_target_transform(scene_config.applied_transform)

        return scene

    @staticmethod
    def from_config(scene_config: "SceneConfig") -> "Scene":
        """
        Create a scene from the specified scene config.

        Parameters
        ----------
        scene_config
            The scene config to use.

        Returns
        -------
        v
            The loaded scene if the world is loadable and matches the standard scene format. Otherwise the list of the errors.

        Raises
        ------
        InvalidSceneError
           If the world is not loadable or does not match the standard scene format.
        """
        scene_or_error = Scene.from_config_or_error(scene_config)

        if isinstance(scene_or_error, list):
            raise InvalidSceneError(
                f"Failed to load the scene config. Errors: {scene_or_error}"
            )
        else:
            return scene_or_error

    def _get_target_obj_area_on_screen(self) -> TwoDAreas:
        """
        Get the area of the target object on the screen from the currectly selected viewpoint.

        This function is part pf the internal stateful rendering API of this class.

        Returns
        -------
        v
            The area.

        Raises
        ------
        ShowBaseDestroyedError
            If the internal Panda3d show base of the scene is already destroyed.
        """
        if self._show_base is None:
            raise self._show_base_destroyed_error()

        target_pos = self._viewpoints.get_target_position()
        target_obj_threed_area_corners = self.__target_area.get_full_area(
            origin_of_obj=target_pos, include_outside_offset=False
        ).get_corners()
        area = get_bounding_rectangle_on_screen(
            target_obj_threed_area_corners,
            base=self._show_base,
            rendering_resolution=self._rendering_resolution,
        )
        return area

    def _render_rgbd(self) -> RGBsWithDepthsAndMasks:
        """
        Capture a new single RGBD frame using the current camera position.

        This function always initiates (at least one) new render.

        This function is part pf the internal stateful rendering API of this class.

        Returns
        -------
        v
            The captured frame.

        Raises
        ------
        Panda3dAssumptionViolation
            If the camera does not have a PerspectiveLens.
        ShowBaseDestroyedError
            If the internal Panda3d show base of the scene is already destroyed.
        """
        if self._show_base is None:
            raise self._show_base_destroyed_error()

        frame = self._show_base.render_single_RGBB_frame()
        cam_lens = self._show_base.get_cam_lens()

        if not isinstance(cam_lens, PerspectiveLens):
            raise Panda3dAssumptionViolation("Only perspective lens is supported.")

        depth_and_mask = zbuf_2_depth_and_mask(
            zbuf_data=frame.zbufs, camera_lens=cam_lens, max_depth=self.__depth_cap
        )

        frame = RGBsWithDepthsAndMasks(
            rgbs=frame.rgbs,
            depths=depth_and_mask.depths,
            masks=depth_and_mask.masks,
        )

        return frame

    def get_sample(self, idx: int, sample_type: SampleType) -> "SceneSamples":
        return self.get_samples([idx], sample_type)

    def show_depths_as_point_clouds_in_browser(
        self, depths: dict[str, DepthsWithMasks]
    ) -> None:
        """
        Show the depth maps as point cloud in a new browser window.

        The individual depth maps are added as separate traces to a single figure.

        Parameters
        ----------
        depths
            A dictionary, where the keys are the names of the depth maps. The values are the depth maps (with the corresponding masks).

        Raises
        ------
        ShowBaseDestroyedError
            If the internal Panda3d show base of the scene is already destroyed.

        Notes
        -----
        This function internally creates a Plotly figure, then saves it to a temporary HTML file, then opens it using the default browser.

        This means that this function needs the global temporary dictionary to be activated.
        """
        if self._show_base is None:
            raise self._show_base_destroyed_error()

        cam_proj_spec = get_cam_proj_spec_for_showbase(self._show_base)
        show_depths_as_point_clouds_in_browser(
            cam_proj_spec=cam_proj_spec, depths=depths
        )

    def get_depth_cap(self) -> float:
        return self.__depth_cap

    def get_samples(
        self, idxs: Sequence[int] | slice, sample_type: SampleType
    ) -> "SceneSamples":
        if self.get_n_samples_for_type(sample_type) == 0:
            raise SampleTypeError(
                f'There is no viewpoint for sample type "{sample_type.public_name}"'
            )

        if self._show_base is None:
            raise self._show_base_destroyed_error()

        initial_pos = self._last_set_viewpoint_idx
        initial_viewpt_type = self._last_set_viewpt_type
        if isinstance(idxs, slice):
            idx_iterable = range(
                *idxs.indices(self.get_n_samples_for_type(sample_type))
            )
        else:
            idx_iterable = idxs

        rgbd_list: list[RGBsWithDepthsAndMasks] = []
        area_list: list[TwoDAreas] = []
        for idx in idx_iterable:
            self._select_viewpoint(int(idx), sample_type)
            rgbd = self._render_rgbd()
            area = self._get_target_obj_area_on_screen()
            rgbd_list.append(rgbd)
            area_list.append(area)

        rgbds = RGBsWithDepthsAndMasks(
            rgbs=np.concatenate([sample.rgbs for sample in rgbd_list], axis=DIM_IM_N),
            masks=np.concatenate([sample.masks for sample in rgbd_list], axis=DIM_IM_N),
            depths=np.concatenate(
                [sample.depths for sample in rgbd_list], axis=DIM_IM_N
            ),
        )
        target_obj_areas_on_screen = TwoDAreas(
            x_maxes_excluding=np.concatenate(
                [area.x_maxes_excluding for area in area_list]
            ),
            x_mins_including=np.concatenate(
                [area.x_mins_including for area in area_list]
            ),
            y_maxes_excluding=np.concatenate(
                [area.y_maxes_excluding for area in area_list]
            ),
            y_mins_including=np.concatenate(
                [area.y_mins_including for area in area_list]
            ),
        )

        self._select_viewpoint(initial_pos, initial_viewpt_type)

        return SceneSamples(
            rgbds=rgbds,
            target_obj_areas_on_screen=target_obj_areas_on_screen,
        )

    def temporary_target_transform(
        self, vector_field_spec: PointBasedVectorFieldSpec
    ) -> "_TemporaryTargetTransform":
        """
        Return a monitor that applies the transform on enter restores the pre-enter target object on exit.

        Parameters
        ----------
        vector_field_spec
            The vector field that should be used by the transformation. The control points of the vector field are in the local coordinate system of the mesh of the target object (but they are not relative).

        Raises
        ------
        ShowBaseDestroyedError
            If the internal Panda3d show base of the scene is already destroyed.
        """
        if self._show_base is None:
            raise self._show_base_destroyed_error()

        return _TemporaryTargetTransform(
            scene=self, vector_field_spec=vector_field_spec
        )

    def live_preview_then_quit(self) -> None:
        """
        Starts a live preview window that contains the transformed target with the current configuration of the scene, then quits the application at the end.

        Due to the constraints of Panda3d, this function destroys the internal show base of this scene before the creation of the live preview window.
        """
        self.destroy_showbase()

        preview_base = Panda3dShowBase(
            offscreen=False, win_size=self._rendering_resolution
        )
        preview_base.load_world_from_blender(self._scene_path)
        if self.__applied_transform is not None:
            transform_result = self._target_object_transform.transform_obj_new(
                vector_field_spec=self.__applied_transform
            )
            put_obj(
                path=preview_base.get_target_obj_mesh_path(),
                new_obj=transform_result.new_obj.node(),
            )

        LivePreviewController.attach(
            base=preview_base,
            cam_points_at=self._viewpoints.get_target_position(),
            initial_cam_pos=ThreeDPoint(4, 4, 4),
            viewpoint_split=self._viewpoint_split,
            max_depth=self.__depth_cap,
        )
        preview_base.run()
        sys.exit(0)

    def destroy_showbase(self) -> None:
        """
        Destroy the internal ShowBase of the scene. This is required, since Panda3d supports only one ShowBase instance at a time.

        You can safely call this function multiple times.
        """
        if self._show_base is not None:
            self._show_base.destroy()
        self._show_base = None

    def get_cam_proj_spec(self) -> CamProjSpec:
        """
        Get the projection of the camera of the scene.

        Raises
        ------
        ShowBaseDestroyedError
            If the internal Panda3d show base of the scene is already destroyed.
        """
        if self._show_base is None:
            raise self._show_base_destroyed_error()

        return get_cam_proj_spec_for_showbase(self._show_base)

    def _show_base_destroyed_error(self) -> "ShowBaseDestroyedError":
        """
        A new instance of `ShowBaseDestroyedError` with a default error message.
        """
        return ShowBaseDestroyedError(
            "The internal Panda3d show base of this scene is already destroyed."
        )


class _TemporaryTargetTransform:
    """
    A monitor that stores the target object at enter and restores it on exit.

    Parameters
    ----------
    show_base
        The show base in which the target object should be replaced.
    vector_field_spec
        The vector field that specifies the transform to apply.
    transform
        The transform operation to apply.
    bounds
        The bounds of the target object.
    target_obj
        The preexisting target object to restore after exit.
    set_target_n_bodies
        The function to set the number of separate bodies at the application and restoring of the transform.
    TODO update docs
    """

    def __init__(
        self, scene: Scene, vector_field_spec: PointBasedVectorFieldSpec
    ) -> None:
        self.__scene = scene
        self.__vector_field_spec = vector_field_spec

    def __enter__(self) -> None:
        self.__scene.set_target_transform(self.__vector_field_spec)

    def __exit__(self, *args, **kwargs) -> None:
        self.__scene.set_target_transform(None)


def calc_raw_loss_values_on_scene(
    scene: Scene,
    predictor: AsyncDepthPredictor,
    eval_on_test: bool,
    logging_freq_fn: LoggingFreqFunction,
    progress_logger: Callable[[str], None],
) -> dict[tuple[SampleType, RawLossFn], np.ndarray]:
    """
    Calculate all raw loss function values for the training and validation (and possibly the test) viewpoints of the specified scene.

    This function uses all viewpoints to calculate the values.

    Parameters
    ----------
    scene
        The scene on which the values should be calculated.
    predictor
        The depth predictor to evaluate.
    eval_on_test
        Do the evaluation on the test set too.
    logging_freq_fn
        The logging frequency for each viewpoint type.
    progress_logger
        The function to log the progress.

    Returns
    -------
    v
        The calculated losses for each viewpoint type. Format: ``Scalars::Float``
    """
    viewpt_type_texts: dict[SampleType, str] = {
        SampleType.Train: "training",
        SampleType.Val: "validation",
        SampleType.Test: "testing",
    }
    results: dict[tuple[SampleType, RawLossFn], np.ndarray] = dict()

    if eval_on_test:
        eval_sets = SampleType
    else:
        eval_sets = [SampleType.Train, SampleType.Val]

    for viewpt_type in eval_sets:
        n_viewpoints = scene.get_n_samples_for_type(viewpt_type)

        existing_losses = None

        viewpt_type_text = viewpt_type_texts[viewpt_type]
        for viewpoint_idx in range(n_viewpoints):
            if logging_freq_fn.needs_logging(viewpoint_idx):
                progress_logger(
                    f"Processing {viewpt_type_text} viewpoint {viewpoint_idx} of {n_viewpoints}"
                )

            sample = scene.get_sample(viewpoint_idx, viewpt_type)
            pred_depth = predict_aligned(
                predictor=predictor,
                depth_cap=scene.get_depth_cap(),
                images=sample.rgbds,
            )
            loss_values_for_viewpt = calculate_loss_values(
                pred_depths=pred_depth,
                gt=sample.rgbds.get_depths_with_masks(),
                loss_fns=set(RawLossFn),
                target_obj_areas=sample.target_obj_areas_on_screen,
            )
            if existing_losses is not None:
                existing_losses = concat_losses(existing_losses, loss_values_for_viewpt)
            else:
                existing_losses = loss_values_for_viewpt
        assert existing_losses is not None
        for loss_fn, loss_val in existing_losses.items():
            results[viewpt_type, loss_fn] = loss_val

    return results


def calc_aggr_delta_loss_dict_from_losses_on_scene(
    scene: Scene,
    initial_raw_losses: dict[tuple[SampleType, RawLossFn], np.ndarray],
    predictor: AsyncDepthPredictor,
    eval_on_test: bool,
    logging_freq_fn: LoggingFreqFunction,
    progress_logger: Callable[[str], None],
) -> dict[tuple[LossDerivationMethod, LossPrecision, SampleType, RawLossFn], float]:
    """
    Calculate all aggregated loss values for the training and validation (and possibly) viewpoints of the specified scene.

    This function uses all viewpoints to calculate the values.

    Parameters
    ----------
    scene
        The scene on which the values should be calculated.
    initial_raw_losses
        The loss values for the non-modified scene.
    predictor
        The depth predictor to evaluate.
    eval_on_test
        Do the evaluation on the test set too.
    logging_freq_fn
        The logging frequency for each viewpoint type.
    progress_logger
        The function to log the progress.

    Returns
    -------
    v
        The calculated aggregated loss values.
    """
    result: dict[
        tuple[LossDerivationMethod, LossPrecision, SampleType, RawLossFn], float
    ] = dict()
    new_raw_losses = calc_raw_loss_values_on_scene(
        scene=scene,
        logging_freq_fn=logging_freq_fn,
        predictor=predictor,
        progress_logger=progress_logger,
        eval_on_test=eval_on_test,
    )
    for viewpt_type in SampleType:
        orig_viewpt_losses = collect_raw_losses_with_viewpt_type(
            losses=initial_raw_losses, viewpt_type=viewpt_type
        )

        new_viewpt_losses = collect_raw_losses_with_viewpt_type(
            losses=new_raw_losses, viewpt_type=viewpt_type
        )

        aggr_delta_train_losses_dict = get_aggr_delta_loss_dict_from_losses(
            new_losses=new_viewpt_losses,
            orig_losses=orig_viewpt_losses,
            loss_precision=LossPrecision.Exact,
            viewpt_type=viewpt_type,
        )

        result = result | aggr_delta_train_losses_dict
    return result


class SceneConfig:
    """
    A class that provides a complete description about a specified scene.

    Raises
    ------
    ValueError
        If ``n_volume_sampling_steps_along_shortest_axis`` is non-positive.

        If ``depth_cap`` is non-positive.

        If ``target_size_multiplier`` is not greater than 1.
    """

    def __init__(
        self,
        world_path: Path,
        resolution: TwoDSize,
        viewpt_counts: DesiredViewpointCounts,
        object_transform_type: ObjectTransformType,
        n_volume_sampling_steps_along_shortest_axis: int,
        target_size_multiplier: float,
        depth_cap: float,
        applied_transform: "PointBasedVectorFieldSpec | None",
    ) -> None:
        if n_volume_sampling_steps_along_shortest_axis <= 0:
            raise ValueError(
                f'The value ({n_volume_sampling_steps_along_shortest_axis}) of parameter "n_volume_sampling_steps_along_shortest_axis" is non-positive.'
            )

        if depth_cap <= 0:
            raise ValueError(
                f'The value ({depth_cap}) of parameter "depth_cap" is non-positive.'
            )

        if target_size_multiplier <= 1:
            raise ValueError(
                f'The value ({target_size_multiplier}) of parameter "target_size_multiplier" is not greater than 1.'
            )

        self.__world_path = world_path
        self.__resolution = resolution
        self.__viewpt_counts = viewpt_counts
        self.__object_transform_type = object_transform_type
        self.__n_volume_sampling_steps_along_shortest_axis = (
            n_volume_sampling_steps_along_shortest_axis
        )
        self.__target_size_multiplier = target_size_multiplier
        self.__depth_cap = depth_cap
        self.__applied_transform = applied_transform

    @property
    def world_path(self) -> Path:
        """
        The path of the glb file of the world.
        """
        return self.__world_path

    @property
    def resolution(self) -> TwoDSize:
        """
        The resolution of the rendered images.
        """
        return self.__resolution

    @property
    def viewpt_counts(self) -> DesiredViewpointCounts:
        """
        The desired viewpoint counts for each sample type.
        """
        return self.__viewpt_counts

    @property
    def object_transform_type(self) -> ObjectTransformType:
        """
        The type of the transformation applied on the target object.
        """
        return self.__object_transform_type

    @property
    def n_volume_sampling_steps_along_shortest_axis(self) -> int:
        """
        If the volume of the object is transformed, then this parameter controls the number of sampling steps alongside the shortest axis. The shortest axis is the axis alongside which the transformed object in the relevant area the shortest is. Ignored if the mesh of the target object is directly transformed instead. This value is always positive.
        """
        return self.__n_volume_sampling_steps_along_shortest_axis

    @property
    def target_size_multiplier(self) -> float:
        """
        A value that controls the connection between the original bounding box of the target object and the area in which the transformed target object should stay (the relevant area of the transformation). This value is always greater than 1.
        """
        return self.__target_size_multiplier

    @property
    def depth_cap(self) -> float:
        """
        The maximal depth of the scene. The greater depth values are masked out. This emulates the maximal depth of some real depth sensors. This value is always positive.
        """
        return self.__depth_cap

    @property
    def applied_transform(self) -> PointBasedVectorFieldSpec | None:
        """
        If this value is not None, then this vector field controls the transformation applied on the scene.
        """
        return self.__applied_transform

    def save_json(self, target: Path | TextIO) -> None:
        """
        Save the specified scene config to the specified path using json.

        This function does not impse any restrictions on the extension of the file.
        """
        if isinstance(target, Path):
            with target.open(mode="w+") as f:
                self.save_json(f)
        else:
            transform_dict = None
            if self.applied_transform is not None:
                control_points_io = io.BytesIO()
                np.save(control_points_io, self.applied_transform.control_points)
                control_points_bytes = control_points_io.getvalue()
                control_points_str = base64.b64encode(control_points_bytes).decode(
                    "ascii"
                )
                vectors_io = io.BytesIO()
                np.save(vectors_io, self.applied_transform.vectors)
                vectors_str = base64.b64encode(vectors_io.getvalue()).decode("ascii")

                transform_dict = {
                    "control_points": control_points_str,
                    "vectors": vectors_str,
                }

            json.dump(
                obj={
                    "world_path": str(self.world_path),
                    "resulution": {
                        "x_size": self.resolution.x_size,
                        "y_size": self.resolution.y_size,
                    },
                    "viewpt_counts_train": self.viewpt_counts.n_train_samples,
                    "viewpt_counts_test": self.viewpt_counts.n_test_samples,
                    "viewpt_counts_val": self.viewpt_counts.n_val_samples,
                    "object_transform_type": self.object_transform_type.public_name,
                    "n_volume_sampling_steps_along_shortest_axis": self.n_volume_sampling_steps_along_shortest_axis,
                    "target_size_multiplier": self.target_size_multiplier,
                    "max_depth": self.depth_cap,
                    "applied_transform": transform_dict,
                },
                fp=target,
            )

    @staticmethod
    def from_json(source: Path | TextIO) -> "SceneConfig":
        """
        Load the scene config from the specified path.

        This function does not check the validity of the loaded json data.

        This function does not impse any restrictions on the extension of the file.
        """
        if isinstance(source, Path):
            with source.open(mode="r") as f:
                return SceneConfig.from_json(f)
        else:
            data = json.load(source)

            world_path = Path(data["world_path"])
            resolution = TwoDSize(
                x_size=data["resulution"]["x_size"],
                y_size=data["resulution"]["y_size"],
            )
            viewpt_counts = DesiredViewpointCounts(
                n_train_samples=data["viewpt_counts_train"],
                n_test_samples=data["viewpt_counts_test"],
                n_val_samples=data["viewpt_counts_val"],
            )
            object_transform_type = get_object_transform_type_by_name(
                data["object_transform_type"]
            )
            n_volume_sampling_steps_along_shortest_axis = data[
                "n_volume_sampling_steps_along_shortest_axis"
            ]
            target_size_multiplier = data["target_size_multiplier"]
            max_depth = data["max_depth"]

            if data["applied_transform"] is not None:
                control_points_str: str = data["applied_transform"]["control_points"]
                vectors_str: str = data["applied_transform"]["vectors"]

                control_points = np.load(
                    io.BytesIO(base64.decodebytes(control_points_str.encode("ascii")))
                )
                vectors = np.load(
                    io.BytesIO(base64.decodebytes(vectors_str.encode("ascii")))
                )
                applied_transform = PointBasedVectorFieldSpec(
                    control_points=control_points, vectors=vectors
                )
            else:
                applied_transform = None

            return SceneConfig(
                world_path=world_path,
                resolution=resolution,
                viewpt_counts=viewpt_counts,
                object_transform_type=object_transform_type,
                n_volume_sampling_steps_along_shortest_axis=n_volume_sampling_steps_along_shortest_axis,
                target_size_multiplier=target_size_multiplier,
                depth_cap=max_depth,
                applied_transform=applied_transform,
            )
    
    def without_transform(self) -> "SceneConfig":
        return SceneConfig(
            applied_transform=None,
            world_path=self.__world_path,
            depth_cap=self.__depth_cap,
            n_volume_sampling_steps_along_shortest_axis=self.__n_volume_sampling_steps_along_shortest_axis,
            object_transform_type=self.__object_transform_type,
            resolution=self.__resolution,
            target_size_multiplier=self.__target_size_multiplier,
            viewpt_counts=self.__viewpt_counts
        )


class SceneSamples(SamplesBase):
    def __init__(
        self, rgbds: RGBsWithDepthsAndMasks, target_obj_areas_on_screen: TwoDAreas
    ):
        super().__init__(rgbds)
        self.target_obj_areas_on_screen = target_obj_areas_on_screen


class ShowBaseDestroyedError(Exception):
    """
    An error that means that the internal show base of the scene is already destroyed.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        super().__init__(*args)


class InvalidSceneError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


if TYPE_CHECKING:
    v: DatasetLike[SceneSamples] = type_instance(Scene)
