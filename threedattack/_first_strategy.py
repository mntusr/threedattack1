from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, NamedTuple, TypeGuard, cast

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from ._attack_commons import (
    AttackStrategy,
    CMAEvolutionStrategyArgs,
    ExactMetrics,
    GenerationEvaluationResult,
    SolutionInfo,
)
from ._attack_primitives import (
    BestSolutionKeeper,
    FreezableIndexSelector,
    ParallelRunner,
)
from ._logging import LoggingFreqFunction
from ._scene import (
    Scene,
    SceneSamples,
    calc_aggr_delta_loss_dict_from_losses_on_scene,
    calc_raw_loss_values_on_scene,
)
from ._typing import type_instance
from .dataset_model import DepthsWithMasks, RGBsWithDepthsAndMasks, SampleType
from .losses import (
    LossPrecision,
    RawLossFn,
    calculate_loss_values,
    collect_raw_losses_with_viewpt_type,
    concat_losses,
    derived_loss_dict_2_str_float_dict,
    get_aggr_delta_loss_dict_from_losses,
    get_aggr_delta_loss_dict_from_preds,
    get_derived_loss_by_name,
    get_derived_loss_name,
    idx_losses,
    sign_loss_to_make_smaller_mean_worse_predictor,
)
from .rendering_core import (
    ObjectTransformType,
    PointBasedVectorFieldSpec,
    ScaledStandingAreas,
    TwoDAreas,
)
from .target_model import (
    AlignmentFunction,
    AsyncDepthPredictor,
    DepthFuture,
    predict_aligned,
)
from .tensor_types.npy import *


class FirstStrategySharedInfo(NamedTuple):
    initial_losses: dict[tuple[SampleType, RawLossFn], np.ndarray]
    viewpt_idx_generator: FreezableIndexSelector
    transform_change_amount_score_penality: "TransformChangeAmountScorePenality"


class FirstStrategy:
    """
    A first implemented adversarial attack strategy.

    This strategy uses interpolated vector fields to create the adversarial example from the original example.

    Parameters
    ----------
    n_control_points
        The number of points to control the vector field.
    n_estim_viewpts
        The number of viewpoints to estimate the metrics for each sample.
    freeze_estim
        Values: "free", the set of the viewpoints to estimate the metrics for each sample is not frozen; "frozen_random", the set of the viewpoints to estimate the metrics for each sample is frozen, but randomly selected; list[int], the set of the viewpoints to estimate the metrics for each sample is manually specified.
    sigma0
        The sigma0 property of the CMA algorithm.
    max_pos_change_per_coord
        The maximal relative values of the coordinates of the vectors that control the interpolated vector field. The actual maximal change = ``max_pos_change_per_coord*min_size``, where ``min_size = the maximum of the sizes of the target object alongside the X, Y and Z axes``
    max_transform_change_amount_score
        A soft maximum for the change amount score of the transformation. This argument is ignored if the scene is configured to transform the mesh directly.
    """

    def __init__(
        self,
        n_control_points: int,
        n_estim_viewpts: int,
        freeze_estim: object,
        sigma0: float,
        optimized_metric: str,
        max_pos_change_per_coord: float,
        max_transform_change_amount_score: float | None,
    ) -> None:
        if n_estim_viewpts < 1:
            raise ValueError(
                f"The number of viewpoints during expected value estimation should be at least 1. Current value: {n_estim_viewpts}"
            )
        if n_control_points <= 1:
            raise ValueError(
                f"The number of viewpoints to control the vector field should be greater than 1. Current value: {n_control_points}"
            )
        if sigma0 <= 0:
            raise ValueError(f"Argument sigma0 should be positive instead of {sigma0}")
        self.optimized_metric = get_derived_loss_by_name(optimized_metric)
        if max_pos_change_per_coord <= 0:
            raise ValueError(
                f"The maximal relative position change per coordinate should be positive. Current value: {max_pos_change_per_coord}"
            )

        self.n_control_points = n_control_points
        self.n_estim_viewpts = n_estim_viewpts
        self.sigma0 = sigma0
        self.max_pos_change_per_coord = max_pos_change_per_coord
        self.max_transform_change_amount_score = max_transform_change_amount_score

        if _is_int_list(freeze_estim):
            self.is_frozen = True
            self.estim_viewpt_idx_list = freeze_estim
        else:
            match freeze_estim:
                case "frozen_random":
                    self.is_frozen = True
                    self.estim_viewpt_idx_list = None
                case "free":
                    self.is_frozen = False
                    self.estim_viewpt_idx_list = None
                case _:
                    raise ValueError(
                        f'Unknown value for "freeze_estim" ({freeze_estim}). Supported values: "frozen_raondom", "free", the list of the viewpoint indices.'
                    )

    def apply_solution_transform(self, x: np.ndarray, scene: Scene) -> None:
        vector_field_spec = self._get_vector_field_spec_from_x(
            x=x, target_area=scene.get_target_areas()
        )
        scene.set_target_transform(vector_field_spec)

    def calc_shared_info_for_scene(
        self,
        scene: Scene,
        predictor: AsyncDepthPredictor,
        eval_on_test: bool,
        progress_logger: Callable[[str], None],
        logging_freq_fn: LoggingFreqFunction,
        seed: int,
    ) -> FirstStrategySharedInfo:
        initial_losses = calc_raw_loss_values_on_scene(
            scene=scene,
            logging_freq_fn=logging_freq_fn,
            predictor=predictor,
            progress_logger=progress_logger,
            eval_on_test=eval_on_test,
        )

        viewpt_idx_generator = FreezableIndexSelector(
            n_values=scene.get_n_samples_for_type(SampleType.Train),
            is_frozen=self.is_frozen,
            n_indices=self.n_estim_viewpts,
            seed_if_frozen_random=seed,
            manual_indices=self.estim_viewpt_idx_list,
        )

        transform_change_amount_score_penality = TransformChangeAmountScorePenality(
            threshold=self.max_transform_change_amount_score,
            transform_type=scene.get_transform_type(),
        )

        return FirstStrategySharedInfo(
            initial_losses=initial_losses,
            viewpt_idx_generator=viewpt_idx_generator,
            transform_change_amount_score_penality=transform_change_amount_score_penality,
        )

    def _get_all_target_areas_on_screen(
        self, scene: Scene
    ) -> dict[SampleType, TwoDAreas]:
        areas_dict: dict[SampleType, TwoDAreas] = dict()
        for viewpt_type in [SampleType.Train, SampleType.Val]:
            scene.get_samples(slice(None), viewpt_type)
        return areas_dict

    def evaluate_generation(
        self,
        xes: list[np.ndarray],
        scene: Scene,
        shared_info_for_scene: FirstStrategySharedInfo,
        predictor: AsyncDepthPredictor,
        generation_seed: int,
    ) -> GenerationEvaluationResult:
        # optimizations to reduce memory usage:
        #
        # * the algorithm keeps only the first 5 rendered images after the start of the unaligned depth predictions
        # * the algorithm keeps only the first 5 gt depth maps and masks after the end of the fitness calculations
        # * the algorithm keeps only the first 5 aligned depth maps after the end of the fitness calculations

        viewpoint_indices = shared_info_for_scene.viewpt_idx_generator.generate_unique(
            seed_if_not_frozen=generation_seed
        )
        kept_slice = slice(0, 5)
        solution_infos: list[SolutionInfo] = []
        cam_proj_spec = scene.get_cam_proj_spec()
        best_solution_extra_keeper = BestSolutionKeeper[_BestSolutionExtra]()

        def render_fn(
            idx: int,
        ) -> tuple[SceneSamples, _SharedInfoFromRender]:
            samples, shared_info_from_render = self._render_solution(
                x=xes[idx],
                scene=scene,
                viewpoint_indices=viewpoint_indices,
                solution_idx=idx,
            )
            return samples, shared_info_from_render

        def prediction_fn_async(
            render_result: SceneSamples,
            _: _SharedInfoFromRender,
        ) -> _AsyncPredResult:
            depth_future = predictor.predict_native(render_result.rgbds.rgbs)
            return _AsyncPredResult(
                gt_full=render_result.rgbds.get_depths_with_masks(),
                rgbs_reduced=idx_im_rgbs(render_result.rgbds.rgbs, n=kept_slice),
                target_areas_full=render_result.target_obj_areas_on_screen,
                depth_future=depth_future,
            )

        def acquire_pred_future_fn(
            async_pred_result: _AsyncPredResult, _: _SharedInfoFromRender
        ) -> _GotPredResult:
            got_native_preds_full = async_pred_result.depth_future.get()
            return _GotPredResult(
                gt_full=async_pred_result.gt_full,
                rgbs_reduced=async_pred_result.rgbs_reduced,
                target_areas_full=async_pred_result.target_areas_full,
                got_native_preds_full=got_native_preds_full,
            )

        def eval_sink_fn(
            got_pred_result: _GotPredResult,
            shared_info_from_render: _SharedInfoFromRender,
        ) -> None:
            (
                aligned_depth_preds,
                str_delta_losses,
                fitness_val,
            ) = self._get_delta_losses_and_fitness_val_for_unaligned_preds_estim_train(
                alignment_function=predictor.alignment_function,
                gt=got_pred_result.gt_full,
                native_preds=got_pred_result.got_native_preds_full,
                scene=scene,
                shared_info=shared_info_for_scene,
                target_obj_areas_on_screen=got_pred_result.target_areas_full,
                viewpoint_indices=viewpoint_indices,
                target_n_bodies=shared_info_from_render.target_n_bodies,
                transform_change_amount_score_penality=shared_info_for_scene.transform_change_amount_score_penality,
                transform_change_amount_score=shared_info_from_render.transform_change_amount_score,
            )

            solution_infos.append(
                SolutionInfo(
                    cam_proj_spec=cam_proj_spec,
                    fitness=fitness_val,
                    metrics=str_delta_losses
                    | {"target_n_bodies": shared_info_from_render.target_n_bodies}
                    | {
                        "target_change_amount_score": shared_info_from_render.transform_change_amount_score
                    },
                    pred_depths_aligned=None,
                    rgbds=None,
                    x=xes[shared_info_from_render.solution_idx],
                    change_amount_score=shared_info_from_render.transform_change_amount_score,
                )
            )
            gts_reduced = got_pred_result.gt_full[kept_slice]
            best_solution_extra_keeper.update(
                fitness=fitness_val,
                gt_data=_BestSolutionExtra(
                    rgbds_reduced=RGBsWithDepthsAndMasks(
                        rgbs=got_pred_result.rgbs_reduced,
                        depths=gts_reduced.depths,
                        masks=gts_reduced.masks,
                    ),
                    aligned_depth_preds_reduced=idx_im_depthmaps(
                        aligned_depth_preds, n=kept_slice
                    ),
                    idx=shared_info_from_render.solution_idx,
                ),
            )

        parallel_runner = ParallelRunner(
            render_fn=render_fn,
            prediction_fn_async=prediction_fn_async,
            acquire_pred_future_fn=acquire_pred_future_fn,
            eval_sink_fn=eval_sink_fn,
        )

        parallel_runner.run(len(xes))

        found_best_solution_extra = best_solution_extra_keeper.get_best()
        solution_infos[
            found_best_solution_extra.idx
        ].rgbds = found_best_solution_extra.rgbds_reduced
        solution_infos[
            found_best_solution_extra.idx
        ].pred_depths_aligned = found_best_solution_extra.aligned_depth_preds_reduced

        return GenerationEvaluationResult(
            solution_infos=solution_infos,
            total_pred_calls=self._get_n_pred_calls_for_generation(xes),
        )

    def _get_n_pred_calls_for_generation(self, xes: list[np.ndarray]) -> int:
        return self.n_estim_viewpts * len(xes)

    def _render_solution(
        self,
        x: np.ndarray,
        solution_idx: int,
        scene: Scene,
        viewpoint_indices: np.ndarray,
    ) -> tuple[SceneSamples, "_SharedInfoFromRender"]:
        """
        Render the RGBD images for the specified viewpoints of the specified scene and solution and get the areas of the target object on the screen.

        Parameters
        ----------
        x
            The CMA solution to use. Format: ``Scalars::*``
        scene
            The scene to use.
        viewpoint_indices
            The indexes of the viewpoints to render. Format: ``Scalars::Int``

        Returns
        -------
        rgbds
            The rendered RGBD images with masks.
        areas
            The areas of the bounding rectangles around the target object on the screen.
        target_n_bodies
            The number of separate bodies in the target object.
        """
        vector_field_spec = self._get_vector_field_spec_from_x(
            x=x, target_area=scene.get_target_areas()
        )

        viewpt_idx_list = [int(viewpt_idx) for viewpt_idx in viewpoint_indices]
        with scene.temporary_target_transform(vector_field_spec):
            samples = scene.get_samples(viewpt_idx_list, SampleType.Train)
            target_n_bodies = scene.get_target_n_bodies()
            transform_change_amount_score = scene.get_transform_change_amount_score()
        return (
            samples,
            _SharedInfoFromRender(
                solution_idx=solution_idx,
                target_n_bodies=target_n_bodies,
                transform_change_amount_score=transform_change_amount_score,
            ),
        )

    def _get_delta_losses_and_fitness_val_for_unaligned_preds_estim_train(
        self,
        native_preds: np.ndarray,
        shared_info: FirstStrategySharedInfo,
        alignment_function: AlignmentFunction,
        gt: DepthsWithMasks,
        scene: Scene,
        viewpoint_indices: np.ndarray,
        target_obj_areas_on_screen: TwoDAreas,
        target_n_bodies: int,
        transform_change_amount_score_penality: "TransformChangeAmountScorePenality",
        transform_change_amount_score: float,
    ) -> tuple[np.ndarray, dict[str, float], float,]:
        """
        Get the delta losses and fitness values for the unaligned perdictions.

        This function assumes that the calculated losses are estimations and they are calculated on training viewpoints.

        Parameters
        ----------
        native_preds
            The native predictions. Format: ``ArbSamples::*``
        shared_info
            The shared information for this particular attack.
        alignment_function
            The alignment function of the target model.
        gt
            The ground truth data.
        scene
            The scene in which the RGBD images were rendered.
        viewpoint_indices
            The indices of the rendered viewpoints. ``Scalars::Int``
        transform_change_amount_score_penality
            The function that describes the penality for the "too big" change amount score.
        transform_change_amount_score
            The change amount score.

        Returns
        -------
        aligned_depth_preds
            The aligned depth predictions. Format: ``Im::DepthMaps``
        str_delta_losses
            The aggregated delta losses.
        fitness_val
            The value of the fitness function.
        """
        train_losses = collect_raw_losses_with_viewpt_type(
            shared_info.initial_losses, SampleType.Train
        )
        aligned_depth_preds: np.ndarray = alignment_function(
            native_preds=native_preds,
            gt_depths=gt.depths,
            masks=gt.masks,
            depth_cap=scene.get_depth_cap(),
        )

        aggr_delta_losses_dict = get_aggr_delta_loss_dict_from_preds(
            aligned_depth_preds=aligned_depth_preds,
            gt=gt,
            loss_precision=LossPrecision.PossiblyEstim,
            orig_losses=idx_losses(train_losses, viewpoint_indices),
            viewpt_type=SampleType.Train,
            loss_fns={self.optimized_metric[1]},
            target_obj_areas=target_obj_areas_on_screen,
        )

        fitness_val = sign_loss_to_make_smaller_mean_worse_predictor(
            self.optimized_metric[1],
            aggr_delta_losses_dict[
                self.optimized_metric[0],
                LossPrecision.PossiblyEstim,
                SampleType.Train,
                self.optimized_metric[1],
            ],
        )
        fitness_val += get_target_n_bodies_penality(target_n_bodies)
        fitness_val += transform_change_amount_score_penality.get_penality(
            transform_change_amount_score
        )

        str_delta_losses = derived_loss_dict_2_str_float_dict(aggr_delta_losses_dict)

        return aligned_depth_preds, str_delta_losses, fitness_val

    def get_cma_config(self) -> CMAEvolutionStrategyArgs:
        """
        The CMA configuration of this strategy:

        * Lower bound: 0
        * Upper bound: 1
        * sigma0: ``self.sigma0``
        * ``x0 = full(0.5, self.n_control_points * 3 * 2)``
        * ``tolfunhist = 1e-3``
        """
        options = {"bounds": [[0], [1]], "tolfunhist": 1e-3}

        return {
            "x0": [0.5] * self.n_control_points * 3 * 2,
            "sigma0": self.sigma0,
            "inopts": options,
        }

    def _get_vector_field_spec_from_x(
        self, x: np.ndarray, target_area: ScaledStandingAreas
    ) -> PointBasedVectorFieldSpec:
        """
        Calculate the vector field specification for the specified sample.

        Algorithm pseudocode:

        1. ``rel_points:= the first n_points values``
        2. ``rel_vectors:= (the second n_points values .- 0.5)*2``
        3. ``absolute_points := make_rel_points_absolute(rel_points, full_scaled_standing_area_of_the_target_object) ``
        4. ``absolute_vectors := rel_vectors*min_size_of_the_original_target_object*self.max_pos_change_per_coord``
        5. ``return vector_field(absolute_points, absolute_vectors)``

        Parameters
        ----------
        x
            The sample from which the vector field should be calculated. Format: ``SVals::Float``
        target_area
            The bounding boxes of the target object.

        Returns
        -------
        v
            The vector field.

        Raises
        ------
        ValueError
            If the number of elements in the solution is not divisible by 6.

            If the number of elements in the solution is not greater than 0.
        """
        if len(x) == 0:
            raise ValueError("The solution does not contain any element.")
        if len(x) % 6 != 0:
            raise ValueError("The number of elements in x should be divisible by 6.")

        rel_points = x[: len(x) // 2].reshape((-1, 3))
        rel_vectors = (x[len(x) // 2 :].reshape((-1, 3)) - 0.5) * 2

        min_size = target_area.get_original_area(origin_of_obj=None).get_min_size()

        exact_points = target_area.get_full_area(
            origin_of_obj=None, include_outside_offset=False
        ).make_rel_points_absolute(rel_points=rel_points)
        exact_value_points = rel_vectors * min_size * self.max_pos_change_per_coord

        return PointBasedVectorFieldSpec(
            control_points=exact_points, vectors=exact_value_points
        )

    def get_strat_repr(self) -> str:
        return f'FirstStrategy(n_points={self.n_control_points}, sigma0={self.sigma0}, optimized_metric="{get_derived_loss_name(*self.optimized_metric)}")'

    def get_metrics_exact(
        self,
        x: np.ndarray,
        scene: Scene,
        predictor: AsyncDepthPredictor,
        eval_on_test: bool,
        shared_info_for_scene: FirstStrategySharedInfo,
        progress_logger: Callable[[str], None],
        logging_freq_fn: LoggingFreqFunction,
    ) -> ExactMetrics:
        with scene.temporary_target_transform(
            self._get_vector_field_spec_from_x(x, scene.get_target_areas())
        ):
            derived_loss_dict = calc_aggr_delta_loss_dict_from_losses_on_scene(
                scene=scene,
                predictor=predictor,
                initial_raw_losses=shared_info_for_scene.initial_losses,
                logging_freq_fn=logging_freq_fn,
                progress_logger=progress_logger,
                eval_on_test=eval_on_test,
            )

        minimizably_signed_versions = {
            key: sign_loss_to_make_smaller_mean_worse_predictor(key[-1], value)
            for key, value in derived_loss_dict.items()
        }

        return ExactMetrics(
            actual_metrics=derived_loss_dict_2_str_float_dict(derived_loss_dict),
            minimizably_signed_metrics=derived_loss_dict_2_str_float_dict(
                minimizably_signed_versions
            ),
        )


def get_target_n_bodies_penality(target_n_bodies: int) -> float:
    """
    Returns
    -------
    v
        ``(target_n_bodies-1)*50``

    Raises
    ------
    ValueError
        If the number of bodies in the target object is smaller than 1.
    """
    if target_n_bodies <= 0:
        raise ValueError(
            f"The number of bodies should be greater than 0. Current value: {target_n_bodies}"
        )
    return (target_n_bodies - 1) * 50


class TransformChangeAmountScorePenality:
    """
    A function that penalizes the "too big" change amount score if the volume of the target object is transformed.

    Parameters
    ----------
    transform_type
        The type of the transformation of the target object.
    threshold
        The threshold above which the change amount score should be penalized if the volume of the target object is transformed and not its mesh directly.
    """

    def __init__(
        self, transform_type: ObjectTransformType, threshold: float | None
    ) -> None:
        data_candidate = (transform_type, threshold)
        self.__data: tuple[
            Literal[ObjectTransformType.MeshBased], float | None
        ] | tuple[Literal[ObjectTransformType.VolumeBased], float]

        match data_candidate:
            case ObjectTransformType.MeshBased, _:
                self.__data = data_candidate
            case ObjectTransformType.VolumeBased, None:
                raise ValueError(
                    "The object transform is volume transform, but the threshold is None."
                )
            case ObjectTransformType.VolumeBased, _:
                self.__data = data_candidate

        if (transform_type == ObjectTransformType.VolumeBased) and (threshold is None):
            raise ValueError(
                "The object transform is volume transform, but the threshold is None."
            )

    def get_transform_type(self) -> ObjectTransformType:
        """
        Get the specified type of the transformation of the target object.
        """
        return self.__data[0]

    def get_threshold(self) -> float | None:
        """
        Get the threshold of above which the the change amount score should be penalized if the volume of the target object is transformed.

        This function always returns with the specified threshold, regardless of the specified transformation type.
        """
        return self.__data[1]

    def get_penality(self, score: float) -> float:
        """
        The penality function. Calculation:

        * Constant 0 if the mesh of the target object is transformed.
        * ``relu(score - threshold)*(50 + (score - threshold) * 30)`` if the (change amount) ``score`` is greater than the threshold and the volume of the target object is transformed.
        """
        if self.__data[0] == ObjectTransformType.MeshBased:
            return 0
        else:
            threshold = self.__data[1]
            if score < threshold:
                return 0
            else:
                return 50 + (score - threshold) * 30


@dataclass
class _SharedInfoFromRender:
    solution_idx: int
    """The index of described solution."""
    target_n_bodies: int
    """The number of bodies in case of the described solution."""
    transform_change_amount_score: float
    """The change amount score of the described solution."""


@dataclass
class _AsyncPredResult:
    rgbs_reduced: np.ndarray
    gt_full: DepthsWithMasks
    target_areas_full: TwoDAreas
    depth_future: DepthFuture


@dataclass
class _GotPredResult:
    rgbs_reduced: np.ndarray
    gt_full: DepthsWithMasks
    target_areas_full: TwoDAreas
    got_native_preds_full: np.ndarray


@dataclass
class _BestSolutionExtra:
    rgbds_reduced: RGBsWithDepthsAndMasks
    aligned_depth_preds_reduced: np.ndarray
    idx: int


def _is_int_list(obj: object) -> TypeGuard[list[int]]:
    if not isinstance(obj, list):
        return False

    return all(isinstance(item, int) for item in obj)


# enforce protocol implementation
if TYPE_CHECKING:
    v: AttackStrategy = type_instance(FirstStrategy)
