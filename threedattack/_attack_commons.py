import copy
import datetime
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    NamedTuple,
    Protocol,
    TypedDict,
    TypeVar,
    overload,
)

import cma
import numpy as np

from ._attack_primitives import BestSolutionKeeper, estimate_remaining_time
from ._logging import LoggingFreqFunction
from ._scene import Scene
from .dataset_model import CamProjSpec, RGBsWithDepthsAndMasks, SampleType
from .losses import (
    LossDerivationMethod,
    LossPrecision,
    RawLossFn,
    derived_loss_dict_2_str_float_dict,
    get_derived_loss_name,
)
from .mlflow_util import CustomMlflowClient, RunConfig, RunId
from .target_model import AsyncDepthPredictor
from .tensor_types.idx import *
from .tensor_types.npy import *

T = TypeVar("T")


class AttackStrategy(Protocol[T]):
    """
    This protocol specifies the main logic of the adversarial attacks.

    It does two things:

    * Evaluates a particular adversarial example solution.
    * Applies a particular adversarial example solution.

    It does not implement the actual searching of the adversarial example solutions, however it assumes the CMA evolution strategy implemented by `cma.CMAEvolutionStrategy`.
    """

    def calc_shared_info_for_scene(
        self,
        scene: Scene,
        predictor: AsyncDepthPredictor,
        eval_on_test: bool,
        progress_logger: Callable[[str], None],
        logging_freq_fn: LoggingFreqFunction,
        seed: int,
    ) -> T:
        """
        Calculate the constant shared info that is reused during the whole attack.

        Parameters
        ----------
        scene
            The scene on which the attack occurs.
        predictor
            The attacked depth prediction model.
        eval_on_test
            Do the evaluation on the test set too. If false, then the function should not do any evaluation on the test set.
        progress_logger
            An extra function to log the progress of the shared info calculation. Its argument is the message.
        logging_freq_fn
            The logging frequency function for a single "long calculation". Generally, the whole shared info calculation might be treated as a single "long operation", but this is not necessary.
        seed
            The seed for the non-deterministic operations.

        Returns
        -------
        v
            The shared constant information.
        """
        ...

    def evaluate_generation(
        self,
        xes: list[np.ndarray],
        scene: Scene,
        shared_info_for_scene: T,
        predictor: AsyncDepthPredictor,
        generation_seed: int,
    ) -> "GenerationEvaluationResult":
        """
        Evaluate a particular adversarial attack solution.

        The best solution also contains RGBD images and ground truth depth maps. The other solutions may or may not contain such information.

        Parameters
        ----------
        xes
            The adversarial attack solutions. Format of the elements: ``SVals::Float``
        scene
            The scene to manipulate. This function might modify the state of the scene, however it restores the original state until the function returns.
        shared_info_for_scene
            The constant shared information of this attack.
        predictor
            The target model.
        n_poses
            The number of random camera positions to evaluate the solution.
        iter_idx
            The index of the generation. This is generally not useful for a production-strategy, but unittests can use it to test, whether the correct seed is passed.
        generation_seed
            A generation-dependent seed to control the stochastic operations during the evaluation.

        Returns
        -------
        v
            The results of the generation evaluation.
        """
        ...

    def get_cma_config(self) -> "CMAEvolutionStrategyArgs":
        """Get the initial configuration for the `cma.CMAEvolutionStrategyArgs` to search for solutions.

        This might configure any option of the CMA algorithm except the maximum number of iterations.
        """
        ...

    def get_metrics_exact(
        self,
        x: np.ndarray,
        scene: Scene,
        predictor: AsyncDepthPredictor,
        eval_on_test: bool,
        shared_info_for_scene: T,
        progress_logger: Callable[[str], None],
        logging_freq_fn: LoggingFreqFunction,
    ) -> "ExactMetrics":
        """
        Get the exact metrics for the specified scene and predictor with the specified solution.

        Parameters
        ----------
        x
            The solution to evaluate.
        scene
            The scene to use.
        predictor
            The depth predictor.
        eval_on_test
            Do the evaluation on the test set too. If false, then the function should not do any evaluation on the test set.
        shared_info_for_scene
            The constant shared information of this attack.
        progress_logger
            An extra function to log the progress of the exact metric calculation. Its argument is the message.
        logging_freq_fn
            The logging frequency function for a single "long calculation". Generally, the whole metric calculation might be treated as a single "long operation", but this is not necessary.


        Returns
        -------
        v
            The exact metrics. The names of these metrics is different from the names of the metrics returned in `AttackStrategy.get_fitness`.
        """
        ...

    def get_strat_repr(self) -> str:
        ...

    def apply_solution_transform(self, x: np.ndarray, scene: Scene) -> None:
        """
        Apply the transform specified by the given solution on the scene.

        Parameters
        ----------
        x
            The solution to apply.
        scene
            The scene on which the solution should be applied.

        Raises
        ------
        RuntimeError
            If a transformation is already applied on the scene.
        """
        ...


class CMAEvolutionStrategyArgs(TypedDict):
    """
    The arguments of the constructor of `cma.CMAEvolutionStrategy`.
    """

    x0: list[float] | np.ndarray
    sigma0: float
    inopts: dict[str, Any]


class AdvAttackLearner(Generic[T]):
    """
    The main class of adversarial attack learning.

    It uses CMA-ES to find the adversarial example.

    Parameters
    ----------
    attack_strategy
        This object creates the adversarial examples from the individual solutions and evaluates the adversarial examples too.
    n_poses
        The number of random camera positions to evaluate an adversarial attack.
    mlflow_client
        The mlflow client to use for logging.
    maxiter
        The maximum number of generations for each attack.
    log_im
        Log the RGB images and depth maps during attack.
    eval_on_test
        Do the evaluation on the test split too.
    """

    def __init__(
        self,
        attack_strategy: AttackStrategy,
        mlflow_client: CustomMlflowClient,
        maxiter: int,
        log_im: bool,
        eval_on_test: bool,
    ) -> None:
        self.attack_strategy = attack_strategy
        self.mlflow_client = mlflow_client
        self.maxiter = maxiter
        self.log_im = log_im
        self.eval_on_test = eval_on_test

    def learn_adv_sample(
        self,
        scene: Scene,
        predictor: AsyncDepthPredictor,
        console_logging_freq: LoggingFreqFunction,
        shared_info_calc_logging_freq: LoggingFreqFunction,
        run_config: RunConfig,
    ) -> "AdvSampleLearningResult":
        """
        Learn an adversarial example.

        The adversarial example learning consist of three phases. These are:

        1. Calculate some constants that may be reused during the attack learning.
        2. Do the actual attack learning.
        3. Configure the scene to show the final attack.

        Parameters
        ----------
        scene
            The scene to manipulate.
        predictor
            The target model.
        console_logging_freq
            The frequency of logging to the console at the different generations.
        shared_info_calc_logging_freq
            The logging frequency during the calculation of the shared constants.
        run_config
            The configuration of the Mlflow run corresponding to this attack learning.

        Returns
        -------
        v
            The results of the optimization.
        """
        print(f"Creating mlflow run")
        run_id = self.mlflow_client.create_run(run_config=run_config)
        run_name = self.mlflow_client.get_run_name(run_id)
        print(f'Run name: "{run_name}"')

        best_solution_keeper = BestSolutionKeeper[SolutionInfo]()

        strategy = cma.CMAEvolutionStrategy(**self._get_cma_config())
        rng = np.random.default_rng()

        print("Computing shared info for scene")
        shared_info = self.attack_strategy.calc_shared_info_for_scene(
            scene=scene,
            predictor=predictor,
            logging_freq_fn=shared_info_calc_logging_freq,
            progress_logger=lambda text: print("> " + text),
            seed=rng.integers(low=0, high=10000),
            eval_on_test=self.eval_on_test,
        )

        print("Attack started")

        start_time = None

        generation_idx = 0
        n_pred_calls_total = 0
        strategy_stop = False
        while not strategy_stop:
            if start_time is None:
                start_time = datetime.datetime.now()

            generation_seed = int(rng.integers(low=0, high=10000))
            solution_info, n_pred_calls_for_generation = self._update_strategy(
                strategy=strategy,
                predictor=predictor,
                scene=scene,
                shared_info_for_scene=shared_info,
                generation_seed=generation_seed,
            )
            best_solution_keeper.update(solution_info.fitness, solution_info)
            iter_end_time = datetime.datetime.now()

            strategy_stop = strategy.stop()

            self._log_best(
                best_solution_keeper=best_solution_keeper,
                console_logging_freq=console_logging_freq,
                generation_idx=generation_idx,
                run_id=run_id,
                attack_start_time=start_time,
                current_generation_end_time=iter_end_time,
                maxiter=self.maxiter,
                is_last_iter=strategy_stop,
                scene=scene,
            )
            generation_idx += 1
            n_pred_calls_total += n_pred_calls_for_generation

        self.mlflow_client.set_terminated(run_id=run_id)

        exact_metrics = self.attack_strategy.get_metrics_exact(
            x=strategy.result.xbest,
            scene=scene,
            shared_info_for_scene=shared_info,
            logging_freq_fn=shared_info_calc_logging_freq,
            progress_logger=lambda text: print("> " + text),
            predictor=predictor,
            eval_on_test=self.eval_on_test,
        )

        self.mlflow_client.log_metrics(
            run_id=run_id,
            generation_idx=None,
            metrics=exact_metrics.actual_metrics,
        )

        final_best_solution = best_solution_keeper.get_best()
        return AdvSampleLearningResult(
            n_pred_calls_total=n_pred_calls_total,
            run_name=run_name,
            final_change_amount_score=final_best_solution.change_amount_score,
            best_solution_exact_origsigned_metrics=exact_metrics.actual_metrics,
            best_solution_exact_minimizably_signed_metrics=exact_metrics.minimizably_signed_metrics,
        )

    def _get_cma_config(self) -> CMAEvolutionStrategyArgs:
        cma_config = copy.deepcopy(self.attack_strategy.get_cma_config())
        cma_config["inopts"]["maxiter"] = self.maxiter
        return cma_config

    def _update_strategy(
        self,
        strategy: cma.CMAEvolutionStrategy,
        scene: Scene,
        shared_info_for_scene: T,
        predictor: AsyncDepthPredictor,
        generation_seed: int,
    ) -> "tuple[SolutionInfo, int]":
        """
        Do a single generation update on the specified `cma.CMAEvolutionStrategy`.

        The function might modify the scene, however all modifications are restored until the function returns.

        Parameters
        ----------
        strategy
            The strategy to update.
        scene
            The manipulated scene.
        shared_info_for_scene
            The already calculated strategy-specific shared constants.
        predictor
            The target model.
        generation_idx
            The zero-based index of the current generation.
        generation_seed
            The seed for the stochastic operations of the strategy in the current generation.

        Returns
        -------
        v
            The generation-dependent seed for the stochastic operations during the evaluation of the generation.
        total_pred_calls
            The total number of prediction calls to evaluate this solution.
        """
        solutions = strategy.ask()

        gen_eval_result = self.attack_strategy.evaluate_generation(
            xes=solutions,
            predictor=predictor,
            generation_seed=generation_seed,
            scene=scene,
            shared_info_for_scene=shared_info_for_scene,
        )
        fitnesses_for_solutions = [
            solution_info.fitness for solution_info in gen_eval_result.solution_infos
        ]

        best_sample_idx = np.argmin(fitnesses_for_solutions)

        strategy.tell(solutions, fitnesses_for_solutions)
        return (
            gen_eval_result.solution_infos[best_sample_idx],
            gen_eval_result.total_pred_calls,
        )

    def _log_best(
        self,
        generation_idx: int,
        console_logging_freq: LoggingFreqFunction,
        best_solution_keeper: "BestSolutionKeeper[SolutionInfo]",
        attack_start_time: datetime.datetime,
        current_generation_end_time: datetime.datetime,
        maxiter: int,
        is_last_iter: bool,
        run_id: RunId,
        scene: Scene,
    ) -> None:
        """
        Log various pieces of information about the best solution.

        The function always logs the metrics. It logs the depth maps and rendered images when it prints to the console. It logs the best solution itself at the final iteration.

        The best solution is stored in an npz file artifact, called "best_solution.npz". It has a single key, called "x". The corresponding value contains the best solution itself.

        Parameters
        ----------
        generation_idx
            The zero-based index of the current generation.
        console_logging_freq
            The frequency function of logging to the console.
        best_solution
            The object that keeps track of the best sample.
        attack_start_time
            The time of the start of the first generation.
        current_generation_end_time
            The time of the end of the current generation.
        is_last_iter
            True if this is the last iteration in this attack.
        run_id
            The id of the current Mlflow run.
        scene
            The manipulated scene.

        Raises
        ------
        RuntimeError
            If the target object of the scene is transformed.
        """
        remaining_time = estimate_remaining_time(
            start=attack_start_time,
            iter_end=current_generation_end_time,
            n_elapsed_iter=generation_idx + 1,
            total_iter_count=maxiter,
        )

        best_solution_obj = best_solution_keeper.get_best()

        self.mlflow_client.log_metrics(
            run_id=run_id,
            metrics=best_solution_obj.metrics | {"fitness": best_solution_obj.fitness},
            generation_idx=generation_idx,
        )

        if best_solution_obj.pred_depths_aligned is None:
            raise Exception(
                "Internal error. The best solution has no stored depth map predictions."
            )

        assert (
            best_solution_obj.pred_depths_aligned is not None
        ), "The the best solution has no stored depth map predictions."

        assert (
            best_solution_obj.rgbds is not None
        ), "The best solution has no stored RGB images and ground truth depth maps."

        if is_last_iter:
            self.mlflow_client.log_npz(
                name="best_solution.npz", data={"x": best_solution_obj.x}, run_id=run_id
            )
            scene_data_for_best = self._get_scene_data_for_solution(
                x=best_solution_obj.x, scene=scene
            )
            self.mlflow_client.log_text(
                name="best_solution.scene", run_id=run_id, data=scene_data_for_best
            )

        if console_logging_freq.needs_logging(generation_idx) or is_last_iter:
            if self.log_im or is_last_iter:
                self.mlflow_client.log_gt_depths_and_preds(
                    cam_proj_spec=best_solution_obj.cam_proj_spec,
                    generation_idx=generation_idx,
                    pred_depths_aligned=best_solution_obj.pred_depths_aligned,
                    rgbds=best_solution_obj.rgbds,
                    run_id=run_id,
                )

                self.mlflow_client.log_rgbs(
                    generation_idx=generation_idx,
                    rgbs=best_solution_obj.rgbds.rgbs,
                    run_id=run_id,
                )

            logs_str = "; ".join(
                f"{key}: {value:.7E}"
                for key, value in best_solution_obj.metrics.items()
            )

            print(
                f"Iteration: {generation_idx}; Remaining: {remaining_time}; Best fitness: {best_solution_obj.fitness} | {logs_str}"
            )

    def _get_scene_data_for_solution(self, x: np.ndarray, scene: Scene) -> str:
        """
        Save the resulting scene of the best solution to a string.

        Parameters
        ----------
        x
            The solution to apply.
        scene
            The scene to modify.

        Returns
        -------
        v
            The transformed scene saved into a string.

        Raises
        ------
        RuntimeError
            If the target object of the scene is transformed.

        Notes
        -----
        This function internally does the following:

        1. Apply the specified solution onto the scene.
        2. Save the scene to a string.
        3. Clear the transform of the scene.
        4. Return the previously created string.
        """
        self.attack_strategy.apply_solution_transform(x=x, scene=scene)
        str_io = StringIO()
        scene.save(str_io)
        scene.set_target_transform(None)
        data = str_io.getvalue()
        return data


@dataclass
class ExactMetrics:
    actual_metrics: dict[str, float]
    """
    The values of the metrics calculated using all viewpoints with the corresponding viewpoint types.
    """

    minimizably_signed_metrics: dict[str, float]
    """
    The values of the metrics calculated using all viewpoints with the corresponding viewpoint types, then their sign changed to make sure that the smaller values mean that the depth predictor is worse (i. e. the attack is more successful).
    """


class AdvSampleLearningResult(NamedTuple):
    best_solution_exact_origsigned_metrics: dict[str, float]
    """
    The exact values of all supported metrics on the best solution with their original signs.
    """

    best_solution_exact_minimizably_signed_metrics: dict[str, float]
    """
    The exact values of all supported metrics on the best solution with possibly modified sign to make them minimizable.
    """

    n_pred_calls_total: int
    """
    The total number of depth prediction calls to evaluate all generations.
    """

    final_change_amount_score: float
    """
    The change amount score of the best solution.
    """

    run_name: str
    """
    The name of the created mlflow run.
    """


@dataclass
class SolutionInfo:
    fitness: float
    """
    The fitness function for the sample. Smaller is better.
    """

    change_amount_score: float
    """
    The change amount score for this solution.
    """

    metrics: dict[str, float]
    """
    The different calculated metrics for this sample. The smaller values might be better, but this is not necessary.
    """

    pred_depths_aligned: np.ndarray | None
    """
    The aligned depth predictions for the corresponding rendered RGB images.

    Format: ``Im::RGBs``
    """

    rgbds: RGBsWithDepthsAndMasks | None
    """
    The rendered RGB images and the corresponding ground truth depth maps with masks.
    """

    cam_proj_spec: CamProjSpec
    """
    The projection of the camera of the ground truth depths.
    """

    x: np.ndarray
    """
    The solution itself.
    """


@dataclass
class GenerationEvaluationResult:
    """
    The result of the evaluation of a single generation during attack learning.
    """

    solution_infos: list[SolutionInfo]
    """
    Various information for each solution.
    """

    total_pred_calls: int
    """
    The total number of prediction calls to evaluate this generation.
    """
