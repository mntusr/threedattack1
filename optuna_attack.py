import argparse
import tkinter as tk
from pathlib import Path
from typing import Protocol, Union, cast

import optuna
import optuna.study

from threedattack.losses import LossDerivationMethod, RawLossFn, get_derived_loss_name
from threedattack.rendering_core import MIN_CONTROL_POINTS_FOR_WARP, ObjectTransformType
from threedattack.script_util import (
    AttackDetails,
    attack_details_2_cli_args,
    run_first_strategy_attack,
    set_mlflow_run_name_as_optuna_attribute,
    suggest_with_public_name,
)


class FirstStrategyTargetFunction:
    """
    A target function for optuna that uses `threedattack.FirstStrategy`.

    Parameters
    ----------
    target_scene_path
        The scene on which the attack should be done.
    n_val_viewpoints
        The number of validation viewpoints.
    experiment_name
        The name of the Mlflow experiment to which the results of the runs should be logged.
    target_model_name
        The name of the target model.
    meta_optimized_metric_name
        The name of the metric that Optuna optimizes.
    """

    def __init__(
        self,
        target_scene_path: Path,
        n_val_viewpoints: int,
        experiment_name: str,
        target_model_name: str,
        meta_optimized_metric_name: str,
        max_maxiter: int,
        max_n_estim_viewpts: int,
        max_n_control_points: int,
        no_predcall_target: bool,
    ):
        self.target_scene_path = target_scene_path
        self.n_val_viewpoints = n_val_viewpoints
        self.experiment_name = experiment_name
        self.target_model_name = target_model_name
        self.meta_optimized_metric_name = meta_optimized_metric_name
        self.max_maxiter = max_maxiter
        self.max_n_estim_viewpts = max_n_estim_viewpts
        self.max_n_control_points = max_n_control_points
        self.no_predcall_target = no_predcall_target

    def get_metric_names(self) -> list[str]:
        """Get the names of the metrics returned by the objective function."""
        if self.no_predcall_target:
            return [
                self.meta_optimized_metric_name,
                "change_amount_score",
            ]
        else:
            return [
                self.meta_optimized_metric_name,
                "n_pred_calls_total_per10000",
                "change_amount_score",
            ]

    def run_trial(
        self,
        trial: optuna.Trial,
    ) -> Union[tuple[float, float, float], tuple[float, float]]:
        """
        Run a concrete Optuna trial.

        This function assumes that Optuna **minimizes** the target function.

        Parameters
        ----------
        trial
            The Optuna trial object.

        Returns
        -------
        min_fn_val
            The value of the minimzed function for this trial.
        n_pred_calls_total
            The total number of depth prediction calls to evaluate all generations.
        change_amount_score
            The change amount score of the best solution.
        """
        free_area_multiplier = trial.suggest_float("free_area_multiplier", 1.01, 2)
        maxiter = trial.suggest_int("maxiter", 10, self.max_maxiter)
        n_control_points = trial.suggest_int(
            "n_control_points", MIN_CONTROL_POINTS_FOR_WARP, self.max_n_control_points
        )
        max_pos_change_per_coord = trial.suggest_float(
            "max_pos_change_per_coord", 0.01, 1
        )
        max_change_amount_score = trial.suggest_float(
            "max_change_amount_score", 0.001, 0.4
        )
        sigma0 = trial.suggest_float("sigma0", 0.01, 0.4)
        n_estim_viewpts = trial.suggest_int(
            "n_estim_viewpts", 1, self.max_n_estim_viewpts
        )
        is_estim_free = trial.suggest_categorical("is_estim_free", [True, False])
        if is_estim_free:
            freeze_estim = "free"
        else:
            freeze_estim = list(range(n_estim_viewpts))

        cma_optimized_loss_deriv = suggest_with_public_name(
            trial, "cma_optimized_loss_deriv", LossDerivationMethod
        )
        cma_optimized_raw_loss_fn = suggest_with_public_name(
            trial, "cma_optimized_raw_loss_fn", RawLossFn
        )
        cma_optimized_metric = get_derived_loss_name(
            cma_optimized_loss_deriv, cma_optimized_raw_loss_fn
        )

        attack_details = AttackDetails(
            cma_optimized_metric=cma_optimized_metric,
            experiment_name=self.experiment_name,
            free_area_multiplier=free_area_multiplier,
            freeze_estim=freeze_estim,
            max_pos_change_per_coord=max_pos_change_per_coord,
            maxiter=maxiter,
            n_control_points=n_control_points,
            n_val_viewpoints=self.n_val_viewpoints,
            sigma0=sigma0,
            target_model_name=self.target_model_name,
            target_scene_path=self.target_scene_path,
            n_estim_viewpts=n_estim_viewpts,
            n_train_viewpoints=400,
            transform_type=ObjectTransformType.VolumeBased.public_name,
            n_cubes_steps=20,
            max_shape_change=max_change_amount_score,
            n_test_viewpoints=200,
            eval_on_test=False,
        )
        command = ["python", "manual_attack.py"] + attack_details_2_cli_args(
            attack_details
        )
        attack_result = run_first_strategy_attack(
            attack_details=attack_details,
            run_repro_command=command,
            log_im=False,
        )

        set_mlflow_run_name_as_optuna_attribute(
            trial=trial, run_name=attack_result.run_name
        )

        if self.no_predcall_target:
            return (
                attack_result.best_solution_exact_minimizably_signed_metrics[
                    self.meta_optimized_metric_name
                ],
                attack_result.final_change_amount_score,
            )
        else:
            return (
                attack_result.best_solution_exact_minimizably_signed_metrics[
                    self.meta_optimized_metric_name
                ],
                attack_result.n_pred_calls_total / 10000,
                attack_result.final_change_amount_score,
            )


SAMPLERS: dict[
    str, Union[optuna.samplers.NSGAIISampler, optuna.samplers.TPESampler]
] = {
    "nsga2": optuna.samplers.NSGAIISampler(),
    "tpe": optuna.samplers.TPESampler(),
}


def main(args: "_Args") -> None:
    if args.no_predcall_target:
        directions = [
            optuna.study.StudyDirection.MINIMIZE,
            optuna.study.StudyDirection.MINIMIZE,
        ]
    else:
        directions = [
            optuna.study.StudyDirection.MINIMIZE,
            optuna.study.StudyDirection.MINIMIZE,
            optuna.study.StudyDirection.MINIMIZE,
        ]

    study = optuna.create_study(
        directions=directions,
        storage="sqlite:///optuna_studies.db",
        study_name=args.study,
        load_if_exists=args.resume_without_retry or args.resume_with_retry,
        sampler=SAMPLERS[args.sampler],
    )
    target_fn = FirstStrategyTargetFunction(
        target_scene_path=Path("scenes/room1_subdivided3.glb"),
        experiment_name=args.experiment,
        n_val_viewpoints=200,
        target_model_name=args.target_model,
        meta_optimized_metric_name=args.meta_optimized,
        max_maxiter=args.max_maxiter,
        max_n_estim_viewpts=args.max_n_estim_viewpts,
        max_n_control_points=args.max_n_control_points,
        no_predcall_target=args.no_predcall_target,
    )
    study.set_metric_names(target_fn.get_metric_names())

    # rerun last trial if it failed
    if args.resume_with_retry:
        if len(study.trials) > 0:
            if study.trials[-1].state == optuna.trial.TrialState.FAIL:
                study.enqueue_trial(study.trials[-1].params)

    study.optimize(
        target_fn.run_trial,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False,
        timeout=args.study_timeout,
    )


def parse_args() -> "_Args":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study", required=True, help="The name of the optuna study.", type=str
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="The name of the Mlflow experiment.",
        type=str,
    )
    parser.add_argument(
        "--study-timeout",
        required=True,
        help="The timeout for the whole study in seconds.",
        type=float,
    )

    parser.add_argument(
        "--resume-with-retry",
        action="store_true",
        help="Enables to resume a previously started Optuna study and retry the last failed experiment.",
    )

    parser.add_argument(
        "--resume-without-retry",
        action="store_true",
        help="Enables to resume a previously started Optuna study without retrying the last failed trial.",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="The name of the target model.",
    )
    parser.add_argument(
        "--max-maxiter",
        type=int,
        required=True,
        help="The maximum value of the maxiter parameter.",
    )
    parser.add_argument(
        "--max-n-estim-viewpts",
        type=int,
        required=True,
        help="The maximum value of the n_estim_viewpts parameter (including).",
    )
    parser.add_argument(
        "--max-n-control-points",
        type=int,
        required=True,
        help="The maximum value of the n_control_points parameter (including).",
    )
    parser.add_argument(
        "--meta-optimized",
        type=str,
        required=True,
        help="The name of the meta-optimized metric.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        required=True,
        help=f"The sampler to use. Values: {list(sorted(SAMPLERS.keys()))}. Comparison of the samplers: <https://optuna.readthedocs.io/en/stable/reference/samplers/index.html#module-optuna.samplers>; <https://github.com/optuna/optuna/issues/2537>",
    )
    parser.add_argument(
        "--no-predcall-target",
        action="store_true",
        help="Disable the target function for the number of depth prediction calls.",
    )

    parsed = parser.parse_args()
    return cast(_Args, parsed)


class _Args(Protocol):
    study: str
    experiment: str
    study_timeout: float
    resume_with_retry: bool
    resume_without_retry: bool
    target_model: str
    max_maxiter: int
    max_n_estim_viewpts: int
    max_n_control_points: int
    meta_optimized: str
    sampler: str
    no_predcall_target: bool


if __name__ == "__main__":
    main(parse_args())
