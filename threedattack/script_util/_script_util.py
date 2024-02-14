import argparse
import re
import sys
import tkinter as tk
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, TypeVar, overload

import mlflow
import optuna.trial
import pandas as pd
from matplotlib.axes import Axes

from .._attack_commons import AdvAttackLearner, AdvSampleLearningResult
from .._first_strategy import FirstStrategy
from .._logging import StepLoggingFreqFunction
from .._scene import Scene, SceneConfig, create_scene_or_quit_wit_error
from ..external_datasets import NYUV2_IM_SIZE, NYUV2_MAX_DEPTH
from ..local_config import get_local_config_json
from ..mlflow_util import CustomMlflowClient, RunConfig
from ..rendering_core import (
    DesiredViewpointCounts,
    ObjectTransformType,
    get_object_transform_type_by_name,
    get_supported_object_transform_type_names,
)
from ..target_model import (
    AsyncDepthPredictor,
    get_supported_models,
    load_target_model_by_name,
)


def wait_for_enter(message: str) -> None:
    print(message)
    input()


class AxesArray2D(Protocol):
    @overload
    def __getitem__(self, i: tuple[int, int]) -> Axes:
        ...

    @overload
    def __getitem__(self, i: int) -> "AxesArray1D":
        ...


class AxesArray1D(Protocol):
    def __getitem__(self, i: int, /) -> Axes:
        ...


def show_scene_selector() -> Path:
    scene_paths = [scene for scene in Path("./scenes").rglob("*.glb")] + [
        scene for scene in Path("test_resources").rglob("*.glb")
    ]

    scene_path_strs = [str(scene_path) for scene_path in scene_paths]
    return Path(show_selector(scene_path_strs))


def get_all_world_paths() -> list[Path]:
    return [scene for scene in Path("./scenes").rglob("*.glb")] + [
        scene for scene in Path("test_resources").rglob("*.glb")
    ]


def show_model_selector() -> str:
    supported_models = get_supported_models()
    return show_selector(supported_models)


def show_selector(alternatives: list[str]) -> str:
    def ok_command():
        window.destroy()

    window = tk.Tk()
    selected = tk.StringVar(value=alternatives[0])
    menu = tk.OptionMenu(window, selected, *alternatives)
    menu.pack()
    ok_button = tk.Button(window, text="ok", command=ok_command)
    ok_button.pack()

    window.mainloop()

    return selected.get()


def run_first_strategy_attacks(
    attack_detail_list: "Sequence[tuple[AttackDetails, dict[str, Any]]]",
    run_repro_command_fn: "Callable[[AttackDetails], list[str]]",
    log_im: bool,
) -> pd.DataFrame:
    row_dicts: list[dict[str, Any]] = []
    for i in range(len(attack_detail_list)):
        attack_details, extra_tags = attack_detail_list[i]
        run_repro_command = run_repro_command_fn(attack_details)
        prepared_attack = (
            _prepare_attack_by_attack_details_for_learning_and_print_status(
                attack_details=attack_details,
                log_im=log_im,
                run_repro_command=run_repro_command,
            )
        )

        print(f"Attack {i+1} started.")
        try:
            attack_result = prepared_attack.attack_learner.learn_adv_sample(
                scene=prepared_attack.scene,
                console_logging_freq=StepLoggingFreqFunction(
                    steps=[20, 100],
                    freqencies=[1, 5, 20],
                ),
                predictor=prepared_attack.predictor,
                shared_info_calc_logging_freq=StepLoggingFreqFunction(
                    steps=[], freqencies=[5]
                ),
                run_config=prepared_attack.run_conf,
            )

            row_dict = attack_details_and_exact_origsigned_metrics_2_dict(
                attack_details=attack_details,
                exact_origsigned_metrics=attack_result.best_solution_exact_origsigned_metrics,
                extra_tags=extra_tags,
                run_name=attack_result.run_name,
            )
            row_dicts.append(row_dict)
        finally:
            prepared_attack.scene.destroy_showbase()

    return pd.DataFrame.from_records(row_dicts)


def attack_details_and_exact_origsigned_metrics_2_dict(
    attack_details: "AttackDetails",
    exact_origsigned_metrics: dict[str, float],
    run_name: str,
    extra_tags: dict[str, float | bool | str],
) -> dict[str, float | bool | str]:
    """
    Convert the attack details, the tags and the exact metrics with their original signs to a single dictionary.

    This function converts the non `int`/`bool`/`str`/`float` hyperaprameters to `str`.

    The keys of the dictionary:

    * ``metrics.<metric name>``: The different exact metrics with their original signs.
    * ``params.<param name>``: The different hyperparameters.
    * ``tags.<tag name>``: The different further metadata specified in the ``extra_tags`` argument.
    * ``mlflow.run_name``: The name of the mlflow run for this attack.
    * ``mlflow.experiment_name``: The name of the mlflow experiment for this attack.

    Parameters
    ----------
    attack_details
        The attack details for the attack to store.
    exact_origsigned_metrics
        The different exact metrics with their original signs.
    run_name
        The name of the Mlflow run of the attack.
    extra_tags
        The various extra metadata.

    Returns
    -------
    v
        The created dictionary.
    """
    row_dict: dict[str, float | bool | str] = dict()
    for key, value in attack_details_2_dict(attack_details).items():
        if not (
            isinstance(value, str)
            or isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, bool)
        ):
            value = str(value)
        row_dict["params." + key] = value
    for (
        key,
        value,
    ) in exact_origsigned_metrics.items():
        row_dict["metrics." + key] = value

    for key, value in extra_tags.items():
        row_dict["tags." + key] = value
    row_dict["mlflow.run_name"] = run_name
    row_dict["mlflow.experiment_name"] = attack_details.experiment_name

    return row_dict


def run_first_strategy_attack(
    attack_details: "AttackDetails",
    run_repro_command: list[str],
    log_im: bool,
) -> AdvSampleLearningResult:
    """
    Run an adversarial attack with `threedattack.FirstStrategy` using the specified properties.

    This function prints the progress of the attack to the console.

    Parameters
    ----------
    attack_details
        The different options of the attack.
    run_repro_command
        The command that enables the user to reproduce the run.
    result_metric_name
        The name of the returned metric. The metric sign is modified to make sure that the smaller values mean that the depth predictor is worse (i. e. the attack is more successful).
    log_im
        Log the RGB images and depth maps during attack.

    Returns
    -------
    min_fn_val
        The value of the minimzed function for this trial.
    n_seq_depth_est_calls
        The number of sequential (i. e. non-parallellized) calls to the depth estimator.

    Raises
    ------
    TBD
    """
    prepared_attack = _prepare_attack_by_attack_details_for_learning_and_print_status(
        attack_details=attack_details,
        log_im=log_im,
        run_repro_command=run_repro_command,
    )

    print("Attack started.")
    try:
        attack_result = prepared_attack.attack_learner.learn_adv_sample(
            scene=prepared_attack.scene,
            console_logging_freq=StepLoggingFreqFunction(
                steps=[20, 100],
                freqencies=[1, 5, 20],
            ),
            predictor=prepared_attack.predictor,
            shared_info_calc_logging_freq=StepLoggingFreqFunction(
                steps=[], freqencies=[5]
            ),
            run_config=prepared_attack.run_conf,
        )

        return attack_result
    finally:
        prepared_attack.scene.destroy_showbase()


def _prepare_attack_by_attack_details_for_learning_and_print_status(
    attack_details: "AttackDetails",
    run_repro_command: list[str],
    log_im: bool,
) -> "_PreparedAttackForLearning":
    """
    Creates all objects necessary for an adversarial attack based on the specified attack details.

    Parameters
    ----------
    attack_details
        The attack details to use.
    run_repro_command
        The command that reproduces this particular run.
    log_im
        Log the RGB images and depth maps during attack.

    Returns
    -------
    v
        All objects necessary to the attack.

    Raises
    ------
    TBD
    """
    rendering_resolution = NYUV2_IM_SIZE
    attack_strategy = FirstStrategy(
        n_control_points=attack_details.n_control_points,
        freeze_estim=attack_details.freeze_estim,
        max_pos_change_per_coord=attack_details.max_pos_change_per_coord,
        n_estim_viewpts=attack_details.n_estim_viewpts,
        optimized_metric=attack_details.cma_optimized_metric,
        sigma0=attack_details.sigma0,
        max_transform_change_amount_score=attack_details.max_shape_change,
    )

    viewpoint_counts = DesiredViewpointCounts(
        n_train_samples=attack_details.n_train_viewpoints,
        n_test_samples=attack_details.n_test_viewpoints,
        n_val_samples=attack_details.n_val_viewpoints,
    )

    local_config = get_local_config_json()
    mlflow_client = _configure_mlflow(local_config.mlflow_tracking_url)
    print(f"Waiting for the worker of the target model to start.")
    predictor = load_target_model_by_name(attack_details.target_model_name)

    print("Loading scene.")
    scene = create_scene_or_quit_wit_error(
        SceneConfig(
            world_path=attack_details.target_scene_path,
            resolution=rendering_resolution,
            viewpt_counts=viewpoint_counts,
            object_transform_type=get_object_transform_type_by_name(
                attack_details.transform_type
            ),
            n_volume_sampling_steps_along_shortest_axis=attack_details.n_cubes_steps,
            target_size_multiplier=attack_details.free_area_multiplier,
            depth_cap=NYUV2_MAX_DEPTH,
            applied_transform=None,
        )
    )

    _print_attack_details(attack_details)

    print(f'Checking experiment "{attack_details.experiment_name}" and creating run')
    run_conf = _make_sure_experiment_exists_and_create_run_conf(
        attack_details=attack_details,
        mlflow_client=mlflow_client,
        run_repro_command=run_repro_command,
    )

    print("Creating attack learner")
    attack_learner = AdvAttackLearner(
        attack_strategy=attack_strategy,
        mlflow_client=mlflow_client,
        maxiter=attack_details.maxiter,
        log_im=log_im,
        eval_on_test=attack_details.eval_on_test,
    )

    return _PreparedAttackForLearning(
        attack_strategy=attack_strategy,
        predictor=predictor,
        run_conf=run_conf,
        scene=scene,
        attack_learner=attack_learner,
    )


def _configure_mlflow(tracking_uri: str) -> CustomMlflowClient:
    mlflow_client = CustomMlflowClient(mlflow.MlflowClient(tracking_uri=tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow_client


@dataclass
class _PreparedAttackForLearning:
    attack_strategy: FirstStrategy
    scene: Scene
    predictor: AsyncDepthPredictor
    run_conf: RunConfig
    attack_learner: AdvAttackLearner


def _print_attack_details(attack_details: "AttackDetails") -> None:
    """Print the attack details to the console."""
    print("Attack details")
    attack_detail_dict = attack_details_2_dict(attack_details)
    for key, value in attack_detail_dict.items():
        print(f"\t{key}: {value}")


def _make_sure_experiment_exists_and_create_run_conf(
    attack_details: "AttackDetails",
    mlflow_client: CustomMlflowClient,
    run_repro_command: list[str],
) -> RunConfig:
    attack_detail_dict = attack_details_2_dict(attack_details)
    experiment_id = mlflow_client.get_experiment_id_by_name(
        attack_details.experiment_name
    )

    run_conf = RunConfig(
        experiment_id=experiment_id,
        run_repro_command=run_repro_command,
        run_params=attack_detail_dict,
    )

    return run_conf


def add_attack_details_args(parser: argparse.ArgumentParser) -> None:
    """
    Add the necessary CLI arguments to an argument parser that specify all attack details.

    Parameters
    ----------
    parser
        The argument parser to configure.
    """

    def positive_int(value: str) -> int:
        val_as_int = int(value)
        if val_as_int < 0:
            raise argparse.ArgumentTypeError(f"{val_as_int} is not positive.")
        return val_as_int

    def positive_float(value: str) -> float:
        val_as_float = float(value)
        if val_as_float < 0:
            raise argparse.ArgumentTypeError(f"{val_as_float} is not positive.")
        return val_as_float

    def float_greater_than_1(value: str) -> float:
        val_as_float = float(value)
        if val_as_float <= 1:
            raise argparse.ArgumentTypeError(f"{val_as_float} is not greater than 1.")
        return val_as_float

    # TODO improve argument validation

    parser.add_argument(
        "--n-control-points",
        type=positive_int,
        help="The number of viewpoints to control the interpolated vector field.",
        required=True,
    )
    parser.add_argument(
        "--n-estim-viewpts",
        type=positive_int,
        help="The number of viewpoints to estimate the expected value of the fitness function during training. It should be at least 1. If the viewpoints to estimate the training metrics for each sample is manually set, then this should have the same value as the number of that points.",
        required=True,
    )

    parser.add_argument(
        "--freeze-estim",
        type=str,
        required=True,
        help='Values: "free", the set of the viewpoints to estimate the metrics for each sample is not frozen; "frozen_random", the set of the viewpoints to estimate the metrics for each sample is frozen, but randomly selected; list[int], the set of the viewpoints to estimate the metrics for each sample is manually specified.',
        nargs="+",
    )

    parser.add_argument(
        "--max-pos-change-per-coord",
        type=positive_float,
        required=True,
        help='Values: "free", the set of the viewpoints to estimate the metrics for each sample is not frozen; "frozen_random", the set of the viewpoints to estimate the metrics for each sample is frozen, but randomly selected; list[int], the set of the viewpoints to estimate the metrics for each sample is manually specified.',
    )
    parser.add_argument(
        "--cma-optimized-metric",
        type=str,
        required=True,
        help="The name of the optimized metric for a single run.",
    )
    parser.add_argument(
        "--sigma0",
        type=positive_float,
        required=True,
        help="The sigma0 parameter of the CMA algorithm.",
    )
    parser.add_argument(
        "--n-val-viewpoints",
        type=positive_int,
        required=True,
        help="The number of viewpoints in the validation set to use.",
    )
    parser.add_argument(
        "--n-train-viewpoints",
        type=positive_int,
        required=True,
        help="The number of viewpoints in the training set to use.",
    )
    parser.add_argument(
        "--n-test-viewpoints",
        type=positive_int,
        required=True,
        help="The number of viewpoints in the testing set to use.",
    )
    parser.add_argument(
        "--target-model-name",
        type=str,
        required=True,
        help="The name of the target model.",
    )
    parser.add_argument(
        "--target-scene-path",
        type=Path,
        required=True,
        help="The path of the target scene.",
    )
    parser.add_argument(
        "--free-area-multiplier",
        type=float_greater_than_1,
        required=True,
        help="How many times the area of the target object might increase during attack.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="The name of the experiment to use.",
    )

    parser.add_argument(
        "--maxiter",
        type=positive_int,
        required=True,
        help="The maximum number of generations in the CMA algorithm.",
    )

    object_transform_type_names = get_supported_object_transform_type_names()
    parser.add_argument(
        "--transform-type",
        type=str,
        required=True,
        choices=object_transform_type_names,
        help=f'The name of the type of the transform of the target object. Value "{ObjectTransformType.MeshBased}" means that the mesh of the target object is transformed. Value "{ObjectTransformType.VolumeBased.public_name}" means that the occupacy function of the target object is transformed.',
    )
    parser.add_argument(
        "--n-cubes-steps",
        type=positive_int,
        required=True,
        help=f'The number of steps alognside a single axis, when Marching Cubes is applied on the occupacy function of the target object. This affects only "{ObjectTransformType.VolumeBased.public_name}"',
    )

    parser.add_argument(
        "--max-shape-change",
        type=positive_float,
        required=False,
        help=f'A soft upper bound for the change amount score in case of the type of the transformation is "{ObjectTransformType.VolumeBased.public_name}". Otherwise this argument is ignored.',
    )
    parser.add_argument(
        "--eval-on-test",
        action="store_true",
        help="If set, then the evaluation is done on the test set too.",
    )


def get_attack_details_from_parsed_args(parsed_args: Any) -> "AttackDetails":
    """
    Get the attack details from the parsed CLI args.

    This function might do some (not necessarily complete) additional argument validations and quit with error if a problem was found.

    Parameters
    ----------
    parsed_args
        The parsed CLI args.

    Returns
    -------
    v
        The collected attack details.
    """
    vals_dict: dict[str, Any] = dict()

    for field in fields(AttackDetails):
        vals_dict[field.name] = getattr(parsed_args, field.name)

    if vals_dict["freeze_estim"] == ["free"]:
        vals_dict["freeze_estim"] = "free"
    elif vals_dict["freeze_estim"] == ["frozen_random"]:
        vals_dict["freeze_estim"] = "frozen_random"
    elif all(val.isdigit() for val in vals_dict["freeze_estim"]):
        vals_dict["freeze_estim"] = [int(val) for val in vals_dict["freeze_estim"]]
    else:
        print(
            'Argument --freeze-estim should be either "free" or "frozen_random" or a range of integers.'
        )
        sys.exit(1)

    if (
        vals_dict["max_shape_change"] == ObjectTransformType.VolumeBased.public_name
    ) and (vals_dict["max_shape_change"] is None):
        print(
            f'The argument --max-shape-change should be specified if the transformation type is "{ObjectTransformType.VolumeBased.public_name}"'
        )
        sys.exit(1)
    # TODO improve argument validation

    return AttackDetails(**vals_dict)


def attack_details_2_cli_args(attack_details: "AttackDetails") -> list[str]:
    """
    Convert the attack details to CLI argument list.

    Parameters
    ----------
    attack_details
        The attack details to convert.

    Returns
    -------
    v
        The attack details as a CLI argument list.

    Raises
    ------
    ValueError
        If some of the string values starts with "-".
    """
    details_dict = attack_details_2_dict(attack_details)
    store_true_keys = ["eval_on_test"]

    assert all(
        store_true_key in details_dict.keys() for store_true_key in store_true_keys
    )

    arg_list: list[str] = []
    for key, value in details_dict.items():
        arg_name = "--" + key.replace("_", "-")

        if key in store_true_keys:
            if value == True:
                arg_list.append(arg_name)
        else:
            if value is None:
                continue

            arg_list.append(arg_name)

            if isinstance(value, str):
                if value.startswith("-"):
                    raise ValueError(f'The value "{value}" starts with "-".')

            if isinstance(value, list):
                for val_part in value:
                    arg_list.append(str(val_part))
            else:
                arg_list.append(str(value))

    return arg_list


def attack_details_2_dict(attack_details: "AttackDetails") -> dict[str, Any]:
    result: dict[str, Any] = dict()
    for field in fields(attack_details):
        result[field.name] = getattr(attack_details, field.name)

    return result


@dataclass
class AttackDetails:
    n_control_points: int
    """
    The number of viewpoints to control the interpolated vector field.
    """

    n_estim_viewpts: int
    """
    The number of viewpoints to estimate the expected value of the fitness function during training. It should be at least 1. If the viewpoints to estimate the training metrics for each sample is manually set, then this should have the same value as the number of that points.
    """

    freeze_estim: list[int] | str
    """
    Values: "free", the set of the viewpoints to estimate the metrics for each sample is not frozen; "frozen_random", the set of the viewpoints to estimate the metrics for each sample is frozen, but randomly selected; list[int], the set of the viewpoints to estimate the metrics for each sample is manually specified.
    """

    max_pos_change_per_coord: float
    """
    The maximum relative position change for a single vertex for each coordinate. The actual maximal change = ``max_pos_change_per_coord*min_size``, where ``min_size = the minimum of the sizes of the target object alongside the X, Y and Z axes``
    """

    cma_optimized_metric: str
    """
    The name of the optimized metric for a single run.
    """

    sigma0: float
    """
    The sigma0 parameter of the CMA algorithm.
    """

    n_val_viewpoints: int
    """
    The number of viewpoints in the validation set to use.
    """

    n_train_viewpoints: int
    """
    The number of viewpoints in the training set to use.
    """

    n_test_viewpoints: int
    """
    The number of viewpoints in the testing set to use.
    """

    target_model_name: str
    """
    The name of the target model.
    """

    target_scene_path: Path
    """
    The path of the target scene.
    """

    free_area_multiplier: float
    """
    How many times the area of the target object might increase during attack.
    """

    experiment_name: str
    """
    The name of the experiment to use.
    """

    maxiter: int
    """
    The maximum number of generations in the CMA algorithm.
    """

    transform_type: str
    """
    The name of the type of the transformation of the target object.
    """

    n_cubes_steps: int
    """
    The number of steps alognside a single axis, when Marching Cubes is applied on the occupacy function of the target object.
    """

    max_shape_change: float | None
    """
    A soft upper bound for the change amount score of the transformation if the volume of the target object is transformed. In this case, this value is not None. If the mesh of the target object is directly transformed, then this value is ignored and might be None too.
    """

    eval_on_test: bool
    """
    If true, then the evaluation is done on the test set too.
    """


class _ThingWithPublicName(Protocol):
    public_name: str


T = TypeVar("T", bound=_ThingWithPublicName)


def suggest_with_public_name(trial: optuna.Trial, name: str, choices: Iterable[T]) -> T:
    """
    Implement `optuna.Trial.suggest_categorical` for an iterable of choices, where
    the field ``public_name`` sets their user facing name.

    The names are always sorted before passing to the internal choice function.

    Parameters
    ----------
    trial
        The trial to use.
    name
        Same as the corresponding argument in `optuna.Trial.suggest_categorical`.
    choices
        The alternatives.

    Returns
    -------
    v
        The chosen alternative.
    """
    public_names = list(sorted(val.public_name for val in choices))
    public_name_dict = {val.public_name: val for val in choices}
    chosen_public_name = trial.suggest_categorical(name=name, choices=public_names)

    result = public_name_dict[chosen_public_name]

    return result
