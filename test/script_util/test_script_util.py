import argparse
import unittest
from dataclasses import dataclass, fields
from enum import Enum
from gc import freeze
from pathlib import Path
from typing import cast
from unittest import mock

import mlflow
import requests

from threedattack.dataset_model import SampleType
from threedattack.local_config import get_local_config_json
from threedattack.losses import (
    LossDerivationMethod,
    RawLossFn,
    get_derived_loss_by_name,
)
from threedattack.mlflow_util import ExperimentId
from threedattack.rendering_core import ObjectTransformType, VolumeBasedObjectTransform
from threedattack.script_util import (
    AttackDetails,
    add_attack_details_args,
    attack_details_2_cli_args,
    attack_details_and_exact_origsigned_metrics_2_dict,
    get_attack_details_from_parsed_args,
    suggest_with_public_name,
)
from threedattack.script_util._script_util import (
    _prepare_attack_by_attack_details_for_learning_and_print_status,
    attack_details_2_dict,
)


class TestScriptUtil(unittest.TestCase):
    def setUp(self):
        self.TEST_EXPERIMENT_NAMES = ["Test experiment1", "Test experiment2"]
        self.attack_details = AttackDetails(
            cma_optimized_metric="mean_delta_rmse",
            experiment_name=self.TEST_EXPERIMENT_NAMES[0],
            free_area_multiplier=1.5,
            freeze_estim="free",
            max_pos_change_per_coord=2.1,
            maxiter=300,
            n_control_points=25,
            n_estim_viewpts=13,
            n_train_viewpoints=100,
            n_val_viewpoints=30,
            sigma0=1.1,
            target_model_name="dummy_depth1",
            target_scene_path=Path("test_resources/test_scene.glb"),
            transform_type=ObjectTransformType.MeshBased.public_name,
            max_shape_change=0.2,
            n_cubes_steps=40,
            eval_on_test=False,
            n_test_viewpoints=60,
        )

    def test_cli_functions(self):
        transform_type_configs: list[tuple[str, float | None]] = [
            (ObjectTransformType.MeshBased.public_name, None),
            (ObjectTransformType.MeshBased.public_name, 3),
            (ObjectTransformType.VolumeBased.public_name, 3),
        ]

        subtest_idx = 0
        for freeze_estim in ["free", "frozen_random", [1, 2, 3, 4]]:
            for transform_type_name, max_shape_change in transform_type_configs:
                with self.subTest(subtest_idx + 1):
                    self.attack_details.freeze_estim = freeze_estim
                    self.attack_details.transform_type = transform_type_name
                    self.attack_details.max_shape_change = max_shape_change

                    cli_args = attack_details_2_cli_args(self.attack_details)

                    for cli_arg in cli_args:
                        self.assertIsInstance(cli_arg, str)

                    parser = argparse.ArgumentParser()
                    add_attack_details_args(parser)

                    parsed = parser.parse_args(cli_args)

                    reparsed_attack_details = get_attack_details_from_parsed_args(
                        parsed
                    )

                    self.assertEqual(self.attack_details, reparsed_attack_details)
                subtest_idx += 1

    def test_prepare_attack_by_attack_details_for_learning_and_print_status_all_objects_properly_set_up(
        self,
    ):
        TEST_EXPERIMENT_NAMES = ["Test experiment1", "Test experiment2"]

        attack_detail_dict = attack_details_2_dict(self.attack_details)

        def assert_attack_objs_correctly_prepaired(
            attack_details: AttackDetails,
            log_im: bool,
            run_repro_command: list[str],
            custom_mlflow_client_mock: mock.Mock,
        ):
            prepared_attack = (
                _prepare_attack_by_attack_details_for_learning_and_print_status(
                    attack_details=attack_details,
                    log_im=log_im,
                    run_repro_command=run_repro_command,
                )
            )
            try:
                self.assertEqual(
                    prepared_attack.attack_strategy.optimized_metric,
                    get_derived_loss_by_name(attack_details.cma_optimized_metric),
                )
                self.assertEqual(
                    prepared_attack.scene.get_target_areas().get_size_multiplier(),
                    attack_details.free_area_multiplier,
                )
                self.assertEqual(prepared_attack.attack_strategy.is_frozen, False)
                self.assertEqual(
                    prepared_attack.attack_strategy.max_pos_change_per_coord,
                    attack_details.max_pos_change_per_coord,
                )
                self.assertEqual(
                    prepared_attack.attack_learner.maxiter,
                    attack_details.maxiter,
                )
                self.assertEqual(
                    prepared_attack.attack_strategy.n_control_points,
                    attack_details.n_control_points,
                )
                self.assertEqual(
                    prepared_attack.attack_strategy.n_estim_viewpts,
                    attack_details.n_estim_viewpts,
                )
                self.assertEqual(
                    prepared_attack.scene.get_n_samples_for_type(SampleType.Train),
                    attack_details.n_train_viewpoints,
                )
                self.assertEqual(
                    prepared_attack.scene.get_n_samples_for_type(SampleType.Val),
                    attack_details.n_val_viewpoints,
                )
                self.assertEqual(
                    prepared_attack.attack_strategy.sigma0,
                    attack_details.sigma0,
                )
                self.assertEqual(
                    prepared_attack.predictor.get_name(),
                    attack_details.target_model_name,
                )
                self.assertEqual(
                    prepared_attack.scene.get_world_path(),
                    attack_details.target_scene_path,
                )
                self.assertEqual(
                    prepared_attack.attack_learner.attack_strategy,
                    prepared_attack.attack_strategy,
                )
                self.assertEqual(
                    prepared_attack.run_conf.run_repro_command, run_repro_command
                )
                self.assertEqual(prepared_attack.attack_learner.log_im, log_im)
                self.assertEqual(
                    prepared_attack.run_conf.run_repro_command, run_repro_command
                )
                self.assertEqual(
                    prepared_attack.scene._target_object_transform.get_transform_type().public_name,
                    attack_details.transform_type,
                )
                self.assertEqual(
                    prepared_attack.scene.get_n_samples_for_type(SampleType.Test),
                    attack_details.n_test_viewpoints,
                )
                if (
                    prepared_attack.scene._target_object_transform.get_transform_type()
                    == ObjectTransformType.VolumeBased
                ):
                    assert isinstance(
                        prepared_attack.scene._target_object_transform,
                        VolumeBasedObjectTransform,
                    )
                    volume_based_transform = (
                        prepared_attack.scene._target_object_transform
                    )

                    self.assertEqual(
                        volume_based_transform.get_n_steps_along_shortest_axis(),
                        attack_details.n_cubes_steps,
                    )
                self.assertEqual(
                    prepared_attack.attack_strategy.max_transform_change_amount_score,
                    attack_details.max_shape_change,
                )

                custom_mlflow_client_mock.get_experiment_id_by_name.assert_called_with(
                    attack_details.experiment_name
                )
            finally:
                prepared_attack.scene.destroy_showbase()

        non_float_replacements = {
            "cma_optimized_metric": "mean_delta_cropped_rmse",
            "experiment_name": TEST_EXPERIMENT_NAMES[1],
            "freeze_estim": "frozen_random",
            "target_model_name": "dummy_depth2",
            "target_scene_path": Path("scenes/room1.glb"),
            "transform_type": ObjectTransformType.VolumeBased.public_name,
            "eval_on_test": not attack_detail_dict["eval_on_test"],
        }

        for attack_detail_name in {None}.union(attack_detail_dict.keys()):
            if attack_detail_name is not None:
                attack_detail_val = attack_detail_dict[attack_detail_name]
                if isinstance(attack_detail_val, float) or isinstance(
                    attack_detail_val, int
                ):
                    attack_detail_val += 1
                else:
                    attack_detail_val = non_float_replacements[attack_detail_name]

                new_attack_detail_dict = attack_detail_dict | {
                    attack_detail_name: attack_detail_val
                }
            new_attack_detail_dict = attack_detail_dict

            with self.subTest(f"change_attack_detail_{attack_detail_name}"):
                mflow_mock_objs = _new_configure_mlflow_mock(experiment_exists=True)
                with mock.patch(
                    "threedattack.script_util._script_util._configure_mlflow",
                    mflow_mock_objs.configure_mlflow_fn_mock,
                ):
                    new_attack_details = AttackDetails(**new_attack_detail_dict)
                    assert_attack_objs_correctly_prepaired(
                        attack_details=new_attack_details,
                        log_im=False,
                        run_repro_command=["python", "repro.py"],
                        custom_mlflow_client_mock=mflow_mock_objs.custom_mlflow_client_mock,
                    )

        for log_im in [True, False]:
            with self.subTest(f"{log_im=}"):
                mflow_mock_objs = _new_configure_mlflow_mock(experiment_exists=True)
                with mock.patch(
                    "threedattack.script_util._script_util._configure_mlflow",
                    mflow_mock_objs.configure_mlflow_fn_mock,
                ):
                    assert_attack_objs_correctly_prepaired(
                        attack_details=self.attack_details,
                        log_im=log_im,
                        run_repro_command=["python", "repro.py"],
                        custom_mlflow_client_mock=mflow_mock_objs.custom_mlflow_client_mock,
                    )

        for i, run_repro_command in enumerate(
            [["python", "repro.py"], ["python", "repro2.py"]]
        ):
            with self.subTest(f"run_repro_command_{i}"):
                mflow_mock_objs = _new_configure_mlflow_mock(experiment_exists=True)
                with mock.patch(
                    "threedattack.script_util._script_util._configure_mlflow",
                    mflow_mock_objs.configure_mlflow_fn_mock,
                ):
                    assert_attack_objs_correctly_prepaired(
                        attack_details=self.attack_details,
                        log_im=False,
                        run_repro_command=run_repro_command,
                        custom_mlflow_client_mock=mflow_mock_objs.custom_mlflow_client_mock,
                    )

    def test_suggest_with_public_name(self):
        trial_mock = mock.Mock()
        CHOSEN_VALUE_PUBLIC_NAME = "val2_name"
        NAME = "myparam"

        def suggest_categorical(name: str, choices: list[str]) -> str:
            self.assertEqual(name, NAME)
            self.assertEqual(len(choices), 5)
            self.assertEqual(choices, list(sorted(choices)))
            return CHOSEN_VALUE_PUBLIC_NAME

        trial_mock.suggest_categorical = mock.Mock(side_effect=suggest_categorical)

        class Values(Enum):
            Val1 = ("val1_name",)
            Val2 = ("val2_name",)
            Val3 = ("val3_name",)
            Val4 = ("val4_name",)
            Val5 = ("val5_name",)

            def __init__(self, public_name: str):
                self.public_name = public_name

        chosen_value = suggest_with_public_name(
            trial=trial_mock, name=NAME, choices=Values
        )

        self.assertEqual(chosen_value, Values.Val2)
        trial_mock.suggest_categorical.assert_called_once()

    def test_attack_details_and_exact_origsigned_metrics_2_dict(self):
        run_name = "myrun99"
        extra_tags = {
            "tag1": 4,
            "tag2": "hello",
        }
        exact_metrics = {"exact_metric1": 2.3}
        result_dict = attack_details_and_exact_origsigned_metrics_2_dict(
            attack_details=self.attack_details,
            exact_origsigned_metrics=exact_metrics,
            run_name=run_name,
            extra_tags=extra_tags,
        )

        expected_key_count = (
            len(fields(AttackDetails))
            + len(extra_tags.keys())
            + len(exact_metrics.keys())
            + 2
        )
        self.assertEqual(len(result_dict.keys()), expected_key_count)
        self.assertTrue(
            all(
                key.startswith("params.")
                or key.startswith("metrics.")
                or key.startswith("tags.")
                or key.startswith("mlflow.")
                for key in result_dict.keys()
            )
        )
        self.assertTrue(any(key.startswith("params.") for key in result_dict.keys()))
        self.assertTrue(any(key.startswith("metrics.") for key in result_dict.keys()))
        self.assertTrue(any(key.startswith("tags.") for key in result_dict.keys()))
        self.assertTrue(any(key.startswith("mlflow.") for key in result_dict.keys()))

        self.assertEqual(
            result_dict["params.n_control_points"], self.attack_details.n_control_points
        )
        self.assertEqual(
            result_dict["params.target_scene_path"],
            str(self.attack_details.target_scene_path),
        )
        self.assertEqual(
            result_dict["params.experiment_name"],
            self.attack_details.experiment_name,
        )
        self.assertEqual(
            result_dict["params.eval_on_test"],
            self.attack_details.eval_on_test,
        )

        checked_metric_name = next(iter(exact_metrics.keys()))
        self.assertEqual(
            result_dict["metrics." + checked_metric_name],
            exact_metrics[checked_metric_name],
        )
        checked_tag_name = next(iter(extra_tags.keys()))
        self.assertEqual(
            result_dict["tags." + checked_tag_name],
            extra_tags[checked_tag_name],
        )
        self.assertEqual(result_dict["mlflow.run_name"], run_name)
        self.assertEqual(
            result_dict["mlflow.experiment_name"], self.attack_details.experiment_name
        )


def _new_configure_mlflow_mock(experiment_exists: bool) -> "_ConfigMlflowMocks":
    custom_mlflow_client_mock = mock.Mock(name="custom_mlflow_client_mock")
    if experiment_exists:
        experiment_id = ExperimentId("1234")
    else:
        experiment_id = None
    custom_mlflow_client_mock.get_experiment_id_by_name = mock.Mock(
        return_value=experiment_id, name="configure_mlflow_fn_mock"
    )

    configure_mlflow_fn_mock = mock.Mock(return_value=custom_mlflow_client_mock)

    return _ConfigMlflowMocks(
        configure_mlflow_fn_mock=configure_mlflow_fn_mock,
        custom_mlflow_client_mock=custom_mlflow_client_mock,
    )


@dataclass
class _ConfigMlflowMocks:
    custom_mlflow_client_mock: mock.Mock
    configure_mlflow_fn_mock: mock.Mock
