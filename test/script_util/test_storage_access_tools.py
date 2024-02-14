import io
import unittest
from pathlib import Path
from unittest import mock

from threedattack import SceneConfig
from threedattack.rendering_core import (
    DesiredViewpointCounts,
    ObjectTransformType,
    TwoDSize,
)
from threedattack.script_util import MlflowRunLoader, OptunaTrialLoader


class TestRunLoaders(unittest.TestCase):
    def setUp(self):
        def new_run_mock_with_number_only(number: int):
            new_run_mock = mock.Mock(name=f"run_mock_with_number_only_{number}")
            new_run_mock.info.number = number
            return new_run_mock

        self.mlflow_client_mock = mock.Mock(name="mlflow_client")
        self.experiment_name = "experiment65"
        self.run_id = "373476"

        self.experiment_mock = mock.Mock(name="experiment")
        self.experiment_mock.experiment_id = "675"
        self.run_mock = mock.Mock(name="run")
        self.run_mock.info = mock.Mock(name="run.info")
        self.run_mock.info.run_id = self.run_id
        self.run_mock.data = mock.Mock(name="run.data")
        self.run_mock.data.tags = {
            "mlflow.note.content": "Command: 'python' 'manual_attack.py' '--n-control-points' '191' '--n-estim-viewpts' '3' '--freeze-estim' '0' '1' '2' '--max-pos-change-per-coord' '0.4239080534313835' '--cma-optimized-metric' 'mean_delta_cropped_d1' '--sigma0' '0.22539067077356092' '--n-val-viewpoints' '200' '--n-train-viewpoints' '400' '--target-model-name' 'midas_large' '--target-scene-path' 'scenes\room1_subdivided3.glb' '--free-area-multiplier' '1.1200183315951973' '--experiment-name' 'Midas large optuna attack9' '--maxiter' '191' '--transform-type' 'volume_based' '--n-cubes-steps' '20' '--max-shape-change' '0.34837751590218197'"
        }

        self.scene_config = SceneConfig(
            world_path=Path("test_resources/test_scene.glb"),
            applied_transform=None,
            depth_cap=8.3,
            n_volume_sampling_steps_along_shortest_axis=20,
            object_transform_type=ObjectTransformType.VolumeBased,
            resolution=TwoDSize(x_size=800, y_size=600),
            target_size_multiplier=3.1,
            viewpt_counts=DesiredViewpointCounts(
                n_train_samples=2, n_val_samples=1, n_test_samples=0
            ),
        )
        scene_config_json_io = io.StringIO()
        self.scene_config.save_json(scene_config_json_io)
        scene_config_json = scene_config_json_io.getvalue()

        self.run_n_cubes_steps = 20

        def get_experiment_by_name(name: str) -> mock.Mock | None:
            if name == self.experiment_name:
                return self.experiment_mock
            else:
                return None

        def search_runs(
            experiment_ids: list[str], filter_string: str, max_results: int
        ):
            self.assertEqual(filter_string, 'attributes.run_name="run93"')
            self.assertEqual(experiment_ids, ["675"])
            return [self.run_mock]

        def download_artifacts(run_id: str, path: str, dst_path: str):
            scene_artifact_name = "best_solution.scene"
            self.assertEqual(path, scene_artifact_name)
            self.assertEqual(run_id, self.run_id)
            self.assertTrue(Path(dst_path).is_absolute())
            scene_path = Path(dst_path) / scene_artifact_name
            scene_path.write_text(scene_config_json)

            return str(scene_path)

        def get_run(run_id: str):
            self.assertEqual(run_id, self.run_id)

            return self.run_mock

        self.mlflow_client_mock.get_experiment_by_name = get_experiment_by_name
        self.mlflow_client_mock.search_runs = search_runs
        self.mlflow_client_mock.download_artifacts = download_artifacts
        self.mlflow_client_mock.get_run = get_run

        self.run_loader = MlflowRunLoader(
            client=self.mlflow_client_mock, experiment_name=self.experiment_name
        )
        self.run_name = "run93"
        self.trial_number = 103

        self.study_mock = mock.Mock(name="study")
        self.trial_mock = mock.Mock(name="trial")
        self.trial_mock.user_attrs = {"run_name": self.run_name}
        self.trial_mock.number = self.trial_number
        self.study_mock.trials = [
            new_run_mock_with_number_only(self.trial_number - 10),
            self.trial_mock,
            new_run_mock_with_number_only(self.trial_number + 1000),
            new_run_mock_with_number_only(self.trial_number + 1200),
        ]

        self.optuna_trial_loader = OptunaTrialLoader(
            mlflow_client=self.mlflow_client_mock,
            mlflow_experiment_name=self.experiment_name,
            optuna_study=self.study_mock,
        )

    def test_mlflow_get_modified_attack_details_n_test_samples_not_present(self):
        attack_details = self.run_loader.get_modified_attack_details(
            run_name=self.run_name, changes_dict={"maxiter": 71}
        )
        self.assertEqual(attack_details.n_cubes_steps, self.run_n_cubes_steps)
        self.assertEqual(attack_details.maxiter, 71)
        self.assertEqual(attack_details.n_test_viewpoints, 200)

    def test_mlflow_get_modified_attack_details_n_test_samples_present(self):
        self.run_mock.data.tags["mlflow.note.content"] += " '--n-test-viewpoints' '185'"
        attack_details = self.run_loader.get_modified_attack_details(
            run_name=self.run_name, changes_dict={"maxiter": 71}
        )
        self.assertEqual(attack_details.n_cubes_steps, self.run_n_cubes_steps)
        self.assertEqual(attack_details.maxiter, 71)
        self.assertEqual(attack_details.n_test_viewpoints, 185)

    def test_mlflow_get_run_result_scene_config(self):
        scene_config = self.run_loader.get_run_result_scene_config(
            run_name=self.run_name
        )

        actual_depth_cap = scene_config.depth_cap
        expected_depth_cap = self.scene_config.depth_cap

        self.assertAlmostEqual(actual_depth_cap, expected_depth_cap)

    def test_mlflow_get_trial_result_scene_happy_path(self):
        scene_config = self.optuna_trial_loader.get_trial_result_scene_config(
            trial_number=self.trial_number
        )
        actual_depth_cap = scene_config.depth_cap
        expected_depth_cap = self.scene_config.depth_cap

        self.assertAlmostEqual(actual_depth_cap, expected_depth_cap)

    def test_get_trial_result_scene_trial_not_found(self):
        with self.assertRaises(ValueError):
            self.optuna_trial_loader.get_trial_result_scene_config(
                trial_number=self.trial_number + 1
            )

    def test_optuna_get_modified_attack_details_happy_path(self):
        attack_details = self.optuna_trial_loader.get_modified_attack_details(
            trial_number=self.trial_number, changes_dict={"maxiter": 71}
        )
        self.assertEqual(attack_details.n_cubes_steps, self.run_n_cubes_steps)
        self.assertEqual(attack_details.maxiter, 71)
