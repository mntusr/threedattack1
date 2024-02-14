import unittest
from typing import Any, Mapping
from unittest import mock

import numpy as np

from threedattack.dataset_model import CamProjSpec, RGBsWithDepthsAndMasks
from threedattack.mlflow_util import CustomMlflowClient, ExperimentId, RunConfig, RunId
from threedattack.mlflow_util._mlflow_client import (
    _EXPERIMENT_DESCRIPTION_TAG,
    RUN_DESCRIPTION_TAG,
    _calculate_experiment_description_text,
)
from threedattack.tensor_types.npy import *


class TestCustomMlflowClient(unittest.TestCase):
    def setUp(self):
        self.mlflow_client_mock = mock.Mock()
        self.test_client = CustomMlflowClient(self.mlflow_client_mock)

    def test_get_experiment_happy_path(self):
        EXPERIMENT_NAME = "MyExperiment"
        EXISTING_EXPERIMENT_ID = "13"

        experiment_mock = mock.Mock()
        experiment_mock.experiment_id = EXISTING_EXPERIMENT_ID

        self.mlflow_client_mock.get_experiment_by_name = mock.Mock(
            return_value=experiment_mock
        )

        actual_experiment_id = self.test_client.get_experiment_id_by_name(
            experiment_name=EXPERIMENT_NAME,
        )

        self.assertEqual(actual_experiment_id.experiment_id, EXISTING_EXPERIMENT_ID)
        self.mlflow_client_mock.create_experiment.assert_not_called()

    def test_get_experiment_experiment_does_not_exist(self):
        EXPERIMENT_NAME = "MyExperiment"
        self.mlflow_client_mock.get_experiment_by_name = mock.Mock(return_value=None)

        with self.assertRaises(ValueError):
            self.test_client.get_experiment_id_by_name(
                experiment_name=EXPERIMENT_NAME,
            )

    def test_calculate_experiment_description_text(self):
        data1 = {"a": "3", "b": "5"}
        data2 = {"b": "5", "a": "3"}

        text1 = _calculate_experiment_description_text(data1)
        text2 = _calculate_experiment_description_text(data2)

        self.assertEqual(text1, text2)
        self.assertIn("3", text1)
        self.assertIn("5", text1)

    def test_create_run(self):
        EXPERIMENT_ID = ExperimentId("27")
        RUN_ID = RunId("49")
        PARAMS = {"param1": 9, "param2": 7}
        PYTHON_ARGS = ["prog.py", "389"]

        run_mock = mock.Mock()
        run_mock.info = mock.Mock()
        run_mock.info.run_id = RUN_ID.run_id

        experiment_mock = mock.Mock()
        experiment_mock.experiment_id = EXPERIMENT_ID.experiment_id

        self.mlflow_client_mock.get_experiment = mock.Mock(return_value=experiment_mock)
        self.mlflow_client_mock.create_run = mock.Mock(return_value=run_mock)

        self.test_client.create_run(
            run_config=RunConfig(
                experiment_id=EXPERIMENT_ID,
                run_repro_command=["python", "prog.py", "389"],
                run_params=PARAMS,
            ),
        )

        self.mlflow_client_mock.create_run.assert_called_once()
        create_run_args: Mapping[
            str, Any
        ] = self.mlflow_client_mock.create_run.call_args.kwargs
        self.assertEqual(create_run_args["experiment_id"], EXPERIMENT_ID.experiment_id)
        run_tags = create_run_args["tags"]
        run_description = run_tags[RUN_DESCRIPTION_TAG]
        for python_arg in PYTHON_ARGS:
            self.assertIn(python_arg, run_description)

        expected_log_param_calls = [
            mock.call(run_id=RUN_ID.run_id, key=name, value=value, synchronous=True)
            for name, value in PARAMS.items()
        ]
        self.mlflow_client_mock.log_param.assert_has_calls(
            expected_log_param_calls, any_order=True
        )

    def test_log_metrics_more_than_zero_metric(self):
        RUN_ID = RunId("49")
        GENERATION_IDX = 13
        self.test_client.log_metrics(
            run_id=RUN_ID, generation_idx=GENERATION_IDX, metrics={"a": 2, "b": 5}
        )

        metric_log_calls = [
            mock.call(run_id=RUN_ID.run_id, key="a", step=GENERATION_IDX, value=2),
            mock.call(run_id=RUN_ID.run_id, key="b", step=GENERATION_IDX, value=5),
        ]

        self.mlflow_client_mock.log_metric.assert_has_calls(metric_log_calls)

    def test_log_metrics_no_metric(self):
        RUN_ID = RunId("49")
        GENERATION_IDX = 13
        self.test_client.log_metrics(
            run_id=RUN_ID, generation_idx=GENERATION_IDX, metrics=dict()
        )

        self.mlflow_client_mock.log_metric.assert_not_called()

    def test_log_gt_depths_and_preds_more_than_zero_metric(self):
        RUN_ID = RunId("49")
        GENERATION_IDX = 13
        N_SAMPLES = 3
        IM_SHAPE = {"n": N_SAMPLES, "h": 5, "w": 7}
        self.test_client.log_gt_depths_and_preds(
            run_id=RUN_ID,
            cam_proj_spec=CamProjSpec(
                im_left_x_val=-1,
                im_right_x_val=1,
                im_bottom_y_val=1,
                im_top_y_val=-1,
                proj_mat=np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32
                ),
            ),
            generation_idx=GENERATION_IDX,
            pred_depths_aligned=np.ones(
                newshape_im_depthmaps(**IM_SHAPE), dtype=np.float32
            ),
            rgbds=RGBsWithDepthsAndMasks(
                rgbs=np.ones(newshape_im_rgbs(**IM_SHAPE), dtype=np.float32),
                depths=np.ones(newshape_im_depthmaps(**IM_SHAPE), dtype=np.float32),
                masks=np.ones(newshape_im_depthmasks(**IM_SHAPE), dtype=np.bool_),
            ),
        )

        self.assertEqual(self.mlflow_client_mock.log_figure.call_count, N_SAMPLES)

    def test_log_gt_depths_and_preds(self):
        RUN_ID = RunId("49")
        GENERATION_IDX = 13
        self.test_client.log_metrics(
            run_id=RUN_ID, generation_idx=GENERATION_IDX, metrics=dict()
        )

        self.mlflow_client_mock.log_figure.assert_not_called()

    def test_log_rgbs_more_than_zero_images(self):
        RUN_ID = RunId("49")
        GENERATION_IDX = 13
        N_SAMPLES = 3
        IM_SHAPE = {"n": N_SAMPLES, "h": 5, "w": 7}
        self.test_client.log_rgbs(
            run_id=RUN_ID,
            generation_idx=GENERATION_IDX,
            rgbs=np.ones(newshape_im_rgbs(**IM_SHAPE), dtype=np.float32),
        )

        self.assertEqual(self.mlflow_client_mock.log_figure.call_count, 1)

    def test_log_rgbs_zero_images(self):
        RUN_ID = RunId("49")
        GENERATION_IDX = 13
        IM_SHAPE = {"n": 0, "h": 5, "w": 7}
        self.test_client.log_rgbs(
            run_id=RUN_ID,
            generation_idx=GENERATION_IDX,
            rgbs=np.ones(newshape_im_rgbs(**IM_SHAPE), dtype=np.float32),
        )

        self.mlflow_client_mock.log_figure.assert_not_called()

    def test_get_run_name_run_has_name(self):
        EXPECTED_RUN_NAME = "run1"
        RUN_ID = RunId("0123")

        run_mock = mock.Mock()
        run_mock.info = mock.Mock()
        run_mock.info.run_name = EXPECTED_RUN_NAME

        self.mlflow_client_mock.get_run = mock.Mock(return_value=run_mock)

        actual_run_name = self.test_client.get_run_name(RUN_ID)

        self.assertEqual(actual_run_name, EXPECTED_RUN_NAME)
        self.mlflow_client_mock.get_run.assert_called_with(run_id=RUN_ID.run_id)

    def test_get_run_name_run_has_no_name(self):
        RUN_ID = RunId("0123")

        run_mock = mock.Mock()
        run_mock.info = mock.Mock()
        run_mock.info.run_name = None

        self.mlflow_client_mock.get_run = mock.Mock(return_value=run_mock)

        actual_run_name = self.test_client.get_run_name(RUN_ID)

        self.assertIsInstance(actual_run_name, str)
        self.mlflow_client_mock.get_run.assert_called_with(run_id=RUN_ID.run_id)

    def test_log_text_happy_path(self):
        run_id = RunId("123")
        data = "abcdefg"
        name = "myfile1.abc"
        self.test_client.log_text(run_id=run_id, data=data, name=name)

        self.mlflow_client_mock.log_text.assert_called_with(
            run_id=run_id.run_id, artifact_file=name, text=data
        )

    def test_log_text_invalid_name(self):
        run_id = RunId("123")
        data = "abcdefg"
        invalid_names = [
            "myfile1./abc",
            "myfile1@.abc",
        ]
        for invalid_name in invalid_names:
            with self.subTest(invalid_name):
                with self.assertRaises(ValueError):
                    self.test_client.log_text(
                        run_id=run_id, data=data, name=invalid_name
                    )

                self.mlflow_client_mock.log_text.assert_not_called()
