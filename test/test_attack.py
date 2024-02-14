import itertools
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Callable, TextIO
from unittest import mock

import numpy as np

import threedattack.mlflow_util
from threedattack import (
    AdvAttackLearner,
    AttackStrategy,
    CMAEvolutionStrategyArgs,
    ExactMetrics,
    FirstStrategy,
    GenerationEvaluationResult,
    LoggingFreqFunction,
    Scene,
    SolutionInfo,
    StepLoggingFreqFunction,
    TransformChangeAmountScorePenality,
    get_target_n_bodies_penality,
)
from threedattack._typing import type_instance
from threedattack.losses import LossDerivationMethod, LossPrecision, RawLossFn
from threedattack.mlflow_util import ExperimentId, RunConfig, RunId
from threedattack.rendering_core import (
    ObjectTransformType,
    ScaledStandingAreas,
    ThreeDSize,
)
from threedattack.target_model import AsyncDepthPredictor
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *


class TestAdvAttackLearner(unittest.TestCase):
    def test_learn_adv_sample(self):
        for log_im, eval_on_test, result_metric_name in itertools.product(
            [True, False], [True, False], [None, TestAttackStrategy.TARGET_METRIC_NAME]
        ):
            with self.subTest(f"{log_im=};{eval_on_test=};{result_metric_name=}"):
                RUN_ID = RunId("45")
                RUN_CONFIG = RunConfig(
                    run_repro_command=["myscript.py", "2"],
                    experiment_id=ExperimentId("0"),
                    run_params={"param1": 17},
                )
                RUN_NAME = "testrun"
                SERIALIZED_SCENE = "serialized scene text"

                custom_mlflow_client_mock = mock.Mock()
                custom_mlflow_client_mock.create_run = mock.Mock(return_value=RUN_ID)
                custom_mlflow_client_mock.get_run_name = mock.Mock(
                    return_value=RUN_NAME
                )
                custom_mlflow_client_mock.log_text = mock.Mock(
                    name="custom_mlflow_client.log_text"
                )

                test_strategy = TestAttackStrategy(
                    testcase=self,
                    client_mock=custom_mlflow_client_mock,
                    run_config=RUN_CONFIG,
                    expected_eval_on_test_conf=eval_on_test,
                )
                attack_learner = AdvAttackLearner(
                    attack_strategy=test_strategy,
                    mlflow_client=custom_mlflow_client_mock,
                    maxiter=49,
                    log_im=log_im,
                    eval_on_test=eval_on_test,
                )

                def save_scene(target: TextIO | Path):
                    self.assertNotIsInstance(target, Path)
                    assert not isinstance(target, Path)
                    target.write(SERIALIZED_SCENE)

                scene_mock = mock.Mock(name="scene")
                scene_mock.save = save_scene
                scene_mock.set_target_transform = mock.Mock(
                    name="scene.set_target_transform"
                )

                learning_results = attack_learner.learn_adv_sample(
                    scene=scene_mock,
                    console_logging_freq=StepLoggingFreqFunction(
                        steps=[], freqencies=[1]
                    ),
                    predictor=mock.Mock(),
                    shared_info_calc_logging_freq=StepLoggingFreqFunction(
                        steps=[], freqencies=[7]
                    ),
                    run_config=RUN_CONFIG,
                )

                custom_mlflow_client_mock.set_terminated.assert_called_with(
                    run_id=RUN_ID
                )

                self._asssert_im_log_call_count(
                    custom_mlflow_client_mock.log_rgbs, log_im
                )
                self._asssert_im_log_call_count(
                    custom_mlflow_client_mock.log_gt_depths_and_preds, log_im
                )
                self.assertTrue(test_strategy.apply_final_called)
                self.assertEqual(
                    learning_results.n_pred_calls_total,
                    test_strategy.get_expected_n_pred_calls_total(),
                )
                self.assertGreater(
                    len(learning_results.best_solution_exact_origsigned_metrics.keys()),
                    0,
                )
                self.assertGreater(
                    len(
                        learning_results.best_solution_exact_minimizably_signed_metrics.keys()
                    ),
                    0,
                )
                self.assertTrue(
                    any(
                        learning_results.best_solution_exact_origsigned_metrics[key]
                        != learning_results.best_solution_exact_minimizably_signed_metrics[
                            key
                        ]
                        for key in learning_results.best_solution_exact_origsigned_metrics.keys()
                    )
                )
                self.assertEqual(learning_results.run_name, RUN_NAME)
                self.assertEqual(
                    learning_results.final_change_amount_score,
                    TestAttackStrategy.CHANGE_AMOUNT_SCORE,
                )
                self.assertEqual(custom_mlflow_client_mock.log_text.call_count, 1)

                logged_scene_data = custom_mlflow_client_mock.log_text.call_args_list[
                    0
                ].kwargs["data"]
                self.assertIsNone(scene_mock.set_target_transform.call_args[0][0])
                self.assertEqual(logged_scene_data, SERIALIZED_SCENE)
                self.assertTrue(test_strategy.apply_solution_transform_called)

    def _asssert_im_log_call_count(self, im_logging_fn_mock: mock.Mock, log_im: bool):
        if log_im:
            self.assertGreater(im_logging_fn_mock.call_count, 1)
        else:
            self.assertEqual(im_logging_fn_mock.call_count, 1)

    def test_get_cma_config(self):
        expected_maxiter = 20

        strategy_provided_cma_config = {
            "x0": [1.0] * 2,
            "sigma0": 0.1,
            "inopts": {"bounds": [[0.5], [1.5]], "tolfunhist": 1e-3},
        }

        strategy_mock = mock.Mock()
        strategy_mock.get_cma_config = mock.Mock(
            return_value=strategy_provided_cma_config
        )

        attack_learner = AdvAttackLearner(
            attack_strategy=strategy_mock,
            maxiter=expected_maxiter,
            mlflow_client=mock.Mock(),
            log_im=False,
            eval_on_test=False,
        )

        provided_maxiter = attack_learner._get_cma_config()["inopts"].get(
            "maxiter", None
        )
        self.assertEqual(expected_maxiter, provided_maxiter)


class TestFirstStrategy(unittest.TestCase):
    def setUp(self):
        self.MAX_TRANSFORM_CHANGE_AMOUNT_SCORE = 0.7
        self.VALID_PARAMS = {
            "n_control_points": 9,
            "n_estim_viewpts": 8,
            "freeze_estim": "free",
            "sigma0": 0.5,
            "optimized_metric": "median_delta_rmse",
            "max_pos_change_per_coord": 0.7,
            "max_transform_change_amount_score": self.MAX_TRANSFORM_CHANGE_AMOUNT_SCORE,
        }
        self.optimized_metric_in_valid_params = (
            LossDerivationMethod.MedianDelta,
            RawLossFn.RMSE,
        )
        self.INVALID_PARAMS = {
            "n_control_points": 0,
            "sigma0": 0,
            "optimized_metric": "invalid_metric",
            "max_pos_change_per_coord": 0,
        }

    def test_init_happy_path(self):
        strategy = FirstStrategy(**self.VALID_PARAMS)

        self.assertEqual(
            strategy.n_control_points, self.VALID_PARAMS["n_control_points"]
        )
        self.assertEqual(strategy.sigma0, self.VALID_PARAMS["sigma0"])
        self.assertEqual(
            strategy.optimized_metric, self.optimized_metric_in_valid_params
        )
        self.assertFalse(strategy.is_frozen)
        self.assertEqual(strategy.estim_viewpt_idx_list, None)
        self.assertEqual(
            strategy.max_transform_change_amount_score,
            self.MAX_TRANSFORM_CHANGE_AMOUNT_SCORE,
        )

    def test_freeze_estim_validation(self):
        for invalid_freeze_estim in ["invalid", ["hello"], 0]:
            with self.subTest(f"{invalid_freeze_estim=}"):
                params = self.VALID_PARAMS | {"freeze_estim": invalid_freeze_estim}
                with self.assertRaises(ValueError):
                    FirstStrategy(**params)

    def test_freeze_handling(self):
        freeze_confs: list[tuple[object, bool, list[int] | None]] = [
            ("free", False, None),
            ("frozen_random", True, None),
            (
                [0] * self.VALID_PARAMS["n_estim_viewpts"],
                True,
                [0] * self.VALID_PARAMS["n_estim_viewpts"],
            ),
        ]

        for (
            freeze_estim,
            expected_is_frozen,
            expected_estim_viewpt_idx_list,
        ) in freeze_confs:
            with self.subTest(str(freeze_estim).replace(" ", "")):
                params = self.VALID_PARAMS | {"freeze_estim": freeze_estim}
                strategy = FirstStrategy(**params)

                self.assertEqual(strategy.is_frozen, expected_is_frozen)
                self.assertEqual(
                    strategy.estim_viewpt_idx_list, expected_estim_viewpt_idx_list
                )

    def test_get_cma_config(self):
        current_params = self.VALID_PARAMS.copy()
        strategy = FirstStrategy(**current_params)
        cma_config = strategy.get_cma_config()
        self.assertEqual(len(cma_config["inopts"]["bounds"][0]), 1)
        self.assertEqual(len(cma_config["inopts"]["bounds"][1]), 1)
        self.assertAlmostEqual(cma_config["inopts"]["bounds"][0][0], 0)
        self.assertAlmostEqual(cma_config["inopts"]["bounds"][1][0], 1)
        self.assertAlmostEqual(cma_config["inopts"]["tolfunhist"], 1e-3)
        self.assertAlmostEqual(cma_config["sigma0"], self.VALID_PARAMS["sigma0"])
        self.assertTrue(
            np.allclose(
                np.array(cma_config["x0"]),
                np.full(
                    fill_value=0.5, shape=self.VALID_PARAMS["n_control_points"] * 3 * 2
                ),
            )
        )

    def test_init_invalid_val(self):
        for key in self.INVALID_PARAMS.keys():
            with self.subTest(key):
                with self.assertRaises(ValueError):
                    current_params = self.VALID_PARAMS.copy()
                    current_params[key] = self.INVALID_PARAMS[key]

                    FirstStrategy(**current_params)

    def test_call_count_calculation(self):
        N_SOLUTIONS = 10
        strategy = FirstStrategy(**self.VALID_PARAMS)

        actual_call_count = strategy._get_n_pred_calls_for_generation(
            xes=[np.arange(5)] * N_SOLUTIONS
        )

        expected_call_count = N_SOLUTIONS * strategy.n_estim_viewpts

        self.assertEqual(actual_call_count, expected_call_count)

    def test_get_vector_field_spec_from_x_happy_path(self):
        strategy = FirstStrategy(**self.VALID_PARAMS)

        target_area = ScaledStandingAreas(
            original_size=ThreeDSize(x_size=5.1, y_size=3.0, z_size=1.9),
            size_multiplier=2,
            extra_offset_after_size_mult=0.4,
        )

        x = np.array([0.1, 0.9, 0.6, 0.3, 0.2, 0.8] + [0.7, 0.1, 0.8, 0.1, 0.7, 0.4])

        rel_points = x[0:6].reshape(newshape_points_space(n=2))
        rel_vectors = (x[6:12].reshape(newshape_points_space(n=2)) - 0.5) * 2

        expected_control_points = target_area.get_full_area(
            origin_of_obj=None,
            include_outside_offset=False,
        ).make_rel_points_absolute(rel_points)
        expected_vectors = (
            rel_vectors
            * target_area.get_original_area(origin_of_obj=None).get_min_size()
            * self.VALID_PARAMS["max_pos_change_per_coord"]
        )

        actual_vector_field = strategy._get_vector_field_spec_from_x(
            x=x, target_area=target_area
        )
        self.assertTrue(
            np.allclose(expected_control_points, actual_vector_field.control_points)
        )
        self.assertTrue(np.allclose(expected_vectors, actual_vector_field.vectors))

    def test_get_vector_field_spec_from_x_invalid_x(self):
        strategy = FirstStrategy(**self.VALID_PARAMS)
        target_area = ScaledStandingAreas(
            original_size=ThreeDSize(x_size=5.1, y_size=3.0, z_size=1.9),
            size_multiplier=2,
            extra_offset_after_size_mult=0.4,
        )
        cases: list[tuple[str, np.ndarray]] = [
            ("invalid_modulo", np.linspace(0, 1, 13)),
            ("empty_solution", np.array([], dtype=np.float32)),
        ]
        for subtest_name, x in cases:
            with self.subTest(subtest_name):
                with self.assertRaises(ValueError):
                    strategy._get_vector_field_spec_from_x(x=x, target_area=target_area)


class TestTransformChangeAmountScorePenality(unittest.TestCase):
    def test_mesh_transform(self):
        THRESHOLD = 0.7
        penality_fn = TransformChangeAmountScorePenality(
            threshold=THRESHOLD, transform_type=ObjectTransformType.MeshBased
        )

        self.assertAlmostEqual(penality_fn.get_penality(THRESHOLD + 1), 0, places=4)
        self.assertEqual(
            penality_fn.get_transform_type(), ObjectTransformType.MeshBased
        )
        self.assertEqual(penality_fn.get_threshold(), THRESHOLD)

    def test_volume_transform_getters(self):
        THRESHOLD = 0.7
        penality_fn = TransformChangeAmountScorePenality(
            threshold=THRESHOLD, transform_type=ObjectTransformType.MeshBased
        )

        self.assertAlmostEqual(penality_fn.get_penality(THRESHOLD + 1), 0, places=4)
        self.assertEqual(
            penality_fn.get_transform_type(), ObjectTransformType.MeshBased
        )
        self.assertEqual(penality_fn.get_threshold(), THRESHOLD)

    def test_volume_transform_score_below_threshold(self):
        THRESHOLD = 0.7
        penality_fn = TransformChangeAmountScorePenality(
            threshold=THRESHOLD, transform_type=ObjectTransformType.VolumeBased
        )

        self.assertAlmostEqual(penality_fn.get_penality(THRESHOLD / 2), 0, places=4)

    def test_volume_transform_score_above_threshold(self):
        THRESHOLD = 0.7
        penality_fn = TransformChangeAmountScorePenality(
            threshold=THRESHOLD, transform_type=ObjectTransformType.VolumeBased
        )

        score = THRESHOLD + 0.1

        expected_penality = 50 + (score - THRESHOLD) * 30

        self.assertAlmostEqual(
            penality_fn.get_penality(score), expected_penality, places=4
        )


class TestAttackStrategy:
    TARGET_METRIC_NAME = "target_minimizable_metric"
    EXPECTED_BEST_X0 = 0.5
    EXPECTED_BEST_X1 = 1.5
    SCENE_SPECIFIC_STATE = "mystate"
    TARGET_METRIC_VAL = 31
    PRED_CALL_MULTIPLIER = 13
    CHANGE_AMOUNT_SCORE = 0.2

    def __init__(
        self,
        testcase: unittest.TestCase,
        expected_eval_on_test_conf: bool,
        client_mock: mock.Mock,
        run_config: RunConfig,
    ):
        self.apply_final_called = False
        self.testcase = testcase
        self.shared_info_calc_called = False
        self.client_mock = client_mock
        self.run_config = run_config
        self.stored_pred_calls_for_generations: dict[str, int] = dict()
        self.apply_solution_transform_called = False
        self.expected_eval_on_test_conf = expected_eval_on_test_conf

    def get_expected_n_pred_calls_total(self) -> int:
        return sum(self.stored_pred_calls_for_generations.values())

    def calc_shared_info_for_scene(
        self,
        scene: Scene,
        predictor: AsyncDepthPredictor,
        eval_on_test: bool,
        progress_logger: Callable[[str], None],
        logging_freq_fn: LoggingFreqFunction,
        seed: int,
    ) -> str:
        self.shared_info_calc_called = True

        self.testcase.assertEqual(eval_on_test, self.expected_eval_on_test_conf)

        self.client_mock.create_run.assert_called_with(run_config=self.run_config)
        return self.SCENE_SPECIFIC_STATE

    def get_cma_config(self) -> CMAEvolutionStrategyArgs:
        return {
            "x0": [1.0] * 2,
            "sigma0": 0.1,
            "inopts": {"bounds": [[0.5], [1.5]], "tolfunhist": 1e-3},
        }

    def evaluate_generation(
        self,
        xes: list[np.ndarray],
        scene: Scene,
        shared_info_for_scene: str,
        predictor: AsyncDepthPredictor,
        generation_seed: int,
    ) -> GenerationEvaluationResult:
        solution_infos: list[SolutionInfo] = []

        for x in xes:
            self.testcase.assertTrue(generation_seed >= 0)
            self.testcase.assertEqual(shared_info_for_scene, self.SCENE_SPECIFIC_STATE)

            myloss = TestAttackStrategy._fitness_fn(x)

            self.client_mock.set_terminated.assert_not_called()

            solution_infos.append(
                SolutionInfo(
                    fitness=myloss,
                    cam_proj_spec=mock.Mock(),
                    metrics={"myloss": myloss},
                    pred_depths_aligned=mock.Mock(),
                    rgbds=mock.Mock(),
                    x=x,
                    change_amount_score=self.CHANGE_AMOUNT_SCORE,
                )
            )

        n_pred_calls = TestAttackStrategy.PRED_CALL_MULTIPLIER * len(xes)

        self.stored_pred_calls_for_generations[str(xes)] = n_pred_calls

        return GenerationEvaluationResult(
            solution_infos=solution_infos,
            total_pred_calls=n_pred_calls,
        )

    @staticmethod
    def _fitness_fn(x: np.ndarray):
        """
        The fitness function of this test strategy.

        Parameters
        ----------
        x
            The solution to evaluete. Format: ``SVals::Float[v=2]``
        """
        return abs(x[0] - 0) + abs(x[1] - 2)

    def get_metrics_exact(
        self,
        x: np.ndarray,
        scene: Scene,
        predictor: AsyncDepthPredictor,
        eval_on_test: bool,
        shared_info_for_scene: str,
        progress_logger: Callable[[str], None],
        logging_freq_fn: LoggingFreqFunction,
    ) -> ExactMetrics:
        self.testcase.assertEqual(shared_info_for_scene, self.SCENE_SPECIFIC_STATE)
        self.testcase.assertTrue(match_scalars_float(x, shape={"n": 2}))

        self.testcase.assertEqual(eval_on_test, self.expected_eval_on_test_conf)

        self.apply_final_called = True

        x0 = x[0]
        x1 = x[1]

        self.testcase.assertAlmostEqual(x0, self.EXPECTED_BEST_X0, places=4)
        self.testcase.assertAlmostEqual(x1, self.EXPECTED_BEST_X1, places=4)
        self.client_mock.log_metrics.assert_called()

        return ExactMetrics(
            actual_metrics={
                "d": 4,
                "e": 9,
                self.TARGET_METRIC_NAME: -self.TARGET_METRIC_VAL,
            },
            minimizably_signed_metrics={
                "d": 4,
                "e": 9,
                self.TARGET_METRIC_NAME: self.TARGET_METRIC_VAL,
            },
        )

    def get_strat_repr(self) -> str:
        return "dummy"

    def apply_solution_transform(self, x: np.ndarray, scene: Scene) -> None:
        self.testcase.assertAlmostEqual(x[0], self.EXPECTED_BEST_X0, places=4)
        self.testcase.assertAlmostEqual(x[1], self.EXPECTED_BEST_X1, places=4)
        self.apply_solution_transform_called = True


class TestFunction(unittest.TestCase):
    def test_get_target_n_bodies_penality_happy_path(self):
        actual_penality = get_target_n_bodies_penality(4)
        expected_penality = 150
        self.assertEqual(actual_penality, expected_penality)

    def test_get_target_n_bodies_penality_invaid_n_bodies(self):
        invalid_vals = [0, -1]
        for invalid_val in invalid_vals:
            with self.subTest(f"{invalid_val=}"):
                with self.assertRaises(ValueError):
                    get_target_n_bodies_penality(invalid_val)


if TYPE_CHECKING:
    v: AttackStrategy = type_instance(TestAttackStrategy)
