import math
import unittest
from typing import TYPE_CHECKING, Sequence
from unittest import mock

import numpy as np

from threedattack._typing import type_instance
from threedattack.dataset_model import (
    ExactSampleCounts,
    RGBsWithDepthsAndMasks,
    SamplesBase,
    SampleType,
    SampleTypeError,
)
from threedattack.losses import RawLossFn, log10_loss, rmse_loss
from threedattack.rendering_core import DepthsWithMasks
from threedattack.script_util import calculate_mean_losses_of_predictor_on_dataset
from threedattack.target_model import AsyncDepthPredictor
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *


class TestVerification(unittest.TestCase):
    def test_calculate_mean_losses_of_predictor_on_dataset_happy_path(self):
        n_train_samples = 7
        n_val_samples = 12

        training_rgbs = np.zeros(
            shape=newshape_im_rgbs(n=n_train_samples, h=5, w=8), dtype=np.float32
        )
        for i in range(n_train_samples):
            upd_im_rgbs(training_rgbs, n=i, value_=n_train_samples / (i + 0.3))
        validation_rgbs = np.zeros(
            shape=newshape_im_rgbs(n=n_val_samples, h=5, w=8), dtype=np.float32
        )
        for i in range(n_val_samples):
            upd_im_rgbs(validation_rgbs, n=i, value_=n_val_samples / (i + 0.3))

        training_gt_depths = np.ones(
            newshape_im_depthmaps(n=n_train_samples, h=5, w=8), dtype=np.float32
        )
        validation_gt_depths = (
            np.ones(newshape_im_depthmaps(n=n_val_samples, h=5, w=8), dtype=np.float32)
            * 3
        )

        training_masks = np.ones_like(training_gt_depths, dtype=np.bool_)
        validation_masks = np.ones_like(validation_gt_depths, dtype=np.bool_)

        all_training_rgbds = RGBsWithDepthsAndMasks(
            depths=training_gt_depths, masks=training_masks, rgbs=training_rgbs
        )
        all_validation_rgbds = RGBsWithDepthsAndMasks(
            depths=validation_gt_depths, masks=validation_masks, rgbs=validation_rgbs
        )

        all_rgbds_by_type = {
            SampleType.Train: all_training_rgbds,
            SampleType.Val: all_validation_rgbds,
        }

        counters_by_type = {SampleType.Train: 0, SampleType.Val: 0}

        depth_cap = 31
        predictor = _FakePredictor(self, depth_cap)
        dataset_mock = mock.Mock()
        dataset_mock.get_depth_cap = mock.Mock(return_value=depth_cap)

        def get_samples(
            idx: Sequence[int] | slice, sample_type: SampleType
        ) -> SamplesBase:
            counters_by_type[sample_type] += 1
            return SamplesBase(all_rgbds_by_type[sample_type][idx])

        dataset_mock.get_samples = get_samples
        dataset_mock.get_n_samples = mock.Mock(
            return_value=ExactSampleCounts(
                n_train_samples=n_train_samples,
                n_test_samples=0,
                n_val_samples=n_val_samples,
            )
        )

        batch_size = 3

        expected_losses = {
            (SampleType.Val, RawLossFn.Log10): float(
                np.mean(
                    log10_loss(
                        pred_depths=_FakePredictor.aligned_transform_fn(
                            validation_rgbs
                        ),
                        gt=DepthsWithMasks(
                            depths=validation_gt_depths, masks=validation_masks
                        ),
                    )
                )
            ),
            (SampleType.Val, RawLossFn.RMSE): float(
                np.mean(
                    rmse_loss(
                        pred_depths=_FakePredictor.aligned_transform_fn(
                            validation_rgbs
                        ),
                        gt=DepthsWithMasks(
                            depths=validation_gt_depths, masks=validation_masks
                        ),
                    )
                )
            ),
            (SampleType.Train, RawLossFn.Log10): float(
                np.mean(
                    log10_loss(
                        pred_depths=_FakePredictor.aligned_transform_fn(training_rgbs),
                        gt=DepthsWithMasks(
                            depths=training_gt_depths, masks=training_masks
                        ),
                    )
                )
            ),
        }

        expected_train_batch_count = math.ceil(n_train_samples / batch_size)
        expected_val_batch_count = math.ceil(n_val_samples / batch_size)

        actual_losses = calculate_mean_losses_of_predictor_on_dataset(
            predictor=predictor,
            dataset=dataset_mock,
            batch_size=batch_size,
            sample_types_and_fns=set(expected_losses.keys()),
        )

        self.assertEqual(set(actual_losses.keys()), set(expected_losses.keys()))

        for sample_type_and_loss_fn in expected_losses.keys():
            expected_loss = expected_losses[sample_type_and_loss_fn]
            actual_loss = actual_losses[sample_type_and_loss_fn]

            self.assertAlmostEqual(expected_loss, actual_loss, places=5)

        self.assertEqual(expected_train_batch_count, counters_by_type[SampleType.Train])
        self.assertEqual(expected_val_batch_count, counters_by_type[SampleType.Val])

    def test_calculate_mean_losses_of_predictor_on_dataset_dataset_empty(self):
        dataset = mock.Mock()

        def get_samples(*args, **kwargs):
            raise SampleTypeError()

        dataset.get_samples = mock.Mock(return_value=get_samples)
        dataset.get_n_samples = mock.Mock(
            return_value=ExactSampleCounts(
                n_train_samples=0, n_test_samples=0, n_val_samples=0
            )
        )
        predictor = mock.Mock()
        with self.assertRaises(SampleTypeError):
            calculate_mean_losses_of_predictor_on_dataset(
                predictor=predictor,
                dataset=dataset,
                batch_size=3,
                sample_types_and_fns={(SampleType.Train, RawLossFn.Log10)},
            )

    def test_calculate_mean_losses_of_predictor_on_dataset_nonpositive_batch_size(self):
        batch_sizes = [-3, 0]
        for batch_size in batch_sizes:
            with self.subTest(f"{batch_size=}"):
                dataset = mock.Mock()
                dataset.get_n_samples = mock.Mock(
                    return_value=ExactSampleCounts(
                        n_train_samples=100, n_test_samples=50, n_val_samples=76
                    )
                )
                predictor = mock.Mock()
                with self.assertRaises(ValueError):
                    calculate_mean_losses_of_predictor_on_dataset(
                        predictor=predictor,
                        dataset=dataset,
                        batch_size=batch_size,
                        sample_types_and_fns={(SampleType.Train, RawLossFn.Log10)},
                    )


class _FakePredictor:
    def __init__(self, test_case: unittest.TestCase, expected_depth_cap: float):
        self.test_case = test_case
        self.expected_depth_cap = expected_depth_cap

    @staticmethod
    def aligned_transform_fn(rgbs: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.mean(rgbs, axis=DIM_IM_C), axis=1) * 7.1

    def get_name(self) -> str:
        return "fake_predictor"

    def predict_native(self, rgbs: np.ndarray) -> "ImmediateNext":
        native_preds = self.aligned_transform_fn(rgbs) / 2
        return ImmediateNext(native_preds)

    def alignment_function(
        self,
        native_preds: np.ndarray,
        gt_depths: np.ndarray,
        masks: np.ndarray,
        depth_cap: float,
    ) -> np.ndarray:
        self.test_case.assertEqual(depth_cap, self.expected_depth_cap)
        return native_preds * 2


class ImmediateNext:
    def __init__(self, val: np.ndarray):
        self.val = val

    def get(self) -> np.ndarray:
        return self.val


if TYPE_CHECKING:
    v: AsyncDepthPredictor = type_instance(_FakePredictor)
