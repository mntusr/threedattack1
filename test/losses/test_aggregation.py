import unittest

import numpy as np

from threedattack.dataset_model import DepthsWithMasks, RGBsWithDepthsAndMasks
from threedattack.losses import (
    RawLossFn,
    calculate_loss_values,
    concat_losses,
    cropped_d1_loss,
    cropped_log10_loss,
    cropped_rmse_loss,
    d1_loss,
    get_loss_val_mean,
    get_loss_val_median,
    get_loss_val_min,
    idx_losses,
    log10_loss,
    rmse_loss,
    subtract_losses,
)
from threedattack.rendering_core import TwoDAreas
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *


class TestAggregation(unittest.TestCase):
    def test_calculate_loss_values_happy_path(self):
        rng = np.random.default_rng(50)
        N_IMAGES = 4

        gt_depths = np.ones(
            shape=newshape_im_depthmaps(n=N_IMAGES, h=11, w=17), dtype=np.float32
        )
        masks = np.ones_like(gt_depths, dtype=np.bool_)

        pred_depths = rng.uniform(1, 3, size=gt_depths.shape).astype(gt_depths.dtype)
        target_obj_areas = TwoDAreas(
            x_mins_including=np.arange(0, N_IMAGES),
            x_maxes_excluding=np.arange(0, N_IMAGES) + 6,
            y_mins_including=np.arange(0, N_IMAGES),
            y_maxes_excluding=np.arange(0, N_IMAGES) + 6,
        )

        actual_loss_values = calculate_loss_values(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            loss_fns=set(RawLossFn),
            target_obj_areas=target_obj_areas,
        )

        expected_loss_values = {
            RawLossFn.D1: d1_loss(
                pred_depths=pred_depths,
                gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            ),
            RawLossFn.RMSE: rmse_loss(
                pred_depths=pred_depths,
                gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            ),
            RawLossFn.Log10: log10_loss(
                pred_depths=pred_depths,
                gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            ),
            RawLossFn.CroppedD1: cropped_d1_loss(
                pred_depths=pred_depths,
                gt=DepthsWithMasks(depths=gt_depths, masks=masks),
                target_obj_areas=target_obj_areas,
            ),
            RawLossFn.CroppedRMSE: cropped_rmse_loss(
                pred_depths=pred_depths,
                gt=DepthsWithMasks(depths=gt_depths, masks=masks),
                target_obj_areas=target_obj_areas,
            ),
            RawLossFn.CroppedLog10: cropped_log10_loss(
                pred_depths=pred_depths,
                gt=DepthsWithMasks(depths=gt_depths, masks=masks),
                target_obj_areas=target_obj_areas,
            ),
        }

        _assert_allclose_for_all_losses(self, actual_loss_values, expected_loss_values)

    def test_calculate_loss_values_conflicting_losses(self):
        rng = np.random.default_rng(50)
        N_IMAGES = 4

        gt_depths = np.ones(
            shape=newshape_im_depthmaps(n=N_IMAGES, h=11, w=17), dtype=np.float32
        )
        masks = np.ones_like(gt_depths, dtype=np.bool_)

        pred_depths = rng.uniform(1, 3, size=gt_depths.shape).astype(gt_depths.dtype)

        with self.assertRaises(ValueError):
            calculate_loss_values(
                pred_depths=pred_depths,
                gt=DepthsWithMasks(depths=gt_depths, masks=masks),
                loss_fns=set(RawLossFn),
                target_obj_areas=None,
            )

    def test_get_loss_val_mean(self):
        loss_values = {
            RawLossFn.D1: np.array([0.2, 0.9, 0.7]),
            RawLossFn.RMSE: np.array([8, 5, 3]),
            RawLossFn.Log10: np.array([2, 10, 0.1]),
        }
        expected_means_dict = {
            key: float(np.mean(value)) for key, value in loss_values.items()
        }

        actual_means_dict = get_loss_val_mean(loss_values)

        for key in expected_means_dict.keys():
            self.assertAlmostEqual(expected_means_dict[key], actual_means_dict[key])

    def test_get_loss_val_median(self):
        loss_values = {
            RawLossFn.D1: np.array([0.2, 0.9, 0.7]),
            RawLossFn.RMSE: np.array([8, 5, 3]),
            RawLossFn.Log10: np.array([2, 10, 0.1]),
        }
        expected_medians_dict = {
            key: float(np.median(value)) for key, value in loss_values.items()
        }

        actual_means_dict = get_loss_val_median(loss_values)

        for key in expected_medians_dict.keys():
            self.assertAlmostEqual(expected_medians_dict[key], actual_means_dict[key])

    def test_get_loss_val_min(self):
        loss_values = {
            RawLossFn.D1: np.array([0.2, 0.9, 0.7]),
            RawLossFn.RMSE: np.array([8, 5, 3]),
            RawLossFn.Log10: np.array([2, 10, 0.1]),
        }
        expected_mins_dict = {
            key: float(np.min(value)) for key, value in loss_values.items()
        }

        actual_means_dict = get_loss_val_min(loss_values)

        for key in expected_mins_dict.keys():
            self.assertAlmostEqual(expected_mins_dict[key], actual_means_dict[key])

    def test_subtract_losses_happy_path(self):
        losses_left = {
            RawLossFn.D1: np.array([0.3]),
            RawLossFn.RMSE: np.array([5.1]),
            RawLossFn.Log10: np.array([6.1]),
        }
        losses_right = {
            RawLossFn.D1: np.array([0.1]),
            RawLossFn.RMSE: np.array([1.0]),
            RawLossFn.Log10: np.array([1.1]),
        }

        expected_result = {
            RawLossFn.D1: np.array([0.2]),
            RawLossFn.RMSE: np.array([4.1]),
            RawLossFn.Log10: np.array([5.0]),
        }

        actual_result = subtract_losses(losses_left, losses_right)

        _assert_allclose_for_all_losses(self, expected_result, actual_result)

    def test_subtract_losses_not_matching_keys(self):
        losses_left = {
            RawLossFn.D1: np.array([0.3]),
        }
        losses_right = {
            RawLossFn.D1: np.array([0.1]),
            RawLossFn.RMSE: np.array([1.0]),
        }

        with self.assertRaises(ValueError):
            subtract_losses(losses_left, losses_right)

    def test_concat_losses_happy_path(self):
        losses1 = {
            RawLossFn.D1: np.array([0.3]),
            RawLossFn.RMSE: np.array([5.1]),
            RawLossFn.Log10: np.array([6.1]),
        }
        losses2 = {
            RawLossFn.D1: np.array([0.1]),
            RawLossFn.RMSE: np.array([1.0]),
            RawLossFn.Log10: np.array([1.1]),
        }

        expected_result = {
            key: np.concatenate([losses1[key], losses2[key]], axis=DIM_SCALARS_N)
            for key in losses1.keys()
        }
        losses_concat = concat_losses(losses1, losses2)

        _assert_allclose_for_all_losses(self, expected_result, losses_concat)

    def test_concat_losses_not_matching_keys(self):
        losses_left = {
            RawLossFn.D1: np.array([0.3]),
        }
        losses_right = {
            RawLossFn.D1: np.array([0.1]),
            RawLossFn.RMSE: np.array([1.0]),
        }

        with self.assertRaises(ValueError):
            subtract_losses(losses_left, losses_right)

    def test_getitem(self):
        losses = {
            RawLossFn.D1: np.array([0.3, 0.5]),
            RawLossFn.Log10: np.array([6.1, 8.5]),
        }
        INDEX = 1

        actual_item = idx_losses(losses, INDEX)
        expected_item = {key: value[INDEX] for key, value in losses.items()}

        _assert_allclose_for_all_losses(self, actual_item, expected_item)


def _assert_allclose_for_all_losses(
    test_case: unittest.TestCase,
    losses1: dict[RawLossFn, np.ndarray],
    losses2: dict[RawLossFn, np.ndarray],
):
    test_case.assertEqual(losses1.keys(), losses2.keys())
    for key in losses1.keys():
        test_case.assertTrue(np.allclose(losses1[key], losses2[key], atol=1e-4))
