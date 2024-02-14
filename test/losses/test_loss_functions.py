import unittest

import numpy as np

from threedattack.dataset_model import DepthsWithMasks
from threedattack.losses import (
    cropped_d1_loss,
    cropped_d1_map,
    cropped_log10_loss,
    cropped_log10_map,
    cropped_rmse_loss,
    cropped_se_map,
    d1_loss,
    d1_map,
    log10_loss,
    log10_map,
    rmse_loss,
    se_map,
)
from threedattack.losses._loss_functions import _masked_samplewise_mean
from threedattack.rendering_core import TwoDAreas, TwoDSize, get_twod_area_masks
from threedattack.tensor_types.npy import *


class TestLossFunctions(unittest.TestCase):
    def test_d1_loss(self):
        num_depths = 5

        pred_depths = np.ones(
            newshape_im_depthmaps(n=num_depths, h=7, w=9), dtype=np.float32
        )
        gt_depths = np.ones_like(pred_depths, dtype=np.float32)
        masks = np.ones_like(pred_depths, dtype=np.bool_)

        upd_im_depthmaps(pred_depths, n=1, h=slice(None, 1), value_=30)
        upd_im_depthmaps(pred_depths, n=2, h=slice(None, 2), value_=30)
        upd_im_depthmaps(pred_depths, n=3, h=slice(None, 3), value_=30)
        upd_im_depthmaps(pred_depths, n=4, h=slice(None, 4), value_=30)

        upd_im_depthmasks(masks, n=4, h=slice(None, 4), value_=0)

        losses = d1_loss(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
        )

        self.assertTrue(match_scalars_float(losses, shape={"n": num_depths}))

        loss_float_list = [float(loss) for loss in losses]

        self.assertAlmostEqual(loss_float_list[0], loss_float_list[-1], places=6)
        self.assertGreater(loss_float_list[0], loss_float_list[1])
        self.assertGreater(loss_float_list[1], loss_float_list[2])

    def test_cropped_d1_loss(self):
        rng = np.random.default_rng(30)
        n_ims = 5
        im_shape = TwoDSize(x_size=41, y_size=40)
        pred_depths = np.ones(
            newshape_im_depthmaps(n=n_ims, h=im_shape.y_size, w=im_shape.x_size),
            dtype=np.float32,
        )
        gt_depths = rng.choice([1, 2], size=pred_depths.shape).astype(np.float32)
        masks = rng.choice([0, 1], size=pred_depths.shape).astype(np.bool_)

        target_obj_areas = TwoDAreas(
            x_maxes_excluding=np.arange(0, n_ims) + 9,
            x_mins_including=np.arange(0, n_ims) + 3,
            y_maxes_excluding=np.arange(0, n_ims) + 20,
            y_mins_including=np.arange(0, n_ims) + 1,
        )
        target_obj_masks = get_twod_area_masks(
            areas=target_obj_areas, im_shape=im_shape
        )

        losses1 = cropped_d1_loss(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            target_obj_areas=target_obj_areas,
        )
        losses2 = d1_loss(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks & target_obj_masks),
        )

        self.assertTrue(np.allclose(losses1, losses2, atol=1e-4))

    def test_rmse_loss(self):
        pred_depths = np.ones(newshape_im_depthmaps(n=4, h=11, w=7), dtype=np.float32)
        gt_depths = np.ones_like(pred_depths, dtype=np.float32)
        upd_im_depthmaps(gt_depths, n=1, value_=2)
        upd_im_depthmaps(gt_depths, n=2, value_=3)
        upd_im_depthmaps(gt_depths, n=3, h=slice(0, 5), value_=8)

        masks = scast_im_depthmasks(np.ones_like(gt_depths, dtype=np.bool_))
        upd_im_depthmasks(masks, n=3, h=slice(0, 5), value_=0)

        losses = rmse_loss(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
        )
        self.assertTrue(match_scalars_float(losses))

        loss_list: list[float] = [float(val) for val in losses]

        self.assertAlmostEqual(loss_list[0], 0)
        self.assertGreater(loss_list[1], loss_list[0])
        self.assertGreater(loss_list[2], loss_list[1])
        self.assertAlmostEqual(loss_list[3], 0)

    def test_cropped_rmse_loss(self):
        rng = np.random.default_rng(30)
        n_ims = 5
        im_shape = TwoDSize(x_size=41, y_size=40)
        pred_depths = np.ones(
            newshape_im_depthmaps(n=n_ims, h=im_shape.y_size, w=im_shape.x_size),
            dtype=np.float32,
        )
        gt_depths = rng.choice([1, 2], size=pred_depths.shape).astype(np.float32)
        masks = rng.choice([0, 1], size=pred_depths.shape).astype(np.bool_)

        target_obj_areas = TwoDAreas(
            x_maxes_excluding=np.arange(0, n_ims) + 9,
            x_mins_including=np.arange(0, n_ims) + 3,
            y_maxes_excluding=np.arange(0, n_ims) + 20,
            y_mins_including=np.arange(0, n_ims) + 1,
        )
        target_obj_masks = get_twod_area_masks(
            areas=target_obj_areas, im_shape=im_shape
        )

        losses1 = cropped_rmse_loss(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            target_obj_areas=target_obj_areas,
        )
        losses2 = rmse_loss(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks & target_obj_masks),
        )

        self.assertTrue(np.allclose(losses1, losses2, atol=1e-4))

    def test_log10_loss(self):
        pred_depths = np.ones(newshape_im_depthmaps(n=6, h=11, w=7), dtype=np.float32)
        upd_im_depthmaps(pred_depths, n=4, value_=5)
        upd_im_depthmaps(pred_depths, n=5, value_=10)

        gt_depths = np.ones_like(pred_depths, dtype=np.float32)
        upd_im_depthmaps(gt_depths, n=1, value_=2)
        upd_im_depthmaps(gt_depths, n=2, value_=3)
        upd_im_depthmaps(gt_depths, n=3, h=slice(0, 5), value_=8)
        upd_im_depthmaps(gt_depths, n=4, value_=idx_im_depthmaps(pred_depths, n=4) * 2)
        upd_im_depthmaps(gt_depths, n=5, value_=idx_im_depthmaps(pred_depths, n=5) * 2)

        masks = scast_im_depthmasks(np.ones_like(gt_depths, dtype=np.bool_))
        upd_im_depthmasks(masks, n=3, h=slice(0, 5), value_=0)

        losses = log10_loss(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
        )

        self.assertTrue(match_scalars_float(losses))

        loss_list: list[float] = [float(val) for val in losses]

        self.assertAlmostEqual(loss_list[0], 0)
        self.assertGreater(loss_list[1], loss_list[0])
        self.assertGreater(loss_list[2], loss_list[1])
        self.assertAlmostEqual(loss_list[3], 0)
        self.assertAlmostEqual(loss_list[4], loss_list[5], places=4)

    def test_cropped_log10_loss(self):
        rng = np.random.default_rng(30)
        n_ims = 5
        im_shape = TwoDSize(x_size=41, y_size=40)
        pred_depths = np.ones(
            newshape_im_depthmaps(n=n_ims, h=im_shape.y_size, w=im_shape.x_size),
            dtype=np.float32,
        )
        gt_depths = rng.choice([1, 2], size=pred_depths.shape).astype(np.float32)
        masks = rng.choice([0, 1], size=pred_depths.shape).astype(np.bool_)

        target_obj_areas = TwoDAreas(
            x_maxes_excluding=np.arange(0, n_ims) + 9,
            x_mins_including=np.arange(0, n_ims) + 3,
            y_maxes_excluding=np.arange(0, n_ims) + 20,
            y_mins_including=np.arange(0, n_ims) + 1,
        )
        target_obj_masks = get_twod_area_masks(
            areas=target_obj_areas, im_shape=im_shape
        )

        losses1 = cropped_log10_loss(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            target_obj_areas=target_obj_areas,
        )
        losses2 = log10_loss(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks & target_obj_masks),
        )

        self.assertTrue(np.allclose(losses1, losses2, atol=1e-4))

    def test_se_map(self):
        width = 11
        height = 10
        n_samples = 4
        pred_depths = np.linspace(0, 5, width * height * n_samples).reshape(
            newshape_im_depthmaps(n=n_samples, w=width, h=height)
        )
        gt_depths = np.linspace(1, 8, width * height * n_samples).reshape(
            newshape_im_depthmaps(n=n_samples, w=width, h=height)
        )
        masks = np.ones_like(pred_depths, dtype=np.bool_)
        upd_im_depthmasks(masks, w=3, value_=0)
        upd_im_depthmasks(masks, w=5, value_=0)

        got_se_map = se_map(
            pred_depths=pred_depths, gt=DepthsWithMasks(depths=gt_depths, masks=masks)
        )
        self.assertTrue(match_im_floatmap(got_se_map))

        expected_rmse_losses = rmse_loss(
            pred_depths=pred_depths, gt=DepthsWithMasks(depths=gt_depths, masks=masks)
        )
        actual_rmse_losses = np.sqrt(
            _masked_samplewise_mean(ims=got_se_map, masks=masks)
        )

        self.assertTrue(
            np.allclose(expected_rmse_losses, actual_rmse_losses, atol=1e-4)
        )

    def test_cropped_se_map(self):
        width = 11
        height = 10
        n_samples = 4
        pred_depths = np.linspace(0, 5, width * height * n_samples).reshape(
            newshape_im_depthmaps(n=n_samples, w=width, h=height)
        )
        gt_depths = np.linspace(1, 8, width * height * n_samples).reshape(
            newshape_im_depthmaps(n=n_samples, w=width, h=height)
        )
        masks = np.ones_like(pred_depths, dtype=np.bool_)
        upd_im_depthmasks(masks, w=3, value_=0)
        upd_im_depthmasks(masks, w=5, value_=0)

        target_obj_areas = TwoDAreas(
            x_mins_including=np.array([0, 2, 3, 1]),
            x_maxes_excluding=np.array([8, 9, 7, 8]),
            y_mins_including=np.array([0, 1, 0, 2]),
            y_maxes_excluding=np.array([6, 7, 8, 9]),
        )

        got_se_map = cropped_se_map(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            target_obj_areas=target_obj_areas,
        )
        self.assertTrue(match_im_floatmap(got_se_map))

        expected_rmse_losses = cropped_rmse_loss(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            target_obj_areas=target_obj_areas,
        )

        target_obj_masks = get_twod_area_masks(
            areas=target_obj_areas, im_shape=TwoDSize(x_size=width, y_size=height)
        )
        actual_rmse_losses = np.sqrt(
            _masked_samplewise_mean(ims=got_se_map, masks=masks & target_obj_masks)
        )

        self.assertTrue(
            np.allclose(expected_rmse_losses, actual_rmse_losses, atol=1e-4)
        )

    def test_log10_map(self):
        width = 11
        height = 10
        n_samples = 4
        pred_depths = np.linspace(0, 5, width * height * n_samples).reshape(
            newshape_im_depthmaps(n=n_samples, w=width, h=height)
        )
        gt_depths = np.linspace(1, 8, width * height * n_samples).reshape(
            newshape_im_depthmaps(n=n_samples, w=width, h=height)
        )
        masks = np.ones_like(pred_depths, dtype=np.bool_)
        upd_im_depthmasks(masks, w=3, value_=0)
        upd_im_depthmasks(masks, w=5, value_=0)

        got_se_map = log10_map(
            pred_depths=pred_depths, gt=DepthsWithMasks(depths=gt_depths, masks=masks)
        )
        self.assertTrue(match_im_floatmap(got_se_map))

        expected_log10_losses = log10_loss(
            pred_depths=pred_depths, gt=DepthsWithMasks(depths=gt_depths, masks=masks)
        )
        actual_log10_losses = _masked_samplewise_mean(ims=got_se_map, masks=masks)

        self.assertTrue(
            np.allclose(expected_log10_losses, actual_log10_losses, atol=1e-4)
        )

    def test_cropped_log10_map(self):
        width = 11
        height = 10
        n_samples = 4
        pred_depths = np.linspace(0, 5, width * height * n_samples).reshape(
            newshape_im_depthmaps(n=n_samples, w=width, h=height)
        )
        gt_depths = np.linspace(1, 8, width * height * n_samples).reshape(
            newshape_im_depthmaps(n=n_samples, w=width, h=height)
        )
        masks = np.ones_like(pred_depths, dtype=np.bool_)
        upd_im_depthmasks(masks, w=3, value_=0)
        upd_im_depthmasks(masks, w=5, value_=0)

        target_obj_areas = TwoDAreas(
            x_mins_including=np.array([0, 2, 3, 1]),
            x_maxes_excluding=np.array([8, 9, 7, 8]),
            y_mins_including=np.array([0, 1, 0, 2]),
            y_maxes_excluding=np.array([6, 7, 8, 9]),
        )

        got_se_map = cropped_log10_map(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            target_obj_areas=target_obj_areas,
        )
        self.assertTrue(match_im_floatmap(got_se_map))

        expected_rmse_losses = cropped_log10_loss(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            target_obj_areas=target_obj_areas,
        )
        target_obj_masks = get_twod_area_masks(
            areas=target_obj_areas, im_shape=TwoDSize(x_size=width, y_size=height)
        )
        actual_rmse_losses = _masked_samplewise_mean(
            ims=got_se_map, masks=masks & target_obj_masks
        )

        self.assertTrue(
            np.allclose(expected_rmse_losses, actual_rmse_losses, atol=1e-4)
        )

    def test_d1_map(self):
        width = 11
        height = 10
        n_samples = 4
        pred_depths = np.linspace(0, 5, width * height * n_samples).reshape(
            newshape_im_depthmaps(n=n_samples, w=width, h=height)
        )
        rng = np.random.default_rng(65)
        gt_depths = pred_depths + rng.choice([0, 0.01], size=pred_depths.shape)
        masks = np.ones_like(pred_depths, dtype=np.bool_)
        upd_im_depthmasks(masks, w=3, value_=0)
        upd_im_depthmasks(masks, w=5, value_=0)

        got_d1_map = d1_map(
            pred_depths=pred_depths, gt=DepthsWithMasks(depths=gt_depths, masks=masks)
        )
        self.assertTrue(match_im_floatmap(got_d1_map))

        expected_d1_losses = d1_loss(
            pred_depths=pred_depths, gt=DepthsWithMasks(depths=gt_depths, masks=masks)
        )
        actual_d1_losses = _masked_samplewise_mean(ims=got_d1_map, masks=masks)

        self.assertTrue(np.allclose(expected_d1_losses, actual_d1_losses, atol=1e-4))

    def test_cropped_d1_map(self):
        width = 11
        height = 10
        n_samples = 4
        pred_depths = np.linspace(0, 5, width * height * n_samples).reshape(
            newshape_im_depthmaps(n=n_samples, w=width, h=height)
        )
        rng = np.random.default_rng(65)
        gt_depths = pred_depths + rng.choice([0, 0.01], size=pred_depths.shape)
        masks = np.ones_like(pred_depths, dtype=np.bool_)
        upd_im_depthmasks(masks, w=3, value_=0)
        upd_im_depthmasks(masks, w=5, value_=0)

        target_obj_areas = TwoDAreas(
            x_mins_including=np.array([0, 2, 3, 1]),
            x_maxes_excluding=np.array([8, 9, 7, 8]),
            y_mins_including=np.array([0, 1, 0, 2]),
            y_maxes_excluding=np.array([6, 7, 8, 9]),
        )

        got_se_map = cropped_d1_map(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            target_obj_areas=target_obj_areas,
        )
        self.assertTrue(match_im_floatmap(got_se_map))

        expected_d1_losses = cropped_d1_loss(
            pred_depths=pred_depths,
            gt=DepthsWithMasks(depths=gt_depths, masks=masks),
            target_obj_areas=target_obj_areas,
        )
        target_obj_masks = get_twod_area_masks(
            areas=target_obj_areas, im_shape=TwoDSize(x_size=width, y_size=height)
        )
        actual_d1_losses = _masked_samplewise_mean(
            ims=got_se_map, masks=masks & target_obj_masks
        )

        self.assertTrue(np.allclose(expected_d1_losses, actual_d1_losses, atol=1e-4))
