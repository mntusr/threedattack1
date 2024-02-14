import unittest

import numpy as np

from threedattack.dataset_model import DepthsWithMasks
from threedattack.rendering_core import scale_depth_maps_with_masks
from threedattack.tensor_types.npy import *


class TestIm(unittest.TestCase):
    def test_scale_depth_maps_with_masks_happy_path(self) -> None:
        original_depths = np.zeros((3, 1, 10, 10), dtype=np.float32)
        upd_im(original_depths, h=slice(0, 5), value_=7)

        original_masks = np.ones_like(original_depths, dtype=np.bool_)
        upd_im(original_masks, h=slice(0, 3), w=slice(0, 3), value_=0)

        depths_with_masks = DepthsWithMasks(
            depths=original_depths, masks=original_masks
        )

        scaled = scale_depth_maps_with_masks(depths_with_masks, new_size_rel=0.5)

        self.assertTrue(
            match_im_depthmaps(scaled.depths, shape={"h": 5, "w": 5, "n": 3})
        )
        self.assertTrue(
            match_im_depthmasks(scaled.masks, shape={"h": 5, "w": 5, "n": 3})
        )

        scaled_depths_val_set = {float(val) for val in np.unique(scaled.depths)}
        self.assertSetEqual(scaled_depths_val_set, {0.0, 7.0})

        self.assertTrue(np.all(idx_im(scaled.masks, h=0, w=-1)))
        self.assertTrue(np.all(~idx_im(scaled.masks, h=0, w=0)))
