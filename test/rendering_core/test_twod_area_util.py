import unittest
from unittest import mock

import numpy as np

from threedattack.rendering_core import TwoDAreas, TwoDSize, get_twod_area_masks, imshow
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *


class TestTwodAreaUtil(unittest.TestCase):
    def setUp(self):
        self.im_shape = TwoDSize(200, 150)

        self.areas = TwoDAreas(
            x_mins_including=np.array([5, 20, -5, 600, 5]),
            x_maxes_excluding=np.array([50, 110, 8, 650, 50]),
            y_mins_including=np.array([10, -1, 70, 10, 600]),
            y_maxes_excluding=np.array([55, 160, 88, 55, 650]),
        )
        self.area_count = self.areas.x_maxes_excluding.shape[DIM_SCALARS_N]
        self.area_copletely_outside = [False, False, False, True, True]

    def test_get_twod_area_masks_happy_path(self):
        masks = get_twod_area_masks(areas=self.areas, im_shape=self.im_shape)

        self.assertTrue(
            match_im_depthmasks(
                masks,
                shape={
                    "n": self.area_count,
                    "h": self.im_shape.y_size,
                    "w": self.im_shape.x_size,
                },
            )
        )

        for area_idx in range(self.area_count):
            if not self.area_copletely_outside[area_idx]:
                x_min_including = idx_scalars_int(
                    self.areas.x_mins_including, n=area_idx
                )
                x_max_excluding = idx_scalars_int(
                    self.areas.x_maxes_excluding, n=area_idx
                )
                y_min_including = idx_scalars_int(
                    self.areas.y_mins_including, n=area_idx
                )
                y_max_excluding = idx_scalars_int(
                    self.areas.y_maxes_excluding, n=area_idx
                )

                x_min_including_clip = np.clip(x_min_including, 0, self.im_shape.x_size)
                x_max_excluding_clip = np.clip(x_max_excluding, 0, self.im_shape.x_size)
                y_min_including_clip = np.clip(y_min_including, 0, self.im_shape.y_size)
                y_max_excluding_clip = np.clip(y_max_excluding, 0, self.im_shape.y_size)

                mask_inside = idx_im_depthmasks(
                    masks,
                    n=[area_idx],
                    w=slice(x_min_including_clip, x_max_excluding_clip),
                    h=slice(y_min_including_clip, y_max_excluding_clip),
                )

                self.assertTrue(np.all(mask_inside))

                mask_outsides: list[np.ndarray] = []

                if x_min_including_clip != 0:
                    mask_outsides.append(
                        idx_im_depthmasks(
                            masks,
                            n=area_idx,
                            w=slice(0, x_min_including_clip),
                        )
                    )
                if x_max_excluding_clip != self.im_shape.x_size:
                    mask_outsides.append(
                        idx_im_depthmasks(
                            masks,
                            n=area_idx,
                            w=slice(x_max_excluding_clip, self.im_shape.x_size),
                        )
                    )
                if y_min_including_clip != 0:
                    mask_outsides.append(
                        idx_im_depthmasks(
                            masks,
                            n=area_idx,
                            h=slice(0, y_min_including_clip),
                        )
                    )
                if y_max_excluding_clip != self.im_shape.y_size:
                    mask_outsides.append(
                        idx_im_depthmasks(
                            masks,
                            n=area_idx,
                            h=slice(y_max_excluding_clip, self.im_shape.y_size),
                        )
                    )

                for outside_part in mask_outsides:
                    self.assertTrue(np.all(~outside_part))
            else:
                mask_for_area = idx_im_depthmasks(
                    masks,
                    n=[area_idx],
                )
                self.assertTrue(np.all(~mask_for_area))

    def test_get_twod_area_masks_no_image(self):
        with self.assertRaises(ValueError):
            get_twod_area_masks(
                areas=TwoDAreas(
                    x_mins_including=np.array([], dtype=np.float32),
                    x_maxes_excluding=np.array([], dtype=np.float32),
                    y_maxes_excluding=np.array([], dtype=np.float32),
                    y_mins_including=np.array([], dtype=np.float32),
                ),
                im_shape=self.im_shape,
            )

    def test_get_twod_area_masks_non_pos_size(self):
        with self.assertRaises(ValueError):
            get_twod_area_masks(
                areas=self.areas,
                im_shape=TwoDSize(0, 9),
            )
