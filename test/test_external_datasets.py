import unittest
from typing import Any

import numpy as np

from threedattack.dataset_model import (
    RGBsWithDepthsAndMasks,
    SampleType,
    SampleTypeError,
)
from threedattack.external_datasets import (
    NYUV2_IM_SIZE,
    NyuV2Samples,
    nyu_depthv2_dataset_from_default_paths,
)
from threedattack.rendering_core import imshow
from threedattack.tensor_types.idx import *
from threedattack.tensor_types.npy import *


class TestNyuDepthv2Dataset(unittest.TestCase):
    def setUp(self):
        self.dataset = nyu_depthv2_dataset_from_default_paths(add_black_frame=True)

    def test_get_sample(self):
        sample = self.dataset.get_sample(110, SampleType.Test)

        expected_names = ["test/bedroom/rgb_00280.jpg"]
        expected_mean_rgb = 0.25761288

        self.assertTrue(
            match_im_rgbs(
                sample.rgbds.rgbs,
                shape={"n": 1, "w": NYUV2_IM_SIZE.x_size, "h": NYUV2_IM_SIZE.y_size},
            )
        )
        self.assertTrue(
            match_im_depthmaps(
                sample.rgbds.depths,
                shape={"n": 1, "w": NYUV2_IM_SIZE.x_size, "h": NYUV2_IM_SIZE.y_size},
            )
        )
        self.assertTrue(
            match_im_depthmasks(
                sample.rgbds.masks,
                shape={"n": 1, "w": NYUV2_IM_SIZE.x_size, "h": NYUV2_IM_SIZE.y_size},
            )
        )

        actual_mean_rgb = float(sample.rgbds.rgbs.mean())
        self.assertAlmostEqual(expected_mean_rgb, actual_mean_rgb, 3)

        self.assertEqual(expected_names, sample.names)

    def test_get_n_samples(self):
        expected_n_train_samples = 654
        expected_n_test_samples = 0
        expected_n_val_samples = 0
        actual_n_train_samples = self.dataset.get_n_samples().n_train_samples
        actual_n_test_samples = self.dataset.get_n_samples().n_test_samples
        actual_n_val_samples = self.dataset.get_n_samples().n_val_samples

        self.assertEqual(expected_n_train_samples, actual_n_test_samples)
        self.assertEqual(expected_n_test_samples, actual_n_train_samples)
        self.assertEqual(expected_n_val_samples, actual_n_val_samples)

    def test_get_samples_happy_path(self):
        cases: list[tuple[Any, list[int], int]] = [
            (slice(0, 3), [0, 1, 2], 3),
            (slice(3, 0, -1), [3, 2, 1], 3),
            ([0, 1, 2], [0, 1, 2], 3),
        ]

        for indices, raw_idx_list, num_ims in cases:
            case_name = str(indices).replace(" ", "")
            with self.subTest(case_name):
                expected_samples: list[NyuV2Samples] = []
                for raw_idx in raw_idx_list:
                    sample = self.dataset.get_sample(raw_idx, SampleType.Test)
                    expected_samples.append(sample)

                actual_batch = self.dataset.get_samples(indices, SampleType.Test)

                self.assertTrue(
                    match_im_rgbs(
                        actual_batch.rgbds.rgbs,
                        shape={
                            "n": num_ims,
                            "w": NYUV2_IM_SIZE.x_size,
                            "h": NYUV2_IM_SIZE.y_size,
                        },
                    )
                )
                self.assertTrue(
                    match_im_depthmaps(
                        actual_batch.rgbds.depths,
                        shape={
                            "n": num_ims,
                            "w": NYUV2_IM_SIZE.x_size,
                            "h": NYUV2_IM_SIZE.y_size,
                        },
                    )
                )
                self.assertTrue(
                    match_im_depthmasks(
                        actual_batch.rgbds.masks,
                        shape={
                            "n": num_ims,
                            "w": NYUV2_IM_SIZE.x_size,
                            "h": NYUV2_IM_SIZE.y_size,
                        },
                    )
                )

                self.assertEqual(
                    len(expected_samples), actual_batch.rgbds.rgbs.shape[DIM_IM_N]
                )
                self.assertEqual(len(expected_samples), len(actual_batch.names))
                for i, expected_sample in enumerate(expected_samples):
                    self.assertTrue(
                        np.allclose(
                            expected_sample.rgbds.rgbs, actual_batch.rgbds[[i]].rgbs
                        )
                    )
                    self.assertTrue(
                        np.allclose(
                            expected_sample.rgbds.depths, actual_batch.rgbds[[i]].depths
                        )
                    )
                    shp1 = expected_sample.rgbds.masks.shape
                    shp2 = actual_batch.rgbds[[i]].masks.shape
                    self.assertTrue(
                        np.array_equal(
                            expected_sample.rgbds.masks, actual_batch.rgbds[[i]].masks
                        )
                    )
                    self.assertEqual(expected_sample.names[0], actual_batch.names[i])

    def test_get_samples_too_big_idx(self):
        with self.assertRaises(IndexError):
            self.dataset.get_samples([60000], SampleType.Test)

    def test_get_samples_invalid_sample_type(self):
        invalid_types = [SampleType.Train, SampleType.Val]

        for sample_type in invalid_types:
            with self.subTest(sample_type.public_name):
                with self.assertRaises(SampleTypeError):
                    self.dataset.get_samples([0], sample_type)
