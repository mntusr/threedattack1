import unittest
from unittest import mock

import numpy as np

from threedattack.dataset_model import (
    ExactSampleCounts,
    RGBsWithDepthsAndMasks,
    SamplesBase,
    SampleType,
)
from threedattack.script_util import calculate_dataset_depth_stats
from threedattack.tensor_types.npy import *


class TestDatasetStats(unittest.TestCase):
    def test_calculate_dataset_depth_stats_happy_path(self):
        for expected_sample_type in SampleType:
            with self.subTest(f"{expected_sample_type=}"):
                reference_depth_map = np.linspace(0.6, 5, 25).reshape(
                    newshape_im_depthmaps(n=1, h=5, w=5)
                )
                reference_mask = np.ones_like(reference_depth_map, dtype=np.bool_)
                upd_im_depthmaps(reference_depth_map, h=[1, 2, 3], value_=0)
                upd_im_depthmasks(reference_mask, h=[1, 2, 3], value_=0)

                mean_ref_distance = float(np.mean(reference_depth_map[reference_mask]))
                median_ref_distance = float(
                    np.median(reference_depth_map[reference_mask])
                )
                min_ref_distance = float(np.min(reference_depth_map[reference_mask]))
                max_ref_distance = float(np.max(reference_depth_map[reference_mask]))

                depth_offsets = [3, 9, 2]

                depth_maps = [reference_depth_map + offset for offset in depth_offsets]

                def get_sample(idx: int, sample_type: SampleType) -> SamplesBase:
                    self.assertEqual(sample_type, expected_sample_type)
                    return SamplesBase(
                        rgbds=RGBsWithDepthsAndMasks(
                            rgbs=mock.Mock(),
                            depths=depth_maps[idx],
                            masks=reference_mask,
                        )
                    )

                test_dataset = mock.Mock()
                test_dataset.get_sample = get_sample
                test_dataset.get_n_samples = mock.Mock(
                    return_value=self._get_sample_counts_for_dataset_that_only_contains(
                        n_samples=len(depth_maps), sample_type=expected_sample_type
                    )
                )

                expected_mean_depths = [
                    mean_ref_distance + offset for offset in depth_offsets
                ]
                expected_median_depths = [
                    median_ref_distance + offset for offset in depth_offsets
                ]
                expected_min_depths = [
                    min_ref_distance + offset for offset in depth_offsets
                ]
                expected_max_depths = [
                    max_ref_distance + offset for offset in depth_offsets
                ]

                actual_depth_stats = calculate_dataset_depth_stats(
                    dataset=test_dataset, sample_type=expected_sample_type
                )

                self._assert_list_allclose(
                    expected_min_depths, actual_depth_stats.min_depths, atol=1e-4
                )
                self._assert_list_allclose(
                    expected_max_depths, actual_depth_stats.max_depths, atol=1e-4
                )
                self._assert_list_allclose(
                    expected_mean_depths, actual_depth_stats.mean_depths, atol=1e-4
                )
                self._assert_list_allclose(
                    expected_median_depths, actual_depth_stats.median_depths, atol=1e-4
                )

    def _assert_list_allclose(
        self, list1: list[float], list2: list[float], atol: float
    ):
        self.assertTrue(np.allclose(np.array(list1), np.array(list2), atol=atol))

    def _get_sample_counts_for_dataset_that_only_contains(
        self, n_samples: int, sample_type: SampleType
    ) -> ExactSampleCounts:
        match sample_type:
            case SampleType.Train:
                return ExactSampleCounts(
                    n_train_samples=n_samples, n_val_samples=0, n_test_samples=0
                )
            case SampleType.Test:
                return ExactSampleCounts(
                    n_train_samples=0, n_val_samples=0, n_test_samples=n_samples
                )
            case SampleType.Val:
                return ExactSampleCounts(
                    n_train_samples=0, n_val_samples=n_samples, n_test_samples=0
                )
