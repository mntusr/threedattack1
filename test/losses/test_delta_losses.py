import unittest

import numpy as np

from threedattack.dataset_model import DepthsWithMasks, SampleType
from threedattack.losses import (
    LossDerivationMethod,
    LossPrecision,
    RawLossFn,
    calculate_loss_values,
    derived_loss_dict_2_str_float_dict,
    divide_losses,
    get_aggr_delta_loss_dict_from_losses,
    get_aggr_delta_loss_dict_from_preds,
    get_all_derived_loss_name_list,
    get_derived_loss_by_name,
    get_derived_loss_name,
    subtract_losses,
)
from threedattack.rendering_core import TwoDAreas
from threedattack.tensor_types.npy import *


class TestDeltaLosses(unittest.TestCase):
    def setUp(self):
        self.N_SAMPLES = 10
        self.new_losses = {
            RawLossFn.D1: np.linspace(0, 1, self.N_SAMPLES),
            RawLossFn.CroppedRMSE: np.linspace(1, 5, self.N_SAMPLES),
        }
        self.orig_losses = {
            RawLossFn.D1: np.linspace(0, 1, self.N_SAMPLES),
            RawLossFn.CroppedRMSE: np.linspace(3, 5, self.N_SAMPLES),
        }
        self.loss_fns_in_test_inputs = [RawLossFn.D1, RawLossFn.CroppedRMSE]

    def test_get_aggr_delta_loss_dict_from_preds(self):
        for loss_precision in LossPrecision:
            for loss_fns in [
                set(self.loss_fns_in_test_inputs),
                {self.loss_fns_in_test_inputs[0]},
            ]:
                for viewpt_type in SampleType:
                    with self.subTest(f"{loss_fns=};{viewpt_type=};{loss_precision=}"):
                        pred_depths = np.ones(
                            newshape_im_depthmaps(n=self.N_SAMPLES, w=9, h=7),
                            dtype=np.float32,
                        )
                        for i in range(self.N_SAMPLES):
                            upd_im_depthmaps(pred_depths, n=i, value_=i + 0.5)

                        masks = np.ones_like(pred_depths, dtype=np.bool_)
                        gt_depths = np.ones(
                            newshape_im_depthmaps(n=self.N_SAMPLES, w=9, h=7),
                            dtype=np.float32,
                        )
                        gt_data = DepthsWithMasks(depths=gt_depths, masks=masks)
                        target_obj_areas = TwoDAreas(
                            x_maxes_excluding=np.full(
                                shape=newshape_scalars_int(n=self.N_SAMPLES),
                                fill_value=6,
                            ),
                            x_mins_including=np.full(
                                shape=newshape_scalars_int(n=self.N_SAMPLES),
                                fill_value=1,
                            ),
                            y_maxes_excluding=np.full(
                                shape=newshape_scalars_int(n=self.N_SAMPLES),
                                fill_value=5,
                            ),
                            y_mins_including=np.full(
                                shape=newshape_scalars_int(n=self.N_SAMPLES),
                                fill_value=0,
                            ),
                        )

                        expected_agg_delta = get_aggr_delta_loss_dict_from_preds(
                            gt=gt_data,
                            aligned_depth_preds=pred_depths,
                            loss_precision=loss_precision,
                            orig_losses=self.orig_losses,
                            viewpt_type=viewpt_type,
                            loss_fns=loss_fns,
                            target_obj_areas=target_obj_areas,
                        )

                        new_losses = calculate_loss_values(
                            pred_depths=pred_depths,
                            gt=gt_data,
                            loss_fns=loss_fns,
                            target_obj_areas=target_obj_areas,
                        )
                        filtered_orig_losses = {
                            loss_fn: vals
                            for loss_fn, vals in self.orig_losses.items()
                            if loss_fn in loss_fns
                        }
                        actual_agg_delta = get_aggr_delta_loss_dict_from_losses(
                            orig_losses=filtered_orig_losses,
                            new_losses=new_losses,
                            loss_precision=loss_precision,
                            viewpt_type=viewpt_type,
                        )

                        self.assertEqual(
                            expected_agg_delta.keys(), actual_agg_delta.keys()
                        )

                        for key in actual_agg_delta.keys():
                            self.assertAlmostEqual(
                                expected_agg_delta[key], actual_agg_delta[key]
                            )

    def test_get_aggr_delta_loss_dict_from_losses(self):
        for loss_precision in LossPrecision:
            for viewpt_type in SampleType:
                with self.subTest(f"{loss_precision=};{viewpt_type=}"):
                    agg_dict = get_aggr_delta_loss_dict_from_losses(
                        new_losses=self.new_losses,
                        orig_losses=self.orig_losses,
                        loss_precision=loss_precision,
                        viewpt_type=viewpt_type,
                    )

                    expected_loss_num = len(self.new_losses.keys()) * 6
                    actual_loss_num = len(agg_dict.keys())

                    self.assertEqual(expected_loss_num, actual_loss_num)

                    self.assertAlmostEqual(
                        float(
                            np.mean(
                                subtract_losses(self.new_losses, self.orig_losses)[
                                    RawLossFn.CroppedRMSE
                                ]
                            )
                        ),
                        agg_dict[
                            LossDerivationMethod.MeanDelta,
                            loss_precision,
                            viewpt_type,
                            RawLossFn.CroppedRMSE,
                        ],
                    )
                    self.assertAlmostEqual(
                        float(
                            np.median(
                                subtract_losses(self.new_losses, self.orig_losses)[
                                    RawLossFn.CroppedRMSE
                                ]
                            )
                        ),
                        agg_dict[
                            LossDerivationMethod.MedianDelta,
                            loss_precision,
                            viewpt_type,
                            RawLossFn.CroppedRMSE,
                        ],
                    )
                    self.assertAlmostEqual(
                        float(
                            np.min(
                                subtract_losses(self.new_losses, self.orig_losses)[
                                    RawLossFn.CroppedRMSE
                                ]
                            )
                        ),
                        agg_dict[
                            LossDerivationMethod.MinDelta,
                            loss_precision,
                            viewpt_type,
                            RawLossFn.CroppedRMSE,
                        ],
                    )
                    self.assertAlmostEqual(
                        float(
                            np.mean(
                                divide_losses(
                                    subtract_losses(self.new_losses, self.orig_losses),
                                    self.orig_losses,
                                )[RawLossFn.CroppedRMSE]
                            )
                        ),
                        agg_dict[
                            LossDerivationMethod.MeanReldelta,
                            loss_precision,
                            viewpt_type,
                            RawLossFn.CroppedRMSE,
                        ],
                    )
                    self.assertAlmostEqual(
                        float(
                            np.median(
                                divide_losses(
                                    subtract_losses(self.new_losses, self.orig_losses),
                                    self.orig_losses,
                                )[RawLossFn.CroppedRMSE]
                            )
                        ),
                        agg_dict[
                            LossDerivationMethod.MedianReldelta,
                            loss_precision,
                            viewpt_type,
                            RawLossFn.CroppedRMSE,
                        ],
                    )
                    self.assertAlmostEqual(
                        float(
                            np.min(
                                divide_losses(
                                    subtract_losses(self.new_losses, self.orig_losses),
                                    self.orig_losses,
                                )[RawLossFn.CroppedRMSE]
                            )
                        ),
                        agg_dict[
                            LossDerivationMethod.MinReldelta,
                            loss_precision,
                            viewpt_type,
                            RawLossFn.CroppedRMSE,
                        ],
                    )

    def test_get_derived_loss_by_name_happy_path(self):
        expected_loss = (LossDerivationMethod.MedianDelta, RawLossFn.Log10)

        name = expected_loss[0].public_name + "_" + expected_loss[1].public_name

        actual_loss = get_derived_loss_by_name(name)

        self.assertEqual(expected_loss, actual_loss)

    def test_get_derived_loss_by_name_invalid_name(self):
        name = "invalid_loss"

        with self.assertRaises(ValueError):
            get_derived_loss_by_name(name)

    def test_get_derived_loss_name(self):
        loss = (LossDerivationMethod.MedianDelta, RawLossFn.Log10)

        expected_name = loss[0].public_name + "_" + loss[1].public_name
        actual_name = get_derived_loss_name(loss[0], loss[1])

        self.assertEqual(expected_name, actual_name)

    def test_derived_loss_dict_2_str_float_dict(self):
        derived_loss_dict: dict[
            tuple[LossDerivationMethod, LossPrecision, SampleType, RawLossFn], float
        ] = {
            (
                LossDerivationMethod.MeanDelta,
                LossPrecision.PossiblyEstim,
                SampleType.Val,
                RawLossFn.RMSE,
            ): 35.1,
            (
                LossDerivationMethod.MinDelta,
                LossPrecision.Exact,
                SampleType.Train,
                RawLossFn.D1,
            ): 19.1,
        }

        expected_dict = {
            "estim_mean_delta_val_rmse": 35.1,
            "exact_min_delta_train_d1": 19.1,
        }

        actual_dict = derived_loss_dict_2_str_float_dict(derived_loss_dict)

        self.assertEqual(expected_dict, actual_dict)

    def test_get_all_derived_loss_name_list(self):
        got_loss_names = get_all_derived_loss_name_list()

        self.assertEqual(got_loss_names, sorted(got_loss_names))
        self.assertEqual(
            len(set(got_loss_names)), len(LossDerivationMethod) * len(RawLossFn)
        )

        for got_loss_name in got_loss_names:
            # this should not raise any error
            get_derived_loss_by_name(got_loss_name)
