import unittest

from threedattack.losses import (
    RawLossFn,
    sign_loss_to_make_smaller_mean_worse_predictor,
)


class TestLossSigns(unittest.TestCase):
    def test_sign_loss_to_make_smaller_mean_worse_predictor(self):
        needs_sign_swap = {
            RawLossFn.CroppedD1: False,
            RawLossFn.D1: False,
            RawLossFn.CroppedRMSE: True,
            RawLossFn.RMSE: True,
            RawLossFn.CroppedLog10: True,
            RawLossFn.Log10: True,
        }
        ORIGINAL_VAL = 1.1

        for loss_fn in RawLossFn:
            with self.subTest(loss_fn.public_name):
                actual_proper_val = sign_loss_to_make_smaller_mean_worse_predictor(
                    loss_fn=loss_fn, val=ORIGINAL_VAL
                )
                expected_proper_val = (
                    -ORIGINAL_VAL if needs_sign_swap[loss_fn] else ORIGINAL_VAL
                )

                self.assertAlmostEqual(actual_proper_val, expected_proper_val)
