import unittest

from threedattack import StepLoggingFreqFunction


class TestLogging(unittest.TestCase):
    def test_needs_logging_happy_path(self):
        fn = StepLoggingFreqFunction(
            steps=[9, 5],
            freqencies=[1, 3, 6],
        )

        self.assertTrue(fn.needs_logging(0))
        self.assertTrue(fn.needs_logging(1))
        self.assertTrue(fn.needs_logging(2))
        self.assertFalse(fn.needs_logging(5))
        self.assertTrue(fn.needs_logging(6))
        self.assertTrue(fn.needs_logging(9))
        self.assertFalse(fn.needs_logging(10))
        self.assertTrue(fn.needs_logging(12))

    def test_needs_logging_invalid_step_count(self):
        with self.assertRaises(ValueError):
            StepLoggingFreqFunction(
                steps=[9, 5, 4],
                freqencies=[1, 3, 6],
            )

    def test_needs_logging_negative_freq(self):
        with self.assertRaises(ValueError):
            StepLoggingFreqFunction(
                steps=[9, 5],
                freqencies=[1, 3, -3],
            )
