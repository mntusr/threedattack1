import datetime
import unittest

from threedattack._attack_primitives import estimate_remaining_time


class TestTiming(unittest.TestCase):
    def setUp(self) -> None:
        iter_length_sec = 10
        self.n_total_iter = 150
        self.n_elapsed_iter = 50
        self.start_time = datetime.datetime(
            year=2023, month=6, day=15, hour=10, minute=20, second=32, microsecond=150
        )
        n_remaining_iter = self.n_total_iter - self.n_elapsed_iter
        self.expected_remaining_time = datetime.timedelta(
            seconds=iter_length_sec * n_remaining_iter
        )
        self.iter_end_time = self.start_time + datetime.timedelta(
            seconds=iter_length_sec * self.n_elapsed_iter
        )

    def test_estimate_remaining_time_happy_path(self):
        actual_remaining_time = estimate_remaining_time(
            n_elapsed_iter=self.n_elapsed_iter,
            iter_end=self.iter_end_time,
            start=self.start_time,
            total_iter_count=self.n_total_iter,
        )

        self.assertEqual(
            self.expected_remaining_time.total_seconds(),
            actual_remaining_time.total_seconds(),
        )

    def test_estimate_remaining_time_negative_elapsed_time(self):
        with self.assertRaises(ValueError):
            estimate_remaining_time(
                n_elapsed_iter=self.n_elapsed_iter,
                iter_end=self.start_time - datetime.timedelta(seconds=10),
                start=self.start_time,
                total_iter_count=self.n_total_iter,
            )

    def test_estimate_remaining_time_nonpos_elapsed_iter(self):
        for invalid_n_elapsed_iter in [0, -1]:
            with self.subTest(f"n_elapsed_iter={invalid_n_elapsed_iter}"):
                with self.assertRaises(ValueError):
                    estimate_remaining_time(
                        n_elapsed_iter=invalid_n_elapsed_iter,
                        iter_end=self.iter_end_time,
                        start=self.start_time,
                        total_iter_count=self.n_total_iter,
                    )

    def test_estimate_remaining_time_nonpos_total_iter(self):
        for invalid_total_iter_count in [0, -1]:
            with self.subTest(f"total_iter_count={invalid_total_iter_count}"):
                with self.assertRaises(ValueError):
                    estimate_remaining_time(
                        n_elapsed_iter=self.n_elapsed_iter,
                        iter_end=self.iter_end_time,
                        start=self.start_time,
                        total_iter_count=invalid_total_iter_count,
                    )
