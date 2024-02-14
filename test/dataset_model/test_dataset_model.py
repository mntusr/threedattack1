import unittest

from threedattack.dataset_model import ExactSampleCounts


class TestExactSampleCounts(unittest.TestCase):
    def test_init_invalid_sample_count(self):
        valid_sample_count = 5
        invalid_sample_count = -1

        sample_counts: list[tuple[int, int, int]] = [
            (invalid_sample_count, valid_sample_count, valid_sample_count),
            (valid_sample_count, invalid_sample_count, valid_sample_count),
            (valid_sample_count, valid_sample_count, invalid_sample_count),
        ]

        for n_train, n_test, n_val in sample_counts:
            with self.subTest(f"{n_train=};{n_test=};{n_val=}"):
                with self.assertRaises(ValueError):
                    ExactSampleCounts(
                        n_train_samples=n_train,
                        n_test_samples=n_test,
                        n_val_samples=n_val,
                    )

    def test_properties(self):
        n_train = 5
        n_test = 6
        n_val = 7

        sample_counts = ExactSampleCounts(
            n_train_samples=n_train, n_test_samples=n_test, n_val_samples=n_val
        )

        self.assertEqual(n_train, sample_counts.n_train_samples)
        self.assertEqual(n_test, sample_counts.n_test_samples)
        self.assertEqual(n_val, sample_counts.n_val_samples)

    def test_sum(self):
        sample_counts = ExactSampleCounts(
            n_train_samples=5, n_test_samples=2, n_val_samples=1
        )

        expected_sum = 8
        actual_sum = sample_counts.sum()

        self.assertEqual(expected_sum, actual_sum)

    def test_is_all_smaller_or_equal_true(self):
        smaller = ExactSampleCounts(
            n_train_samples=5, n_test_samples=2, n_val_samples=1
        )
        greater_or_equals: list[ExactSampleCounts] = [
            ExactSampleCounts(
                n_train_samples=smaller.n_train_samples + 1,
                n_test_samples=smaller.n_test_samples + 1,
                n_val_samples=smaller.n_val_samples + 1,
            ),
            ExactSampleCounts(
                n_train_samples=smaller.n_train_samples,
                n_test_samples=smaller.n_test_samples + 1,
                n_val_samples=smaller.n_val_samples + 1,
            ),
            ExactSampleCounts(
                n_train_samples=smaller.n_train_samples + 1,
                n_test_samples=smaller.n_test_samples,
                n_val_samples=smaller.n_val_samples + 1,
            ),
            ExactSampleCounts(
                n_train_samples=smaller.n_train_samples + 1,
                n_test_samples=smaller.n_test_samples + 1,
                n_val_samples=smaller.n_val_samples,
            ),
        ]

        for greater_on_equal in greater_or_equals:
            subtest_name = repr(greater_on_equal).replace(" ", "")
            with self.subTest(subtest_name):
                self.assertTrue(smaller.is_all_smaller_or_equal(greater_on_equal))

    def test_is_all_smaller_or_equal_false(self):
        smaller = ExactSampleCounts(
            n_train_samples=5, n_test_samples=2, n_val_samples=1
        )
        others = [
            ExactSampleCounts(
                n_train_samples=smaller.n_train_samples - 1,
                n_test_samples=smaller.n_test_samples + 1,
                n_val_samples=smaller.n_val_samples + 1,
            ),
            ExactSampleCounts(
                n_train_samples=smaller.n_train_samples + 1,
                n_test_samples=smaller.n_test_samples - 1,
                n_val_samples=smaller.n_val_samples + 1,
            ),
            ExactSampleCounts(
                n_train_samples=smaller.n_train_samples + 1,
                n_test_samples=smaller.n_test_samples + 1,
                n_val_samples=smaller.n_val_samples - 1,
            ),
        ]

        for other in others:
            other_repr = repr(other).replace(" ", "")
            with self.subTest(other_repr):
                self.assertFalse(smaller.is_all_smaller_or_equal(other))

    def test_equality_equal(self):
        sample_counts1 = ExactSampleCounts(
            n_train_samples=3, n_test_samples=5, n_val_samples=0
        )
        sample_counts2 = ExactSampleCounts(
            n_train_samples=3, n_test_samples=5, n_val_samples=0
        )

        self.assertTrue(sample_counts1 == sample_counts2)
        self.assertFalse(sample_counts1 != sample_counts2)

    def test_equality_not_equal(self):
        sample_counts1 = ExactSampleCounts(
            n_train_samples=3, n_test_samples=5, n_val_samples=0
        )
        other_value = 9
        not_equal_sample_counts = [
            ExactSampleCounts(
                n_train_samples=other_value,
                n_test_samples=sample_counts1.n_test_samples,
                n_val_samples=sample_counts1.n_val_samples,
            ),
            ExactSampleCounts(
                n_train_samples=sample_counts1.n_train_samples,
                n_test_samples=other_value,
                n_val_samples=sample_counts1.n_val_samples,
            ),
            ExactSampleCounts(
                n_train_samples=sample_counts1.n_train_samples,
                n_test_samples=sample_counts1.n_test_samples,
                n_val_samples=other_value,
            ),
        ]

        for i, sample_counts2 in enumerate(not_equal_sample_counts):
            with self.subTest(str(i)):
                self.assertFalse(sample_counts1 == sample_counts2)
                self.assertTrue(sample_counts1 != sample_counts2)

    def test_repr(self):
        sample_counts = ExactSampleCounts(
            n_train_samples=3, n_test_samples=5, n_val_samples=0
        )
        repr_str = repr(sample_counts)
        self.assertIn("n_train_samples=3", repr_str)
        self.assertIn("n_test_samples=5", repr_str)
        self.assertIn("n_val_samples=0", repr_str)
        self.assertTrue(repr_str.startswith(type(sample_counts).__name__ + "("))
        self.assertTrue(repr_str.endswith(")"))
        self.assertEqual(repr_str.count(","), 2)

    def test_str(self):
        sample_counts = ExactSampleCounts(
            n_train_samples=3, n_test_samples=5, n_val_samples=0
        )
        repr_str = repr(sample_counts)
        str_str = repr(sample_counts)
        self.assertEqual(repr_str, str_str)
