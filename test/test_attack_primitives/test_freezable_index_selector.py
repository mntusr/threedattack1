import unittest
from typing import Callable

import numpy as np

from threedattack._attack_primitives import FreezableIndexSelector


class TestFreezableIndexSelector(unittest.TestCase):
    def setUp(self):
        self.n_values = 50
        self.n_indices = 45
        self.constant_seed = 65

        self.frozen_random_generate_fn = lambda seed: FreezableIndexSelector(
            n_values=self.n_values,
            n_indices=self.n_indices,
            seed_if_frozen_random=seed,
            is_frozen=True,
            manual_indices=None,
        ).generate_unique(seed_if_not_frozen=13)

        self.free_generate_fn = lambda seed: FreezableIndexSelector(
            n_values=self.n_values,
            n_indices=self.n_indices,
            seed_if_frozen_random=90,
            is_frozen=False,
            manual_indices=None,
        ).generate_unique(seed_if_not_frozen=seed)

        self.MANUALLY_SET_IDX_LIST = list(range(self.n_indices))
        self.manually_set_idx_generate_fn = lambda seed: FreezableIndexSelector(
            n_values=self.n_values,
            n_indices=self.n_indices,
            seed_if_frozen_random=90,
            is_frozen=True,
            manual_indices=self.MANUALLY_SET_IDX_LIST,
        ).generate_unique(seed_if_not_frozen=seed)

        self.generate_fns_with_names: list[tuple[str, Callable[[int], np.ndarray]]] = [
            ("frozen_generate_fn", self.frozen_random_generate_fn),
            ("non_frozen_generate_fn", self.free_generate_fn),
        ]

    def test_generation_deterministic(self):
        for gen_fn_name, generate_fn in self.generate_fns_with_names:
            with self.subTest(gen_fn_name):
                self._assert_generation_deterministic(generate_fn)

    def test_generation_seed_dependence(self):
        for gen_fn_name, generate_fn in self.generate_fns_with_names:
            with self.subTest(gen_fn_name):
                self._assert_generation_seed_dependent(generate_fn)

    def test_generation_range_validity(self):
        for gen_fn_name, generate_fn in self.generate_fns_with_names:
            with self.subTest(gen_fn_name):
                self._assert_generation_in_range(
                    n_values=self.n_values, gen_fn=generate_fn
                )

    def test_generation_unique(self):
        for gen_fn_name, generate_fn in self.generate_fns_with_names:
            with self.subTest(gen_fn_name):
                self._assert_generation_unique(generate_fn)

    def test_freeze_has_effect(self):
        for manual_indices in [None, self.MANUALLY_SET_IDX_LIST]:
            with self.subTest(str(manual_indices).replace(" ", "")):
                generator = FreezableIndexSelector(
                    n_values=self.n_values,
                    n_indices=self.n_indices,
                    seed_if_frozen_random=9,
                    is_frozen=True,
                    manual_indices=manual_indices,
                )

                idxs1 = generator.generate_unique(seed_if_not_frozen=13)
                idxs2 = generator.generate_unique(seed_if_not_frozen=50)

                self.assertTrue(np.array_equal(idxs1, idxs2))

    def test_argument_validation(self):
        cases: list[tuple[str, tuple[int, int, bool, list[int] | None]]] = [
            ("negative_n_values", (-1, self.n_indices, True, None)),
            ("zero_n_values", (0, self.n_indices, True, None)),
            ("negative_n_indices", (self.n_values - 1, -1, True, None)),
            ("zero_n_indices", (self.n_values, 0, True, None)),
            ("n_indices_too_big", (self.n_values, self.n_values + 1, True, None)),
            (
                "too_many_manual_idxs",
                (
                    self.n_values,
                    self.n_values + 1,
                    True,
                    list(range(self.n_indices + 1)),
                ),
            ),
            (
                "too_small_manual_indices",
                (
                    self.n_values,
                    self.n_values + 1,
                    True,
                    [0] * (self.n_values - 1) + [-1],
                ),
            ),
            (
                "too_big_manual_indices",
                (
                    self.n_values,
                    self.n_values + 1,
                    True,
                    self.MANUALLY_SET_IDX_LIST[0 : self.n_indices - 1]
                    + [self.n_values + 1],
                ),
            ),
            (
                "duplicate_manual_indices",
                (self.n_values, self.n_values + 1, True, [0] * self.n_indices),
            ),
            (
                "free_generation_with_manual_indices_set",
                (self.n_values, self.n_values + 1, False, self.MANUALLY_SET_IDX_LIST),
            ),
        ]

        for name, (n_values, n_indices, is_frozen, manual_indices) in cases:
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    FreezableIndexSelector(
                        n_values=n_values,
                        n_indices=n_indices,
                        seed_if_frozen_random=9,
                        is_frozen=is_frozen,
                        manual_indices=manual_indices,
                    )

    def test_manual_idx_setting(self):
        actual_indices = self.manually_set_idx_generate_fn(43)
        expected_indices = np.array(self.MANUALLY_SET_IDX_LIST)
        self.assertTrue(np.array_equal(actual_indices, expected_indices))

    def _assert_generation_deterministic(self, gen_fn: Callable[[int], np.ndarray]):
        SEED = 5
        idxs1 = gen_fn(SEED)
        idxs2 = gen_fn(SEED)

        self.assertTrue(np.array_equal(idxs1, idxs2))

    def _assert_generation_seed_dependent(self, gen_fn: Callable[[int], np.ndarray]):
        idxs1 = gen_fn(3)
        idxs2 = gen_fn(9)

        self.assertFalse(np.array_equal(idxs1, idxs2))

    def _assert_generation_in_range(
        self, n_values: int, gen_fn: Callable[[int], np.ndarray]
    ):
        idxs = gen_fn(15)
        self.assertTrue(np.all(idxs < n_values))

    def _assert_generation_unique(self, gen_fn: Callable[[int], np.ndarray]):
        idxs = gen_fn(3)

        unique_idxs = np.unique(idxs)

        self.assertEqual(idxs.shape, unique_idxs.shape)
