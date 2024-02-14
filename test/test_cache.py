import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from threedattack.script_util import CsvCache


class TestCache(unittest.TestCase):
    def test_cache_happy_path(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            cache = CsvCache(td_path)
            call_counts = {"fn1": 0, "fn2": 0}

            @cache.cached(Path("cached_fn1.csv"))
            def cached_fn1(v: str) -> pd.DataFrame:
                call_counts["fn1"] += 1
                return pd.DataFrame({"a1": [1, 2], "b1": [3, 4]})

            @cache.cached(Path("cached_fn2.csv"))
            def cached_fn2(v: str) -> pd.DataFrame:
                call_counts["fn2"] += 1
                return pd.DataFrame({"a2": [3, 4], "b2": [5, 6]})

            fn1_frame1 = cached_fn1("a")
            fn1_frame2 = cached_fn1("b")

            fn2_frame1 = cached_fn2("a")
            fn2_frame2 = cached_fn2("a")

            self.assertEqual(list(fn1_frame1.columns), ["a1", "b1"])
            self.assertEqual(list(fn1_frame2.columns), ["a1", "b1"])
            self.assertEqual(list(fn2_frame1.columns), ["a2", "b2"])
            self.assertEqual(list(fn2_frame2.columns), ["a2", "b2"])

            self.assertIsNot(fn1_frame1, fn1_frame2)
            self.assertIsNot(fn2_frame1, fn2_frame2)

            self.assertEqual(call_counts["fn1"], 1)
            self.assertEqual(call_counts["fn2"], 1)

    def test_cache_path_not_exists(self):
        cache = CsvCache(Path("not_existing_dir"))
        call_counts = {"fn1": 0, "fn2": 0}

        @cache.cached(Path("cached_fn1.csv"))
        def cached_fn1(v: str) -> pd.DataFrame:
            call_counts["fn1"] += 1
            return pd.DataFrame({"a1": [1, 2], "b1": [3, 4]})

        with self.assertRaises(RuntimeError):
            cached_fn1("a")
