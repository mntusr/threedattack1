from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pandas as pd
from mlflow.utils.file_utils import is_directory

T = TypeVar("T")


class CsvCache:
    """
    Adds a simple way to cache the results of calculations to a csv file.

    Parameters
    ----------
    cache_dir
        The directory to store the cached results.
    """

    def __init__(self, cache_dir: Path):
        self.__cache_file_set: set[Path] = set()
        self.__cache_dir = cache_dir

    def cached(
        self, csv_path: Path
    ) -> Callable[[Callable[[T], pd.DataFrame]], Callable[[T], pd.DataFrame],]:
        """
        A decorator that automatically saves the returned data frame and saves it to a csv file.

        This function skips the indexes during the save.

        The return values provided by the latter calls of the decorated function will be reloaded from the csv file instead.

        The caching does not care the arguments of the function.

        The calls of the decorated functions will raise `RuntimeError` if the cache directory does not exist or it is not a file.
        """
        abs_cache_path = (self.__cache_dir / csv_path).resolve()
        if abs_cache_path in self.__cache_file_set:
            raise ValueError(
                f'The path "{abs_cache_path}" is already used for an other cache.'
            )
        else:
            self.__cache_file_set.add(abs_cache_path)

        def fn2(in_fn: Callable[[T], pd.DataFrame]) -> Callable[[T], pd.DataFrame]:
            def fn(
                arg1: T,
            ) -> pd.DataFrame:
                if not self.__cache_dir.is_dir():
                    raise RuntimeError(
                        f'The cache directory "{self.__cache_dir}" does not exist.'
                    )

                if abs_cache_path.is_file():
                    return pd.read_csv(abs_cache_path)
                else:
                    frame = in_fn(arg1)
                    frame.to_csv(abs_cache_path, index=False)

                    return frame

            return fn

        return fn2
