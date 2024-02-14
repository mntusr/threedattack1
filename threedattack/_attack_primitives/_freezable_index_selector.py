from enum import Enum

import numpy as np
from numpy.random import f


class FreezableIndexSelector:
    """
    Generate random indices.

    This index selector might be frozen. If it is frozen, then

    Parameters
    ----------
    n_values
        The number of the values with index.
    n_indices
        The number of indices to generate.
    seed_if_frozen_random
        The seed to generate the indices if the index selector is frozen, but randomly selected.
    is_frozen
        True if the index selector should be frozen.

    Raises
    ------
    ValueError
        If the number of values with index is less than 1, or the number of indices to generate is less than 1.
        If there are manually specified indices, but the index selection is not frozen.
        If the number of the manually set indices is not equal the number of indices to generate.
    """

    def __init__(
        self,
        n_values: int,
        n_indices: int,
        seed_if_frozen_random: int | None,
        manual_indices: list[int] | None,
        is_frozen: bool,
    ):
        if n_values < 1:
            raise ValueError(
                f"The number of the values with index ({n_values}) is less than 1."
            )
        if n_indices < 1:
            raise ValueError(
                f"The number of indices to generate ({n_values}) is less than 1."
            )

        if n_indices > n_values:
            raise ValueError(
                f"The number of the indices to generate ({n_indices}) is greater than the total number of values with index ({n_values})."
            )

        if is_frozen:
            if manual_indices is not None:
                if len(manual_indices) != n_indices:
                    raise ValueError(
                        f"The length of the manually set indices ({len(manual_indices)}) is not equal the number of indices to generate ({n_indices})."
                    )

                manual_idx_array = np.array(manual_indices)

                if not (
                    np.all(manual_idx_array >= 0)
                    and np.all(manual_idx_array < n_values)
                ):
                    raise ValueError(
                        f"The list of manually set indices ({manual_indices}) contains elements that are smaller than 0 or greater than the number of values with index ({n_values})."
                    )

                if not (len(np.unique(manual_idx_array)) == len(manual_idx_array)):
                    raise ValueError(
                        f"The list of manually set indices ({manual_indices}) contains duplicate elements."
                    )
                self._frozen_viewpoint_indices = np.array(manual_indices)
            else:
                self._frozen_viewpoint_indices = self.__generate_new_unique_indices(
                    n_indices=n_indices,
                    n_values=n_values,
                    seed=seed_if_frozen_random,
                )
        else:
            if manual_indices is not None:
                raise ValueError(
                    "There are manually set indices, but the index selection is not frozen."
                )
            self._frozen_viewpoint_indices = None

        self._n_viewpoints_in_scene = n_values
        self._n_indices = n_indices

    def get_n_values(self) -> int:
        """Get the number of the number of the values with index."""
        return self._n_viewpoints_in_scene

    def get_n_indices(self) -> int:
        """Get the number of the indexes to generate."""
        return self._n_indices

    def is_frozen(self) -> bool:
        """Returns true if the object is frozen."""
        return self._frozen_viewpoint_indices is not None

    def generate_unique(self, seed_if_not_frozen: int | None) -> np.ndarray:
        """
        Generate random indices.

        If the object is frozen then this function always returns with the copy of the same values.

        Parameters
        ----------
        seed_if_not_frozen
            The seed of the index generation if the object is not frozen.

        Returns
        -------
        v
            The array of the generated indices. It always returns with a separate array.Format: ``Scalars::Int``
        -----
        """
        if self._frozen_viewpoint_indices is not None:
            return self._frozen_viewpoint_indices.copy()
        else:
            return self.__generate_new_unique_indices(
                n_indices=self._n_indices,
                n_values=self._n_viewpoints_in_scene,
                seed=seed_if_not_frozen,
            )

    @staticmethod
    def __generate_new_unique_indices(
        n_indices: int, n_values: int, seed: int | None
    ) -> np.ndarray:
        """
        Generate new unique indces.

        Parameters
        ----------
        n_indices
            The number of indices to generate.
        n_values
            The number of values with index.
        seed
            The seed for the random index generation.

        Returns
        -------
        v
            The array of generated indexes. Fromat: ``Scalars::Int``

        Raises
        ------
        ValueError
            If the number of values with index is less than 1, or the number of indices to generate is less than 1.


        See Also
        --------

        Notes
        -----
        """
        if n_values < 1:
            raise ValueError(
                f"The number of the values with index ({n_values}) is less than 1."
            )
        if n_indices < 1:
            raise ValueError(
                f"The number of indices to generate ({n_values}) is less than 1."
            )

        rng = np.random.default_rng(seed)

        return rng.choice(n_values, size=n_indices, replace=False)
