from typing import TYPE_CHECKING, Protocol

import numpy as np

from ._typing import type_instance
from .tensor_types.npy import *


class LoggingFreqFunction(Protocol):
    def needs_logging(self, iteration: int) -> bool:
        """
        Check whether the results at the current iteration should be logged.

        Parameters
        ----------
        iteration
            The current iteration.

        Returns
        -------
        v
            True if the results should be logged, otherwise False.

        Raises
        ------
        ValueError
            If the iteration is negative.
        """
        ...

    def get_fn_repr(self) -> str:
        """
        Get the string representation of the function.
        """
        ...


class StepLoggingFreqFunction:
    """
    The logging frequency function that specifies the logging frequency in discrete steps.

    Parameters
    ----------
    steps
        The start of the steps after 0.
    freqencies
        The logging frequencies at each step. The element at index 0 belongs to the logging frequency at generation 0. The latter values belong to the corresponding step starts.

    Raises
    ------
    ValueError
        If the number of steps after 0 is not equal to the number of frequencies or at least one frequency is non-positive.
    """

    def __init__(self, steps: list[int], freqencies: list[int]) -> None:
        if len(steps) != len(freqencies) - 1:
            raise ValueError(
                f"The number of frequency step starts after 0 ({steps}) should be equal to the number of the specified logging frequencies-1."
            )
        if any(freq <= 0 for freq in freqencies):
            raise ValueError("At least one logging frequency is non-positive.")

        steps_with_0_array = np.array([0] + steps)
        frequencies_array = np.array(freqencies)

        self._step_sort_idxs = np.argsort(steps_with_0_array)

        self._sorted_steps = np.array(
            idx_svals_int(steps_with_0_array, v=self._step_sort_idxs)
        )
        """
        Format: ``svals::int``
        """

        self._freqs_for_sorted_steps = np.array(
            idx_svals_int(frequencies_array, v=self._step_sort_idxs)
        )
        """
        Format: ``svals::int``
        """

    def needs_logging(self, iteration: int) -> bool:
        if iteration < 0:
            raise ValueError(f"The iteration is negative ({iteration}).")

        step_idx = len(self._sorted_steps[self._sorted_steps <= iteration]) - 1
        logging_frequency = int(idx_svals_int(self._freqs_for_sorted_steps, v=step_idx))

        return iteration % logging_frequency == 0

    def get_fn_repr(self) -> str:
        step_list = [int(v) for v in self._sorted_steps][1:]
        freq_list = [int(v) for v in self._freqs_for_sorted_steps]
        return f"LoggingFreqFunction(steps={step_list}, freqencies={freq_list})"


if TYPE_CHECKING:
    v: LoggingFreqFunction = type_instance(StepLoggingFreqFunction)
