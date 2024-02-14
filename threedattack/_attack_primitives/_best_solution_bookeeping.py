from typing import Generic, TypeVar

T = TypeVar("T")


class BestSolutionKeeper(Generic[T]):
    """
    A class to keep track of arbitrary data for the best solutions in an evolutionary algorithm.
    """

    def __init__(self):
        self._best_solution: tuple[float, T] | None = None

    def update(self, fitness: float, gt_data: T) -> None:
        """
        Update the data for the best solution if the fitness of the new solution is smaller than the fitness of the current best solution.

        Parameters
        ----------
        data
            The data for the new solution.
        """
        if (
            (self._best_solution is not None)
            and (self._best_solution[0] > fitness)
            or (self._best_solution is None)
        ):
            self._best_solution = (fitness, gt_data)

    def get_best(self) -> T:
        """
        Get the data for the best solution if present.

        Returns
        -------
        v
            The data for the best solution.

        Raises
        ------
        Exception
            If there is no stored data yet.
        """
        if self._best_solution is not None:
            return self._best_solution[1]
        else:
            raise Exception("There is no logged solution yet.")
