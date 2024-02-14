import datetime


def estimate_remaining_time(
    start: datetime.datetime,
    iter_end: datetime.datetime,
    n_elapsed_iter: int,
    total_iter_count: int,
) -> datetime.timedelta:
    """
    Estimate the remaining time for iteration-based calculations.

    This function assumes that each iteration takes about the same time to complete.

    Parameters
    ----------
    start
        The start time of the whole operation.
    current_iter_end
        The end time of the just ended iteration.
    n_elapsed_iter
        The number of the elapsed iterations.
    total_iter_count
        The total number of iterations.

    Returns
    -------
    v
        The estimated remaining time.

    Raises
    ------
    ValueError
        If the number of the elapsed iterations is not positive or the total number of iterations is not positive or the elapsed time is not positive.
    """
    if n_elapsed_iter < 1:
        raise ValueError(
            f"The number of the elapsed iterations is not positive. Value: {n_elapsed_iter}"
        )

    if total_iter_count < 1:
        raise ValueError(
            f"The total count of iterations is not positive. Value: {total_iter_count}"
        )

    elapsed_time = iter_end - start

    if elapsed_time.total_seconds() < 0:
        raise ValueError(f"The elapsed time is negative. Elapsed time: {elapsed_time}")

    total_estim_time = elapsed_time / n_elapsed_iter * total_iter_count

    remaining_estim_time = total_estim_time - elapsed_time
    return remaining_estim_time
