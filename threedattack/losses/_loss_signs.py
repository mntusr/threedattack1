from ._loss_functions import RawLossFn


def sign_loss_to_make_smaller_mean_worse_predictor(
    loss_fn: RawLossFn, val: float
) -> float:
    """
    This function changes the sign of the loss function to make sure that the smaller values mean that the depth predictor is worse (i. e. an adversarial attack is more successful).

    Parameters
    ----------
    loss_fn
        The loss function to which the value belongs.
    val
        The value.

    Returns
    -------
    v
        The new value with the proper sign.
    """
    if loss_fn in [RawLossFn.D1, RawLossFn.CroppedD1]:
        return val
    else:
        return -val
