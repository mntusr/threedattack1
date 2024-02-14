from typing import TypeVar

T = TypeVar("T")


def type_instance(_: type[T], /) -> T:
    """
    A simple function to create a fake instance of the specified type for static type checking.

    This function should not be invoked runtime.
    """
    raise NotImplementedError(
        "The goal of this function is to implement fake instances for static type checking. This function should not be invoked runtime."
    )
