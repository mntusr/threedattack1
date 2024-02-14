class Panda3dAssumptionViolation(Exception):
    """
    This exception means that a simplifying assumption about a Panda3d scene is violated.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
