import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional, TypeVar

_temp_file_ctr = 0
_temp_folder_path: Optional[Path] = None


def new_temp_file(suffix: str) -> Path:
    """
    Create a new file in the automatically managed temporary directory.

    This function requires the automatically managed temporary directory to be created.

    Parameters
    ----------
    suffix
        The end of the name of the file to create.

    Returns
    -------
    v
        The path of the created file.
    """
    global _temp_file_ctr
    assert _temp_folder_path is not None, "No temporary folder is specified."

    file_path = _temp_folder_path / f"{_temp_file_ctr}{suffix}"

    file_path.touch()

    _temp_file_ctr += 1

    return file_path


class GlobalTempFolder:
    """
    This monitor initializes the temporary directory on enter and deletes it recursively on exit.

    This module should be used before any temporary directory access via this subpackage.
    """

    def __enter__(self) -> None:
        global _temp_folder_path

        assert (
            _temp_folder_path is None
        ), "The temporary directory should not be initialized twice."

        temp_folder_path_str = tempfile.mkdtemp()

        _temp_folder_path = Path(temp_folder_path_str)

    def __exit__(self, *argv, **argc) -> None:
        global _temp_folder_path

        shutil.rmtree(str(_temp_folder_path))
        _temp_folder_path = None
