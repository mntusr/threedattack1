"""
This module manages an automatically (recursively) deleted temporary directory that can be used throughout the whole program.
"""

from ._tempfolder import GlobalTempFolder, new_temp_file

__all__ = ["GlobalTempFolder", "new_temp_file"]
