import subprocess
import sys
import unittest

import numpy as np

from foreign_model_commons import RemoteDepthEstWorkerMainLoop
from threedattack.target_model import (
    RemoteDepthEstController,
    RemoteProcessingShutDownError,
)
from threedattack.tensor_types.npy import *


class TestCommunication(unittest.TestCase):
    def test_communication_happy_path(self):
        remote_code = "import sys; from foreign_model_commons import RemoteDepthEstWorkerMainLoop; RemoteDepthEstWorkerMainLoop('testmem').wait_for_input(lambda a: a.mean(1, keepdims=True)+1 if a.max()<100 else exec('raise KeyboardInterrupt()'));sys.exit(0)"

        rem_proc = None
        USE_EXTERNALLY_SPAWNED_REM_PROC = False  # useful for debugging purposes

        if not USE_EXTERNALLY_SPAWNED_REM_PROC:
            current_python = sys.executable
            rem_proc = subprocess.Popen(
                [current_python, "-c", remote_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        main = RemoteDepthEstController("testmem")

        image = np.zeros(newshape_im_rgbs(n=3, h=5, w=7))
        for _ in range(40):
            depth = main.process_async(image).get()
            self.assertTrue(match_im_depthmaps(depth, shape={"n": 3, "h": 5, "w": 7}))
            image = np.broadcast_to(depth, image.shape)

        data_val: float = float(image.max())

        self.assertAlmostEqual(data_val, 40, places=4)

        with self.assertRaises(RemoteProcessingShutDownError):
            main.process_async(np.full_like(image, fill_value=611)).get()

        if rem_proc is not None:
            return_code = rem_proc.wait()
            self.assertEqual(return_code, 0)
