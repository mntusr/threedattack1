import socket
from multiprocessing import shared_memory
from time import sleep
from typing import Any

import numpy as np


class RemoteDepthEstController:
    MAX_BATCH_SIZE = 100
    IM_DTYPE = np.float32
    DIM_DATA_DTYPE = np.int16
    DIMS_ARR_SHAPE = (3,)
    MAX_WIDTH = 700
    MAX_HEIGHT = 650

    def __init__(
        self,
        shared_mem_stem: str,
    ):
        in_shape = (self.MAX_BATCH_SIZE, 3, self.MAX_HEIGHT, self.MAX_WIDTH)
        out_shape = (self.MAX_BATCH_SIZE, 1, self.MAX_HEIGHT, self.MAX_WIDTH)
        dims_size = _get_array_size(self.DIM_DATA_DTYPE, self.DIMS_ARR_SHAPE)
        in_size = _get_array_size(self.IM_DTYPE, in_shape)
        out_size = _get_array_size(self.IM_DTYPE, out_shape)
        self._pred_in_progress = False

        while True:
            try:
                self._dims_shared_memory = shared_memory.SharedMemory(
                    name=f"{shared_mem_stem}_dims", create=False, size=dims_size
                )
                self._input_shared_memory = shared_memory.SharedMemory(
                    name=f"{shared_mem_stem}_input", create=False, size=in_size
                )
                self._output_shared_memory = shared_memory.SharedMemory(
                    name=f"{shared_mem_stem}_output", create=False, size=out_size
                )
                break
            except:
                sleep(0.1)

        self._dims_shm_array = np.ndarray(
            shape=self.DIMS_ARR_SHAPE,
            dtype=self.DIM_DATA_DTYPE,
            buffer=self._dims_shared_memory.buf,
        )
        self._input_shm_array = np.ndarray(
            shape=in_shape,
            dtype=self.IM_DTYPE,
            buffer=self._input_shared_memory.buf,
        )
        self._output_shm_array = np.ndarray(
            shape=out_shape,
            dtype=self.IM_DTYPE,
            buffer=self._output_shared_memory.buf,
        )

    def process_async(self, ims: np.ndarray) -> "RemoteDepthEstControllerPredFuture":
        """
        Do a depth prediction.

        Parameters
        ----------
        im
            The images. Format: ``Im::RGBs``

        Returns
        -------
        v
            An object that resolves the native depth predictions.

        Raises
        ------
        PredInProgressError
            A prediction is already in progress.
        """
        if self._pred_in_progress:
            raise PredInProgressError("A depth prediction is already in progress.")

        current_batch_size = ims.shape[0]
        n_samples, h, w = ims.shape[0], ims.shape[2], ims.shape[3]
        self._dims_shm_array[0] = n_samples
        self._dims_shm_array[1] = h
        self._dims_shm_array[2] = w

        self._input_shm_array[:current_batch_size, :, :h, :w] = ims[:]

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("localhost", 7515))
        s.send(bytes([1]))
        self._pred_in_progress = True

        return RemoteDepthEstControllerPredFuture(
            socket=s, current_batch_size=current_batch_size, h=h, w=w, owner=self
        )


class RemoteDepthEstControllerPredFuture:
    def __init__(
        self,
        socket: socket.socket,
        current_batch_size: int,
        h: int,
        w: int,
        owner: RemoteDepthEstController,
    ):
        self._socket = socket
        self._owner = owner
        self._current_batch_size = current_batch_size
        self._h = h
        self._w = w

    def get(self) -> np.ndarray:
        """
        Block until the native depth predictions are created, then return.

        This function returns immediately if the predictions are already created.

        Returns
        -------
        v
            The native depth predictions. ``ArbSamples::*``

        Raises
        ------
        RemoteProcessingShutDownError
            The remote processing shut down before processing this request.
        """
        try:
            response_code = self._socket.recv(1)
        finally:
            self._owner._pred_in_progress = False

        if response_code == bytes([2]):
            raise RemoteProcessingShutDownError(
                "The remote processing failed with some error. See the logs of the remote process for more details."
            )
        self._socket.close()

        return self._owner._output_shm_array[
            : self._current_batch_size, :, : self._h, : self._w
        ].copy()


class PredInProgressError(Exception):
    """
    This means that a depth prediction is already in progress.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class RemoteProcessingShutDownError(Exception):
    """
    This error means that the worker shut down while processing the request.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def _get_array_size(dtype: Any, shape: tuple[int, ...]) -> int:
    v = np.array((0,), dtype=dtype)
    return int(np.prod(shape)) * v.dtype.itemsize
