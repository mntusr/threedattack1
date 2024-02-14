import socket
from http import client
from multiprocessing import shared_memory
from time import sleep
from typing import Any, Callable

import numpy as np


class RemoteDepthEstWorkerMainLoop:
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

        self._dims_shared_memory = shared_memory.SharedMemory(
            name=f"{shared_mem_stem}_dims", create=True, size=dims_size
        )
        self._input_shared_memory = shared_memory.SharedMemory(
            name=f"{shared_mem_stem}_input", create=True, size=in_size
        )
        self._output_shared_memory = shared_memory.SharedMemory(
            name=f"{shared_mem_stem}_output", create=True, size=out_size
        )

        self._dims_shm_array = np.ndarray(
            shape=self.DIMS_ARR_SHAPE,
            dtype=self.DIM_DATA_DTYPE,
            buffer=self._dims_shared_memory.buf,
        )
        self._input_shm_array = np.ndarray(
            shape=in_shape, dtype=self.IM_DTYPE, buffer=self._input_shared_memory.buf
        )
        self._output_shm_array = np.ndarray(
            shape=out_shape, dtype=self.IM_DTYPE, buffer=self._output_shared_memory.buf
        )

    def wait_for_input(self, processor: Callable[[np.ndarray], np.ndarray]) -> None:
        print("Waiting for requests. Press Ctrl+C to quit")
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind(("localhost", 7515))
        serversocket.listen(1)
        serversocket.settimeout(2)

        client_socket = None

        try:
            while True:
                try:
                    try:
                        (client_socket, _) = serversocket.accept()
                    except socket.timeout:
                        pass
                    else:
                        successful_processing = False
                        _ = client_socket.recv(1)[0]

                        n_items, h, w = self._dims_shm_array
                        try:
                            current_input_array = self._input_shm_array[
                                :n_items, :, :h, :w
                            ]
                            proc_result = processor(current_input_array)
                            self._output_shm_array[:n_items, :, :h, :w] = proc_result[:]
                            successful_processing = True
                        finally:
                            if successful_processing:
                                client_socket.send(bytes([1]))
                            else:
                                client_socket.send(bytes([2]))
                            client_socket = None
                except ConnectionResetError:
                    client_socket = None
        except KeyboardInterrupt:
            if client_socket is not None:
                # The worker may send two bytes if the keyboard interrupt happens
                # before the client socket is set to None.
                # However, this is not a problem, since the
                # controller expects only one byte, then closes the connection.
                client_socket = client_socket.send(bytes([2]))

        print("Shutting down...")


def _get_array_size(dtype: Any, shape: tuple[int, ...]) -> int:
    v = np.array((0,), dtype=dtype)
    return int(np.prod(shape)) * v.dtype.itemsize
