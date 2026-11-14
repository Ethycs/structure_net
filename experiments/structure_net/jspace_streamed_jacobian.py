"""Read-only, bounded row access for the unchanged task-Jacobian fitter.

The fitter's randomized SVD consumes tensor metadata and contiguous row slices;
it does not require the complete Jacobian to reside on its compute device.
Each access copies at most ``max_rows`` from an .npy memmap into owned CPU
storage and performs a blocking device transfer. No asynchronous staging buffer
is reused. CUDA callers remain responsible for acquiring their experiment lease.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class StreamedJacobianRows:
    """Tensor-like matrix metadata plus bounded slices for Gröbner anchor fitting.

    Files must be nonempty native-endian FP32/FP64 .npy matrices. Finiteness is
    deliberately checked by the existing fitter as it visits each chunk, rather
    than scanning or materializing the entire file here. Set max_rows at least
    as large as the fitter's chunk_rows. Constructor/device metadata access does
    not allocate anything on CUDA; the first requested CUDA slice transfers data.
    """

    def __init__(self, path, *, device="cpu", max_rows=256):
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
            raise ValueError("max_rows must be a positive integer")
        self.path = Path(path).resolve()
        self.device = torch.device(device)
        self.max_rows = max_rows
        matrix = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if not isinstance(matrix, np.memmap):
            if hasattr(matrix, "close"):
                matrix.close()
            raise ValueError("Jacobian source must be a read-only .npy memmap")
        if matrix.ndim != 2 or min(matrix.shape) <= 0:
            matrix._mmap.close()
            raise ValueError("Jacobian source must be a nonempty matrix")
        if matrix.dtype not in (np.dtype("float32"), np.dtype("float64")) or not matrix.dtype.isnative:
            matrix._mmap.close()
            raise ValueError("Jacobian source must use native-endian float32 or float64")
        if matrix.flags.writeable:
            matrix._mmap.close()
            raise ValueError("Jacobian memmap must be read-only")
        self._matrix = matrix
        self.shape = tuple(matrix.shape)
        self.ndim = 2
        self.dtype = torch.float64 if matrix.dtype == np.dtype("float64") else torch.float32
        self.read_calls = self.rows_read = self.bytes_read = self.maximum_read_rows = 0

    def __len__(self):
        return self.shape[0]

    def is_floating_point(self):
        return True

    @property
    def readonly(self):
        return True

    def __getitem__(self, index):
        if self._matrix is None:
            raise ValueError("Jacobian source is closed")
        if not isinstance(index, slice) or index.step not in (None, 1):
            raise TypeError("Jacobian source supports contiguous row slices only")
        start, stop, _ = index.indices(len(self))
        rows = max(stop - start, 0)
        if rows > self.max_rows:
            raise ValueError("Requested Jacobian slice exceeds max_rows; use bounded fitter chunks")
        # The small owned copy avoids writable-Tensor aliases of a read-only
        # mmap and keeps returned CPU tensors valid after the source closes.
        cpu = torch.from_numpy(np.array(self._matrix[start:stop], copy=True, order="C"))
        result = cpu.to(device=self.device, dtype=self.dtype, non_blocking=False)
        self.read_calls += 1
        self.rows_read += rows
        self.bytes_read += cpu.numel() * cpu.element_size()
        self.maximum_read_rows = max(self.maximum_read_rows, rows)
        return result

    def metadata(self):
        return {"path": str(self.path), "shape": list(self.shape), "dtype": str(self.dtype),
                "device": str(self.device), "readonly": True, "max_rows": self.max_rows,
                "transfer": "owned bounded CPU copy; blocking device transfer",
                "read_calls": self.read_calls, "rows_read": self.rows_read,
                "bytes_read": self.bytes_read, "maximum_read_rows": self.maximum_read_rows}

    def close(self):
        if self._matrix is not None:
            self._matrix._mmap.close()
            self._matrix = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
