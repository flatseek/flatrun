"""Core types shared across the storage and runtime layers.

Anything in this package must be free of dependencies on the storage
backend or the scheduler - that is what makes it possible for the
backend module to expose :class:`TensorHandle` to the runtime without
a circular import.
"""

from .tensor import (
    BufferTensorHandle,
    HandleSource,
    MmapTensorHandle,
    TensorHandle,
    TensorView,
)

__all__ = [
    "BufferTensorHandle",
    "HandleSource",
    "MmapTensorHandle",
    "TensorHandle",
    "TensorView",
]