from __future__ import annotations

import random as _random
from typing import Iterable, List, Sequence

float32 = "float32"
int32 = "int32"
int8 = "int8"


class ndarray:
    def __init__(self, data: Iterable):
        if isinstance(data, ndarray):
            self._data = list(data._data)
        else:
            self._data = list(data)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._data),)

    @property
    def size(self) -> int:
        return len(self._data)

    def sum(self) -> float:
        return sum(self._data)

    def astype(self, _dtype: str):
        if _dtype in (int32, int8):
            return ndarray(int(bool(item)) for item in self._data)
        if _dtype == float32:
            return ndarray(float(item) for item in self._data)
        return ndarray(self._data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return ndarray(self._data[key])
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __gt__(self, other):
        return ndarray(item > other for item in self._data)

    def __repr__(self) -> str:
        return f"array({self._data})"


class Generator:
    def __init__(self, seed=None):
        self._rng = _random.Random(seed)

    def choice(self, seq: Sequence):
        return self._rng.choice(list(seq))

    def integers(self, low: int, high: int | None = None) -> int:
        if high is None:
            high = low
            low = 0
        return self._rng.randrange(low, high)


class _RandomModule:
    Generator = Generator

    @staticmethod
    def default_rng(seed=None) -> Generator:
        return Generator(seed)


random = _RandomModule()


def asarray(data: Iterable, dtype: str | None = None) -> ndarray:
    arr = ndarray(data)
    if dtype is not None:
        return arr.astype(dtype)
    return arr


def flatnonzero(data: Iterable) -> List[int]:
    return [idx for idx, value in enumerate(data) if value]


def argmax(data: Iterable) -> int:
    seq = list(data)
    if not seq:
        raise ValueError("argmax of empty sequence")
    max_value = max(seq)
    return seq.index(max_value)
