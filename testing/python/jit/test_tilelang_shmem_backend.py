# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""SHMEM host/device ABI selection must also separate cached kernels."""

import pytest

from tilelang.jit.adapter.libgen import get_shmem_backend


def test_shmem_backend_default(monkeypatch):
    monkeypatch.delenv("TL_SHMEM_BACKEND", raising=False)
    assert get_shmem_backend() == "hybm"


@pytest.mark.parametrize("backend", ["default", "hybm"])
def test_shmem_backend_selection(monkeypatch, backend):
    monkeypatch.setenv("TL_SHMEM_BACKEND", backend)
    assert get_shmem_backend() == backend


def test_shmem_backend_rejects_invalid(monkeypatch):
    monkeypatch.setenv("TL_SHMEM_BACKEND", "unknown")
    with pytest.raises(ValueError, match="TL_SHMEM_BACKEND"):
        get_shmem_backend()
