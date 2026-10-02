# Copyright (c) 2026 Tile-AI.
# Licensed under the MIT License. See LICENSE in the repository root.

"""Source-only regressions for public SHMEM resources and signal ordering."""

import re

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tir


_MANUAL_CONFIG = {
    "tl.ascend_auto_sync": False,
    "tl.ascend_memory_planning": False,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def _buffer(name: str, extent: int, dtype: str = "float32", scope: str = "global"):
    return tir.decl_buffer((extent,), dtype, name=name, scope=scope)


def _region(buffer, start: int, extent: int):
    return tir.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(start, extent)])


def _lower(func) -> str:
    with tvm.transform.PassContext(opt_level=3, config=_MANUAL_CONFIG):
        return tilelang.lower(func, target="ascendc", platform="A3").kernel_source


@pytest.mark.parametrize("operation", [T.shmem_get_nbi, T.shmem_put_nbi])
def test_gm_nbi_has_explicit_offsets_scratch_bytes_and_event(operation):
    dst = _buffer("dst", 128)
    src = _buffer("src", 128)
    scratch = _buffer("scratch", 64, scope="shared.ub")
    call = operation(
        _region(dst, 17, 32),
        _region(src, 9, 32),
        32,
        1,
        scratch=_region(scratch, 8, 16),
        event_id=3,
    )
    assert len(call.args) == 8
    assert int(call.args[1].args[2]) == 17
    assert int(call.args[2].args[2]) == 9
    assert int(call.args[3].args[2]) == 8
    assert int(call.args[3].args[3]) == 16
    assert int(call.args[4]) == 64
    assert int(call.args[5]) == 32
    assert int(call.args[6]) == 1
    assert int(call.args[7]) == 3
    assert int(call.op.get_attr("TCallEffectKind")) == 3  # kOpaque


@pytest.mark.parametrize("dtype", ["float16", "float32", "bfloat16"])
def test_scratch_capacity_uses_sliced_arena(dtype: str):
    payload = _buffer("payload", 64, dtype)
    scratch = _buffer("scratch", 256, dtype, "shared.ub")
    element_bytes = tvm.DataType(dtype).itemsize()
    first = 32 // element_bytes
    extent = 64 // element_bytes
    call = T.shmem_get_nbi(
        payload,
        payload,
        16,
        1,
        scratch=_region(scratch, first, extent),
        event_id=1,
    )
    assert int(call.args[4]) == 64


@pytest.mark.parametrize(
    "scratch,match",
    [
        (_buffer("scratch", 64), "scope shared.ub"),
        (_buffer("scratch", 0, scope="shared.ub"), "positive multiple"),
        (_buffer("scratch", 7, scope="shared.ub"), "positive multiple"),
        (_buffer("scratch", 32, "int32", "shared.ub"), "dtype must match"),
        (_region(_buffer("scratch", 64, scope="shared.ub"), 1, 16), "32-byte aligned"),
        (_region(_buffer("scratch", 64, scope="shared.ub"), 56, 16), "exceeds backing"),
    ],
)
def test_invalid_scratch_fails_before_lowering(scratch, match: str):
    payload = _buffer("payload", 64)
    with pytest.raises((TypeError, ValueError), match=match):
        T.shmem_get_nbi(payload, payload, 16, 1, scratch=scratch, event_id=0)


def test_nbi_resources_are_mandatory_and_invalid_endpoints_fail():
    payload = _buffer("payload", 64)
    scratch = _buffer("scratch", 64, scope="shared.ub")
    with pytest.raises(TypeError, match="scratch"):
        T.shmem_get_nbi(payload, payload, 16, 1, event_id=0)
    with pytest.raises(TypeError, match="event_id"):
        T.shmem_get_nbi(payload, payload, 16, 1, scratch=scratch)
    with pytest.raises(ValueError, match="dtypes must match"):
        T.shmem_get_nbi(
            _buffer("dst", 64, "float16"), payload, 16, 1, scratch=scratch, event_id=0
        )
    with pytest.raises(ValueError, match="positive"):
        T.shmem_get_nbi(payload, payload, 0, 1, scratch=scratch, event_id=0)
    with pytest.raises(ValueError, match="nonnegative int32"):
        T.shmem_get_nbi(payload, payload, 16, -1, scratch=scratch, event_id=0)
    with pytest.raises(TypeError, match="int32 expression"):
        T.shmem_get_nbi(payload, payload, 16, 1, scratch=scratch, event_id=tir.Var("id", "int64"))
    pitched = tir.decl_buffer((64,), "float32", name="pitched", strides=(2,))
    with pytest.raises(ValueError, match="compact row-major"):
        T.shmem_get_nbi(pitched, payload, 16, 1, scratch=scratch, event_id=0)


def test_ub_nbi_requires_and_preserves_event():
    gm = _buffer("gm", 128)
    ub = _buffer("ub", 64, scope="shared.ub")
    put = T.shmem_ub_put_nbi(_region(ub, 8, 16), _region(gm, 17, 16), 16, 1, 4, event_id=2)
    get = T.shmem_ub_get_nbi(_region(ub, 8, 16), _region(gm, 17, 16), 16, 1, event_id=3)
    assert len(put.args) == 7
    assert int(put.args[-1]) == 2
    assert len(get.args) == 6
    assert int(get.args[-1]) == 3
    with pytest.raises(TypeError, match="event_id"):
        T.shmem_ub_put_nbi(ub, gm, 16, 1)
    with pytest.raises(TypeError, match="event_id"):
        T.shmem_ub_get_nbi(ub, gm, 16, 1)


def test_signal_int32_slot_offsets_values_and_effects():
    flags = _buffer("flags", 128, "int32")
    generation = tir.Var("generation", "int32")
    publish = T.shmem_signal_op(_region(flags, 32, 1), generation, signal_op=0, pe=1)
    wait = T.shmem_signal_wait_until(_region(flags, 64, 1), cmp=0, value=generation)
    assert len(publish.args) == 5
    assert len(wait.args) == 4
    assert int(publish.args[1].args[2]) == 32
    assert int(wait.args[1].args[2]) == 64
    assert publish.args[2].same_as(generation)
    assert wait.args[3].same_as(generation)
    assert int(publish.op.get_attr("TCallEffectKind")) == 3
    assert int(wait.op.get_attr("TCallEffectKind")) == 3


def test_invalid_signal_contracts_fail_before_lowering():
    flags = _buffer("flags", 64, "int32")
    with pytest.raises(ValueError, match="dtype int32"):
        T.shmem_signal_op(_buffer("bad", 64, "uint64"), 1, signal_op=0, pe=1)
    with pytest.raises(ValueError, match="scope global"):
        T.shmem_signal_wait_until(_buffer("bad", 64, "int32", "shared.ub"), cmp=0, value=1)
    with pytest.raises(TypeError, match="int32 expression"):
        T.shmem_signal_op(flags, tir.Var("value", "uint64"), signal_op=0, pe=1)
    with pytest.raises(ValueError, match="SHMEM_SIGNAL_SET"):
        T.shmem_signal_op(flags, 1, signal_op=2, pe=1)
    with pytest.raises(ValueError, match="SHMEM_CMP"):
        T.shmem_signal_wait_until(flags, cmp=6, value=1)


def test_lowering_preserves_nbi_completion_publish_and_wait_order():
    @T.prim_func
    def kernel(
        source: T.Tensor((128,), "float32"),
        output: T.Tensor((128,), "float32"),
        signals: T.Tensor((128,), "int32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _:
            with T.Scope("V"):
                scratch = T.alloc_ub((64,), "float32")
                T.annotate_address({scratch: 0})
                T.shmem_get_nbi(
                    output[17:49], source[9:41], 32, 1, scratch=scratch[8:24], event_id=3
                )
                T.set_flag("mte3", "s", 3)
                T.wait_flag("mte3", "s", 3)
                T.shmem_signal_op(signals[32:33], 7, signal_op=0, pe=1)
                T.shmem_signal_wait_until(signals[64:65], cmp=0, value=7)

    source = _lower(kernel)
    calls = [
        "tl::ascend::shmem_get_nbi<float>",
        "AscendC::SetFlag<AscendC::HardEvent::MTE3_S>",
        "AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>",
        "tl::ascend::shmem_signal_op",
        "tl::ascend::shmem_signal_wait_until",
    ]
    positions = [source.index(call) for call in calls]
    assert positions == sorted(positions)
    assert re.search(r"shmem_get_nbi<float>\([^;]*64,\s*32,\s*1,\s*3\)", source)
    assert re.search(r"shmem_signal_op\([^;]*7,\s*0,\s*1\)", source)
    assert "EVENT_ID0" not in source


def test_signal_is_vector_owned_not_cube_owned():
    flags = _buffer("flags", 128, "int32")
    statement = tir.Evaluate(T.shmem_signal_op(_region(flags, 32, 1), 7, signal_op=0, pe=1))
    for owner in (0, 1):
        scoped = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", owner, statement)
        block = tir.Block([], [], [], "tilelang_root", scoped)
        func = tir.PrimFunc([flags.data], tir.BlockRealize([], True, block))
        module = tvm.IRModule({"main": func})
        if owner == 0:
            with pytest.raises(Exception, match="Vector operation must be inside"):
                tilelang.transform.AscendResourceScopeVerify()(module)
        else:
            tilelang.transform.AscendResourceScopeVerify()(module)
