# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Frontend and source-code regressions for genuine Vector-only kernels."""

import re

import pytest
from tvm import IRModule, ir, tir

import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
    "tl.ascend_auto_cv_combine": True,
    "tl.ascend_memory_planning": True,
}


def _copy_kernel(kernel_type: str | None = "aiv", threads: int = 1) -> tir.PrimFunc:
    @T.prim_func
    def main(A: T.Tensor((4, 32), "float32"), B: T.Tensor((4, 32), "float32")):
        with T.Kernel(4, threads=threads, is_npu=True, kernel_type=kernel_type) as bid:
            with T.Scope("V"):
                ub = T.alloc_ub((32,), "float32")
                T.copy(A[bid, :], ub)
                T.copy(ub, B[bid, :])

    return main


def _lower(function: tir.PrimFunc) -> str:
    with tilelang.transform.PassContext(config=PASS_CONFIGS):
        return tilelang.lower(function, target="ascendc", platform="A3").kernel_source


def test_aiv_frontend_sets_kernel_type() -> None:
    function = _copy_kernel()
    assert function.attrs["npu_kernel_type"].value == "aiv"
    assert function.attrs["npu_cv_ratio"].value == "cv_1_1"


def test_aiv_codegen_has_one_physical_vector_per_block() -> None:
    source = _lower(_copy_kernel())
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV);" in source
    assert "KERNEL_TYPE_MIX" not in source
    assert "AscendC::GetBlockIdx()" in source
    assert "GetSubBlockIdx" not in source
    assert " / 2" not in source
    assert "ascend_ub" in source
    for resource in ("ascend_l0a", "ascend_l0b", "ascend_l0c", "ascend_l1"):
        assert resource not in source
    assert re.search(r"main_kernel<<<4, nullptr, stream>>>", source)


def test_aiv_rejects_unsupported_codegen_backend() -> None:
    with pytest.raises(ValueError, match="only supported by the AscendC backend"):
        tilelang.lower(_copy_kernel(), target="pto", platform="A3")


def test_default_kernel_keeps_mix_codegen() -> None:
    function = _copy_kernel(kernel_type=None)
    assert "npu_kernel_type" not in function.attrs
    source = _lower(function)
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);" in source
    assert "GetSubBlockIdx" in source
    assert "ascend_l0a" in source


def test_unused_block_binding_keeps_aiv_launch_extent() -> None:
    @T.prim_func
    def main(A: T.Tensor((32,), "int32")):
        with T.Kernel(4, threads=1, is_npu=True, kernel_type="aiv") as _:
            with T.Scope("V"):
                ub = T.alloc_ub((32,), "int32")
                T.tile.fill(ub, 1)

    source = _lower(main)
    assert re.search(r"main_kernel<<<4, nullptr, stream>>>", source)
    assert "GetSubBlockIdx" not in source


@pytest.mark.parametrize(
    "options,message",
    [
        ({"kernel_type": "cube", "is_npu": True, "threads": 1}, "Unsupported kernel_type"),
        ({"kernel_type": "aiv", "is_npu": False, "threads": 1}, "requires is_npu=True"),
        ({"kernel_type": "aiv", "is_npu": True, "is_cpu": True, "threads": 1}, "is_cpu=False"),
        ({"kernel_type": "aiv", "is_npu": True, "threads": None}, "requires threads=1"),
        ({"kernel_type": "aiv", "is_npu": True, "threads": 2}, "requires threads=1"),
        ({"kernel_type": "aiv", "is_npu": True, "threads": [1, 1]}, "requires threads=1"),
    ],
)
def test_invalid_aiv_frontend_options(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        T.Kernel(4, **options)


def test_aiv_rejects_explicit_cube_scope() -> None:
    @T.prim_func
    def main(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=1, is_npu=True, kernel_type="aiv"):
            with T.Scope("C"):
                A[0] = 1.0

    with pytest.raises(Exception, match="AIV kernel cannot contain T.Scope"):
        _lower(main)


@pytest.mark.parametrize("scope", ["shared.l1", "wmma.matrix_a", "wmma.matrix_b", "wmma.accumulator"])
def test_aiv_verifier_rejects_cube_allocation(scope: str) -> None:
    var = tir.Var("cube_storage", "handle", type_annotation=ir.PointerType(ir.PrimType("float32"), scope))
    allocation = tir.Allocate(var, "float32", [16], True, tir.Evaluate(0))
    function = tir.PrimFunc([], allocation).with_attr("npu_kernel_type", tir.StringImm("aiv"))
    with pytest.raises(Exception, match="AIV kernel cannot allocate Cube storage"):
        tilelang.transform.AscendResourceScopeVerify()(IRModule({"main": function}))


def test_aiv_verifier_rejects_cube_operation_in_vector_scope() -> None:
    call = tir.call_intrin("handle", tir.op.Op.get("tl.ascend_pipe_barrier"), "M")
    body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, tir.Evaluate(call))
    function = tir.PrimFunc([], body).with_attr("npu_kernel_type", tir.StringImm("aiv"))
    with pytest.raises(Exception, match="AIV kernel cannot contain Cube resources or operations"):
        tilelang.transform.AscendResourceScopeVerify()(IRModule({"main": function}))
