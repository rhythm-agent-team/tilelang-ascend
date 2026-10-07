"""Two-rank READY/ACK handshake using native int32 GM waits."""
import argparse
import multiprocessing as mp
import socket
import time

import shmem
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

ELEMENTS, SLOT_ELEMENTS, READY, ACK = 64, 128, 72, 80
CONFIG = {"tl.ascend_auto_sync": False, "tl.ascend_memory_planning": True,
          "tl.ascend_auto_cross_core_sync": False, "tl.ascend_auto_cv_combine": False}


@tilelang.jit(target="ascendc", platform="A3", pass_configs=CONFIG)
def shmem_wait_kernel(rank, rounds):
    @T.prim_func
    def main(slots: T.Tensor((SLOT_ELEMENTS,), "int32"),
             output: T.Tensor((rounds * ELEMENTS,), "int32"), generation: T.int32):
        with T.Kernel(1, threads=1, is_npu=True):
            data = T.alloc_ub((ELEMENTS,), "int32")
            flag = T.alloc_ub((8,), "int32")
            with T.Scope("V"):
                if rank == 0:
                    for i in T.serial(ELEMENTS):
                        data[i] = generation * 1000 + i
                    T.set_flag("s", "mte3", 0)
                    T.wait_flag("s", "mte3", 0)
                    T.shmem_ub_put_nbi(data, slots, ELEMENTS, 1)
                    T.set_flag("mte3", "s", 0)
                    T.wait_flag("mte3", "s", 0)
                    flag[0] = generation
                    T.set_flag("s", "mte3", 0)
                    T.wait_flag("s", "mte3", 0)
                    T.shmem_ub_put_nbi(flag, slots, 1, 1, strelem=READY)
                    T.set_flag("mte3", "s", 0)
                    T.wait_flag("mte3", "s", 0)
                    T.shmem_int32_wait_until(T.address_of(slots[ACK]), T.ACLSHMEM_CMP_EQ,
                                             generation)
                else:
                    T.shmem_int32_wait_until(T.address_of(slots[READY]), T.ACLSHMEM_CMP_EQ,
                                             generation)
                    T.copy(slots[:ELEMENTS], data)
                    T.set_flag("mte2", "mte3", 0)
                    T.wait_flag("mte2", "mte3", 0)
                    T.copy(data, output[(generation - 1) * ELEMENTS:generation * ELEMENTS])
                    T.set_flag("mte3", "s", 0)
                    T.wait_flag("mte3", "s", 0)
                    flag[0] = generation
                    T.set_flag("s", "mte3", 0)
                    T.wait_flag("s", "mte3", 0)
                    T.shmem_ub_put_nbi(flag, slots, 1, 0, strelem=ACK)
                    T.set_flag("mte3", "s", 0)
                    T.wait_flag("mte3", "s", 0)
    return main


def worker(rank, args, barrier, endpoint, generations):
    device = args.devices[rank]
    torch.npu.set_device(device)
    tilelang.disable_cache()
    kernel = shmem_wait_kernel(rank, args.rounds)
    attributes = shmem.InitAttr()
    attributes.my_rank, attributes.n_ranks = rank, 2
    attributes.local_mem_size = 32 * 1024 * 1024
    attributes.option_attr.data_op_engine_type = shmem.OpEngineType.MTE
    if rank == 0:
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(2)
        port = listener.getsockname()[1]
        endpoint.put(port)
        attributes.option_attr.sockFd = listener.detach()
    else:
        port = endpoint.get(timeout=args.timeout)
    attributes.ip_port = f"tcp://127.0.0.1:{port}"
    if shmem.set_conf_store_tls(False, "") != 0 or shmem.aclshmem_init(attributes) != 0:
        raise RuntimeError(f"rank={rank}: SHMEM initialization failed")
    slots = shmem.aclshmem_create_tensor([SLOT_ELEMENTS], dtype=torch.int32, device_id=device)
    slots.zero_()
    output = torch.full((args.rounds * ELEMENTS,), -1, dtype=torch.int32,
                        device=f"npu:{device}")
    torch.npu.synchronize()
    barrier.wait()
    for generation in range(1, args.rounds + 1):
        generations[rank] = generation
        kernel(slots, output, generation)
        torch.npu.synchronize()  # Each slot is reused only after the READY/ACK handshake.
    observed = int(slots[ACK if rank == 0 else READY].cpu())
    if observed != args.rounds:
        raise RuntimeError(f"rank={rank}: final flag={observed}, expected={args.rounds}")
    if rank == 1:
        expected = (torch.arange(1, args.rounds + 1, dtype=torch.int32)[:, None] * 1000
                    + torch.arange(ELEMENTS, dtype=torch.int32)).flatten()
        actual = output.cpu()
        if not torch.equal(actual, expected):
            index = int(torch.nonzero(actual != expected)[0])
            raise RuntimeError(f"rank=1 generation={index // ELEMENTS + 1}: "
                               f"element={index % ELEMENTS}, got={actual[index]}, "
                               f"expected={expected[index]}")
    generations[rank] = -1
    barrier.wait()
    shmem.aclshmem_free_tensor(slots)
    if shmem.aclshmem_finalize() != 0:
        raise RuntimeError(f"rank={rank}: SHMEM finalize failed")
    print(f"rank={rank}: PASS, {args.rounds} generations", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs=2, required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if len(set(args.devices)) != 2 or not 1 <= args.rounds <= 999 or args.timeout <= 0:
        parser.error("use two distinct devices, 1..999 rounds and a positive timeout")
    context = mp.get_context("spawn")
    barrier, endpoint = context.Barrier(2, timeout=args.timeout), context.Queue()
    generations = context.Array("i", [0, 0])  # Host diagnostics; no device progress tensor.
    processes = [context.Process(target=worker, args=(r, args, barrier, endpoint, generations))
                 for r in range(2)]
    deadline = time.monotonic() + args.timeout
    try:
        for process in processes:
            process.start()
        while any(p.is_alive() for p in processes):
            for rank, process in enumerate(processes):
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"rank={rank}: exit={process.exitcode}")
            if time.monotonic() >= deadline:
                pending = [(r, generations[r]) for r, p in enumerate(processes) if p.is_alive()]
                raise TimeoutError(f"HANG: (rank, generation)={pending}; 0=setup, -1=finalize")
            time.sleep(0.1)
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError(f"rank exit codes: {[p.exitcode for p in processes]}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                if process.is_alive():
                    raise TimeoutError(f"HANG: test process pid={process.pid} survived cleanup")
        endpoint.close()
        endpoint.join_thread()
    print("READY/ACK handshake passed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as error:
        print(error, flush=True)
        raise SystemExit(124)
