# SHMEM examples

## Native int32 GM wait

This example follows the existing `examples/shmem/` pattern: a TileLang JIT
kernel, SHMEM worker initialization and symmetric tensor allocation, then two
Python processes and an exact result check.

Rank 0 sends payload, completes the transfer, publishes READY and waits for ACK.
Rank 1 waits for READY, saves payload and publishes ACK after consumption.
Both use `T.shmem_int32_wait_until(flag_ptr, T.ACLSHMEM_CMP_EQ, value)` with a
positive generation per round. MTE/scalar events order payload and flag writes;
the wait itself does not complete outstanding transfers. Every round's payload
is retained and checked. Each rank runs one AIV block.

Run inside an NPU container with TileLang and SHMEM initialized in the environment:

```bash
python examples/shmem/shmem_wait_until.py --devices 0 1 --rounds 100 --timeout 180
```

Choose two available devices. The launcher reports rank/generation on timeout
and terminates only its own child processes.
