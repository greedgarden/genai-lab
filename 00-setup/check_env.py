"""Check the PyTorch and GPU setup, and compare matrix multiplication on CPU and GPU."""

import time

import torch

GIB = 1024 ** 3
N = 4096                 # matrix size: N x N
RUNS = 10                # timed runs per measurement; the average is reported
FLOPS = 2 * N ** 3       # floating-point operations in one N x N matmul


def bench(device, sync=True):
    """Return the average time of one N x N matmul on `device`, in seconds.

    CUDA kernels run asynchronously: `a @ b` returns as soon as the work is
    queued. With sync=False, the timer stops before the GPU has finished, so
    the result only shows how long it takes to queue the work.
    """
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)

    # Warm-up: the first call includes one-time setup, so it is not timed.
    a @ b
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(RUNS):
        a @ b
    if device == "cuda" and sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / RUNS


def report(name, seconds):
    print(f"{name:<18} {seconds * 1000:8.2f} ms  {FLOPS / seconds / 1e12:6.2f} TFLOPS")


def main():
    print(f"PyTorch version:   {torch.__version__}")

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    free, total = torch.cuda.mem_get_info()
    print(f"GPU:               {torch.cuda.get_device_name()}")
    print(f"VRAM total:        {total / GIB:.2f} GiB")
    print(f"VRAM free:         {free / GIB:.2f} GiB")
    print()
    print(f"{N}x{N} float32 matmul, average of {RUNS} runs:")

    cpu = bench("cpu")
    gpu = bench("cuda")
    gpu_nosync = bench("cuda", sync=False)

    report("CPU", cpu)
    report("GPU", gpu)
    print(f"GPU speedup:       {cpu / gpu:.1f}x")
    print()
    print(f"GPU without sync:  {gpu_nosync * 1000:8.3f} ms  (only the time to queue the work)")


if __name__ == "__main__":
    main()
