# 00-setup

Set up Python and PyTorch with CUDA on WSL2, check that the GPU works, and compare a matrix multiplication on the CPU and the GPU.

## Requirements

- WSL2 with an NVIDIA GPU
- Python 3 with the `venv` module. On Ubuntu:

  ```bash
  sudo apt install python3.14-venv
  ```

Tested on WSL2 with Ubuntu 26.04, Python 3.14.4, and an NVIDIA GeForce RTX 3070 Laptop GPU (8 GiB).

## Check the GPU

```bash
nvidia-smi
```

The `CUDA Version` in the header is the newest CUDA version the driver supports.

On WSL2, the GPU driver comes from Windows. Do not install a Linux NVIDIA driver inside WSL.

## Create a virtual environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install PyTorch

1. Open <https://pytorch.org/get-started/locally/>.
2. Select Linux, Pip, Python, and a CUDA version. The CUDA version must not be newer than the one shown by `nvidia-smi`.
3. Run the command it shows. For CUDA 13.4:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu134
   ```

## Run the check

```bash
python 00-setup/check_env.py
```

The script prints the PyTorch version, the GPU name, and the total and free VRAM. Then it times a 4096×4096 float32 matrix multiplication on the CPU and on the GPU, and reports the average of 10 runs.

## Results

```
PyTorch version:   2.15.0+cu134
GPU:               NVIDIA GeForce RTX 3070 Laptop GPU
VRAM total:        8.00 GiB
VRAM free:         6.95 GiB

4096x4096 float32 matmul, average of 10 runs:
CPU                  400.54 ms    0.34 TFLOPS
GPU                   18.18 ms    7.56 TFLOPS
GPU speedup:       22.0x

GPU without sync:     0.083 ms  (only the time to queue the work)
```

`VRAM free` and the timings change from run to run. Across several runs, the GPU was about 20–27 times faster than the CPU.

TFLOPS is the number of floating-point operations per second, in trillions. One N×N matrix multiplication takes about 2N³ operations, which is about 1.37 × 10¹¹ for N = 4096.

### GPU timing needs `torch.cuda.synchronize()`

CUDA kernels run asynchronously. `a @ b` returns as soon as the work is queued on the GPU, before the GPU has finished. Without `torch.cuda.synchronize()` before stopping the timer, the GPU time was 0.083 ms, more than 200 times shorter than the real 18.18 ms. That number only shows how long it takes to queue the work, and it would make the GPU look thousands of times faster than the CPU.

The script also runs one untimed multiplication first, because the first CUDA call includes one-time setup.
