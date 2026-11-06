"""GPU and CUDA pre-flight inspection utility for RecoverBrain."""

from __future__ import annotations

import sys


def check_gpu() -> bool:
    """Inspect environment, PyTorch CUDA build, GPU compute capability, and execute matmul test."""
    print("=" * 65)
    print(" RecoverBrain GPU Environment Pre-flight Inspection")
    print("=" * 65)

    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is not installed in the current environment.")
        return False

    print(f"PyTorch Version:         {torch.__version__}")
    print(f"CUDA Available:          {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[CRITICAL ERROR] CUDA is not available to PyTorch!")
        print("An RTX 50-series GPU requires a PyTorch build compiled for CUDA 12.8+.")
        print("Training cannot fall back silently to CPU for --preset base.")
        return False

    device_count = torch.cuda.device_count()
    device_id = 0
    device_name = torch.cuda.get_device_name(device_id)
    major, minor = torch.cuda.get_device_capability(device_id)
    compute_capability = f"{major}.{minor}"

    total_mem = torch.cuda.get_device_properties(device_id).total_memory
    free_mem, _ = torch.cuda.mem_get_info(device_id)

    total_gb = total_mem / (1024**3)
    free_gb = free_mem / (1024**3)

    print(f"CUDA Runtime Version:    {torch.version.cuda}")
    print(f"GPU Device Count:        {device_count}")
    print(f"Primary Device Name:     {device_name}")
    print(f"Compute Capability:      sm_{major}{minor} ({compute_capability})")
    print(f"Total Dedicated VRAM:    {total_gb:.2f} GB")
    print(f"Free Dedicated VRAM:     {free_gb:.2f} GB")

    # Check for Blackwell (sm_120) or modern architectures
    if major < 8:
        print(f"[WARNING] Compute capability {compute_capability} is below recommended sm_80+.")

    # Execute a real GPU tensor matmul validation
    try:
        print("\nExecuting GPU matmul verification test (bf16 tensor ops)...")
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print(f"[SUCCESS] GPU Tensor Core matmul test completed. Result shape: {tuple(c.shape)}")
    except Exception as e:
        print(f"[CRITICAL ERROR] GPU matmul test failed: {e}")
        return False

    # Optional pynvml telemetry probe
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        print(f"NVML Telemetry:          Temp={temp}C, GPU Core={util.gpu}%, Mem={util.memory}%")
    except Exception:
        print("NVML Telemetry:          pynvml not initialized (optional fallback)")

    print("=" * 65)
    print("Pre-flight check PASSED: NVIDIA GPU is ready for RecoverBrain training.")
    print("=" * 65)
    return True


def main() -> None:
    success = check_gpu()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

