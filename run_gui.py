#!/usr/bin/env python
"""Launch the Audio Visualizer GUI."""

import os
import sys
from pathlib import Path


def find_cuda_path():
    """Find CUDA installation path, preferring CUDA 12.x for cupy-cuda12x compatibility."""
    cuda_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if not cuda_base.exists():
        return None

    # Find all CUDA versions
    versions = list(cuda_base.glob("v*"))
    if not versions:
        return None

    # Prefer CUDA 12.x (compatible with cupy-cuda12x)
    cuda12_versions = [v for v in versions if v.name.startswith("v12")]
    if cuda12_versions:
        # Pick highest 12.x version
        return sorted(cuda12_versions, reverse=True)[0]

    # Fall back to highest available version
    return sorted(versions, reverse=True)[0]


# Set CUDA path if available (for CuPy)
cuda_path = find_cuda_path()
if cuda_path:
    os.environ["CUDA_PATH"] = str(cuda_path)
    os.environ["PATH"] = str(cuda_path / "bin") + ";" + os.environ.get("PATH", "")

from src.gui import main

if __name__ == "__main__":
    main()
