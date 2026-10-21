import os
import sys
import ctypes
from pathlib import Path


def load_plugins(plugin_path: str | None = None) -> ctypes.CDLL:
    """
    Loads the TensorRT DFT Plugin library into memory.
    """
    if plugin_path is not None:
        p = Path(plugin_path)
    else:
        cur_dir = Path(__file__).resolve().parent
        if sys.platform == "win32":
            p = cur_dir / "trt_dft_plugins.dll"
            if not p.exists():
                # check prebuilt
                p = cur_dir.parents[2] / "prebuilt" / "win64_cuda13_trt10" / "trt_dft_plugins.dll"
        else:
            p = cur_dir / "libtrt_dft_plugins.so"

    if not p.exists():
        raise FileNotFoundError(f"TensorRT DFT Plugin not found at: {p}")

    if sys.platform == "win32":
        # Load with standard search flags
        lib = ctypes.CDLL(str(p))
    else:
        lib = ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)

    return lib
