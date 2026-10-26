"""Spatially aware property inspection: damage detection, camera tracking, 3D reconstruction and investigation."""

import os
import sys

import cv2

__version__ = "0.1.0"

# OpenCV uses every core by default. Several analysis processes at full load can overheat a laptop, so each process
# gets a fixed number of threads (SI_THREADS, default 4) and, on a workstation, a lower scheduling priority.
cv2.setNumThreads(int(os.environ.get("SI_THREADS", "4")))


def lower_priority() -> None:
    """Run this process below normal priority so the machine stays responsive (skipped when SI_PRIORITY=normal)."""
    if os.environ.get("SI_PRIORITY", "low") != "low":
        return
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            # the process handle is pointer-sized; without these types it is truncated on 64-bit Windows
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            BELOW_NORMAL_PRIORITY_CLASS = 0x4000
            k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(10)
    except (OSError, AttributeError):
        pass


lower_priority()
