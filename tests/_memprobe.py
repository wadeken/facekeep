"""Cross-platform peak-RSS probe for the large-image memory test (Phase 2).

The memory test needs the process's *peak* resident set — and crucially one that
includes the codec's C-level allocations (libaom for AVIF does the heavy lifting
outside Python), so ``tracemalloc`` (Python-only) is not enough and ``psutil``
would add a dependency the project deliberately avoids. We read the OS peak
directly with the stdlib:

  * Windows: ``PeakWorkingSetSize`` from ``GetProcessMemoryInfo`` via ctypes.
  * Linux/macOS: ``ru_maxrss`` from ``resource.getrusage`` (KB on Linux, bytes
    on macOS — normalized here).

If neither is available the probe returns ``None`` and the memory-bound test
skips (offline-graceful, like the corpus/YuNet tests). The peak is monotonic
(high-water mark) for the life of the process, so the test measures a *delta*
across the operation under test, not the absolute peak (which carries fixed
import overhead and is not portable).

Gotcha baked in from getting this wrong once: the ctypes call MUST set
``argtypes``/``restype``. Without them the ``c_size_t`` out-pointer is truncated
to 32-bit and the call returns 0 / garbage; with them it tracks real growth.
"""

import ctypes
import sys
from typing import Optional


def _peak_working_set_windows() -> Optional[int]:
    """Peak working set in bytes via Win32 GetProcessMemoryInfo, or None."""
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        # K32GetProcessMemoryInfo is exported by kernel32 (no psapi.dll dep).
        fn = ctypes.WinDLL("kernel32", use_last_error=True).K32GetProcessMemoryInfo
        fn.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        fn.restype = wintypes.BOOL

        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not fn(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    except (OSError, AttributeError):
        return None


def _peak_rss_unix() -> Optional[int]:
    """Peak RSS in bytes via resource.getrusage, or None.

    ``ru_maxrss`` is kilobytes on Linux but bytes on macOS; normalize to bytes.
    """
    try:
        import resource
    except ImportError:
        return None
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if maxrss <= 0:
        return None
    return maxrss if sys.platform == "darwin" else maxrss * 1024


def peak_rss_bytes() -> Optional[int]:
    """Return the process peak RSS in bytes, or None if unmeasurable here.

    Monotonic high-water mark for the process lifetime, so subtract two readings
    around an operation to get that operation's contribution to the peak.
    """
    if sys.platform == "win32":
        return _peak_working_set_windows()
    return _peak_rss_unix()
