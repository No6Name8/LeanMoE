"""Backend measurements for the dashboard, with no host/user/path identifiers."""

from datetime import datetime, timezone
import platform
import sys

import psutil

GB = 1e9


def optional(call):
    """An unavailable measurement is null, never a guessed value or an error string."""
    try:
        return call()
    except (OSError, ValueError, AttributeError, RuntimeError, ImportError, psutil.Error):
        return None


def cpu_name():
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip() or None
    return platform.processor().strip() or None


def windows_commit():
    """System-wide commit limit and remaining headroom (not free physical RAM)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class PerformanceInfo(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
                "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
                "KernelNonpaged", "PageSize")
        ] + [(name, wintypes.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]

    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
        return None
    return {"limit_gb": info.CommitLimit * info.PageSize / GB,
            "headroom_gb": (info.CommitLimit - info.CommitTotal) * info.PageSize / GB}


def snapshot(execution_location, threads, loaded, active):
    vm = optional(psutil.virtual_memory)
    mem = optional(lambda: psutil.Process().memory_info())
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "execution_location": execution_location,
        "os": optional(lambda: f"{platform.system()} {platform.release()} ({platform.version()})"),
        "cpu": optional(cpu_name),
        "logical_cores": optional(lambda: psutil.cpu_count(logical=True)),
        "ram_total_gb": vm.total / GB if vm else None,
        "ram_available_gb": vm.available / GB if vm else None,
        "windows_commit": optional(windows_commit),
        "process_private_gb": getattr(mem, "private", None) / GB if mem is not None and hasattr(mem, "private") else None,
        "process_working_set_gb": mem.rss / GB if mem else None,
        "threads": threads,
        "loaded_model": ({"real": "Qwen1.5-MoE-A2.7B-Chat (int8)",
                          "synthetic": "Engine test (random weights)"}.get(loaded["source"]) if loaded else "none"),
        "execution_mode": loaded["mode"] if loaded else None,
        "active_mode": active.mode if active else None,
    }
