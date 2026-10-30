"""Preflight check: can this machine run LeanMoE's quick verification and the real model?

    .\\.venv\\Scripts\\python.exe scripts\\preflight.py
    .\\.venv\\Scripts\\python.exe scripts\\preflight.py --require-model   (also fail if the real model isn't ready)

Reports the operating system, Python, packages, memory (physical RAM and, on
Windows, commit headroom), disk space, thread settings and the model store.
Each problem comes with what to do next. Passing preflight does not guarantee
a run will fit in memory: other programs can take memory while it runs, and
the engine's budget is a planning target, not an enforced limit.

Exit code 0: quick verification can run (and, with --require-model, the real
model store is complete). Exit code 1: something required is missing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import the engine package before numpy: it sets the OpenBLAS thread default,
# which only applies if numpy hasn't been imported yet.
import src.engine  # noqa: E402,F401

STORE_NAME = "qwen1.5-moe-a2.7b-chat-int8"
STORE_FILES = ["store.json", "experts.bin", "resident.bin", "experts_index.json", "resident_index.json"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json"]
TESTED_PYTHON = (3, 13)
# Observed on the 8 GB development laptop at the demo defaults (results/EVIDENCE.md):
# sampled peak private memory 3.18 GB. Used only to say whether headroom looks
# comfortable, never as a guarantee.
OBSERVED_PEAK_PRIVATE_GB = 3.18
GB = 1e9

problems: list[str] = []
warnings: list[str] = []


def line(label: str, value: str) -> None:
    print(f"  {label:<28} {value}")


def windows_commit() -> tuple[float, float] | None:
    """(commit limit, commit available) in GB, or None off Windows."""
    if sys.platform != "win32":
        return None
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]  # fmt: skip

    m = MemoryStatus()
    m.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return None
    return m.ullTotalPageFile / GB, m.ullAvailPageFile / GB


def check_system() -> None:
    print("System")
    line("Operating system", f"{platform.system()} {platform.release()} ({platform.version()}), {platform.machine()}")
    if sys.platform != "win32":
        problems.append(
            "This is not Windows. LeanMoE's engine reads experts with Win32 unbuffered I/O, so the engine "
            "tests and the real model only run on Windows (tested on Windows 11 x64). On other systems only "
            "the saved-evidence checker works."
        )
    v = sys.version_info
    line("Python", f"{platform.python_version()} at {sys.executable}")
    if v[:2] != TESTED_PYTHON:
        if (3, 10) <= v[:2] < (3, 14):
            warnings.append(f"Python {v.major}.{v.minor} is untested here; LeanMoE was tested with Python 3.13.")
        else:
            problems.append(f"Python {v.major}.{v.minor} is outside the expected range. Install Python 3.13 and rerun the setup script.")
    if sys.prefix == sys.base_prefix:
        warnings.append("Not running inside a virtual environment. Use the setup script, which creates .venv in the repository.")


def check_packages() -> None:
    print("Packages")
    required = {"numpy": "numpy", "numba": "numba", "psutil": "psutil", "tokenizers": "tokenizers", "jinja2": "jinja2"}
    for name, module in required.items():
        try:
            mod = importlib.import_module(module)
            line(name, getattr(mod, "__version__", "installed"))
        except ImportError:
            line(name, "MISSING")
            problems.append(f"Package '{name}' is missing. Run the setup script again (it installs requirements-dev.txt).")
    for name, why in (("pytest", "needed for the engine tests"), ("torch", "optional, only for the toy fallback")):
        try:
            mod = importlib.import_module(name)
            line(name, getattr(mod, "__version__", "installed"))
        except ImportError:
            line(name, f"not installed ({why})")
            if name == "pytest":
                problems.append("pytest is missing, so the engine tests can't run. Run the setup script again.")


def check_memory_and_disk(models_root: Path | None) -> None:
    print("Memory and disk")
    try:
        import psutil
    except ImportError:
        line("Memory", "unknown (psutil missing)")
        return
    vm = psutil.virtual_memory()
    line("Physical RAM", f"{vm.total / GB:.2f} GB total, {vm.available / GB:.2f} GB available now")
    commit = windows_commit()
    if commit:
        limit, avail = commit
        line("Windows commit headroom", f"{avail:.2f} GB available of a {limit:.2f} GB commit limit")
        if avail < OBSERVED_PEAK_PRIVATE_GB + 0.5:
            warnings.append(
                f"Commit headroom is {avail:.2f} GB. The real-model demo committed up to {OBSERVED_PEAK_PRIVATE_GB} GB "
                "on the development laptop; close other programs or lower --start-gb / --start-cache-gb / --cut-gb."
            )
    if vm.available / GB < 2.0:
        warnings.append(
            f"Only {vm.available / GB:.2f} GB of physical RAM is free. Real-model runs will page and be slower, "
            "and the cache will be smaller (it is capped by free RAM)."
        )
    for label, path in (("Repository drive", REPO_ROOT), ("Models drive", models_root)):
        if path is None:
            continue
        probe = path if path.exists() else path.anchor or path
        try:
            free = shutil.disk_usage(probe).free / GB
            line(f"{label} free space", f"{free:.1f} GB ({probe})")
        except OSError:
            line(f"{label} free space", f"unknown ({probe})")


def check_threads() -> None:
    print("Threads")
    try:
        from src.engine.benchmarking import openblas_threads

        line("OPENBLAS_NUM_THREADS", os.environ.get("OPENBLAS_NUM_THREADS", "unset"))
        line("OpenBLAS threads in effect", str(openblas_threads()))
    except ImportError:
        line("OpenBLAS threads", "unknown (numpy missing)")
    try:
        import numba

        line("numba threads", str(numba.get_num_threads()))
    except ImportError:
        pass
    line("Logical CPUs", str(os.cpu_count()))


def check_model_store() -> tuple[str, Path | None]:
    print("Real-model store (optional for quick verification)")
    root_env = os.environ.get("LEANMOE_MODELS_ROOT")
    root = Path(root_env) if root_env else REPO_ROOT / "models"
    line("LEANMOE_MODELS_ROOT", root_env or f"not set (looking in {root})")
    store = root / STORE_NAME
    if not store.exists():
        line("Store", f"not found at {store}")
        return (
            f"Real model: NOT AVAILABLE. No store at {store}. That is expected for the quick verification. "
            "To run the real model, put the Qwen1.5-MoE-A2.7B-Chat int8 store there (see README, 'Full model') "
            "or set LEANMOE_MODELS_ROOT to the folder that contains it.",
            root,
        )
    missing = [f for f in STORE_FILES if not (store / f).exists()]
    if missing:
        return f"Real model: INCOMPLETE STORE at {store}; missing {missing}. Rebuild or recopy the store.", root
    try:
        manifest = json.loads((store / "store.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return f"Real model: store.json at {store} can't be read ({e}).", root
    sizes = manifest.get("summary", {}).get("sizes", {})
    size_problems = []
    for name, key in (("experts.bin", "experts_bin_bytes"), ("resident.bin", "resident_bin_bytes")):
        actual = (store / name).stat().st_size
        expected = sizes.get(key)
        line(name, f"{actual / GB:.2f} GB" + ("" if expected is None else f" (store.json says {expected / GB:.2f} GB)"))
        if expected is not None and actual != expected:
            size_problems.append(f"{name} is {actual} bytes, store.json expects {expected}")
    source = manifest.get("source", {})
    line("Built from", f"{source.get('repo_id')} at revision {source.get('revision')}")
    line("Marked complete", str(manifest.get("complete")))
    if not manifest.get("complete") or size_problems:
        return f"Real model: STORE NOT COMPLETE at {store}. {'; '.join(size_problems) or 'store.json is not marked complete.'}", root
    no_tok = [f for f in TOKENIZER_FILES if not (store / f).exists()]
    if no_tok:
        warnings.append(
            f"Tokenizer files {no_tok} are not in the store; the engine downloads them from Hugging Face on first "
            "use (needs internet once)."
        )
    calibration = REPO_ROOT / "configs" / "calibration" / f"prefetch_calibration_{STORE_NAME}.json"
    if not calibration.exists():
        return f"Real model: prefetch calibration {calibration} is missing from the repository.", root
    return (
        f"Real model: STORE FOUND AND COMPLETE at {store}. Sizes match store.json "
        "(contents not re-hashed; run the builder's --verify for that).",
        root,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require-model", action="store_true", help="also fail if the real-model store isn't ready")
    args = ap.parse_args()
    print(f"LeanMoE preflight ({REPO_ROOT})\n")
    check_system()
    check_packages()
    model_status, models_root = check_model_store()
    check_memory_and_disk(models_root)
    check_threads()
    print()
    quick_ok = not problems
    print("Quick verification (engine tests on a tiny synthetic store, saved-evidence check):",
          "READY" if quick_ok else "NOT READY")  # fmt: skip
    print(model_status)
    for p in problems:
        print(f"  PROBLEM: {p}")
    for w in warnings:
        print(f"  NOTE: {w}")
    print("\nPassing preflight does not guarantee a real-model run fits in memory; see README, 'Limitations'.")
    model_ok = "COMPLETE" in model_status
    return 0 if quick_ok and (model_ok or not args.require_model) else 1


if __name__ == "__main__":
    sys.exit(main())
