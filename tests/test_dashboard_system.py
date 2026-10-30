"""Backend snapshot: missing values, identity privacy and model/mode labels."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.dashboard import system


def test_missing_measurements_are_null(monkeypatch):
    def denied():
        raise OSError("C:/Users/private-user/private-path")
    monkeypatch.setattr(system, "cpu_name", denied)
    monkeypatch.setattr(system.psutil, "virtual_memory", denied)
    monkeypatch.setattr(system.psutil, "Process", denied)
    monkeypatch.setattr(system, "windows_commit", denied)
    s = system.snapshot("server", {"openblas": None, "numba": None}, None, None)
    assert s["execution_location"] == "server"
    assert all(s[k] is None for k in ("cpu", "ram_total_gb", "ram_available_gb", "windows_commit",
                                     "process_private_gb", "process_working_set_gb"))
    assert "private-user" not in str(s)
    assert datetime.fromisoformat(s["measured_at"]).utcoffset().total_seconds() == 0


@pytest.mark.parametrize("source,mode", [("real", "cached"), ("synthetic", "mmap")])
def test_loaded_and_active_modes(source, mode):
    s = system.snapshot("local", {}, {"source": source, "mode": mode}, SimpleNamespace(mode=mode))
    assert s["execution_mode"] == mode and s["active_mode"] == mode
    assert ("Qwen" if source == "real" else "Engine test") in s["loaded_model"]
    assert not {"hostname", "username", "path", "executable"} & s.keys()


def test_windows_commit_is_capacity_not_ram():
    import sys
    if sys.platform != "win32":
        assert system.windows_commit() is None
    else:
        s = system.windows_commit()
        assert s is not None and 0 <= s["headroom_gb"] <= s["limit_gb"]
