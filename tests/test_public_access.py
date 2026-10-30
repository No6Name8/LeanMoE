from src.dashboard.access import ProxyAccess
import pytest


def test_sessions_are_signed_account_bound_and_expire():
    access = ProxyAccess("https://demo.example.test", "x" * 40, frozenset({"alice", "bob"}))
    h = {"X-LeanMoE-Proxy-Key": "x" * 40, "X-LeanMoE-Visitor": "alice"}
    owner, cookie = access.identify(h, create=True, now=100)
    h["Cookie"] = cookie.split(";", 1)[0]
    assert access.identify(h, now=101)[0] == owner
    for changed in ({**h, "X-LeanMoE-Visitor": "bob"}, {**h, "Cookie": h["Cookie"] + "bad"}):
        with pytest.raises(PermissionError):
            access.identify(changed, now=101)
    with pytest.raises(PermissionError):
        access.identify(h, now=100 + access.session_seconds)
    assert access.identify({k:v for k,v in h.items() if k != "Cookie"}, create=True, now=100)[0] != owner


def test_rate_limit_and_origin_configuration():
    for origin in ("http://example.test", "https://example.test/path", "https://user@example.test"):
        with pytest.raises(ValueError):
            ProxyAccess(origin, "x" * 40, frozenset({"alice"}))
    access = ProxyAccess("https://example.test", "x" * 40, frozenset({"alice"}))
    assert all(access.admit("alice", now=1) for _ in range(3))
    assert not access.admit("alice", now=2)
    assert access.admit("alice", now=62)
