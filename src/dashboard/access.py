"""Opt-in protection for a localhost backend behind the configured HTTPS proxy.

The proxy authenticates individual visitor accounts in private mode or assigns
the fixed public identity in the opt-in demo mode. A signed browser cookie
separates visitors in either mode. No prompt, account name or request identifier
is published to other sessions.
"""
from dataclasses import dataclass, field
import hashlib
import hmac
from http.cookies import SimpleCookie, CookieError
import re
import secrets
import threading
import time
from urllib.parse import urlsplit

COOKIE = "__Host-leanmoe"


@dataclass
class ProxyAccess:
    origin: str
    secret: str = field(repr=False)
    visitors: frozenset[str]
    public_demo: bool = False
    max_tokens: int = 96
    max_seconds: int = 600  # entire comparison, including both modes
    max_budget_gb: float = 4.0
    session_seconds: int = 7200
    _rates: dict = field(default_factory=dict, repr=False)
    _lock: object = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        url = urlsplit(self.origin)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.path or url.query or url.fragment or self.origin.endswith("/")):
            raise ValueError("public origin must be one exact HTTPS origin without a path")
        if len(self.secret) < 32 or not self.visitors or len(self.visitors) > 50:
            raise ValueError("provide a proxy secret of at least 32 characters and 1-50 visitor accounts")
        if not all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", v) for v in self.visitors):
            raise ValueError("invalid visitor account identifier")
        if self.public_demo and self.visitors != frozenset({"public"}):
            raise ValueError("public demo requires only the fixed public visitor identity")

    def _sign(self, value):
        return hmac.new(self.secret.encode(), value.encode(), hashlib.sha256).hexdigest()

    def identify(self, headers, *, create=False, now=None):
        """Return opaque owner and optional Set-Cookie; reject forged proxy/cookie data."""
        now = int(time.time() if now is None else now)
        visitor = headers.get("X-LeanMoE-Visitor", "")
        key = headers.get("X-LeanMoE-Proxy-Key", "")
        if not hmac.compare_digest(key.encode(), self.secret.encode()) or visitor not in self.visitors:
            raise PermissionError("trusted proxy required")
        cookie = SimpleCookie()
        try:
            cookie.load(headers.get("Cookie", ""))
            token = cookie[COOKIE].value if COOKIE in cookie else ""
            stamp, nonce, signature = token.split(".")
            valid = (re.fullmatch(r"[a-f0-9]{32}", nonce) is not None
                     and 0 <= now - int(stamp) < self.session_seconds
                     and hmac.compare_digest(signature, self._sign(f"{visitor}:{stamp}.{nonce}")))
        except (ValueError, KeyError, CookieError):
            valid = False
        set_cookie = None
        if not valid:
            if not create:
                raise PermissionError("reload the dashboard to establish a valid browser session")
            stamp, nonce = str(now), secrets.token_hex(16)
            token = f"{stamp}.{nonce}.{self._sign(f'{visitor}:{stamp}.{nonce}')}"
            set_cookie = (f"{COOKIE}={token}; Path=/; Secure; HttpOnly; SameSite=Strict; "
                          f"Max-Age={self.session_seconds}")
        return self._sign(f"owner:{visitor}:{stamp}.{nonce}"), set_cookie

    def admit(self, owner, now=None):
        """Three admissions/minute/proxy identity (global in public mode)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._rates = {k: v for k, v in self._rates.items() if now - v[0] < 60}
            start, count = self._rates.get(owner, (now, 0))
            if count >= 3 or (owner not in self._rates and len(self._rates) >= 500):
                return False
            self._rates[owner] = (start, count + 1)
        return True
