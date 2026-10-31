"""Loopback-only, fixed-origin relay for the protected cloud dashboard API.

The browser's Basic-auth challenge belongs to this loopback relay. Credentials
are forwarded over verified HTTPS for each request and never stored here. The
cloud's signed browser cookie stays server-side, keyed by a separate opaque,
HttpOnly loopback cookie, so cloud visitor isolation remains authoritative.
"""

import hashlib
import http.client
from http.cookies import SimpleCookie
import secrets
import ssl
import threading
import time
from urllib.parse import urlsplit

DEFAULT_CLOUD_ORIGIN = "https://leanmoe-demo-1.eastus.cloudapp.azure.com"
SESSION_COOKIE = "leanmoe-cloud-session"
PATHS = frozenset({"/api/status", "/api/generate", "/api/compare", "/api/cancel", "/api/budget"})


class CloudProxy:
    def __init__(self, origin=DEFAULT_CLOUD_ORIGIN):
        url = urlsplit(origin)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.path or url.query or url.fragment:
            raise ValueError("cloud origin must be one exact HTTPS origin")
        self.origin = origin
        self.host = url.hostname
        self.port = url.port or 443
        self.sessions = {}
        self.lock = threading.Lock()

    def _session(self, cookie_header, authorization):
        cookies = SimpleCookie()
        try:
            cookies.load(cookie_header or "")
            token = cookies[SESSION_COOKIE].value if SESSION_COOKIE in cookies else ""
        except Exception:
            token = ""
        digest = hashlib.sha256(authorization.encode("ascii")).digest()
        with self.lock:
            item = self.sessions.get(token)
            if item and item[0] == digest and time.monotonic() - item[2] < 7200:
                return token, item[1]
        return None, None

    def open(self, method, path, raw, authorization, cookie_header):
        if path not in PATHS or method not in ("GET", "POST") or (method == "GET") != (path == "/api/status"):
            raise ValueError("unsupported cloud API path")
        if not authorization.startswith("Basic ") or len(authorization) > 1024 or not authorization.isascii():
            raise PermissionError("Cloud credentials required")
        token, remote_cookie = self._session(cookie_header, authorization)
        if method == "POST" and not remote_cookie:
            raise PermissionError("Cloud session expired; select Cloud again to reconnect")
        conn = http.client.HTTPSConnection(self.host, self.port, timeout=30, context=ssl.create_default_context())
        headers = {"Authorization": authorization, "Accept": "application/json, text/event-stream"}
        if remote_cookie:
            headers["Cookie"] = remote_cookie
        if method == "POST":
            headers.update({"Origin": self.origin, "Content-Type": "application/json"})
        try:
            conn.request(method, path, body=raw if method == "POST" else None, headers=headers)
            response = conn.getresponse()
            issued = response.getheader("Set-Cookie")
            if response.status == 200 and method == "GET":
                payload = response.read(2_000_001)
                if len(payload) > 2_000_000:
                    raise ValueError("cloud status too large")
                import json
                status = json.loads(payload)
                if status.get("system", {}).get("execution_location") != "server" or not status.get("access", {}).get("protected"):
                    raise ValueError("cloud backend did not report protected server execution")
                if issued:
                    remote_cookie = issued.split(";", 1)[0]
                if not remote_cookie:
                    raise ValueError("cloud did not establish a visitor session")
                if not token:
                    token = secrets.token_urlsafe(32)
                digest = hashlib.sha256(authorization.encode("ascii")).digest()
                with self.lock:
                    now = time.monotonic()
                    self.sessions = {k: v for k, v in self.sessions.items() if now - v[2] < 7200}
                    if len(self.sessions) >= 500 and token not in self.sessions:
                        raise ValueError("too many cloud sessions")
                    self.sessions[token] = (digest, remote_cookie, now)
                return conn, response, payload, token
            return conn, response, None, None
        except Exception:
            conn.close()
            raise
