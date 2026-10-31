"""The loopback cloud relay keeps credentials out of code and sessions separate."""

import base64
import http.client
import json

import pytest

from src.dashboard import cloud


class Response:
    status = 200

    def __init__(self, body, cookie=None, status=200):
        self.body = body
        self.cookie = cookie
        self.status = status

    def getheader(self, name, default=None):
        return self.cookie if name == "Set-Cookie" else default

    def read(self, amount):
        return self.body


class Connection:
    calls = []
    response = None

    def __init__(self, host, port, timeout, context):
        assert host == "demo.example.test" and port == 443
        assert context.verify_mode == 2 and context.check_hostname

    def request(self, method, path, body=None, headers=None):
        self.calls.append((method, path, body, headers))

    def getresponse(self):
        return self.response

    def close(self):
        pass


def auth(user):
    return "Basic " + base64.b64encode(f"{user}:private-password".encode()).decode()


def test_cloud_status_requires_protected_server_and_keeps_cookie_server_side(monkeypatch):
    monkeypatch.setattr(cloud.http.client, "HTTPSConnection", Connection)
    proxy = cloud.CloudProxy("https://demo.example.test")
    with pytest.raises(PermissionError):
        proxy.open("GET", "/api/status", b"", "", "")
    Connection.response = Response(json.dumps({"system": {"execution_location": "server"},
                                               "access": {"protected": True}}).encode(), "__Host-leanmoe=signed; Secure; HttpOnly")
    conn, response, payload, token = proxy.open("GET", "/api/status", b"", auth("alice"), "")
    assert response.status == 200 and token
    assert b"signed" not in payload
    assert "Authorization" in Connection.calls[-1][3]
    assert "Cookie" not in Connection.calls[-1][3]
    Connection.response = Response(json.dumps({"system": {"execution_location": "server"},
                                               "access": {"protected": True}}).encode(), "__Host-leanmoe=other; Secure; HttpOnly")
    _, _, _, other_browser = proxy.open("GET", "/api/status", b"", auth("alice"), "")
    assert other_browser != token
    assert proxy.sessions[other_browser][1] != proxy.sessions[token][1]
    _, _, _, second = proxy.open("GET", "/api/status", b"", auth("bob"), f"{cloud.SESSION_COOKIE}={token}")
    assert second != token  # another account cannot inherit this cloud visitor session
    Connection.response = Response(b"{}")
    proxy.open("POST", "/api/cancel", b'{}', auth("alice"), f"{cloud.SESSION_COOKIE}={token}")
    assert Connection.calls[-1][3]["Cookie"] == "__Host-leanmoe=signed"
    assert Connection.calls[-1][3]["Origin"] == "https://demo.example.test"
    with pytest.raises(PermissionError):
        proxy.open("POST", "/api/compare", b'{}', auth("bob"), f"{cloud.SESSION_COOKIE}={token}")


def test_cloud_relay_rejects_unprotected_or_wrong_location(monkeypatch):
    monkeypatch.setattr(cloud.http.client, "HTTPSConnection", Connection)
    proxy = cloud.CloudProxy("https://demo.example.test")
    for location, protected in (("local", True), ("server", False)):
        Connection.response = Response(json.dumps({"system": {"execution_location": location},
                                                   "access": {"protected": protected}}).encode(), "__Host-leanmoe=signed")
        with pytest.raises(ValueError, match="protected server"):
            proxy.open("GET", "/api/status", b"", auth("alice"), "")
    with pytest.raises(ValueError, match="unsupported"):
        proxy.open("POST", "/api/other", b"", auth("alice"), "")


def test_local_http_cloud_route_requires_auth_and_same_origin(tmp_path):
    from src.dashboard.server import serve
    server, engine = serve(0, models_root=tmp_path)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        # No background thread needed: handle_request serves one request at a time.
        import threading
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        conn.request("GET", "/cloud/api/status")
        response = conn.getresponse()
        assert response.status == 401 and response.getheader("WWW-Authenticate")
        response.read()
        conn.request("POST", "/cloud/api/generate", "{}", {"Content-Type": "application/json", "Origin": "https://evil.test"})
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        conn.close()
        server.shutdown()
        thread.join(5)
    finally:
        server.server_close()
        engine.close()
