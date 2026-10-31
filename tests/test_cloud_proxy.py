"""The public cloud relay uses separate browser sessions and verified HTTPS."""

import http.client
import json
import threading

import pytest

from src.dashboard import cloud


class Response:
    def __init__(self, body, cookie=None, status=200):
        self.body, self.cookie, self.status = body, cookie, status

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


def status_response(cookie="__Host-leanmoe=signed; Secure; HttpOnly", *, location="server", public=True):
    return Response(json.dumps({"system": {"execution_location": location},
                                "access": {"protected": True, "public_demo": public}}).encode(), cookie)


def test_cloud_relay_is_anonymous_and_keeps_sessions_separate(monkeypatch):
    monkeypatch.setattr(cloud.http.client, "HTTPSConnection", Connection)
    proxy = cloud.CloudProxy("https://demo.example.test")
    Connection.response = status_response()
    _, response, payload, token = proxy.open("GET", "/api/status", b"", "")
    assert response.status == 200 and token and b"signed" not in payload
    assert "Authorization" not in Connection.calls[-1][3]
    assert "Cookie" not in Connection.calls[-1][3]
    Connection.response = status_response("__Host-leanmoe=other; Secure; HttpOnly")
    _, _, _, other_browser = proxy.open("GET", "/api/status", b"", "")
    assert other_browser != token
    assert proxy.sessions[other_browser][0] != proxy.sessions[token][0]
    Connection.response = Response(b"{}")
    proxy.open("POST", "/api/cancel", b'{}', f"{cloud.SESSION_COOKIE}={token}")
    assert Connection.calls[-1][3]["Cookie"] == "__Host-leanmoe=signed"
    assert Connection.calls[-1][3]["Origin"] == "https://demo.example.test"
    assert "Authorization" not in Connection.calls[-1][3]
    with pytest.raises(PermissionError, match="expired"):
        proxy.open("POST", "/api/compare", b'{}', "")


def test_cloud_relay_rejects_login_required_or_wrong_backend(monkeypatch):
    monkeypatch.setattr(cloud.http.client, "HTTPSConnection", Connection)
    proxy = cloud.CloudProxy("https://demo.example.test")
    Connection.response = Response(b"login required", status=401)
    with pytest.raises(ValueError, match="still requires login"):
        proxy.open("GET", "/api/status", b"", "")
    for location, public in (("local", True), ("server", False)):
        Connection.response = status_response(location=location, public=public)
        with pytest.raises(ValueError, match="public server"):
            proxy.open("GET", "/api/status", b"", "")
    with pytest.raises(ValueError, match="unsupported"):
        proxy.open("POST", "/api/other", b"", "")


def test_local_http_cloud_route_has_no_login_and_requires_same_origin(tmp_path, monkeypatch):
    from src.dashboard.server import serve
    def unavailable(*args):
        raise ValueError("public release not live")
    monkeypatch.setattr(cloud.CloudProxy, "open", unavailable)
    server, engine = serve(0, models_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        conn.request("GET", "/cloud/api/status")
        response = conn.getresponse()
        assert response.status == 502 and response.getheader("WWW-Authenticate") is None
        response.read()
        conn.request("POST", "/cloud/api/generate", "{}", {"Content-Type": "application/json", "Origin": "https://evil.test"})
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        conn.close()
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()
        engine.close()
