"""Downloads: certificate stores are tried in turn and checking is never switched off."""

import ssl
import urllib.error

import pytest

from tse_core import net


class _Resp:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_falls_back_to_next_store(monkeypatch):
    calls = []

    def fake_contexts():
        return [("system", lambda: "sys"), ("certifi", lambda: "cert")]

    def fake_urlopen(req, timeout, context):
        calls.append(context)
        if context == "sys":
            raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED"))
        return _Resp(b"ok")

    monkeypatch.setattr(net, "_contexts", fake_contexts)
    monkeypatch.setattr(net.urllib.request, "urlopen", fake_urlopen)
    assert net.fetch("https://example.org") == b"ok"
    assert calls == ["sys", "cert"]


def test_fetch_reports_certificate_error_when_all_fail(monkeypatch):
    def fake_urlopen(req, timeout, context):
        raise urllib.error.URLError(ssl.SSLCertVerificationError(
            1, "[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate in certificate chain"))

    monkeypatch.setattr(net, "_contexts", lambda: [("a", lambda: 1), ("b", lambda: 2)])
    monkeypatch.setattr(net.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.URLError) as info:
        net.fetch("https://example.org")
    assert "CERTIFICATE" in str(info.value.reason).upper()


def test_other_errors_are_not_retried(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout, context):
        calls.append(context)
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(net, "_contexts", lambda: [("a", lambda: 1), ("b", lambda: 2)])
    monkeypatch.setattr(net.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.URLError):
        net.fetch("https://example.org")
    assert calls == [1]


def test_contexts_always_verify():
    for _name, make in net._contexts():
        ctx = make()
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname


def test_help_mentions_offline_route():
    text = net.certificate_help("https://x/y.csv", "tse elements-update --from <file>")
    assert "antivirus" in text and "tse elements-update --from" in text
