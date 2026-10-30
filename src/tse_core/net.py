"""Downloads with certificate checks that work on ordinary home and office computers.

Python does not always trust the same certificates as the web browser:

* the python.org installer on macOS ships without any (fixed by certifi);
* antivirus programs (for example Avast, AVG, Kaspersky, ESET, Bitdefender),
  company networks and some VPNs inspect HTTPS traffic by re-signing it with
  their own root certificate. They add that root to the operating system's
  store, so browsers accept it, but certifi's list does not have it and the
  download fails with "self-signed certificate in certificate chain".

So each download tries the operating system's store first (via the
``truststore`` package, when installed), then certifi's list, then Python's
default. Certificates are always verified; nothing here turns checking off.
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from typing import Callable, Union

Request = Union[str, urllib.request.Request]


def _contexts() -> list[tuple[str, Callable[[], ssl.SSLContext]]]:
    out: list[tuple[str, Callable[[], ssl.SSLContext]]] = []
    try:
        import truststore

        out.append(("system", lambda: truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)))
    except ImportError:
        pass
    try:
        import certifi

        out.append(("certifi", lambda: ssl.create_default_context(cafile=certifi.where())))
    except ImportError:
        pass
    out.append(("python", ssl.create_default_context))
    return out


def _is_cert_error(exc: BaseException) -> bool:
    reason = getattr(exc, "reason", exc)
    return isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(reason)


def fetch(request: Request, timeout: float = 60.0) -> bytes:
    """Download ``request`` and return the body, trying each certificate store in turn."""
    last: BaseException | None = None
    for _name, make in _contexts():
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=make()) as resp:
                return resp.read()
        except (urllib.error.URLError, ssl.SSLError) as exc:
            if not _is_cert_error(exc):
                raise
            last = exc
    assert last is not None
    if isinstance(last, urllib.error.URLError):
        raise last
    raise urllib.error.URLError(last)


def certificate_help(url: str, offline_hint: str) -> str:
    """What to tell the user when no certificate store accepts the server."""
    have_truststore = any(name == "system" for name, _ in _contexts())
    lines = [
        "Python could not verify the server's certificate.",
        "If the message says 'self-signed certificate in certificate chain', something on this",
        "computer or network is inspecting secure connections: usually an antivirus program's web",
        "or HTTPS scanning, a company network, or a VPN.",
    ]
    if not have_truststore:
        lines += ["Install the package that lets Python use the same certificates as your browser,",
                  "then run this command again:",
                  "  pip install truststore"]
    else:
        lines += ["Python already uses your system's certificates and still could not verify this site."]
    lines += [f"Or download {url} in a browser and run:", f"  {offline_hint}"]
    return "\n".join(lines)
