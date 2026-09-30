import contextlib
import json
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

from feln_clm.server import IDLE

SERVER = """
import sys, time, types
from feln_clm import cli
fake = types.SimpleNamespace(cat=types.SimpleNamespace(sha="x", layers={}), cfg={"base": "fake"},
                             threshold=0.5, ask=lambda t: (time.sleep(1.5), {"meta": {"text": t}})[1])
cli.translator = lambda a: fake
sys.argv = ["feln-clm", "serve", "okf", "model", "--port", sys.argv[1]]
cli.main()
"""


@contextlib.contextmanager
def serving():
    """A real `feln-clm serve` process over a slow fake translator; always killed at the end."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-c", SERVER, str(port)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)  # fmt: skip
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/info", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("server never came up")
        yield proc, port
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()


def stop(proc, sig, timeout: float) -> str:
    """Send ``sig``; return the output once the process exits, or fail if it does not."""
    proc.send_signal(sig)
    try:
        return proc.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        pytest.fail(f"still running {timeout} s after {sig.name}")


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_signal_finishes_the_request_in_flight_then_exits(sig):
    with serving() as (proc, port):
        got = {}

        def ask():
            data = json.dumps({"text": "hi"}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/ask", data=data)
            with urllib.request.urlopen(req, timeout=10) as r:
                got["status"], got["body"] = r.status, json.load(r)

        t = threading.Thread(target=ask)
        t.start()
        time.sleep(0.5)  # the request is now inside the slow ask
        out = stop(proc, sig, 10)
        t.join(10)
        assert got == {"status": 200, "body": {"meta": {"text": "hi"}}}, out
        assert proc.returncode == 128 + sig, out
        assert "shutting down" in out and f"stopped by {sig.name}" in out, out


def test_idle_connection_does_not_block_shutdown():
    """A client that connects and sends nothing is dropped after server.IDLE seconds."""
    with serving() as (proc, port), socket.create_connection(("127.0.0.1", port)):
        time.sleep(0.2)
        t0 = time.time()
        out = stop(proc, signal.SIGTERM, IDLE + 5)
        assert proc.returncode == 128 + signal.SIGTERM, out
        assert time.time() - t0 < IDLE + 2, out
