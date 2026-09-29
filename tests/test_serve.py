import json
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

SERVER = """
import sys, time, types
from feln_clm import cli
fake = types.SimpleNamespace(cat=types.SimpleNamespace(sha="x", layers={}), cfg={"base": "fake"},
                             threshold=0.5, ask=lambda t: (time.sleep(1.5), {"meta": {"text": t}})[1])
cli.translator = lambda a: fake
sys.argv = ["feln-clm", "serve", "okf", "model", "--port", sys.argv[1]]
cli.main()
"""


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_signal_finishes_the_request_in_flight_then_exits(sig):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-c", SERVER, str(port)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)  # fmt: skip
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{base}/api/info", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    got = {}

    def ask():
        req = urllib.request.Request(f"{base}/api/ask", data=json.dumps({"text": "hi"}).encode())
        with urllib.request.urlopen(req, timeout=10) as r:
            got["status"], got["body"] = r.status, json.load(r)

    t = threading.Thread(target=ask)
    t.start()
    time.sleep(0.5)  # the request is now inside the slow ask
    proc.send_signal(sig)
    t.join(10)
    out, _ = proc.communicate(timeout=10)
    assert got == {"status": 200, "body": {"meta": {"text": "hi"}}}, out
    assert proc.returncode == 128 + sig, out
    assert "shutting down" in out and f"stopped by {sig.name}" in out, out
