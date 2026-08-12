"""Minimal MJPEG + status HTTP server for Reachy 1.2 stereo camera preview.

R12-303 — serves a browser-accessible stereo view from JPEG frame files written
by frame_file_writer().  Requires no extra dependencies beyond the Python stdlib.

Endpoints:
    GET /               HTML page with left/right MJPEG feeds and status bar
    GET /stream/left    MJPEG stream for the left camera
    GET /stream/right   MJPEG stream for the right camera
    GET /status         JSON status (backend, FPS, frame age, connection)

Port is configurable via --port or REACHY_SIM_CAMERA_WEB_PORT (default 8080).
The server binds to 127.0.0.1 by default (REACHY_SIM_CAMERA_WEB_HOST to override).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Optional


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Each request (MJPEG stream) runs in its own thread so both cameras stream concurrently."""
    daemon_threads = True

_LEFT_FILE = "/tmp/reachy_left.jpg"
_RIGHT_FILE = "/tmp/reachy_right.jpg"

_MJPEG_BOUNDARY = b"--reachyframe"
_MJPEG_HEADER = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: multipart/x-mixed-replace; boundary=reachyframe\r\n"
    b"Cache-Control: no-cache\r\n"
    b"Connection: keep-alive\r\n"
    b"\r\n"
)

_INDEX_HTML = b"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Reachy 1.2 Stereo View</title>
  <style>
    body { background: #1a1a2e; color: #e0e0e0; font-family: monospace; margin: 0; }
    h1 { text-align: center; padding: 12px 0 4px; font-size: 1.1em; color: #90caf9; }
    .cameras { display: flex; justify-content: center; gap: 12px; padding: 8px; }
    .cam-wrap { display: flex; flex-direction: column; align-items: center; }
    .cam-label { font-size: 0.85em; margin-bottom: 4px; color: #80cbc4; }
    img { max-width: 48vw; border: 1px solid #333; background: #111; }
    #status { text-align: center; padding: 8px; font-size: 0.78em; color: #aaa; }
    .ok { color: #81c784; }
    .warn { color: #ffb74d; }
  </style>
</head>
<body>
  <h1>Reachy 1.2 &mdash; Live Stereo View</h1>
  <div class="cameras">
    <div class="cam-wrap">
      <div class="cam-label">LEFT CAMERA</div>
      <img src="/stream/left" alt="left camera" />
    </div>
    <div class="cam-wrap">
      <div class="cam-label">RIGHT CAMERA</div>
      <img src="/stream/right" alt="right camera" />
    </div>
  </div>
  <div id="status">Connecting&hellip;</div>
  <script>
    async function poll() {
      try {
        const r = await fetch('/status');
        const d = await r.json();
        const el = document.getElementById('status');
        const cls = d.left_age_ms < 500 ? 'ok' : 'warn';
        el.innerHTML =
          'backend: <span class="' + cls + '">' + d.backend + '</span> &nbsp;|&nbsp; ' +
          'left age: ' + d.left_age_ms + ' ms &nbsp;|&nbsp; ' +
          'right age: ' + d.right_age_ms + ' ms &nbsp;|&nbsp; ' +
          'seq: ' + d.left_seq + ' / ' + d.right_seq;
      } catch(e) { /* ignore fetch errors during startup */ }
    }
    setInterval(poll, 1000);
    poll();
  </script>
</body>
</html>
"""


def _read_jpeg(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return b""


def _frame_age_ms(path: str) -> int:
    try:
        mtime = os.stat(path).st_mtime
        return max(0, int((time.time() - mtime) * 1000))
    except OSError:
        return -1


def _frame_seq(path: str) -> int:
    """Approximate sequence number from file mtime (not the actual fixture seq)."""
    try:
        mtime = os.stat(path).st_mtime
        return int(mtime * 15) % 100000  # rough 15 Hz seq
    except OSError:
        return 0


class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler; no routing library needed for three endpoints."""

    log_message = lambda *args: None  # silence per-request access log

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._serve_index()
        elif path == "/stream/left":
            self._serve_mjpeg(_LEFT_FILE)
        elif path == "/stream/right":
            self._serve_mjpeg(_RIGHT_FILE)
        elif path == "/status":
            self._serve_status()
        else:
            self.send_error(404, "Not Found")

    def _serve_index(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(_INDEX_HTML)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(_INDEX_HTML)

    def _serve_mjpeg(self, path: str):
        self.wfile.write(_MJPEG_HEADER)
        dt = 1.0 / 15.0
        while True:
            try:
                data = _read_jpeg(path)
                if data:
                    chunk = (
                        _MJPEG_BOUNDARY + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(data)).encode() + b"\r\n"
                        b"\r\n"
                        + data + b"\r\n"
                    )
                    self.wfile.write(chunk)
                    self.wfile.flush()
                time.sleep(dt)
            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception:
                break

    def _serve_status(self):
        payload = {
            "backend": "fixture",
            "left_age_ms": _frame_age_ms(_LEFT_FILE),
            "right_age_ms": _frame_age_ms(_RIGHT_FILE),
            "left_seq": _frame_seq(_LEFT_FILE),
            "right_seq": _frame_seq(_RIGHT_FILE),
        }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reachy stereo camera web server")
    parser.add_argument(
        "--host",
        default=os.environ.get("REACHY_SIM_CAMERA_WEB_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("REACHY_SIM_CAMERA_WEB_PORT", "8080")),
    )
    args = parser.parse_args()

    server = _ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"Camera web server on http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
