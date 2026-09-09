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
    h2 { font-size: .8em; color: #90caf9; margin: 0 0 8px; letter-spacing: .08em; }
    .cameras { display: flex; justify-content: center; gap: 12px; padding: 8px; }
    .cam-wrap { display: flex; flex-direction: column; align-items: center; }
    .cam-label { font-size: 0.85em; margin-bottom: 4px; color: #80cbc4; }
    img { max-width: 42vw; border: 1px solid #333; background: #111; }
    #status { text-align: center; padding: 8px; font-size: 0.78em; color: #aaa; }
    .ok { color: #81c784; } .warn { color: #ffb74d; } .bad { color: #e57373; }

    #panel { max-width: 960px; margin: 4px auto 32px; padding: 0 12px;
             display: grid; grid-template-columns: 300px 1fr; gap: 18px; }
    .card { background: #16213e; border: 1px solid #2b3a63; border-radius: 6px;
            padding: 12px; }
    #grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
    .cell { background: #0f3460; border: 1px solid #2b3a63; border-radius: 4px;
            padding: 8px 4px; text-align: center; cursor: pointer;
            font-size: .72em; line-height: 1.5; color: #cfd8dc; }
    .cell:hover { border-color: #90caf9; }
    .cell .nm { font-weight: bold; color: #90caf9; }
    .cell .oc { color: #81c784; }
    .cell.busy { background: #1b3a5c; }
    .cell.unreach { background: #2a1f2f; color: #8d6e78; cursor: not-allowed; }
    .cell.unreach:hover { border-color: #7a4a5a; }
    .hint { font-size: .68em; color: #8a95b2; margin-top: 8px; line-height: 1.6; }
    select, button { font-family: monospace; font-size: .78em; background: #0f3460;
                     color: #e0e0e0; border: 1px solid #2b3a63; border-radius: 4px;
                     padding: 6px 8px; }
    button { cursor: pointer; } button:hover { border-color: #90caf9; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .row { display: flex; gap: 6px; align-items: center; margin-bottom: 8px; }
    .row label { font-size: .72em; color: #8a95b2; min-width: 54px; }
    select { flex: 1; }
    #log { height: 132px; overflow-y: auto; font-size: .7em; line-height: 1.7;
           background: #0d1b2e; border: 1px solid #2b3a63; border-radius: 4px;
           padding: 8px; margin-top: 10px; }
    #log div { white-space: pre-wrap; word-break: break-word; }
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

  <div id="panel">
    <div class="card">
      <h2>THE BOARD</h2>
      <div id="grid">loading&hellip;</div>
      <div class="hint">
        Laid out as the camera sees it: row 3 farthest at the top, column 1
        (the robot's left, +y) on the left. Click a cell to place the selected
        object on it.
      </div>
    </div>

    <div class="card">
      <h2>SCENE CONTROL</h2>
      <div class="row">
        <label>object</label>
        <select id="obj"></select>
        <button id="recall">recall</button>
      </div>
      <div class="row">
        <label>yaw</label>
        <select id="yaw">
          <option>0</option><option>15</option><option>30</option>
          <option>45</option><option>90</option>
        </select>
        <button id="clear">clear board</button>
      </div>
      <div class="row">
        <label>sim</label>
        <input id="wsurl" style="flex:1;font-family:monospace;font-size:.78em;
               background:#0f3460;color:#e0e0e0;border:1px solid #2b3a63;
               border-radius:4px;padding:6px 8px" />
        <button id="reconnect">connect</button>
      </div>
      <div id="log"></div>
    </div>
  </div>

  <script>
  (function () {
    var scene = null, ws = null, occupancy = {}, lastRender = 0;

    function log(msg, cls) {
      var el = document.getElementById('log');
      var d = document.createElement('div');
      if (cls) { d.className = cls; }
      d.textContent = msg;
      el.appendChild(d);
      while (el.childNodes.length > 60) { el.removeChild(el.firstChild); }
      el.scrollTop = el.scrollHeight;
    }

    function cellByName(n) {
      for (var i = 0; i < scene.cells.length; i++) {
        if (scene.cells[i].name === n) { return scene.cells[i]; }
      }
      return null;
    }

    // Rows descend (r3 at the top) so the grid reads the way the camera sees
    // the board; columns ascend left to right because col 1 is +y, the robot's
    // left, which is also the left of the frame.
    function drawGrid() {
      var g = document.getElementById('grid');
      g.innerHTML = '';
      for (var r = 3; r >= 1; r--) {
        for (var c = 1; c <= 3; c++) {
          (function (name) {
            var cell = cellByName(name);
            var d = document.createElement('div');
            d.className = 'cell' + (cell && !cell.reachable ? ' unreach' : '') +
                          (occupancy[name] ? ' busy' : '');
            var occ = occupancy[name] || '';
            d.innerHTML = '<div class="nm">' + name + '</div>' +
                          '<div>' + (cell ? cell.distance_m.toFixed(2) + ' m' : '&mdash;') + '</div>' +
                          '<div class="oc">' + (occ ? occ : '&nbsp;') + '</div>';
            d.title = cell && !cell.reachable
              ? name + ' was measured unreachable for the right arm - click to place anyway'
              : 'place the selected object on ' + name;
            d.onclick = function () { place(name); };
            g.appendChild(d);
          })('r' + r + 'c' + c);
        }
      }
    }

    function send(msg) {
      if (!ws || ws.readyState !== 1) { log('not connected', 'bad'); return false; }
      ws.send(JSON.stringify(msg));
      return true;
    }

    function place(name) {
      var obj = document.getElementById('obj').value;
      var yaw = parseFloat(document.getElementById('yaw').value) || 0;
      var cell = cellByName(name);
      // An unreachable cell is refused by the server unless asked for
      // explicitly.  Clicking one here IS that explicit ask, so say so in the
      // log rather than silently passing the override.
      var force = cell && !cell.reachable;
      if (force) { log('! ' + name + ' is unreachable - placing anyway', 'warn'); }
      send({type: 'place_object', object_id: obj, cell: name, yaw_deg: yaw,
            allow_unreachable: !!force, allow_occupied: false,
            reshape: null, request_id: obj + '@' + name});
    }

    function recall(obj) {
      send({type: 'place_object', object_id: obj, cell: null, yaw_deg: 0,
            allow_unreachable: false, allow_occupied: false,
            reshape: null, request_id: obj + '@pool'});
    }

    function onState(m) {
      // Throttle: state arrives ~30/s and redrawing the DOM that often is
      // pure waste for a board that changes when someone clicks.
      var now = Date.now();
      if (now - lastRender < 200) { return; }
      lastRender = now;
      var next = {};
      (m.objects || []).forEach(function (o) {
        var p = o.pos_xyz;
        scene.cells.forEach(function (c) {
          if (Math.abs(p[0] - c.x) <= c.half_extent &&
              Math.abs(p[1] - c.y) <= c.half_extent &&
              p[2] > c.top_z - 0.01) { next[c.name] = o.object_id; }
        });
      });
      if (JSON.stringify(next) !== JSON.stringify(occupancy)) {
        occupancy = next;
        drawGrid();
      }
    }

    function connect() {
      var url = document.getElementById('wsurl').value;
      if (ws) { try { ws.close(); } catch (e) {} }
      log('connecting to ' + url + ' ...');
      ws = new WebSocket(url);
      ws.onopen = function () {
        // The server drops any connection whose first frame is not `hello`.
        ws.send(JSON.stringify({type: 'hello', protocol_version: 1,
                                client_id: 'camera-panel'}));
      };
      ws.onmessage = function (ev) {
        var m = JSON.parse(ev.data);
        if (m.type === 'hello_ack') {
          log('connected - protocol ' + m.protocol_version +
              ', server ' + m.server_version, 'ok');
        } else if (m.type === 'heartbeat') {
          // MUST reply.  The server drops any client it has not heard from
          // within 6 s (_HB_DEADLINE), and a panel that only listens says
          // nothing at all -- so the socket died silently ~6 s after
          // connecting and every click after that went nowhere.
          ws.send(JSON.stringify({type: 'heartbeat_ack',
                                  echo_ns: m.sent_ns || 0}));
        } else if (m.type === 'state') {
          onState(m);
        } else if (m.type === 'place_ack') {
          if (m.accepted) {
            var pl = m.placement;
            log('OK   ' + pl.object_id + ' -> ' +
                (pl.cell || 'pool') + '  step ' + m.sim_step, 'ok');
          } else {
            log('X    ' + m.error, 'bad');
          }
        } else if (m.type === 'error') {
          log('server error: ' + m.message, 'bad');
        }
      };
      ws.onclose = function () { log('disconnected', 'warn'); };
      ws.onerror = function () {
        log('cannot reach ' + url + ' - is the native sim running?', 'bad');
      };
    }

    fetch('/scene').then(function (r) { return r.json(); }).then(function (d) {
      scene = d;
      if (d.error) { log('scene: ' + d.error, 'bad'); }
      var sel = document.getElementById('obj');
      d.objects.forEach(function (o) {
        var opt = document.createElement('option');
        opt.value = o; opt.textContent = o; sel.appendChild(opt);
      });
      // The sim's websocket runs on the HOST, not in this container, so the
      // browser reaches it at the page's own hostname.
      document.getElementById('wsurl').value =
        'ws://' + location.hostname + ':' + (d.ws_port || 8765);
      drawGrid();
      log('scene: ' + (d.scene || '?') + ' - ' + d.cells.length + ' cells, ' +
          d.objects.length + ' placeable objects');
      connect();
    });

    document.getElementById('recall').onclick = function () {
      recall(document.getElementById('obj').value);
    };
    document.getElementById('clear').onclick = function () {
      scene.objects.forEach(recall);
    };
    document.getElementById('reconnect').onclick = connect;

    async function poll() {
      try {
        var d = await (await fetch('/status')).json();
        var cls = d.left_age_ms < 500 ? 'ok' : 'warn';
        document.getElementById('status').innerHTML =
          'frames <span class="' + cls + '">' + d.left_age_ms + ' ms</span> old' +
          ' &nbsp;|&nbsp; seq: ' + d.left_seq + ' / ' + d.right_seq +
          ' &nbsp;|&nbsp; sim link: ' +
          (ws && ws.readyState === 1
            ? '<span class="ok">connected</span>'
            : '<span class="bad">down</span>');
      } catch (e) { /* ignore fetch errors during startup */ }
    }
    setInterval(poll, 1000);
    poll();
  })();
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


_SCENE_FILE = os.environ.get("REACHY_SIM_SCENE_FILE",
                             "/opt/scenes/tabletop_demo.yaml")
# The sim's websocket runs on the HOST, not in this container, so the browser
# reaches it at the page's own hostname.  Overridable in the UI for the case
# where the page and the sim are on different machines.
_SIM_WS_PORT = int(os.environ.get("REACHY_SIM_WS_PORT", "8765"))


def _scene_payload() -> dict:
    """Cells + placeable objects, or an `error` the page can show plainly."""
    for candidate in ("/opt", os.path.join(os.path.dirname(__file__), "..",
                                           "native_mujoco")):
        if os.path.isfile(os.path.join(candidate, "placement.py")):
            if candidate not in sys.path:
                sys.path.append(candidate)
            break
    try:
        from placement import cells_from_scene, pool_ids
        from scene_io import load_scene
    except ImportError as exc:
        return {"error": f"scene tools unavailable: {exc}",
                "cells": [], "objects": [], "ws_port": _SIM_WS_PORT}
    try:
        doc = load_scene(_SCENE_FILE)
    except Exception as exc:                      # bad path, bad YAML, cycle
        return {"error": f"{_SCENE_FILE}: {exc}",
                "cells": [], "objects": [], "ws_port": _SIM_WS_PORT}

    cells = cells_from_scene(doc)
    placeable = pool_ids(doc) or [
        o["id"] for o in doc.get("objects", [])
        if "manipulable" in (o.get("tags") or [])
    ]
    return {
        "scene": doc.get("name", os.path.basename(_SCENE_FILE)),
        "ws_port": _SIM_WS_PORT,
        "cells": [
            {"name": c.name, "x": c.x, "y": c.y, "top_z": c.top_z,
             "half_extent": c.half_extent, "reachable": c.reachable,
             "distance_m": round(c.shoulder_distance_m, 3)}
            for c in sorted(cells.values(), key=lambda c: c.name)
        ],
        "objects": sorted(placeable),
    }


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
        elif path == "/scene":
            self._serve_scene()
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

    def _serve_scene(self):
        """Grid cells and placeable objects of the scene the sim is running.

        Served rather than hardcoded in the page so the panel and the physics
        cannot disagree about where a cell is or which ones the arm can reach —
        both read the same YAML.  Reachability comes from the cell's own tag,
        which is where FWDCenterLabMCC's measured IK result lives.
        """
        body = json.dumps(_scene_payload()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

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
