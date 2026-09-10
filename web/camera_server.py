"""Minimal MJPEG + status HTTP server for Reachy 1.2 stereo camera preview.

R12-303 — serves a browser-accessible stereo view from JPEG frame files written
by frame_file_writer().  Requires no extra dependencies beyond the Python stdlib.

Endpoints:
    GET /               HTML page with left/right MJPEG feeds and status bar
    GET /stream/left    MJPEG stream for the left camera
    GET /stream/right   MJPEG stream for the right camera
    GET /status         JSON status (backend, FPS, frame age, connection)
    GET /scene          JSON grid cells + placeable objects for the panel
    GET /capabilities   JSON server capabilities (see web/panel_routes.py)
    GET /tasks/{id}     JSON task status, transcript and proposal
    POST /tasks         Submit a typed command  (issue #49)
    POST /tasks/{id}/reply | /confirm | /cancel

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


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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

    /* TALK TO REACHY (issue #49) */
    #talk { grid-column: 1 / -1; }
    .cardhead { display: flex; align-items: baseline; justify-content: space-between;
                gap: 10px; flex-wrap: wrap; }
    .badge { font-size: .62em; letter-spacing: .06em; color: #b0bec5;
             border: 1px solid #2b3a63; border-radius: 10px; padding: 2px 8px; }
    #convo { height: 190px; overflow-y: auto; background: #0d1b2e;
             border: 1px solid #2b3a63; border-radius: 4px; padding: 10px;
             font-size: .74em; line-height: 1.7; }
    .turn { margin-bottom: 8px; white-space: pre-wrap; word-break: break-word; }
    .turn .who { color: #80cbc4; }
    .turn.reachy .who { color: #90caf9; }
    .chips { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 5px; }
    .chips button { font-size: .68em; padding: 3px 8px; }
    #ask { display: flex; gap: 8px; margin-top: 10px; align-items: flex-end; }
    #msg { flex: 1; min-height: 46px; max-height: 140px; resize: vertical;
           font-family: monospace; font-size: .78em; background: #0f3460;
           color: #e0e0e0; border: 1px solid #2b3a63; border-radius: 4px;
           padding: 7px 8px; line-height: 1.5; }
    #proposal { margin-top: 10px; padding: 9px 10px; border-radius: 4px;
                background: #10253f; border: 1px solid #2b3a63; font-size: .74em;
                line-height: 1.6; }
    #proposal .why { color: #8a95b2; }
    #proposal .acts { margin-top: 8px; display: flex; gap: 6px; }
    #taskstatus { margin-top: 8px; font-size: .68em; color: #8a95b2;
                  min-height: 1.2em; }
    #taskstatus.bad { color: #e57373; } #taskstatus.ok { color: #81c784; }
    .sendhint { font-size: .62em; color: #6f7a96; margin-top: 4px; }

    /* Narrow viewport: one column, and let the feeds shrink with it. */
    @media (max-width: 760px) {
      #panel { grid-template-columns: 1fr; }
      .cameras { flex-wrap: wrap; }
      img { max-width: 92vw; }
      #ask { flex-direction: column; align-items: stretch; }
    }
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

    <div class="card" id="talk">
      <div class="cardhead">
        <h2>TALK TO REACHY</h2>
        <span class="badge" id="mode">planning only &middot; scene data</span>
      </div>
      <div id="convo" role="log" aria-live="polite" aria-label="Conversation"></div>
      <div id="proposal" hidden></div>
      <div id="ask">
        <textarea id="msg" rows="2" aria-label="Command or answer"
                  placeholder="Tell me what you would like me to do&hellip;"></textarea>
        <button id="send">Send</button>
      </div>
      <div class="sendhint">Enter sends &middot; Shift+Enter for a new line</div>
      <div id="taskstatus" role="status" aria-live="polite"></div>
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

    // ---------------------------------------------------------------------
    // TALK TO REACHY (issue #49)
    //
    // Planning only: the server has no executor yet, so Confirm resolves to
    // "no movement performed" and this code must never suggest otherwise.
    // Capabilities are read from the server rather than assumed, so when an
    // executor does land (issue #51) the wording follows it without a second
    // source of truth here.
    //
    // Every piece of user or model text reaches the DOM through textContent.
    // The transcript is the one place arbitrary strings are rendered, and
    // innerHTML there would make a planner message an injection vector.
    // ---------------------------------------------------------------------
    var task = null;             // latest task snapshot from the server
    var caps = {live_execution: false, max_text_chars: 2000};
    var pollTimer = null, busy = false, composing = false;
    var TERMINAL = {confirmed_no_motion: 1, completed: 1, cancelled: 1,
                    failed: 1, expired: 1};

    // Only these states change without the human doing anything, so only
    // these are worth polling.  Polling through needs_clarification or
    // awaiting_confirmation was not merely wasteful: each poll redrew the
    // transcript, which destroyed and recreated the choice buttons roughly
    // twice a second, so a click could land on a node that no longer existed.
    function serverWillMove(state) {
      return state === 'planning' || state === 'executing';
    }

    var convoEl = document.getElementById('convo');
    var msgEl = document.getElementById('msg');
    var sendEl = document.getElementById('send');
    var propEl = document.getElementById('proposal');
    var statusEl = document.getElementById('taskstatus');

    function tstatus(text, cls) {
      statusEl.className = cls || '';
      statusEl.textContent = text || '';
    }

    function turn(role, text, choices) {
      var d = document.createElement('div');
      d.className = 'turn ' + (role === 'user' ? 'user' : 'reachy');
      var who = document.createElement('span');
      who.className = 'who';
      who.textContent = (role === 'user' ? 'You: ' : 'Reachy: ');
      d.appendChild(who);
      d.appendChild(document.createTextNode(text));
      if (choices && choices.length) {
        var box = document.createElement('div');
        box.className = 'chips';
        choices.forEach(function (c) {
          var b = document.createElement('button');
          b.type = 'button';
          b.textContent = c;
          // A chip is a shortcut for typing the answer, not a separate path:
          // it fills the box and sends, so the same validation runs either way.
          b.onclick = function () { msgEl.value = c; submit(); };
          box.appendChild(b);
        });
        d.appendChild(box);
      }
      convoEl.appendChild(d);
      while (convoEl.childNodes.length > 60) {
        convoEl.removeChild(convoEl.firstChild);
      }
      convoEl.scrollTop = convoEl.scrollHeight;
    }

    // The server owns the transcript, so redraw from it rather than appending
    // locally: a refresh, a reconnect, or a reply that lost its response then
    // shows the same conversation the server actually has.
    function renderConvo(t) {
      convoEl.textContent = '';
      var last = t.events.length - 1;
      t.events.forEach(function (e, i) {
        var open = (i === last && t.state === 'needs_clarification');
        turn(e.role, e.text, open ? e.choices : null);
      });
    }

    function renderProposal(t) {
      var p = t.proposal;
      if (!p || t.state !== 'awaiting_confirmation') {
        propEl.hidden = true;
        propEl.textContent = '';
        return;
      }
      propEl.hidden = false;
      propEl.textContent = '';

      var head = document.createElement('div');
      head.textContent = 'Proposed action: ' + (p.summary || p.task_type);
      propEl.appendChild(head);

      var why = document.createElement('div');
      why.className = 'why';
      why.textContent = p.brief_reason + '  [' + p.semantic_source + ']';
      propEl.appendChild(why);

      var acts = document.createElement('div');
      acts.className = 'acts';
      var ok = document.createElement('button');
      // Confirm carries the plan id AND the version it was drawn at, so a
      // click on a button rendered before some other change is refused by the
      // server rather than applied to whatever the plan became.
      ok.textContent = caps.live_execution ? 'Confirm plan' : 'Confirm plan (no motion)';
      ok.onclick = function () {
        post('/tasks/' + t.task_id + '/confirm',
             {plan_id: p.plan_id, plan_version: p.plan_version});
      };
      var no = document.createElement('button');
      no.textContent = 'Cancel';
      no.onclick = function () { post('/tasks/' + t.task_id + '/cancel', {}); };
      acts.appendChild(ok);
      acts.appendChild(no);
      propEl.appendChild(acts);
    }

    function render(t) {
      // Redraw only on a real change.  The server bumps `version` on every
      // transition, so an unchanged version means an unchanged conversation
      // and rebuilding it would only throw away scroll position and focus.
      var unchanged = task && task.task_id === t.task_id &&
                      task.version === t.version && task.state === t.state;
      task = t;
      if (!unchanged) {
        renderConvo(t);
        renderProposal(t);
      }
      if (t.state === 'planning') { tstatus('Working\\u2026'); }
      else if (t.detail) {
        tstatus(t.detail, t.state === 'confirmed_no_motion' ? 'ok'
                        : (t.state === 'failed' ? 'bad' : ''));
      } else { tstatus(''); }
      sendEl.disabled = busy || t.state === 'planning';
      if (!serverWillMove(t.state)) { stopPolling(); }
    }

    function stopPolling() {
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    }

    function startPolling(id) {
      stopPolling();
      pollTimer = setInterval(function () {
        fetch('/tasks/' + id).then(function (r) {
          if (!r.ok) { throw new Error('http ' + r.status); }
          return r.json();
        }).then(render).catch(function () {
          // Explicit rather than silent: the brief requires a visible status
          // when the server cannot be reached, and the camera streams keep
          // running independently either way.
          tstatus('Lost contact with the task server. Retrying\\u2026', 'bad');
        });
      }, 700);
    }

    function post(path, body) {
      if (busy) { return; }
      busy = true;
      sendEl.disabled = true;
      return fetch(path, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      }).then(function (r) {
        return r.json().then(function (d) { return {ok: r.ok, d: d}; });
      }).then(function (res) {
        if (!res.ok) {
          tstatus(res.d.error || 'That request was refused.', 'bad');
          // A refusal can also have CHANGED the task: a plan invalidated by
          // the board moving is refused and failed in the same breath.  Re-read
          // rather than redrawing the copy we already had, which would leave a
          // dead proposal on screen next to the reason it was refused.
          if (task) {
            fetch('/tasks/' + task.task_id)
              .then(function (r) { return r.ok ? r.json() : null; })
              .then(function (d) { if (d) { render(d); } })
              .catch(function () {});
          }
          return;
        }
        render(res.d);
        if (serverWillMove(res.d.state)) { startPolling(res.d.task_id); }
      }).catch(function () {
        tstatus('Could not reach the task server.', 'bad');
      }).then(function () {
        busy = false;
        if (task) { sendEl.disabled = task.state === 'planning'; }
        else { sendEl.disabled = false; }
      });
    }

    function submit() {
      var text = msgEl.value;
      if (!text.trim()) { tstatus('Type a command first.', 'bad'); return; }
      if (text.length > caps.max_text_chars) {
        tstatus('That message is too long (limit ' + caps.max_text_chars +
                ' characters).', 'bad');
        return;
      }
      msgEl.value = '';
      var active = task && !TERMINAL[task.state];
      if (active && task.state === 'needs_clarification') {
        post('/tasks/' + task.task_id + '/reply',
             {question_id: task.question_id, text: text});
      } else {
        // A fresh client id per submission makes a retried POST idempotent
        // server-side, so a double click cannot open two tasks.
        post('/tasks', {text: text, client_request_id: newReqId()});
      }
    }

    function newReqId() {
      return Date.now().toString(36) + '-' +
             Math.floor(Math.random() * 1e9).toString(36);
    }

    // Enter sends, Shift+Enter inserts a newline.  `composing` guards an IME:
    // in Japanese, Chinese or Korean input the Enter that commits a candidate
    // must not also send the message.
    msgEl.addEventListener('compositionstart', function () { composing = true; });
    msgEl.addEventListener('compositionend', function () { composing = false; });
    msgEl.addEventListener('keydown', function (ev) {
      if (ev.key !== 'Enter' || ev.shiftKey) { return; }
      if (composing || ev.isComposing || ev.keyCode === 229) { return; }
      ev.preventDefault();
      submit();
    });
    sendEl.onclick = submit;

    // The badge says three things, and all three can change under a running
    // page: whether an executor exists, where semantics come from, and
    // whether the board is actually being seen right now.  So it is refreshed
    // on the status poll rather than read once at load.
    function renderBadge(link) {
      var source = caps.semantic_source === 'scene_data'
        ? 'scene data' : caps.semantic_source;
      var board = (link && link.live) ? 'board live' : 'board unseen';
      document.getElementById('mode').textContent =
        (caps.live_execution ? 'live execution' : 'planning only') +
        ' \\u00b7 ' + source + ' \\u00b7 ' + board;
      document.getElementById('mode').title = (link && link.live)
        ? 'Live object poses from the simulator (scene revision ' +
          (link.scene_revision || '?') + ')'
        : 'No live object poses: ' + ((link && link.detail) || 'link down') +
          '. I will not say what is on the board.';
    }

    function refreshCaps() {
      return fetch('/capabilities').then(function (r) { return r.json(); })
        .then(function (d) {
          caps = d.capabilities || caps;
          renderBadge(d.sim_link);
          return d;
        });
    }

    refreshCaps()
      .then(function () {
        turn('reachy', 'Tell me what you would like me to do.');
      })
      .catch(function () {
        tstatus('The command panel is unavailable on this server.', 'bad');
        msgEl.disabled = true;
        sendEl.disabled = true;
      });

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
    // Same cadence as the frame-age poll, and cheap for the same reason: it
    // reads state the server already holds.
    setInterval(function () { refreshCaps().catch(function () {}); }, 1000);
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

#: Largest POST body accepted on the task routes (see web/panel_routes.py).
_MAX_BODY_BYTES = 8 * 1024


def _load_scene() -> tuple:
    """(doc, cells, placeable_ids, error) from the scene the sim is running.

    Shared by `/scene` and by the command panel's planner so the two cannot
    disagree about what is in the scene — the same reason `/scene` serves the
    grid rather than the page hardcoding it.
    """
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
        return {}, {}, [], f"scene tools unavailable: {exc}"
    try:
        doc = load_scene(_SCENE_FILE)
    except Exception as exc:                      # bad path, bad YAML, cycle
        return {}, {}, [], f"{_SCENE_FILE}: {exc}"

    cells = cells_from_scene(doc)
    placeable = pool_ids(doc) or [
        o["id"] for o in doc.get("objects", [])
        if "manipulable" in (o.get("tags") or [])
    ]
    return doc, cells, placeable, ""


def _scene_payload() -> dict:
    """Cells + placeable objects, or an `error` the page can show plainly."""
    doc, cells, placeable, error = _load_scene()
    if error:
        return {"error": error, "cells": [], "objects": [],
                "ws_port": _SIM_WS_PORT}
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


_SIM_LINK = None


def _scene_view():
    """The planner's view of the scene, rebuilt per request.

    Per request rather than cached because scene control places and recalls
    objects between one command and the next, and a planner reasoning over a
    stale document would propose moving something that is no longer there.

    Semantics come from the scene document; live object poses come from the
    simulator link when it has a fresh snapshot.  With no snapshot the view
    stays non-live and the planner goes on saying it cannot see the board,
    which is the truth rather than an empty board.
    """
    from panel_scene import SceneView, apply_snapshot, scene_view_from_doc
    doc, cells, placeable, error = _load_scene()
    if error:
        return SceneView(error=error)
    view = scene_view_from_doc(doc, cells, placeable=placeable)
    if _SIM_LINK is not None:
        apply_snapshot(view, _SIM_LINK.snapshot())
    return view


_PANEL = None


def _panel():
    """Lazily build the task coordinator, and keep serving frames if it fails.

    The stereo view is the part of this page people rely on; an import error in
    the conversation code must degrade to "no panel" rather than taking the
    camera server down with it.
    """
    global _PANEL, _SIM_LINK
    if _PANEL is None:
        from panel_routes import PanelRoutes
        from panel_sim_link import SimLink
        # Read-only, and its own connection: server.py gives every client its
        # own state/frame/place_ack queues, so this takes nothing away from
        # the browser panel's socket.
        _SIM_LINK = SimLink()
        _SIM_LINK.start()
        _PANEL = PanelRoutes(_scene_view, link=_SIM_LINK)
    return _PANEL


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
        elif path == "/capabilities" or path.startswith("/tasks"):
            self._serve_panel_get(path)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if not path.startswith("/tasks"):
            self.send_error(404, "Not Found")
            return
        self._serve_panel_post(path)

    # -- command panel (issue #49) ----------------------------------------

    def _session_id(self) -> str:
        """The caller's panel session, or "" if the browser sent none."""
        from panel_routes import session_from_cookie
        return session_from_cookie(self.headers.get("Cookie", ""))

    def _serve_panel_get(self, path: str):
        try:
            panel = _panel()
        except Exception as exc:
            self._send_json(503, {"error": f"command panel unavailable: {exc}",
                                  "code": "panel_unavailable"})
            return
        result = panel.handle_get(path, self._session_id())
        if result is None:
            self.send_error(404, "Not Found")
            return
        self._send_json(*result)

    def _serve_panel_post(self, path: str):
        from panel_routes import origin_allowed, parse_body
        from tasks import TaskError

        try:
            panel = _panel()
        except Exception as exc:
            self._send_json(503, {"error": f"command panel unavailable: {exc}",
                                  "code": "panel_unavailable"})
            return

        # Mutating and cookie-authenticated, so a cross-site POST is refused
        # before the body is even read.
        if not origin_allowed(self.headers.get("Origin", ""),
                              self.headers.get("Host", "")):
            self._send_json(403, {"error": "Cross-origin request refused.",
                                  "code": "bad_origin"})
            return

        session_id = self._session_id()
        if not session_id:
            self._send_json(403, {"error": "No panel session. Reload the page.",
                                  "code": "no_session"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > _MAX_BODY_BYTES:
            self._send_json(413, {"error": "That request is too large.",
                                  "code": "body_too_large"})
            return

        try:
            payload = parse_body(self.rfile.read(length))
        except TaskError as exc:
            self._send_json(exc.status, {"error": exc.message, "code": exc.code})
            return

        result = panel.handle_post(path, session_id, payload)
        if result is None:
            self.send_error(404, "Not Found")
            return
        self._send_json(*result)

    def _send_json(self, status: int, payload: dict, extra_headers=()):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in extra_headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _serve_index(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(_INDEX_HTML)))
        self.send_header("Cache-Control", "no-cache")
        # Issue the panel session here rather than on the first POST: the
        # cookie is HttpOnly so page script cannot mint one, and a page that
        # already holds a session survives a refresh without replaying
        # anything.  SameSite=Strict keeps it off cross-site requests.
        if not self._session_id():
            from panel_routes import SESSION_COOKIE
            from tasks import TaskCoordinator
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}={TaskCoordinator.new_session_id()}; "
                "Path=/; HttpOnly; SameSite=Strict; Max-Age=86400",
            )
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
