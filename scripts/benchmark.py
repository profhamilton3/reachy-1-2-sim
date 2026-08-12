#!/usr/bin/env python3
"""
R12-604: Reachy 1.2 Simulator Benchmark Harness.

Connects to the native MuJoCo server via WebSocket and drives it for a
configurable duration, measuring:

  - real-time factor (RTF): sim_time_s / wall_time_s
  - state message rate (Hz)
  - camera frame rate (Hz)
  - state-message age (wall_time - sim_time, ms)
  - command-to-state round-trip latency (ms)
  - CPU and memory (requires psutil)
  - dropped state messages (seq gaps > 1)

Output: prints a summary and writes artifacts/benchmark_report_<timestamp>.json.

Usage:
  python3 scripts/benchmark.py [--url ws://127.0.0.1:8765] [--duration 30]
                               [--report-dir artifacts]

Requires a running native server:
  cd native_mujoco && mjpython server.py
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

# Add native_mujoco to path so we can import protocol
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "native_mujoco"))

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from protocol import PROTOCOL_VERSION


_DEFAULT_URL      = "ws://127.0.0.1:8765"
_DEFAULT_DURATION = 30.0
_DEFAULT_REPORT   = "artifacts"


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class BenchmarkMetrics:
    def __init__(self) -> None:
        self.start_wall = time.monotonic()
        self.state_count = 0
        self.frame_count = 0
        self.dropped_states = 0
        self.prev_seq: Optional[int] = None
        self.prev_sim_time = 0.0
        self.last_sim_time = 0.0
        self.state_ages_ms: List[float] = []
        self.frame_ages_ms: List[float] = []
        self.rtt_samples_ms: List[float] = []
        self._pending_cmd_ts: Optional[float] = None

    def on_state(self, msg: dict) -> None:
        now = time.monotonic()
        seq = msg.get("seq", 0)
        sim_time = float(msg.get("sim_time_s", 0.0))
        wall_time_ns = int(msg.get("wall_time_ns", 0))

        if self.prev_seq is not None and seq > self.prev_seq + 1:
            self.dropped_states += seq - self.prev_seq - 1
        self.prev_seq = seq
        self.last_sim_time = sim_time
        self.state_count += 1

        # State age: gap between sim time and wall time at receipt
        if wall_time_ns > 0:
            age_ms = (time.monotonic_ns() - wall_time_ns) / 1e6
            if -100 < age_ms < 5000:   # sanity bounds (monotonic clocks differ)
                self.state_ages_ms.append(age_ms)

        # RTT: if we sent a command, measure time-to-next-state
        if self._pending_cmd_ts is not None:
            rtt = (now - self._pending_cmd_ts) * 1000.0
            self.rtt_samples_ms.append(rtt)
            self._pending_cmd_ts = None

    def on_frame(self, msg: dict) -> None:
        sim_time = float(msg.get("sim_time_s", 0.0))
        self.frame_count += 1
        age_ms = (time.monotonic() - (sim_time + 0.0)) * 1000.0
        # approximate: we don't have exact wall_time from frame msg in this pass
        self.frame_ages_ms.append(age_ms)

    def mark_command_sent(self) -> None:
        self._pending_cmd_ts = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start_wall

    def rtf(self) -> float:
        wall = self.elapsed()
        if wall <= 0:
            return 0.0
        return self.last_sim_time / wall

    def state_hz(self) -> float:
        return self.state_count / max(self.elapsed(), 1e-9)

    def frame_hz(self) -> float:
        return self.frame_count / max(self.elapsed(), 1e-9)

    @staticmethod
    def _percentiles(data: List[float]) -> Dict[str, float]:
        if not data:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        s = sorted(data)
        n = len(s)
        def pct(p):
            idx = int(math.ceil(p * n / 100)) - 1
            return s[max(0, min(idx, n - 1))]
        return {
            "mean": sum(s) / n,
            "p50": pct(50),
            "p95": pct(95),
            "p99": pct(99),
            "max": s[-1],
        }

    def report(self) -> Dict[str, Any]:
        proc_info: Dict[str, Any] = {}
        if _HAS_PSUTIL:
            proc = psutil.Process()
            proc_info = {
                "cpu_percent": proc.cpu_percent(interval=None),
                "memory_rss_mb": proc.memory_info().rss / (1024 * 1024),
            }
        return {
            "wall_duration_s": round(self.elapsed(), 3),
            "sim_time_s": round(self.last_sim_time, 3),
            "rtf": round(self.rtf(), 4),
            "state_hz": round(self.state_hz(), 2),
            "frame_hz": round(self.frame_hz(), 2),
            "dropped_states": self.dropped_states,
            "state_count": self.state_count,
            "frame_count": self.frame_count,
            "state_age_ms": self._percentiles(self.state_ages_ms),
            "frame_age_ms": self._percentiles(self.frame_ages_ms),
            "cmd_rtt_ms": self._percentiles(self.rtt_samples_ms),
            "process": proc_info,
        }


# ---------------------------------------------------------------------------
# Async benchmark session
# ---------------------------------------------------------------------------

async def run_benchmark(url: str, duration: float) -> Dict[str, Any]:
    metrics = BenchmarkMetrics()
    if _HAS_PSUTIL:
        psutil.Process().cpu_percent(interval=None)  # prime first call

    print(f"Connecting to {url} …")
    async with websockets.connect(url, ping_interval=None, close_timeout=5) as ws:
        # Handshake
        hello = {"type": "hello", "protocol_version": PROTOCOL_VERSION,
                 "client_id": "benchmark"}
        await ws.send(json.dumps(hello))
        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        ack = json.loads(raw)
        if ack.get("type") != "hello_ack":
            raise RuntimeError(f"Handshake failed: {ack}")
        print(f"Connected — sim_fps={ack.get('sim_fps'):.0f}  "
              f"camera_fps={ack.get('camera_fps'):.0f}")

        # Drive for `duration` seconds: send a heartbeat + occasional zero command
        t_end = time.monotonic() + duration
        cmd_seq = 0
        cmd_every = 2.0   # send a joint command every 2 s to measure RTT
        last_cmd = time.monotonic()
        last_hb  = time.monotonic()
        print(f"Benchmarking for {duration:.0f} s …")

        while time.monotonic() < t_end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                raw = None

            if raw is not None:
                try:
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    if mtype == "state":
                        metrics.on_state(msg)
                    elif mtype == "camera_frame":
                        metrics.on_frame(msg)
                    elif mtype == "heartbeat":
                        await ws.send(json.dumps({
                            "type": "heartbeat_ack",
                            "echo_ns": msg.get("sent_ns", 0),
                        }))
                except Exception:
                    pass

            now = time.monotonic()
            # Periodic heartbeat from our side
            if now - last_hb > 2.0:
                await ws.send(json.dumps({"type": "heartbeat",
                                           "sent_ns": time.monotonic_ns()}))
                last_hb = now

            # Periodic zero command for RTT measurement
            if now - last_cmd > cmd_every:
                cmd_seq += 1
                zero_cmd = {
                    "type": "joint_command",
                    "seq": cmd_seq,
                    "target_rad": [0.0] * 21,
                }
                await ws.send(json.dumps(zero_cmd))
                metrics.mark_command_sent()
                last_cmd = now

        await ws.send(json.dumps({"type": "disconnect", "reason": "benchmark_done"}))

    return metrics.report()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Reachy 1.2 simulator benchmark")
    ap.add_argument("--url", default=_DEFAULT_URL,
                    help=f"WebSocket URL (default: {_DEFAULT_URL})")
    ap.add_argument("--duration", type=float, default=_DEFAULT_DURATION,
                    help=f"Benchmark duration in seconds (default: {_DEFAULT_DURATION})")
    ap.add_argument("--report-dir", default=_DEFAULT_REPORT,
                    help=f"Output directory for the report (default: {_DEFAULT_REPORT})")
    args = ap.parse_args()

    try:
        report = asyncio.run(run_benchmark(args.url, args.duration))
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("Is the native MuJoCo server running?")
        print("  cd native_mujoco && mjpython server.py")
        sys.exit(1)

    # Add metadata
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report["benchmark_timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    report["server_url"] = args.url
    report["psutil_available"] = _HAS_PSUTIL

    # Print summary
    print("\n── Benchmark Report ─────────────────────────────────────")
    print(f"  Wall duration:    {report['wall_duration_s']:.1f} s")
    print(f"  Sim time:         {report['sim_time_s']:.1f} s")
    print(f"  RTF:              {report['rtf']:.3f}x")
    print(f"  State rate:       {report['state_hz']:.1f} Hz")
    print(f"  Camera rate:      {report['frame_hz']:.1f} Hz")
    print(f"  Dropped states:   {report['dropped_states']}")
    age = report['state_age_ms']
    print(f"  State age:        mean={age['mean']:.1f}ms  p95={age['p95']:.1f}ms")
    rtt = report['cmd_rtt_ms']
    print(f"  Cmd RTT:          mean={rtt['mean']:.1f}ms  p95={rtt['p95']:.1f}ms")
    if _HAS_PSUTIL and report.get("process"):
        p = report["process"]
        print(f"  CPU:              {p.get('cpu_percent', 0):.1f}%")
        print(f"  Memory (RSS):     {p.get('memory_rss_mb', 0):.1f} MB")

    # Write JSON report
    out_dir = pathlib.Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"benchmark_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
