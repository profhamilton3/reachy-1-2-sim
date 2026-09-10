# Solution for Issue #79

## 🛠️ Proposed Solution (by Aditya Waghamare)

### Analysis
The `grpc.aio` poller completion queue (`PollerCompletionQueue._handle_events`) throws a `BlockingIOError: [Errno 11] Resource temporarily unavailable` inside the web server process when `reachy_sdk` is used across multiple sequential asynchronous/multithreaded operations. This causes the underlying gRPC event loop to die, hanging any active motion future indefinitely and holding the executor's motion lock, leaving subsequent tasks permanently blocked ("another task is already using the arm"). 

Running gRPC aio clients inside a legacy `ThreadingHTTPServer` / mixed synchronous-asynchronous web process causes thread/event-loop collisions and resource exhaustion on the completion queue selector.

### Fix
Isolate all SDK motion commands and gRPC calls into a short-lived subprocess (or dedicated separate worker process) per invocation. This ensures that any `grpc.aio` event loop, completion queue thread, and selector state are fully cleaned up upon exit, preventing resource exhaustion, deadlocks, and hung locks on the shared web server process.

### Implementation
```python
import subprocess
import sys
import json

def execute_motion_command_isolated(command_name: str, host: str = "localhost", port: int = 50051):
    """
    Executes a reachy_sdk motion command in an isolated subprocess to prevent
    grpc.aio PollerCompletionQueue BlockingIOError leaks from wedging the main web server.
    """
    script = f"""
import asyncio
from reachy_sdk import ReachySDK

async def main():
    reachy = ReachySDK(host="{host}", port={port})
    try:
        # Execute motion command
        if "{command_name}" == "stow":
            await reachy.goto_stow()
        elif "{command_name}" == "wave":
            await reachy.goto_wave()
        print(json.dumps({{"status": "success"}}))
    except Exception as e:
        print(json.dumps({{"status": "error", "message": str(e)}}))
    finally:
        await reachy.close()

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        return {"status": "error", "message": result.stderr.strip()}
    
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse subprocess output: {e}"}
```

### Testing
1. Repeatedly invoke alternating commands (`stow` -> `wave` -> `stow` -> `wave`) through the panel interface.
2. Confirm that each command runs in its own process, avoiding accumulation of active file descriptors or `BlockingIOError: [Errno 11]` on the `grpc.aio` completion queue.
3. Verify that the web server process remains healthy, responsive, and able to acquire locks indefinitely.

Signed-off-by: Aditya Waghamare <adityawaghamare7620@gmail.com>


---
*Submitted by Aditya Waghamare*
💰 **Payout Address (Base L2 / EVM):** `0xb61dBcdBc3407F71EaCb64D4CBFAcf9FFfe2415C`