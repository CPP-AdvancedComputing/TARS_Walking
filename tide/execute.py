import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

import websocket


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    result: str = ""
    error: Optional[str] = None
    execution_count: Optional[int] = None
    elapsed_seconds: float = 0.0

    @property
    def output(self) -> str:
        parts = [part for part in (self.stdout, self.result) if part]
        return "\n".join(parts)


def execute(
    ws_base: str,
    kernel_id: str,
    token: str,
    code: str,
    timeout: int = 3600,
    on_output=None,
) -> ExecutionResult:
    ws_url = f"{ws_base}/kernels/{quote(kernel_id)}/channels?token={quote(token, safe='')}"
    ws_headers = [f"Authorization: token {token}"]

    result = ExecutionResult()
    start = time.time()
    msg_id = str(uuid.uuid4())

    ws = websocket.create_connection(ws_url, timeout=30, header=ws_headers)
    try:
        ws.send(json.dumps(_execute_request(msg_id, code)))

        while True:
            if time.time() - start > timeout:
                result.error = f"Timed out after {timeout}s"
                break

            ws.settimeout(5)
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue

            msg = json.loads(raw)
            msg_type = msg.get("msg_type") or msg.get("header", {}).get("msg_type", "")
            parent_id = msg.get("parent_header", {}).get("msg_id", "")
            content = msg.get("content", {})

            if parent_id != msg_id and msg_type != "status":
                continue

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    result.stdout += text
                else:
                    result.stderr += text
                if on_output and text:
                    on_output(text)
            elif msg_type in ("execute_result", "display_data"):
                data = content.get("data", {})
                result.result += data.get("text/plain", "")
                result.execution_count = content.get("execution_count")
            elif msg_type == "error":
                traceback = content.get("traceback", [])
                result.error = "\n".join(traceback) if traceback else content.get("evalue", "error")
            elif msg_type == "status":
                if content.get("execution_state") == "idle" and parent_id == msg_id:
                    break
            elif msg_type == "execute_reply":
                if content.get("status") == "error" and not result.error:
                    result.error = content.get("evalue", "execution error")
                result.execution_count = content.get("execution_count")
    finally:
        ws.close()

    result.elapsed_seconds = round(time.time() - start, 2)
    return result


def _execute_request(msg_id: str, code: str) -> dict:
    return {
        "header": {
            "msg_id": msg_id,
            "username": "tars",
            "session": str(uuid.uuid4()),
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "channel": "shell",
    }
