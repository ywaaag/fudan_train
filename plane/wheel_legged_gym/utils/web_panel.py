"""Small dependency-free HTTP control panel for the Isaac Gym play loop."""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse


CommandGetter = Callable[[], Dict[str, float]]
CommandSetter = Callable[[Dict[str, Any]], Dict[str, float]]
LimitsSetter = Callable[[Dict[str, List[float]]], Dict[str, List[float]]]
RunningGetter = Callable[[], bool]


class TelemetryBuffer:
    """Thread-safe bounded telemetry history used by the browser charts."""

    def __init__(self, maxlen: int = 2400) -> None:
        self._items = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, sample: Dict[str, Any]) -> None:
        clean = {}
        for key, value in sample.items():
            if isinstance(value, bool):
                clean[key] = value
            else:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    clean[key] = number
        with self._lock:
            self._items.append(clean)

    def snapshot(self, limit: int = 600) -> List[Dict[str, Any]]:
        with self._lock:
            if limit <= 0:
                return []
            return list(self._items)[-limit:]


class _PanelHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class WebPanelServer:
    """Serve a local HTML panel and bridge it to the play loop callbacks."""

    def __init__(
        self,
        command_getter: CommandGetter,
        command_setter: CommandSetter,
        telemetry: TelemetryBuffer,
        limits: Dict[str, Iterable[float]],
        running_getter: Optional[RunningGetter] = None,
        limits_setter: Optional[LimitsSetter] = None,
        metadata: Optional[Dict[str, Any]] = None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.command_getter = command_getter
        self.command_setter = command_setter
        self.telemetry = telemetry
        self.limits = {
            key: [float(values[0]), float(values[1])] for key, values in limits.items()
        }
        self._limits_lock = threading.Lock()
        self.running_getter = running_getter or (lambda: True)
        self.limits_setter = limits_setter
        self.metadata = metadata or {}
        self.host = host
        self.port = int(port)
        self._server: Optional[_PanelHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._html_path = Path(__file__).with_name("panel.html")

    @property
    def url(self) -> str:
        bound_port = self.port
        if self._server is not None:
            bound_port = int(self._server.server_address[1])
        return "http://{}:{}/".format(self.host, bound_port)

    def start(self) -> str:
        if self._server is not None:
            return self.url
        if not self._html_path.is_file():
            raise FileNotFoundError("panel.html is missing next to web_panel.py")

        handler = self._make_handler()
        self._server = _PanelHTTPServer((self.host, self.port), handler)
        self._server.panel = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="wheel-leg-web-panel",
            daemon=True,
        )
        self._thread.start()
        return self.url

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def update_limits(self, limits: Dict[str, Iterable[float]]) -> Dict[str, List[float]]:
        with self._limits_lock:
            for key, values in limits.items():
                self.limits[key] = [float(values[0]), float(values[1])]
            return {key: list(values) for key, values in self.limits.items()}

    def _get_limits(self) -> Dict[str, List[float]]:
        with self._limits_lock:
            return {key: list(values) for key, values in self.limits.items()}

    def _make_handler(self):
        panel = self

        class PanelHandler(BaseHTTPRequestHandler):
            server_version = "WheelLegPanel/1.0"

            def log_message(self, format: str, *args: Any) -> None:
                # Isaac Gym already owns the terminal; avoid one line per poll.
                return

            def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                self._send_bytes(body, "application/json; charset=utf-8", status)

            def _state(self) -> Dict[str, Any]:
                return {
                    "ok": True,
                    "running": bool(panel.running_getter()),
                    "command": panel.command_getter(),
                    "limits": panel._get_limits(),
                    "metadata": panel.metadata,
                    "telemetry": panel.telemetry.snapshot(),
                }

            def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler API
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
                route = urlparse(self.path).path
                if route in ("", "/"):
                    try:
                        body = panel._html_path.read_bytes()
                    except OSError as exc:
                        self._send_json({"ok": False, "error": str(exc)}, 500)
                        return
                    self._send_bytes(body, "text/html; charset=utf-8")
                elif route in ("/api/state", "/api/health"):
                    self._send_json(self._state())
                else:
                    self._send_json({"ok": False, "error": "not found"}, 404)

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                route = urlparse(self.path).path
                if route not in ("/api/command", "/api/limits"):
                    self._send_json({"ok": False, "error": "not found"}, 404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 16 * 1024:
                        raise ValueError("invalid request size")
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("request must be a JSON object")
                    if route == "/api/command":
                        command = panel._normalise_command(payload)
                        result = panel.command_setter(command)
                        response = {"ok": True, "command": result}
                    elif panel.limits_setter is None:
                        raise ValueError("runtime limit editing is disabled")
                    else:
                        limits = panel._normalise_limits(payload)
                        result = panel.limits_setter(limits)
                        response = {"ok": True, "limits": panel.update_limits(result)}
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._send_json({"ok": False, "error": str(exc)}, 400)
                    return
                self._send_json(response)

            @property
            def panel(self) -> "WebPanelServer":
                return panel

        return PanelHandler

    def _normalise_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        command: Dict[str, Any] = {"source": str(payload.get("source", "panel"))[:32]}
        limits = self._get_limits()
        for key in ("cmd_x", "ang_vel", "cmd_height"):
            if key not in payload:
                continue
            value = float(payload[key])
            if not math.isfinite(value):
                raise ValueError("{} must be finite".format(key))
            lo, hi = limits[key]
            command[key] = max(lo, min(hi, value))
        if not any(key in command for key in ("cmd_x", "ang_vel", "cmd_height")):
            raise ValueError("no supported command fields")
        return command

    def _normalise_limits(self, payload: Dict[str, Any]) -> Dict[str, List[float]]:
        raw_limits = payload.get("limits")
        if not isinstance(raw_limits, dict):
            raise ValueError("limits must be an object")
        limits = self._get_limits()
        for key in ("cmd_x", "ang_vel", "cmd_height"):
            if key not in raw_limits:
                continue
            values = raw_limits[key]
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError("{} must contain [min, max]".format(key))
            lo, hi = float(values[0]), float(values[1])
            if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                raise ValueError("invalid range for {}".format(key))
            limits[key] = [lo, hi]
        return limits
