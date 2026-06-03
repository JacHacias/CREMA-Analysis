from __future__ import annotations

import json
import socket
import threading
import time
import urllib.parse
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


CHANNELS = {"CH1": 1, "CH2": 2, "CH3": 3}
MODE_NAMES = {"+0": "OFF", "0": "OFF", "+1": "CC", "1": "CC", "+2": "CV", "2": "CV", "+3": "UR", "3": "UR"}


class InstrumentError(RuntimeError):
    pass


class Transport:
    label = "Disconnected"

    def write(self, command: str) -> None:
        raise NotImplementedError

    def query(self, command: str) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SocketTransport(Transport):
    def __init__(self, host: str, port: int = 5555, timeout: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(timeout)
        self.label = f"TCP {host}:{port}"

    def write(self, command: str) -> None:
        self._sock.sendall((command.strip() + "\n").encode("ascii"))

    def query(self, command: str) -> str:
        self.write(command)
        chunks: list[bytes] = []
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                part = self._sock.recv(4096)
            except socket.timeout:
                break
            if not part:
                break
            chunks.append(part)
            if b"\n" in part:
                break
        if not chunks:
            raise InstrumentError(f"No response to {command}")
        return b"".join(chunks).decode("ascii", errors="replace").strip()

    def close(self) -> None:
        self._sock.close()


class VisaTransport(Transport):
    def __init__(self, resource: str, timeout: float = 2.0) -> None:
        try:
            import pyvisa  # type: ignore
        except ImportError as exc:
            raise InstrumentError("PyVISA is not installed. Use TCP mode or install pyvisa/NI-VISA.") from exc
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(resource)
        self._inst.timeout = int(timeout * 1000)
        self._inst.write_termination = "\n"
        self._inst.read_termination = "\n"
        self.label = f"VISA {resource}"

    def write(self, command: str) -> None:
        self._inst.write(command.strip())

    def query(self, command: str) -> str:
        return str(self._inst.query(command.strip())).strip()

    def close(self) -> None:
        self._inst.close()
        self._rm.close()


class SimulatedTransport(Transport):
    def __init__(self) -> None:
        self.label = "Simulator"
        self.channels = {
            ch: {
                "voltage": 0.0,
                "current": 0.1,
                "output": False,
                "ovp": 35.2 if ch != "CH3" else 6.6,
                "ocp": 3.3,
                "ovp_state": True,
                "ocp_state": True,
                "ovp_alarm": False,
                "ocp_alarm": False,
            }
            for ch in CHANNELS
        }

    def write(self, command: str) -> None:
        cmd = command.strip().upper()
        parts = command.replace(",", " ").split()
        channel = _channel_from_command(command)
        if cmd.startswith(":OUTP:OVP:CLE"):
            self.channels[channel]["ovp_alarm"] = False
            return
        if cmd.startswith(":OUTP:OCP:CLE"):
            self.channels[channel]["ocp_alarm"] = False
            return
        if cmd.startswith(":OUTP:OVP:VAL"):
            self.channels[channel]["ovp"] = float(parts[-1])
            return
        if cmd.startswith(":OUTP:OCP:VAL"):
            self.channels[channel]["ocp"] = float(parts[-1])
            return
        if cmd.startswith(":OUTP:OVP"):
            self.channels[channel]["ovp_state"] = _is_on(parts[-1])
            return
        if cmd.startswith(":OUTP:OCP"):
            self.channels[channel]["ocp_state"] = _is_on(parts[-1])
            return
        if cmd.startswith(":OUTP"):
            self.channels[channel]["output"] = _is_on(parts[-1])
            return
        if ":VOLT" in cmd:
            self.channels[channel]["voltage"] = float(parts[-1])
            return
        if ":CURR" in cmd:
            self.channels[channel]["current"] = float(parts[-1])

    def query(self, command: str) -> str:
        cmd = command.strip().upper()
        channel = _channel_from_command(command)
        data = self.channels[channel]
        if cmd == "*IDN?":
            return "RIGOL TECHNOLOGIES,DP932A,SIMULATED,00.00"
        if cmd.startswith(":MEAS") and ":ALL?" in cmd:
            volts = float(data["voltage"]) if data["output"] else 0.0
            amps = min(float(data["current"]) * 0.42, float(data["current"])) if data["output"] else 0.0
            return f"{volts:.4f},{amps:.4f},{volts * amps:.3f}"
        if ":VOLT" in cmd and cmd.endswith("?"):
            return f"{float(data['voltage']):.3f}"
        if ":CURR" in cmd and cmd.endswith("?"):
            return f"{float(data['current']):.3f}"
        if cmd.startswith(":OUTP:MODE?") or cmd.startswith(":OUTP:CVCC?"):
            return "CV" if data["output"] else "OFF"
        if cmd.startswith(":STAT:QUES:INST:ISUM"):
            return "+2" if data["output"] else "+0"
        if cmd.startswith(":OUTP:OVP:ALAR?") or cmd.startswith(":OUTP:OVP:QUES?"):
            return "1" if data["ovp_alarm"] else "0"
        if cmd.startswith(":OUTP:OCP:ALAR?") or cmd.startswith(":OUTP:OCP:QUES?"):
            return "1" if data["ocp_alarm"] else "0"
        if cmd.startswith(":OUTP:OVP:VAL?"):
            return f"{float(data['ovp']):.3f}"
        if cmd.startswith(":OUTP:OCP:VAL?"):
            return f"{float(data['ocp']):.3f}"
        if cmd.startswith(":OUTP:OVP?"):
            return "1" if data["ovp_state"] else "0"
        if cmd.startswith(":OUTP:OCP?"):
            return "1" if data["ocp_state"] else "0"
        if cmd.startswith(":OUTP?"):
            return "1" if data["output"] else "0"
        return "0"


class DP900Controller:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.transport: Transport | None = None
        self._ramp_stop: dict[str, threading.Event] = {}
        self._ramp_threads: dict[str, threading.Thread] = {}
        self._ramping: dict[str, bool] = {ch: False for ch in CHANNELS}
        self._ramp_target: dict[str, float | None] = {ch: None for ch in CHANNELS}
        self._schedule_lock = threading.Lock()
        self._scheduled: dict[str, dict[str, Any] | None] = {ch: None for ch in CHANNELS}
        self._scheduler_stop = threading.Event()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, name="dp900-scheduler", daemon=True)
        self._scheduler_thread.start()

    @property
    def connected(self) -> bool:
        return self.transport is not None

    def connect(self, config: dict[str, Any]) -> dict[str, Any]:
        mode = str(config.get("mode", "sim")).lower()
        timeout = float(config.get("timeout", 2.0))
        with self._lock:
            self.disconnect()
            if mode == "tcp":
                host = str(config.get("host", "")).strip()
                if not host:
                    raise InstrumentError("TCP mode needs an IP address or host name.")
                port = int(config.get("port", 5555))
                self.transport = SocketTransport(host, port, timeout)
            elif mode == "visa":
                resource = str(config.get("resource", "")).strip()
                if not resource:
                    raise InstrumentError("VISA mode needs a resource string.")
                self.transport = VisaTransport(resource, timeout)
            elif mode == "sim":
                self.transport = SimulatedTransport()
            else:
                raise InstrumentError(f"Unknown connection mode: {mode}")
            idn = self.query("*IDN?")
            return {"connected": True, "label": self.transport.label, "idn": idn}

    def disconnect(self) -> None:
        for channel in CHANNELS:
            self._cancel_ramp(channel)
            self.cancel_schedule(channel)
        if self.transport:
            self.transport.close()
            self.transport = None

    def write(self, command: str) -> None:
        if not self.transport:
            raise InstrumentError("Not connected")
        self.transport.write(command)

    def query(self, command: str) -> str:
        if not self.transport:
            raise InstrumentError("Not connected")
        return self.transport.query(command)

    def apply_channel(
        self,
        channel: str,
        voltage: float,
        current: float,
        ovp: float,
        ocp: float,
        output: bool,
        ramp_rate: float = 0.2,
        ramp_seconds: float | None = None,
        ramp_axis: str = "voltage",
    ) -> None:
        ch = _normalize_channel(channel)
        n = CHANNELS[ch]
        _validate_limits(ch, voltage, current, ovp, ocp)
        if ramp_rate < 0:
            raise InstrumentError("Ramp rate must be 0 or greater.")
        axis = _normalize_ramp_axis(ramp_axis)
        self._cancel_ramp(ch)
        with self._lock:
            output_is_on = self._output_state(ch)
            start_value = _float_query(self.query(f":SOUR{n}:{'CURR' if axis == 'current' else 'VOLT'}?"))
            target_value = current if axis == "current" else voltage
            source = "CURR" if axis == "current" else "VOLT"
            effective_ramp_rate = _effective_ramp_rate(start_value, target_value, ramp_rate, ramp_seconds)
            if axis == "current":
                self.write(f":SOUR{n}:VOLT {voltage:.6g}")
            else:
                self.write(f":SOUR{n}:CURR {current:.6g}")
            self.write(f":OUTP:OVP:VAL {ch},{ovp:.6g}")
            self.write(f":OUTP:OCP:VAL {ch},{ocp:.6g}")
            self.write(f":OUTP:OVP {ch},ON")
            self.write(f":OUTP:OCP {ch},ON")
            if not output:
                self.write(f":SOUR{n}:{source} {target_value:.6g}")
                self.write(f":OUTP {ch},OFF")
                return
            if not output_is_on:
                start_value = 0.0
                self.write(f":SOUR{n}:{source} 0")
                self.write(f":OUTP {ch},ON")
                effective_ramp_rate = _effective_ramp_rate(start_value, target_value, ramp_rate, ramp_seconds)
        self._start_ramp(ch, n, start_value, target_value, effective_ramp_rate, source)

    def set_output(
        self,
        channel: str,
        output: bool,
        ramp_rate: float = 0.2,
        ramp_seconds: float | None = None,
        ramp_axis: str = "voltage",
    ) -> None:
        ch = _normalize_channel(channel)
        n = CHANNELS[ch]
        axis = _normalize_ramp_axis(ramp_axis)
        source = "CURR" if axis == "current" else "VOLT"
        self._cancel_ramp(ch)
        with self._lock:
            if output:
                target_value = _float_query(self.query(f":SOUR{n}:{source}?"))
                is_already_on = self._output_state(ch)
                if not is_already_on and target_value > 0 and ramp_rate > 0:
                    effective_ramp_rate = _effective_ramp_rate(0.0, target_value, ramp_rate, ramp_seconds)
                    self.write(f":SOUR{n}:{source} 0")
                    self.write(f":OUTP {ch},ON")
                    self._start_ramp(ch, n, 0.0, target_value, effective_ramp_rate, source)
                    return
            self.write(f":OUTP {ch},{'ON' if output else 'OFF'}")

    def clear_protection(self, channel: str) -> None:
        ch = _normalize_channel(channel)
        with self._lock:
            if isinstance(self.transport, VisaTransport):
                self.write("*CLS")
                return
            self.write(f":OUTP:OVP:CLE {ch}")
            self.write(f":OUTP:OCP:CLE {ch}")

    def schedule_ready_for_tomorrow(
        self,
        channel: str,
        ready_time_text: str,
        voltage: float,
        current: float,
        ovp: float,
        ocp: float,
        output: bool,
        ramp_rate: float,
        ramp_seconds: float | None = None,
        ramp_axis: str = "voltage",
    ) -> dict[str, Any]:
        ch = _normalize_channel(channel)
        _validate_limits(ch, voltage, current, ovp, ocp)
        axis = _normalize_ramp_axis(ramp_axis)
        ready_time = _parse_ready_time(ready_time_text)

        with self._lock:
            source = "CURR" if axis == "current" else "VOLT"
            current_value = _float_query(self.query(f":SOUR{CHANNELS[ch]}:{source}?")) if self.connected else 0.0

        target_value = current if axis == "current" else voltage
        effective_ramp_rate = _effective_ramp_rate(current_value, target_value, ramp_rate, ramp_seconds)
        required_seconds = abs(target_value - current_value) / effective_ramp_rate if effective_ramp_rate > 0 else 0
        start_at = ready_time - timedelta(seconds=required_seconds)
        if start_at <= datetime.now():
            raise InstrumentError("Ready time is too soon for the requested ramp. Choose a later time or a faster ramp rate.")

        payload = {
            "channel": ch,
            "readyTime": ready_time,
            "startAt": start_at,
            "voltage": voltage,
            "current": current,
            "ovp": ovp,
            "ocp": ocp,
            "output": output,
            "rampRate": effective_ramp_rate,
            "rampSeconds": ramp_seconds,
            "rampAxis": axis,
        }
        with self._schedule_lock:
            self._scheduled[ch] = payload
        return self._schedule_status(ch)

    def cancel_schedule(self, channel: str) -> dict[str, Any]:
        ch = _normalize_channel(channel)
        with self._schedule_lock:
            self._scheduled[ch] = None
        return self._schedule_status(ch)

    def _schedule_status(self, channel: str) -> dict[str, Any]:
        ch = _normalize_channel(channel)
        with self._schedule_lock:
            item = self._scheduled[ch]
            if not item:
                return {"scheduled": False, "readyTime": None, "startAt": None}
            return {
                "scheduled": True,
                "readyTime": item["readyTime"].isoformat(timespec="seconds"),
                "startAt": item["startAt"].isoformat(timespec="seconds"),
            }

    def status(self, channel: str) -> dict[str, Any]:
        ch = _normalize_channel(channel)
        n = CHANNELS[ch]
        with self._lock:
            measured = self._measure_all(ch)
            output = self._output_state(ch)
            set_voltage = _float_query(self.query(f":SOUR{n}:VOLT?"))
            set_current = _float_query(self.query(f":SOUR{n}:CURR?"))
            mode = self._mode(ch, n, output, measured[1], set_current)
            return {
                "channel": ch,
                "connected": self.connected,
                "label": self.transport.label if self.transport else "Disconnected",
                "measuredVoltage": measured[0],
                "measuredCurrent": measured[1],
                "measuredPower": measured[2],
                "setVoltage": set_voltage,
                "setCurrent": set_current,
                "output": output,
                "mode": mode,
                "ovpEnabled": _bool_query(self.query(f":OUTP:OVP? {ch}")),
                "ocpEnabled": _bool_query(self.query(f":OUTP:OCP? {ch}")),
                "ovp": _float_query(self.query(f":OUTP:OVP:VAL? {ch}")),
                "ocp": _float_query(self.query(f":OUTP:OCP:VAL? {ch}")),
                "ovpAlarm": _bool_query(self.query(f":OUTP:OVP:ALAR? {ch}")),
                "ocpAlarm": _bool_query(self.query(f":OUTP:OCP:ALAR? {ch}")),
                "ramping": self._ramping[ch],
                "rampTarget": self._ramp_target[ch],
                **self._schedule_status(ch),
            }

    def _measure_all(self, channel: str) -> list[float]:
        if isinstance(self.transport, VisaTransport):
            volts = _float_query(self.query(f":MEAS:VOLT? {channel}"))
            amps = _float_query(self.query(f":MEAS:CURR? {channel}"))
            return [volts, amps, volts * amps]
        return _parse_csv_floats(self.query(f":MEAS:ALL? {channel}"), 3)

    def _output_state(self, channel: str) -> bool:
        if isinstance(self.transport, VisaTransport):
            return _bool_query(self.query(f":OUTP:STAT? {channel}"))
        return _bool_query(self.query(f":OUTP? {channel}"))

    def _mode(self, channel: str, channel_number: int, output: bool, measured_current: float, set_current: float) -> str:
        if not output:
            return "OFF"
        try:
            mode = self.query(f":OUTP:MODE? {channel}").upper()
            if mode in {"CV", "CC", "UR"}:
                return mode
        except Exception:
            mode = ""
        if not isinstance(self.transport, VisaTransport):
            try:
                return MODE_NAMES.get(self.query(f":STAT:QUES:INST:ISUM{channel_number}:COND?").strip(), mode or "CV")
            except Exception:
                pass
        if set_current > 0 and measured_current >= 0.98 * set_current:
            return "CC"
        return "CV"

    def status_all(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "label": self.transport.label if self.transport else "Disconnected",
            "channels": [self.status(ch) for ch in CHANNELS],
        }

    def _cancel_ramp(self, channel: str) -> None:
        event = self._ramp_stop.get(channel)
        if event:
            event.set()
        self._ramping[channel] = False
        self._ramp_target[channel] = None

    def _start_ramp(self, channel: str, channel_number: int, start_value: float, target_value: float, ramp_rate: float, source: str = "VOLT") -> None:
        if abs(target_value - start_value) < 1e-9:
            self._ramping[channel] = False
            self._ramp_target[channel] = None
            return
        if ramp_rate == 0:
            with self._lock:
                self.write(f":SOUR{channel_number}:{source} {target_value:.6g}")
            self._ramping[channel] = False
            self._ramp_target[channel] = None
            return

        stop_event = threading.Event()
        self._ramp_stop[channel] = stop_event
        self._ramping[channel] = True
        self._ramp_target[channel] = target_value

        def worker() -> None:
            interval_s = 0.1
            step_value = max(ramp_rate * interval_s, 0.001)
            current_value = start_value
            direction = 1.0 if target_value > start_value else -1.0
            try:
                while not stop_event.is_set():
                    next_value = current_value + direction * step_value
                    if (direction > 0 and next_value >= target_value) or (direction < 0 and next_value <= target_value):
                        break
                    current_value = next_value
                    with self._lock:
                        if not self.transport:
                            return
                        self.write(f":SOUR{channel_number}:{source} {current_value:.6g}")
                    time.sleep(interval_s)
                if not stop_event.is_set():
                    with self._lock:
                        if not self.transport:
                            return
                        self.write(f":SOUR{channel_number}:{source} {target_value:.6g}")
            finally:
                if self._ramp_stop.get(channel) is stop_event:
                    self._ramping[channel] = False
                    self._ramp_target[channel] = None

        thread = threading.Thread(target=worker, name=f"{channel}-ramp", daemon=True)
        self._ramp_threads[channel] = thread
        thread.start()

    def _scheduler_loop(self) -> None:
        while not self._scheduler_stop.wait(0.5):
            now = datetime.now()
            due: list[dict[str, Any]] = []
            with self._schedule_lock:
                for ch, item in self._scheduled.items():
                    if item and item["startAt"] <= now:
                        due.append(item)
                        self._scheduled[ch] = None
            for item in due:
                try:
                    self.apply_channel(
                        item["channel"],
                        float(item["voltage"]),
                        float(item["current"]),
                        float(item["ovp"]),
                        float(item["ocp"]),
                        bool(item["output"]),
                        float(item["rampRate"]),
                        item.get("rampSeconds"),
                        item.get("rampAxis", "voltage"),
                    )
                except Exception:
                    pass


CONTROLLERS = {profile: DP900Controller() for profile in ("hv", "cec", "custom")}


def _profile_name(value: Any) -> str:
    profile = str(value or "hv").strip().lower()
    return profile if profile in CONTROLLERS else "hv"


def _controller_for(value: Any) -> DP900Controller:
    return CONTROLLERS[_profile_name(value)]


def _normalize_channel(channel: str) -> str:
    ch = str(channel).upper()
    if ch not in CHANNELS:
        raise InstrumentError("Channel must be CH1, CH2, or CH3.")
    return ch


def _normalize_ramp_axis(axis: str) -> str:
    normalized = str(axis or "voltage").strip().lower()
    if normalized not in {"voltage", "current"}:
        raise InstrumentError("Ramp axis must be voltage or current.")
    return normalized


def _channel_from_command(command: str) -> str:
    upper = command.upper()
    for ch, n in CHANNELS.items():
        if ch in upper or f"SOUR{n}" in upper or f"ISUM{n}" in upper:
            return ch
    return "CH1"


def _is_on(value: str) -> bool:
    return str(value).strip().upper() in {"1", "ON", "TRUE"}


def _bool_query(value: str) -> bool:
    return _is_on(value.split(",", 1)[0])


def _float_query(value: str) -> float:
    return float(value.strip().split(",", 1)[0])


def _parse_csv_floats(value: str, expected: int) -> list[float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) < expected:
        raise InstrumentError(f"Expected {expected} comma-separated values, got {value!r}")
    return parts[:expected]


def _validate_limits(channel: str, voltage: float, current: float, ovp: float, ocp: float) -> None:
    max_voltage = 6.6 if channel == "CH3" else 35.2
    if not 0 <= voltage <= max_voltage:
        raise InstrumentError(f"{channel} voltage must be between 0 and {max_voltage} V.")
    if not 0 <= current <= 3.3:
        raise InstrumentError(f"{channel} current limit must be between 0 and 3.3 A.")
    if not 0.001 <= ovp <= max_voltage:
        raise InstrumentError(f"{channel} OVP must be between 0.001 and {max_voltage} V.")
    if not 0.001 <= ocp <= 3.3:
        raise InstrumentError(f"{channel} OCP must be between 0.001 and 3.3 A.")
    if ovp < voltage:
        raise InstrumentError("OVP must be greater than or equal to the set voltage.")
    if ocp < current:
        raise InstrumentError("OCP must be greater than or equal to the current limit.")


def _effective_ramp_rate(start_voltage: float, target_voltage: float, ramp_rate: float, ramp_seconds: float | None) -> float:
    delta = abs(target_voltage - start_voltage)
    if delta < 1e-12:
        return max(ramp_rate, 0.0)
    if ramp_seconds is not None and ramp_seconds > 0:
        return delta / ramp_seconds
    return ramp_rate


def _parse_ready_time(text: str) -> datetime:
    raw = str(text).strip()
    try:
        parsed = datetime.strptime(raw, "%H:%M")
    except ValueError as exc:
        raise InstrumentError("Ready time must use 24-hour HH:MM format.") from exc
    now = datetime.now()
    today_target = now.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
    if today_target > now:
        return today_target
    tomorrow = now + timedelta(days=1)
    return tomorrow.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"/", "/hv", "/cec"}:
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/status":
            params = urllib.parse.parse_qs(parsed.query)
            channel = params.get("channel", ["CH1"])[0]
            profile = params.get("profile", ["hv"])[0]
            self._json_response(lambda: _controller_for(profile).status(channel))
            return
        if parsed.path == "/api/status_all":
            params = urllib.parse.parse_qs(parsed.query)
            profile = params.get("profile", ["hv"])[0]
            self._json_response(lambda: _controller_for(profile).status_all())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        routes = {
            "/api/connect": self._connect,
            "/api/disconnect": self._disconnect,
            "/api/apply": self._apply,
            "/api/output": self._output,
            "/api/clear": self._clear,
            "/api/schedule": self._schedule,
            "/api/cancel_schedule": self._cancel_schedule,
        }
        action = routes.get(urllib.parse.urlparse(self.path).path)
        if not action:
            self.send_error(404)
            return
        self._json_response(action)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _connect(self) -> dict[str, Any]:
        data = self._read_json()
        return _controller_for(data.get("profile")).connect(data)

    def _disconnect(self) -> dict[str, Any]:
        data = self._read_json()
        controller = _controller_for(data.get("profile"))
        controller.disconnect()
        return {"connected": False}

    def _apply(self) -> dict[str, Any]:
        data = self._read_json()
        controller = _controller_for(data.get("profile"))
        controller.apply_channel(
            str(data.get("channel", "CH1")),
            float(data["voltage"]),
            float(data["current"]),
            float(data["ovp"]),
            float(data["ocp"]),
            bool(data.get("output", False)),
            float(data.get("rampRate", 0.2)),
            float(data["rampSeconds"]) if str(data.get("rampSeconds", "")).strip() else None,
            str(data.get("rampAxis", "voltage")),
        )
        return controller.status(str(data.get("channel", "CH1")))

    def _output(self) -> dict[str, Any]:
        data = self._read_json()
        controller = _controller_for(data.get("profile"))
        channel = str(data.get("channel", "CH1"))
        controller.set_output(
            channel,
            bool(data.get("output", False)),
            float(data.get("rampRate", 0.2)),
            float(data["rampSeconds"]) if str(data.get("rampSeconds", "")).strip() else None,
            str(data.get("rampAxis", "voltage")),
        )
        return controller.status(channel)

    def _clear(self) -> dict[str, Any]:
        data = self._read_json()
        controller = _controller_for(data.get("profile"))
        channel = str(data.get("channel", "CH1"))
        controller.clear_protection(channel)
        return controller.status(channel)

    def _schedule(self) -> dict[str, Any]:
        data = self._read_json()
        controller = _controller_for(data.get("profile"))
        controller.schedule_ready_for_tomorrow(
            str(data.get("channel", "CH1")),
            str(data["readyTime"]),
            float(data["voltage"]),
            float(data["current"]),
            float(data["ovp"]),
            float(data["ocp"]),
            bool(data.get("output", True)),
            float(data.get("rampRate", 0.2)),
            float(data["rampSeconds"]) if str(data.get("rampSeconds", "")).strip() else None,
            str(data.get("rampAxis", "voltage")),
        )
        return controller.status(str(data.get("channel", "CH1")))

    def _cancel_schedule(self) -> dict[str, Any]:
        data = self._read_json()
        controller = _controller_for(data.get("profile"))
        channel = str(data.get("channel", "CH1"))
        controller.cancel_schedule(channel)
        return controller.status(channel)

    def _json_response(self, callback: Any) -> None:
        try:
            payload = {"ok": True, "data": callback()}
            status = 200
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
            status = 400
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rigol DP900 Control</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef1f4;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #607080;
      --line: #cbd5df;
      --soft: #f7f9fb;
      --accent: #0f766e;
      --accent-2: #1d4ed8;
      --danger: #b91c1c;
      --ok: #15803d;
      --warn: #a16207;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    main {
      width: 100%;
      margin: 0;
      padding: 12px;
      display: grid;
      gap: 12px;
    }
    header, section, .channel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    h1, h2, h3 { margin: 0; letter-spacing: 0; }
    h1 { font-size: 24px; }
    h2, h3 { font-size: 16px; }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 13px;
    }
    input, select, button {
      min-height: 38px;
      width: 100%;
      border-radius: 6px;
      border: 1px solid var(--line);
      padding: 8px 10px;
      font: inherit;
      background: white;
      color: var(--ink);
    }
    button {
      cursor: pointer;
      border-color: #aab6c2;
      font-weight: 650;
    }
    button.primary { background: var(--accent); border-color: var(--accent); color: white; }
    button.blue { background: var(--accent-2); border-color: var(--accent-2); color: white; }
    button.danger { background: var(--danger); border-color: var(--danger); color: white; }
    .connection-grid {
      display: grid;
      grid-template-columns: 1.1fr 1.1fr 1.2fr .7fr 1.7fr;
      gap: 12px;
      align-items: end;
    }
    .channels {
      display: grid;
      grid-template-columns: repeat(3, minmax(340px, 1fr));
      gap: 12px;
    }
    .channel {
      display: grid;
      gap: 12px;
      padding: 0;
      overflow: hidden;
    }
    .channel-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 14px 14px 0;
    }
    .channel-body {
      display: grid;
      gap: 10px;
      padding: 0 14px 14px;
    }
    .set-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .read-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(104px, 1fr));
      gap: 8px;
    }
    .readout {
      min-height: 78px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      display: grid;
      align-content: space-between;
      background: var(--soft);
      min-width: 0;
    }
    .readout span, .muted {
      color: var(--muted);
      font-size: 13px;
    }
    .readout b {
      font-size: 17px;
      line-height: 1.1;
      letter-spacing: 0;
      white-space: nowrap;
      overflow: visible;
    }
    .row {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .row > button { width: auto; }
    .button-grid {
      display: grid;
      grid-template-columns: 1.05fr 1fr 1fr 1.12fr;
      gap: 8px;
    }
    .schedule-grid {
      display: grid;
      grid-template-columns: 1.15fr 1fr 1fr;
      gap: 8px;
      align-items: end;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 74px;
      min-height: 30px;
      border-radius: 999px;
      padding: 5px 9px;
      background: #e2e8f0;
      color: var(--muted);
      font-weight: 750;
      font-size: 12px;
      white-space: nowrap;
    }
    .pill.on { background: #dcfce7; color: var(--ok); }
    .pill.cv { background: #dbeafe; color: var(--accent-2); }
    .pill.cc, .pill.ur { background: #fef3c7; color: var(--warn); }
    .pill.alarm { background: #fee2e2; color: var(--danger); }
    .log { min-height: 22px; color: var(--muted); font-size: 13px; }
    .error { color: var(--danger); font-weight: 650; }
    input.dirty {
      border-color: var(--accent-2);
      background: #eff6ff;
      box-shadow: 0 0 0 2px rgba(29, 78, 216, 0.10);
    }
    @media (max-width: 1080px) {
      .channels { grid-template-columns: 1fr; }
      .connection-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 620px) {
      main { padding: 10px; }
      header { align-items: flex-start; flex-direction: column; }
      .connection-grid, .set-grid, .read-grid, .button-grid { grid-template-columns: 1fr; }
      .row > button { flex: 1; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1><strong id="supplyName">HV</strong></h1>
        <div class="muted" id="identity">Disconnected</div>
      </div>
      <span class="pill" id="connectionState">Offline</span>
    </header>

    <section>
      <div class="connection-grid">
        <label>Supply profile
          <select id="profile">
            <option value="hv">HV</option>
            <option value="cec">CEC</option>
            <option value="custom">Custom</option>
          </select>
        </label>
        <label>Connection
          <select id="mode">
            <option value="tcp">LAN socket</option>
            <option value="sim">Simulator</option>
            <option value="visa">VISA resource</option>
          </select>
        </label>
        <label>IP / host
          <input id="host" value="192.168.1.181" placeholder="192.168.1.50" />
        </label>
        <label>Port
          <input id="port" type="number" value="5555" min="1" max="65535" />
        </label>
        <label>VISA resource
          <input id="resource" placeholder="USB0::...::INSTR" />
        </label>
      </div>
      <div class="row" style="margin-top: 12px;">
        <button class="primary" id="connect">Connect</button>
        <button id="disconnect">Disconnect</button>
        <button id="refreshAll">Refresh All</button>
        <button class="danger" id="allOff">All Off</button>
        <div class="log" id="message"></div>
      </div>
    </section>

    <div class="channels" id="channels"></div>
  </main>

  <template id="channelTemplate">
    <article class="channel">
      <div class="channel-head">
        <h3 data-role="title"></h3>
        <div class="row">
          <span class="pill" data-role="mode">OFF</span>
          <span class="pill" data-role="output">Output Off</span>
          <span class="pill" data-role="ramp">Idle</span>
          <span class="pill" data-role="schedule">No schedule</span>
          <span class="pill" data-role="alarm">Protection</span>
        </div>
      </div>
      <div class="channel-body">
        <div class="read-grid">
          <div class="readout"><span>Measured V</span><b data-role="measV">--</b></div>
          <div class="readout"><span>Measured I</span><b data-role="measI">--</b></div>
          <div class="readout"><span>Measured P</span><b data-role="measP">--</b></div>
        </div>
        <div class="set-grid">
          <label><span data-role="voltageLabel">Voltage setpoint (V)</span>
            <input data-role="voltage" type="number" min="0" step="0.001" value="0.000" />
          </label>
          <label><span data-role="rampRateLabel">Ramp rate (V/s)</span>
            <input data-role="rampRate" type="number" min="0" step="0.001" value="0.200" />
          </label>
          <label><span data-role="rampSecondsLabel">Ramp interval (s)</span>
            <input data-role="rampSeconds" type="number" min="0" step="1" value="" placeholder="optional" />
          </label>
          <label><span data-role="currentLabel">Current limit (A)</span>
            <input data-role="current" type="number" min="0" max="3.3" step="0.001" value="0.100" />
          </label>
          <label><span data-role="ovpLabel">OVP limit (V)</span>
            <input data-role="ovp" type="number" min="0.001" step="0.001" value="35.200" />
          </label>
          <label><span data-role="ocpLabel">OCP limit (A)</span>
            <input data-role="ocp" type="number" min="0.001" max="3.3" step="0.001" value="3.300" />
          </label>
        </div>
        <div class="read-grid">
          <div class="readout"><span data-role="setVLabel">Set V</span><b data-role="setV">--</b></div>
          <div class="readout"><span data-role="setILabel">Limit I</span><b data-role="setI">--</b></div>
          <div class="readout"><span>OVP</span><b data-role="ovpRead">--</b></div>
          <div class="readout"><span>OCP</span><b data-role="ocpRead">--</b></div>
        </div>
        <div class="button-grid">
          <button class="primary" data-action="apply">Apply</button>
          <button class="blue" data-action="on">On</button>
          <button class="danger" data-action="off">Off</button>
          <button data-action="clear">Clear Trip</button>
        </div>
        <div class="schedule-grid">
          <label>Ready by time
            <input data-role="readyTime" type="time" value="08:00" />
          </label>
          <button data-action="schedule">Schedule</button>
          <button data-action="cancelSchedule">Cancel</button>
        </div>
      </div>
    </article>
  </template>

  <script>
    const $ = (id) => document.getElementById(id);
    const channels = ["CH1", "CH2", "CH3"];
    const panels = new Map();
    const profiles = {
      hv: {
        label: "HV",
        mode: "tcp",
        host: "192.168.1.181",
        port: 5555,
        resource: "",
        rampAxis: "voltage",
        applyText: "CV setup applied.",
        voltageLabel: "Voltage setpoint (V)",
        rampRateLabel: "Voltage ramp (V/s)",
        rampSecondsLabel: "Voltage ramp interval (s)",
        currentLabel: "Current limit (A)",
        ovpLabel: "OVP limit (V)",
        ocpLabel: "OCP limit (A)",
        setVLabel: "Set V",
        setILabel: "Limit I",
      },
      cec: {
        label: "CEC",
        mode: "visa",
        host: "",
        port: 5555,
        resource: "USB0::0x1AB1::0xA4A8::DP9A282M00021::INSTR",
        rampAxis: "current",
        applyText: "current setup applied.",
        voltageLabel: "Voltage compliance (V)",
        rampRateLabel: "Current ramp (A/s)",
        rampSecondsLabel: "Current ramp interval (s)",
        currentLabel: "Current setpoint (A)",
        ovpLabel: "Voltage trip limit (V)",
        ocpLabel: "Current trip limit (A)",
        setVLabel: "Compliance V",
        setILabel: "Set I",
      },
      custom: {
        label: "Custom",
        rampAxis: "voltage",
        applyText: "setup applied.",
        voltageLabel: "Voltage setpoint / compliance (V)",
        rampRateLabel: "Ramp rate (V/s)",
        rampSecondsLabel: "Ramp interval (s)",
        currentLabel: "Current setpoint / limit (A)",
        ovpLabel: "OVP limit (V)",
        ocpLabel: "OCP limit (A)",
        setVLabel: "Set V",
        setILabel: "Set I",
      },
    };
    let activeProfile = "hv";
    let timer = null;

    async function api(path, body) {
      const url = body === undefined && path.startsWith("/api/")
        ? `${path}${path.includes("?") ? "&" : "?"}profile=${encodeURIComponent(activeProfile)}`
        : path;
      const options = body === undefined ? {} : {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({profile: activeProfile, ...body}),
      };
      const res = await fetch(url, options);
      const payload = await res.json();
      if (!payload.ok) throw new Error(payload.error || "Request failed");
      return payload.data;
    }

    function message(text, isError = false) {
      $("message").textContent = text;
      $("message").className = isError ? "log error" : "log";
    }

    function maxVoltage(channel) {
      return channel === "CH3" ? 6.6 : 35.2;
    }

    function fmt(value, places, unit) {
      return Number.isFinite(value) ? `${value.toFixed(places)} ${unit}` : "--";
    }

    function role(panel, name) {
      return panel.querySelector(`[data-role="${name}"]`);
    }

    function updateText(panel, name, text) {
      const el = role(panel, name);
      if (el) el.textContent = text;
    }

    function applyPanelProfile(panel) {
      const profile = profiles[activeProfile] || profiles.custom;
      updateText(panel, "voltageLabel", profile.voltageLabel);
      updateText(panel, "rampRateLabel", profile.rampRateLabel);
      updateText(panel, "rampSecondsLabel", profile.rampSecondsLabel);
      updateText(panel, "currentLabel", profile.currentLabel);
      updateText(panel, "ovpLabel", profile.ovpLabel);
      updateText(panel, "ocpLabel", profile.ocpLabel);
      updateText(panel, "setVLabel", profile.setVLabel);
      updateText(panel, "setILabel", profile.setILabel);
    }

    function applyProfile(name, updateConnection = true) {
      activeProfile = profiles[name] ? name : "hv";
      $("profile").value = activeProfile;
      const profile = profiles[activeProfile];
      $("supplyName").textContent = profile.label;
      document.title = `${profile.label} DP900`;
      if (updateConnection && activeProfile !== "custom") {
        $("mode").value = profile.mode;
        $("host").value = profile.host;
        $("port").value = profile.port;
        $("resource").value = profile.resource;
      }
      for (const panel of panels.values()) applyPanelProfile(panel);
      if (activeProfile === "cec") {
        message("CEC current-control layout selected.");
      }
    }

    function selectCustomProfile() {
      if ($("profile").value !== "custom") {
        applyProfile("custom", false);
      }
    }

    function syncLimits(panel, channel) {
      const max = maxVoltage(channel);
      for (const key of ["voltage", "ovp"]) {
        const input = role(panel, key);
        input.max = max;
        if (Number(input.value) > max) input.value = max.toFixed(3);
      }
    }

    function setPill(el, text, cls = "") {
      el.textContent = text;
      el.className = `pill ${cls}`.trim();
    }

    function applyStatus(data) {
      const panel = panels.get(data.channel);
      if (!panel) return;
      role(panel, "measV").textContent = fmt(data.measuredVoltage, 4, "V");
      role(panel, "measI").textContent = fmt(data.measuredCurrent, 4, "A");
      role(panel, "measP").textContent = fmt(data.measuredPower, 3, "W");
      role(panel, "setV").textContent = fmt(data.setVoltage, 3, "V");
      role(panel, "setI").textContent = fmt(data.setCurrent, 3, "A");
      role(panel, "ovpRead").textContent = fmt(data.ovp, 3, "V");
      role(panel, "ocpRead").textContent = fmt(data.ocp, 3, "A");
      setInputValue(role(panel, "voltage"), data.setVoltage.toFixed(3));
      setInputValue(role(panel, "current"), data.setCurrent.toFixed(3));
      setInputValue(role(panel, "ovp"), data.ovp.toFixed(3));
      setInputValue(role(panel, "ocp"), data.ocp.toFixed(3));
      syncLimits(panel, data.channel);

      const modeClass = data.mode === "CV" ? "cv" : (data.mode === "CC" ? "cc" : (data.mode === "UR" ? "ur" : ""));
      setPill(role(panel, "mode"), data.mode, modeClass);
      setPill(role(panel, "output"), data.output ? "Output On" : "Output Off", data.output ? "on" : "");
      setPill(role(panel, "ramp"), data.ramping ? "Ramping" : "Idle", data.ramping ? "cv" : "");
      setPill(role(panel, "schedule"), data.scheduled ? scheduleLabel(data.startAt, data.readyTime) : "No schedule", data.scheduled ? "cc" : "");
      const alarm = data.ovpAlarm || data.ocpAlarm;
      setPill(role(panel, "alarm"), alarm ? "Trip" : "Protection OK", alarm ? "alarm" : "on");
    }

    function scheduleLabel(startAt, readyTime) {
      const start = new Date(startAt);
      const ready = new Date(readyTime);
      const pad = (n) => String(n).padStart(2, "0");
      return `Start ${pad(start.getHours())}:${pad(start.getMinutes())} Ready ${pad(ready.getHours())}:${pad(ready.getMinutes())}`;
    }

    function setInputValue(input, value) {
      if (input.dataset.dirty === "true") return;
      input.value = value;
    }

    function markDirty(input) {
      input.dataset.dirty = "true";
      input.classList.add("dirty");
    }

    function clearDirty(panel) {
      panel.querySelectorAll("input[data-role]").forEach((input) => {
        input.dataset.dirty = "false";
        input.classList.remove("dirty");
      });
    }

    function buildPanels() {
      const host = $("channels");
      const template = $("channelTemplate");
      for (const channel of channels) {
        const node = template.content.firstElementChild.cloneNode(true);
        role(node, "title").textContent = `${channel} ${channel === "CH3" ? "6 V / 3 A" : "32 V / 3 A"}`;
        applyPanelProfile(node);
        syncLimits(node, channel);
        node.querySelectorAll("input[data-role]").forEach((input) => {
          input.dataset.dirty = "false";
          input.addEventListener("input", () => markDirty(input));
        });
        node.querySelector('[data-action="apply"]').addEventListener("click", () => applyChannel(channel));
        node.querySelector('[data-action="on"]').addEventListener("click", () => setOutput(channel, true));
        node.querySelector('[data-action="off"]').addEventListener("click", () => setOutput(channel, false));
        node.querySelector('[data-action="clear"]').addEventListener("click", () => clearTrip(channel));
        node.querySelector('[data-action="schedule"]').addEventListener("click", () => scheduleChannel(channel));
        node.querySelector('[data-action="cancelSchedule"]').addEventListener("click", () => cancelSchedule(channel));
        panels.set(channel, node);
        host.appendChild(node);
      }
    }

    async function refreshAll(clearPending = false, silent = false) {
      try {
        const data = await api("/api/status_all");
        $("identity").textContent = data.label || "Connected";
        $("connectionState").textContent = "Online";
        $("connectionState").className = "pill on";
        if (clearPending) {
          for (const panel of panels.values()) clearDirty(panel);
        }
        for (const status of data.channels) applyStatus(status);
        if (!silent) message("All channels updated.");
      } catch (err) {
        message(err.message, true);
      }
    }

    async function applyChannel(channel) {
      const panel = panels.get(channel);
      try {
        const data = await api("/api/apply", {
          channel,
          voltage: Number(role(panel, "voltage").value),
          rampRate: Number(role(panel, "rampRate").value),
          rampSeconds: role(panel, "rampSeconds").value,
          rampAxis: (profiles[activeProfile] || profiles.custom).rampAxis,
          current: Number(role(panel, "current").value),
          ovp: Number(role(panel, "ovp").value),
          ocp: Number(role(panel, "ocp").value),
          output: role(panel, "output").textContent === "Output On",
        });
        clearDirty(panel);
        applyStatus(data);
        const profile = profiles[activeProfile] || profiles.custom;
        message(`${channel} ${profile.applyText}`);
      } catch (err) {
        message(`${channel}: ${err.message}`, true);
      }
    }

    async function setOutput(channel, output) {
      const panel = panels.get(channel);
      try {
        const data = await api("/api/output", {
          channel,
          output,
          rampRate: Number(role(panel, "rampRate").value),
          rampSeconds: role(panel, "rampSeconds").value,
          rampAxis: (profiles[activeProfile] || profiles.custom).rampAxis,
        });
        applyStatus(data);
        message(`${channel} output ${output ? "enabled" : "disabled"}.`);
      } catch (err) {
        message(`${channel}: ${err.message}`, true);
      }
    }

    async function clearTrip(channel) {
      try {
        const data = await api("/api/clear", {channel});
        applyStatus(data);
        message(`${channel} protection trips cleared.`);
      } catch (err) {
        message(`${channel}: ${err.message}`, true);
      }
    }

    async function scheduleChannel(channel) {
      const panel = panels.get(channel);
      try {
        const data = await api("/api/schedule", {
          channel,
          readyTime: role(panel, "readyTime").value,
          voltage: Number(role(panel, "voltage").value),
          rampRate: Number(role(panel, "rampRate").value),
          rampSeconds: role(panel, "rampSeconds").value,
          rampAxis: (profiles[activeProfile] || profiles.custom).rampAxis,
          current: Number(role(panel, "current").value),
          ovp: Number(role(panel, "ovp").value),
          ocp: Number(role(panel, "ocp").value),
          output: true,
        });
        applyStatus(data);
        message(`${channel} scheduled for the next matching time.`);
      } catch (err) {
        message(`${channel}: ${err.message}`, true);
      }
    }

    async function cancelSchedule(channel) {
      try {
        const data = await api("/api/cancel_schedule", {channel});
        applyStatus(data);
        message(`${channel} schedule canceled.`);
      } catch (err) {
        message(`${channel}: ${err.message}`, true);
      }
    }

    async function allOff() {
      for (const channel of channels) {
        await setOutput(channel, false);
      }
      message("All outputs disabled.");
    }

    $("connect").addEventListener("click", async () => {
      try {
        const data = await api("/api/connect", {
          mode: $("mode").value,
          host: $("host").value,
          port: Number($("port").value),
          resource: $("resource").value,
          timeout: 2,
        });
        $("identity").textContent = `${data.label} - ${data.idn}`;
        $("connectionState").textContent = "Online";
        $("connectionState").className = "pill on";
        message("Connected.");
        await refreshAll(true, false);
        clearInterval(timer);
        timer = setInterval(() => refreshAll(false, true), 300);
      } catch (err) {
        message(err.message, true);
      }
    });

    $("disconnect").addEventListener("click", async () => {
      clearInterval(timer);
      timer = null;
      await api("/api/disconnect", {});
      $("identity").textContent = "Disconnected";
      $("connectionState").textContent = "Offline";
      $("connectionState").className = "pill";
      message("Disconnected.");
    });

    $("refreshAll").addEventListener("click", () => refreshAll(true, false));
    $("allOff").addEventListener("click", allOff);
    buildPanels();
    $("profile").addEventListener("change", () => applyProfile($("profile").value, true));
    $("mode").addEventListener("change", selectCustomProfile);
    $("host").addEventListener("input", selectCustomProfile);
    $("port").addEventListener("input", selectCustomProfile);
    $("resource").addEventListener("input", selectCustomProfile);
    const initialPath = window.location.pathname.toLowerCase();
    const initialProfile = new URLSearchParams(window.location.search).get("profile");
    const pathProfile = initialPath === "/cec" ? "cec" : (initialPath === "/hv" ? "hv" : null);
    applyProfile(pathProfile || initialProfile || "hv", true);
  </script>
</body>
</html>
"""


def main() -> None:
    host = "127.0.0.1"
    port = 8765
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Rigol DP900 UI running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
