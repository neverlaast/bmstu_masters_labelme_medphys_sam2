"""Lifecycle of a local SAM2 worker process owned by the GUI.

The worker stays a separate process (a PyTorch crash or OOM does not kill the
GUI, and model unloading really frees memory); the GUI only starts it on demand,
waits for /health and stops it on exit.
"""
from __future__ import annotations

import os
import socket
import sys
from collections import deque

from qtpy import QtCore

from app.paths import PROJECT_ROOT
from app.worker_api.client import WorkerClient
from app.worker_api.errors import WorkerConnectionError, WorkerResponseError


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


class WorkerManager(QtCore.QObject):
    """States: stopped -> starting -> ready | failed; 'external' = not owned."""

    stateChanged = QtCore.Signal(str, str)  # state, message

    HOST = "127.0.0.1"
    POLL_MS = 500
    START_TIMEOUT_MS = 180_000  # first checkpoint load on CPU can be slow

    def __init__(self, client: WorkerClient, worker_args=None, parent=None):
        super().__init__(parent)
        self.client = client
        self.worker_args = list(worker_args or [])
        self.state = "stopped"
        self.log = deque(maxlen=500)
        self._process = None
        self._elapsed = QtCore.QElapsedTimer()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._poll)

    # -- public API ---------------------------------------------------------
    @property
    def owns_process(self) -> bool:
        return self._process is not None

    def ensure_running(self) -> None:
        """Use an already running/external worker, otherwise spawn one."""
        if self.owns_process:
            return
        if os.environ.get("MEDPHYS_WORKER_URL") or self._is_alive():
            self._set_state("external", f"Using external worker at {self.client.base_url}")
            return
        self._spawn()

    def restart(self) -> None:
        self.stop()
        self._spawn()

    def stop(self, timeout_ms: int = 5000) -> None:
        self._timer.stop()
        proc, self._process = self._process, None
        if proc is not None and proc.state() != QtCore.QProcess.NotRunning:
            proc.finished.disconnect(self._on_finished)
            proc.terminate()  # SIGTERM -> server loop exits, model unloaded
            if not proc.waitForFinished(timeout_ms):
                proc.kill()
                proc.waitForFinished(1000)
        self._set_state("stopped", "SAM2 worker stopped.")

    def log_text(self) -> str:
        return "".join(self.log)

    # -- internals ------------------------------------------------------------
    def _spawn(self) -> None:
        port = find_free_port(self.HOST)
        self.client.base_url = f"http://{self.HOST}:{port}"
        proc = QtCore.QProcess(self)
        proc.setWorkingDirectory(str(PROJECT_ROOT))
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._read_output)
        proc.finished.connect(self._on_finished)
        self._process = proc
        program, args = self.command(port)
        self.log.append(f"$ {program} {' '.join(args)}\n")
        proc.start(program, args)
        self._elapsed.start()
        self._timer.start()
        self._set_state("starting", "SAM2 worker: loading model...")

    def command(self, port: int):
        """Worker command line; the same interpreter/venv as the GUI."""
        return sys.executable, ["-m", "worker.main", "--host", self.HOST,
                                "--port", str(port), *self.worker_args]

    def _is_alive(self) -> bool:
        try:
            self.client.healthcheck(timeout=0.5)
            return True
        except (WorkerConnectionError, WorkerResponseError):
            return False

    def _poll(self) -> None:
        if self._is_alive():
            self._timer.stop()
            self._set_state("ready", "SAM2 worker ready.")
        elif self._elapsed.elapsed() > self.START_TIMEOUT_MS:
            self.stop()
            self._set_state("failed", "SAM2 worker did not start in time; see worker log.")

    def _read_output(self) -> None:
        if self._process is not None:
            data = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
            self.log.append(data)

    def _on_finished(self, code, _status=None) -> None:
        self._read_output()
        self._timer.stop()
        self._process = None
        self._set_state("failed", f"SAM2 worker exited (code {code}); see worker log.")

    def _set_state(self, state: str, message: str) -> None:
        self.state = state
        self.stateChanged.emit(state, message)
