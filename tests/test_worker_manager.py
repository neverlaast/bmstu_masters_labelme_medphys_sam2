"""WorkerManager: autostart, readiness polling, external worker, shutdown."""
import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtCore, QtWidgets

from app.worker_api.client import WorkerClient
from app.worker_api.errors import WorkerConnectionError
from app.worker_api.manager import WorkerManager

FAKE_PORT = 54321


def _can_signal_children() -> bool:
    import subprocess
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        child.terminate()
        child.wait(3)
        return True
    except (PermissionError, subprocess.TimeoutExpired):
        return False


CAN_SIGNAL = _can_signal_children()
needs_signals = unittest.skipUnless(CAN_SIGNAL, "sandbox forbids signalling child processes")


def wait_until(predicate, ms=10000):
    timer = QtCore.QElapsedTimer()
    timer.start()
    while not predicate() and timer.elapsed() < ms:
        QtWidgets.QApplication.processEvents()
        QtCore.QThread.msleep(20)
    return predicate()


class FakeWorkerManager(WorkerManager):
    POLL_MS = 20
    code = "import time; print('fake worker', flush=True); time.sleep(5)"

    def command(self, port):
        return sys.executable, ["-c", self.code]


class WorkerManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        os.environ.pop("MEDPHYS_WORKER_URL", None)
        self.client = WorkerClient()
        self.alive = False
        port_patch = patch("app.worker_api.manager.find_free_port", return_value=FAKE_PORT)
        port_patch.start()
        self.addCleanup(port_patch.stop)

    def manager(self, cls=None, *args):
        m = (cls or WorkerManager)(self.client, *args)
        self.addCleanup(m.stop, 1000)
        return m

    def tearDown(self):
        self.env.stop()

    def _health(self, timeout=1.5):
        if not self.alive:
            raise WorkerConnectionError("down")

    def test_default_command_uses_gui_interpreter(self):
        program, args = WorkerManager(self.client, ["--device", "cpu"]).command(9000)
        self.assertEqual(program, sys.executable)
        self.assertEqual(args[:2], ["-m", "worker.main"])
        self.assertIn("9000", args)
        self.assertEqual(args[-2:], ["--device", "cpu"])

    def test_env_url_means_external_worker(self):
        os.environ["MEDPHYS_WORKER_URL"] = "http://gpu-host:8765"
        m = self.manager()
        with patch.object(m, "_spawn") as spawn:
            m.ensure_running()
        spawn.assert_not_called()
        self.assertEqual(m.state, "external")

    def test_already_running_worker_is_reused(self):
        self.alive = True
        m = self.manager()
        with patch.object(self.client, "healthcheck", side_effect=self._health), \
                patch.object(m, "_spawn") as spawn:
            m.ensure_running()
        spawn.assert_not_called()
        self.assertFalse(m.owns_process)

    @needs_signals
    @needs_signals
    def test_spawn_ready_and_stop(self):
        m = self.manager(FakeWorkerManager)
        states = []
        m.stateChanged.connect(lambda state, _msg: states.append(state))
        with patch.object(self.client, "healthcheck", side_effect=self._health):
            m.ensure_running()
            self.assertEqual(m.state, "starting")
            self.assertEqual(self.client.base_url, f"http://127.0.0.1:{FAKE_PORT}")
            self.assertTrue(wait_until(lambda: "fake worker" in m.log_text()))
            self.alive = True
            self.assertTrue(wait_until(lambda: m.state == "ready"))
            proc = m._process
            m.stop()
        self.assertEqual(proc.state(), QtCore.QProcess.NotRunning)
        self.assertEqual(states, ["starting", "ready", "stopped"])

    def test_crashed_worker_reports_failed(self):
        m = self.manager(FakeWorkerManager)
        m.code = "print('boom'); raise SystemExit(3)"
        messages = []
        m.stateChanged.connect(lambda _s, msg: messages.append(msg))
        with patch.object(self.client, "healthcheck", side_effect=self._health):
            m.ensure_running()
            self.assertTrue(wait_until(lambda: m.state == "failed"))
        self.assertIn("code 3", messages[-1])
        self.assertIn("boom", m.log_text())
        self.assertFalse(m.owns_process)

    @needs_signals
    @needs_signals
    def test_start_timeout_stops_process(self):
        m = self.manager(FakeWorkerManager)
        m.START_TIMEOUT_MS = 100
        with patch.object(self.client, "healthcheck", side_effect=self._health):
            m.ensure_running()
            self.assertTrue(wait_until(lambda: m.state == "failed"))
        self.assertFalse(m.owns_process)


if __name__ == "__main__":
    unittest.main()
