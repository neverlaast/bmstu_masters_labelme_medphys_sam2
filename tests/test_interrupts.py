import os
import signal
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.main import qt_interrupts
from qtpy import QtWidgets


class InterruptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_sigint_closes_once_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGINT)
        window = Mock()
        with qt_interrupts(window):
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)
            self.app.processEvents()
            window.close.assert_called_once_with()
        self.assertIs(signal.getsignal(signal.SIGINT), previous)
