"""Background file I/O with progress and cooperative cancellation.

The operation runs in a QThread and receives ``progress(done, total, text)``
and ``cancelled()`` callables. It must not touch Qt widgets or mutable GUI
state: callers pass immutable snapshots and apply results in ``on_done`` on
the GUI thread. ``result``/``error`` are stored on the job so that a caller
may ``wait()`` and finish it synchronously before the queued signal arrives.
"""
from __future__ import annotations

from threading import Event

from qtpy import QtCore


class OperationCancelled(Exception):
    """Raised by an operation that observed a cancellation request."""


class IOJob(QtCore.QThread):
    progressed = QtCore.Signal(int, int, str)

    def __init__(self, title, operation, parent=None):
        super().__init__(parent)
        self.title = title
        self.operation = operation
        self.result = None
        self.error = None
        self.handled = False
        self._cancel = Event()

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def _check(self):
        return self._cancel.is_set()

    def _progress(self, done, total, text=""):
        if self._cancel.is_set():
            raise OperationCancelled()
        self.progressed.emit(int(done), int(total), str(text))

    def run(self):
        try:
            self.result = self.operation(self._progress, self._check)
        except BaseException as exc:  # reported on the GUI thread
            self.error = exc
