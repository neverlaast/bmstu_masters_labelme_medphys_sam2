from __future__ import annotations

from threading import Event

from qtpy import QtCore


class WorkerTask(QtCore.QThread):
    """Run blocking HTTP outside Qt's event loop.

    Cancellation discards a result; it does not kill the remote model process.
    """

    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(object)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self._cancelled = Event()

    @property
    def cancelled(self):
        return self._cancelled.is_set()

    def cancel(self):
        self._cancelled.set()

    def run(self):
        try:
            result = self.operation()
        except Exception as exc:
            if not self.cancelled:
                self.failed.emit(exc)
        else:
            if not self.cancelled:
                self.succeeded.emit(result)
