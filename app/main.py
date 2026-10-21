from __future__ import annotations

import argparse
import os
import sys
import signal
from contextlib import contextmanager

from app.paths import LABELME_UPSTREAM_ROOT, ensure_labelme

ensure_labelme()

from qtpy import QtCore, QtWidgets

from app.gui.labelme_window import MedPhysLabelMeWindow
from labelme import __appname__
from labelme.config import get_config
from labelme.utils import newIcon


def install_excepthook():
    """PyQt aborts the process on an exception escaping a slot (timer, signal).

    Log it and show it instead, so one bad object or failed operation cannot
    close the application and lose unsaved annotations.
    """
    import logging
    import traceback

    def hook(kind, value, tb):
        if issubclass(kind, KeyboardInterrupt):
            window = QtWidgets.QApplication.activeWindow()
            if window is not None:
                window.close()
            else:
                QtWidgets.QApplication.quit()
            return
        text = "".join(traceback.format_exception(kind, value, tb))
        logging.getLogger("medphys").error("Unhandled exception\n%s", text)
        sys.stderr.write(text)
        if QtWidgets.QApplication.instance() is not None:
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "Unexpected error",
                                        f"{kind.__name__}: {str(value)[:2000]}")
            box.setDetailedText(text)
            box.exec_()

    sys.excepthook = hook


@contextmanager
def qt_interrupts(window):
    """Translate terminal Ctrl+C to close, preserving save confirmation."""
    pending = False

    def close():
        nonlocal pending
        try:
            window.close()
        finally:
            pending = False

    def interrupt(*_):
        nonlocal pending
        if not pending:
            pending = True
            QtCore.QTimer.singleShot(0, close)

    previous = signal.signal(signal.SIGINT, interrupt)
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)
    try:
        yield
    finally:
        timer.stop()
        signal.signal(signal.SIGINT, previous)


def main() -> int:
    parser = argparse.ArgumentParser(description="LabelMe MedPhys SAM2")
    parser.add_argument("image", nargs="?", help="Image or LabelMe JSON to open")
    parser.add_argument("--no-worker", action="store_true",
                        help="Do not start the SAM2 worker automatically")
    parser.add_argument("--worker-url", help="Use an external worker, e.g. http://gpu-host:8765")
    parser.add_argument("--worker-arg", action="append", default=[],
                        help="Extra argument for worker.main, e.g. --worker-arg=--device=cpu")
    args, qt_args = parser.parse_known_args()
    install_excepthook()
    if args.worker_url:
        os.environ["MEDPHYS_WORKER_URL"] = args.worker_url
    translator = QtCore.QTranslator()
    translator.load(
        QtCore.QLocale.system().name(),
        str(LABELME_UPSTREAM_ROOT / "labelme" / "translate"),
    )

    app = QtWidgets.QApplication([sys.argv[0], *qt_args])
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("icon"))
    app.installTranslator(translator)

    from app.core.video import is_video
    project_file = args.image and args.image.endswith(".medphys.json")
    video_file = args.image and is_video(args.image)
    window = MedPhysLabelMeWindow(config=get_config(),
                                  filename=None if project_file or video_file else args.image)
    if project_file:
        window._project_safe(lambda: window.open_project(args.image))
    if video_file:
        QtCore.QTimer.singleShot(0, lambda: window._import_video_dialog(path=args.image))
    window.setWindowTitle("LabelMe MedPhys SAM2")
    manager = None
    with qt_interrupts(window):
        try:
            if not args.no_worker:
                from app.worker_api.manager import WorkerManager
                manager = WorkerManager(window.worker_client, worker_args=args.worker_arg, parent=app)
                app.aboutToQuit.connect(manager.stop)
                window.attach_worker_manager(manager)
            window.show()
            window.raise_()
            return app.exec_()
        finally:
            if manager is not None:
                manager.stop()


if __name__ == "__main__":
    raise SystemExit(main())
