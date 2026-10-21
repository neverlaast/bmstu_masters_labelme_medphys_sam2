"""Choose frame range/step and output folder for importing a video."""
from __future__ import annotations

from pathlib import Path

from qtpy import QtWidgets

from app.core.video import MAX_EXTRACTED_FRAMES, FrameRange, VideoInfo
from worker.propagation import MAX_FRAMES as MAX_PROPAGATION_FRAMES


class VideoImportDialog(QtWidgets.QDialog):
    def __init__(self, info: VideoInfo, frames_dir: Path, parent=None):
        super().__init__(parent)
        self.info = info
        self.setWindowTitle("Import video as frame sequence")
        layout = QtWidgets.QFormLayout(self)
        fps = f"{info.fps:.2f} fps, {info.duration_s:.1f} s" if info.fps > 0 else "fps unknown"
        layout.addRow(QtWidgets.QLabel(
            f"{Path(info.path).name}: {info.frame_count} frames, "
            f"{info.width}x{info.height}, {fps}"))
        last = info.frame_count - 1
        self.start = QtWidgets.QSpinBox(); self.start.setRange(0, last)
        self.end = QtWidgets.QSpinBox(); self.end.setRange(0, last); self.end.setValue(last)
        self.step = QtWidgets.QSpinBox(); self.step.setRange(1, max(1, info.frame_count))
        # Default step keeps the sequence within one propagation run.
        self.step.setValue(max(1, -(-info.frame_count // MAX_PROPAGATION_FRAMES)))
        layout.addRow("First frame", self.start)
        layout.addRow("Last frame", self.end)
        layout.addRow("Take every N-th frame", self.step)
        self.folder = QtWidgets.QLineEdit(str(frames_dir))
        browse = QtWidgets.QPushButton("...")
        browse.clicked.connect(self._browse)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.folder); row.addWidget(browse)
        layout.addRow("Frames folder (PNG)", row)
        self.summary = QtWidgets.QLabel()
        layout.addRow(self.summary)
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow(self.buttons)
        for box in (self.start, self.end, self.step):
            box.valueChanged.connect(self._update)
        self._update()

    def frame_range(self) -> FrameRange:
        return FrameRange(self.start.value(), self.end.value(), self.step.value())

    def frames_dir(self) -> Path:
        return Path(self.folder.text()).expanduser()

    def count(self) -> int:
        if self.start.value() > self.end.value():
            return 0
        return len(self.frame_range().indices(self.info.frame_count))

    def _update(self):
        count = self.count()
        text = f"{count} frame(s) will be extracted."
        if count > MAX_PROPAGATION_FRAMES:
            text += (f" Propagation handles up to {MAX_PROPAGATION_FRAMES} frames per run; "
                     "longer sequences are processed in several runs.")
        ok = 0 < count <= MAX_EXTRACTED_FRAMES
        if count > MAX_EXTRACTED_FRAMES:
            text = f"{count} frames exceed the limit of {MAX_EXTRACTED_FRAMES}; increase the step."
        self.summary.setText(text)
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(ok)

    def _browse(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Frames folder", self.folder.text())
        if folder:
            self.folder.setText(folder)
