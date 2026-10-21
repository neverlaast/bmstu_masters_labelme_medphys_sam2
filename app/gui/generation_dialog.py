from qtpy import QtWidgets

from app.worker_api.schemas import GenerationOptions


class GenerationDialog(QtWidgets.QDialog):
    def __init__(self, options, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM2 object search: grid and filters")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "Area: px²; diameter: equivalent circular diameter in px.\n"
            "SAM score is a model quality estimate, not measured accuracy."))
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        self.values = {}
        for key, text, value, minimum, maximum, decimals in [
            ("points_per_side", "Grid points per side", options.points_per_side, 2, 20, 0),
            ("max_candidates", "Maximum candidates", options.max_candidates, 1, 128, 0),
            ("min_area", "Minimum area", options.filters.min_area, 0, 1e9, 1),
            ("max_area", "Maximum area (0 = unlimited)", options.filters.max_area, 0, 1e9, 1),
            ("min_diameter", "Minimum diameter", options.filters.min_diameter, 0, 1e6, 1),
            ("max_diameter", "Maximum diameter (0 = unlimited)", options.filters.max_diameter, 0, 1e6, 1),
            ("max_aspect_ratio", "Maximum aspect ratio (0 = unlimited)", options.filters.max_aspect_ratio, 0, 1000, 2),
            ("min_roundness", "Minimum roundness", options.filters.min_roundness, 0, 1, 2),
            ("min_convexity", "Minimum convexity", options.filters.min_convexity, 0, 1, 2),
            ("min_roi_fraction", "Minimum fraction inside ROI", options.filters.min_roi_fraction, 0, 1, 2),
            ("min_score", "Minimum SAM quality score", options.filters.min_score, 0, 1, 2),
            ("duplicate_iou", "Duplicate mask IoU threshold", options.filters.duplicate_iou, .01, 1, 2),
            ("overlap_warning", "Existing mask overlap warning", options.filters.overlap_warning, 0, 1, 2),
        ]:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(decimals)
            spin.setValue(value)
            self.values[key] = spin
            form.addRow(text, spin)
        self.exclude_border = QtWidgets.QCheckBox("Exclude masks touching image border")
        self.exclude_border.setChecked(options.filters.exclude_border)
        form.addRow(self.exclude_border)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self):
        values = {name: spin.value() for name, spin in self.values.items()}
        density, limit = int(values.pop("points_per_side")), int(values.pop("max_candidates"))
        values["exclude_border"] = self.exclude_border.isChecked()
        return GenerationOptions(points_per_side=density, max_candidates=limit, filters=values)

    def accept(self):
        try:
            self.options()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid filter settings", str(exc))
            return
        super().accept()
