from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELME_UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "labelme_upstream"


def ensure_labelme() -> None:
    """Use the pristine vendored tree, or LabelMe bundled into the wheel."""
    if (LABELME_UPSTREAM_ROOT / "labelme" / "__init__.py").is_file():
        root = str(LABELME_UPSTREAM_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
    import labelme

    if labelme.__version__ != "5.2.1":
        raise RuntimeError(
            f"Expected LabelMe 5.2.1, got {labelme.__version__} from {labelme.__file__}."
        )
    from app.compat import install_labelme_compatibility

    install_labelme_compatibility()
