import importlib

import numpy as np


def test_validation_module_imports_without_detection_import_error():
    module = importlib.import_module("src.validation.validate")
    assert hasattr(module, "validate_and_score")
    assert hasattr(module, "run_validation")


def test_find_candidate_regions_detects_bright_target():
    from src.detection.detect import find_candidate_regions

    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:100, 60:120] = 255
    img[50:100, 25:60] = 20

    regions = find_candidate_regions(img)

    assert regions
    assert any(30 <= x <= 70 and 40 <= y <= 80 and w >= 40 and h >= 30 for x, y, w, h in regions)
