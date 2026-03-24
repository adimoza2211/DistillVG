from src.data.grounding import _ensure_image_filename, _resolve_coco_name


def test_referit_filename_keeps_extension() -> None:
    assert _ensure_image_filename("10000.jpg") == "10000.jpg"


def test_referit_filename_adds_extension() -> None:
    assert _ensure_image_filename("10000") == "10000.jpg"


def test_coco_resolution_from_numeric_name() -> None:
    assert _resolve_coco_name("18185.jpg") == "COCO_train2014_000000018185.jpg"


def test_coco_resolution_keeps_prefixed_name() -> None:
    original = "COCO_train2014_000000000009.jpg"
    assert _resolve_coco_name(original) == original
