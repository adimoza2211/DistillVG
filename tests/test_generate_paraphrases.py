from __future__ import annotations

from pathlib import Path
import sys


def test_scene_graph_dataset_aliases() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from extract_scene_graphs import resolve_dataset_names

        assert resolve_dataset_names("dataset=refcocog") == ["gref_umd"]
        assert resolve_dataset_names("refcoco+") == ["unc+"]
        assert resolve_dataset_names("all") == ["referit", "gref", "gref_umd", "unc", "unc+"]
    finally:
        sys.path.pop(0)