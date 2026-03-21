from __future__ import annotations

from pathlib import Path

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from src.data.prepare import (
    default_requirements,
    load_manifest,
    validate_manifest_paths_local,
    validate_raw_data,
    write_manifest_template,
)
from src.utils.logging import get_logger


logger = get_logger("distillvg.prepare_data")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    manifest_path = raw_dir / "dataset_manifest.json"
    requirements = default_requirements()

    raw_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        write_manifest_template(manifest_path=manifest_path, requirements=requirements)
        logger.warning("Created dataset manifest template at %s", manifest_path)
        logger.warning("Fill source_path for each dataset, then rerun this command.")
        raise SystemExit(1)

    manifest = load_manifest(manifest_path=manifest_path)

    path_violations = validate_manifest_paths_local(raw_dir=raw_dir, requirements=requirements, manifest=manifest)
    if path_violations:
        logger.error("Manifest path policy violation. Datasets must be under %s", raw_dir)
        for dataset_name, message in path_violations.items():
            logger.error("- %s: %s", dataset_name, message)
        raise SystemExit(3)

    missing = validate_raw_data(raw_dir=raw_dir, requirements=requirements)
    if missing:
        logger.error("Dataset validation failed. Missing entries:")
        for dataset_name, items in missing.items():
            logger.error("- %s: %s", dataset_name, ", ".join(items))
        raise SystemExit(2)

    logger.info("All required datasets and annotations are present under %s", raw_dir)


if __name__ == "__main__":
    main()
