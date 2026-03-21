from __future__ import annotations

import argparse

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from extract_scene_graphs import generate_augmented_datasets, resolve_dataset_names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for scene-graph augmentation outputs.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        help="Dataset selector: all, refcoco, refcoco+, refcocog, referitgame, or a legacy annotation family name.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset_names = resolve_dataset_names(args.dataset)
    generate_augmented_datasets(dataset_names=dataset_names)


if __name__ == "__main__":
    main()