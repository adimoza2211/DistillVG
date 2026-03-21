from __future__ import annotations

from pathlib import Path

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from src.utils.logging import get_logger

logger = get_logger("distillvg.organize_raw")


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path.resolve(), target_is_directory=target_path.is_dir())


def ensure_annotation_link(dst: Path, src: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if not src.exists():
        logger.warning("Missing annotation source: %s", src)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve(), target_is_directory=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    coco_train = project_root / "data" / "datasets" / "other" / "images" / "mscoco" / "images" / "train2014"
    referit_images = project_root / "data" / "datasets" / "referit" / "images"

    ann_root = project_root / "data" / "annotations"
    raw_root = project_root / "data" / "raw"

    dataset_specs = {
        "refcoco": {
            "image_src": coco_train,
            "annotation_map": {
                "corpus.pth": ann_root / "unc" / "corpus.pth",
                "train.pth": ann_root / "unc" / "unc_train.pth",
                "val.pth": ann_root / "unc" / "unc_val.pth",
                "testA.pth": ann_root / "unc" / "unc_testA.pth",
                "testB.pth": ann_root / "unc" / "unc_testB.pth",
            },
        },
        "refcoco+": {
            "image_src": coco_train,
            "annotation_map": {
                "corpus.pth": ann_root / "unc+" / "corpus.pth",
                "train.pth": ann_root / "unc+" / "unc+_train.pth",
                "val.pth": ann_root / "unc+" / "unc+_val.pth",
                "testA.pth": ann_root / "unc+" / "unc+_testA.pth",
                "testB.pth": ann_root / "unc+" / "unc+_testB.pth",
            },
        },
        "refcocog": {
            "image_src": coco_train,
            "annotation_map": {
                "corpus.pth": ann_root / "gref_umd" / "corpus.pth",
                "train.pth": ann_root / "gref_umd" / "gref_umd_train.pth",
                "val.pth": ann_root / "gref_umd" / "gref_umd_val.pth",
                "test.pth": ann_root / "gref_umd" / "gref_umd_test.pth",
            },
        },
        "referitgame": {
            "image_src": referit_images,
            "annotation_map": {
                "corpus.pth": ann_root / "referit" / "corpus.pth",
                "train.pth": ann_root / "referit" / "referit_train.pth",
                "val.pth": ann_root / "referit" / "referit_val.pth",
                "test.pth": ann_root / "referit" / "referit_test.pth",
            },
        },
    }

    for dataset_name, spec in dataset_specs.items():
        dataset_root = raw_root / dataset_name
        images_dst = dataset_root / "images"
        annotations_dst = dataset_root / "annotations"

        image_src = spec["image_src"]
        if image_src.exists():
            ensure_symlink(images_dst, image_src)
        else:
            logger.warning("Missing image source for %s: %s", dataset_name, image_src)

        annotations_dst.mkdir(parents=True, exist_ok=True)
        for target_name, source_file in spec["annotation_map"].items():
            ensure_annotation_link(annotations_dst / target_name, source_file)

    logger.info("Canonical raw dataset organization completed under %s", raw_root)


if __name__ == "__main__":
    main()
