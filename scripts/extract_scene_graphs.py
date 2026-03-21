from __future__ import annotations

import argparse
from pathlib import Path

import torch


DEFAULT_DATASET_NAMES = ["referit", "gref", "gref_umd", "unc", "unc+"]
DATASET_ALIASES = {
    "all": DEFAULT_DATASET_NAMES,
    "refcoco": ["unc"],
    "refcoco+": ["unc+"],
    "refcocog": ["gref_umd"],
    "referitgame": ["referit"],
}


def norm(p: str) -> str:
    return " ".join(p.lower().strip().split())


def construct_aug_queries(orig_phrase: str, v: dict | None) -> list[str]:
    base = orig_phrase.strip()
    augs: list[str] = []

    if not v:
        return [base, base, base]

    nodes = v.get("nodes", [])
    edges = v.get("edges", [])

    node_map = {n["id"]: n for n in nodes}
    root = node_map.get(0, {})

    for edge in edges:
        if edge.get("subject_id") == 0:
            obj_id = edge.get("object_id")
            if obj_id in node_map:
                obj_node = node_map[obj_id]
                rel = edge.get("relation_name", "").strip()
                obj_name = obj_node.get("name", "").strip()
                attrs = obj_node.get("attributes", [])
                attr = attrs[0].strip() if attrs else ""

                parts = [base, rel, attr, obj_name]
                aug = " ".join([p for p in parts if p != ""])
                if aug not in augs:
                    augs.append(aug)

    if len(augs) < 3 and "attributes" in root:
        for attr in root["attributes"]:
            if attr.strip():
                aug = f"{base} with {attr.strip()}"
                if aug not in augs:
                    augs.append(aug)

    while len(augs) < 3:
        augs.append(base)

    return augs[:3]


def resolve_dataset_names(dataset: str | None) -> list[str]:
    if not dataset:
        return list(DEFAULT_DATASET_NAMES)

    normalized = dataset.strip().lower()
    if normalized.startswith("dataset="):
        normalized = normalized.split("=", 1)[1].strip()

    if normalized in DATASET_ALIASES:
        return list(DATASET_ALIASES[normalized])

    return [normalized]


def generate_augmented_datasets(dataset_names: list[str] | None = None, base_dir: Path | None = None) -> list[Path]:
    dataset_names = dataset_names or list(DEFAULT_DATASET_NAMES)
    base_dir = base_dir or Path("data/annotations")
    written_paths: list[Path] = []

    for ds_name in dataset_names:
        ds_dir = base_dir / ds_name
        if not ds_dir.exists():
            continue

        pth_files = [
            f
            for f in ds_dir.glob(f"{ds_name}_*.pth")
            if "_graphs_" not in f.name and "_aug" not in f.name and "corpus" not in f.name
        ]

        for df in pth_files:
            split = df.name.replace(f"{ds_name}_", "").replace(".pth", "")
            graph_file = ds_dir / f"{ds_name}_graphs_{split}.pth"

            if not graph_file.exists():
                print(f"[{split}] MISSING GRAPH FILE: {graph_file.name}")
                continue

            print(f"[{ds_name} - {split}] Processing...")
            ds = torch.load(df, weights_only=False)
            gs = torch.load(graph_file, weights_only=False)

            gs_index: dict[tuple[str, str], dict] = {}
            for _, v in gs.items():
                if "metadata" not in v:
                    continue
                meta = v["metadata"]
                img = meta.get("image", "")
                phrase = norm(meta.get("phrase", ""))
                mask_id = str(meta.get("mask_id", ""))

                if img and phrase:
                    gs_index[(img, phrase)] = v
                if img and mask_id:
                    gs_index[(img, mask_id)] = v

            aug_dataset = []

            for item in ds:
                img = item[0]
                mask_id = item[1].split(".")[0] if isinstance(item[1], str) else str(item[1])

                phrase_idx = -1
                actual_phrase = ""
                for i, el in enumerate(item):
                    if isinstance(el, str) and not el.endswith(".jpg") and not el.endswith(".pth") and len(el.split()) > 0:
                        actual_phrase = el
                        phrase_idx = i
                        break

                n_phrase = norm(actual_phrase)
                v = None
                if (img, n_phrase) in gs_index:
                    v = gs_index[(img, n_phrase)]
                elif (img, mask_id) in gs_index:
                    v = gs_index[(img, mask_id)]

                augs = construct_aug_queries(actual_phrase, v)

                new_item = list(item)
                if phrase_idx != -1:
                    new_item[phrase_idx] = augs

                aug_dataset.append(tuple(new_item))

            out_path = ds_dir / f"{ds_name}_{split}_aug.pth"
            torch.save(aug_dataset, out_path)
            print(f"  -> Saved {len(aug_dataset)} aug items -> {out_path.name}")
            written_paths.append(out_path)

    return written_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate scene-graph augmented phrase caches.")
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
