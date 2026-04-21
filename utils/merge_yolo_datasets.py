"""
Script pentru combinarea mai multor dataset-uri YOLO intr-unul singur.

Ce face:
- combina split-urile train/valid/test
- combina imaginile si label-urile
- remapeaza corect id-urile claselor in label-uri, pe baza `names` din fiecare data.yaml
- genereaza un `data.yaml` nou pentru dataset-ul rezultat

Utilizare:
    python utils/merge_yolo_datasets.py \
        --datasets "date/glass merge" "date/metal merge" "date/paper merge" "date/plastic-merge-6-v2" \
        --output-name "merge-all-4" \
        --output-root "date"
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetInfo:
    source_dir: Path
    slug: str
    class_names: List[str]
    class_map: Dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combina mai multe dataset-uri YOLO.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Lista de foldere dataset sursa (fiecare trebuie sa contina data.yaml).",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Numele folderului dataset-ului rezultat.",
    )
    parser.add_argument(
        "--output-root",
        default="date",
        help="Folder parinte pentru dataset-ul rezultat (implicit: date).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sterge output-ul existent daca exista deja.",
    )
    return parser.parse_args()


def read_class_names(data_yaml_path: Path) -> List[str]:
    names_line = None
    with data_yaml_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("names:"):
                names_line = stripped.split(":", 1)[1].strip()
                break

    if not names_line:
        raise ValueError(f"Lipseste cheia 'names' in {data_yaml_path}")

    # Roboflow exporteaza de obicei lista inline: names: ['a', 'b', ...]
    try:
        parsed = ast.literal_eval(names_line)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Nu pot interpreta 'names' din {data_yaml_path}: {names_line}"
        ) from exc

    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(parsed, dict):
        # Suport pentru forma {0: class_a, 1: class_b, ...}
        return [str(v) for k, v in sorted(parsed.items(), key=lambda item: int(item[0]))]

    raise ValueError(f"Format invalid pentru 'names' in {data_yaml_path}")


def sanitize_slug(path: Path) -> str:
    return path.name.strip().replace(" ", "_")


def validate_dataset_structure(dataset_path: Path) -> None:
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Lipseste {data_yaml}")

    for split in SPLITS:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"
        if not images_dir.exists():
            raise FileNotFoundError(f"Lipseste folderul {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Lipseste folderul {labels_dir}")


def build_datasets_info(dataset_paths: List[Path]) -> tuple[List[DatasetInfo], List[str]]:
    global_names: List[str] = []
    datasets_info: List[DatasetInfo] = []

    for dataset_path in dataset_paths:
        validate_dataset_structure(dataset_path)

        class_names = read_class_names(dataset_path / "data.yaml")

        for cname in class_names:
            if cname not in global_names:
                global_names.append(cname)

        class_map: Dict[int, int] = {}
        for old_id, cname in enumerate(class_names):
            class_map[old_id] = global_names.index(cname)

        datasets_info.append(
            DatasetInfo(
                source_dir=dataset_path,
                slug=sanitize_slug(dataset_path),
                class_names=class_names,
                class_map=class_map,
            )
        )

    return datasets_info, global_names


def remap_label_file(src_label: Path, dst_label: Path, class_map: Dict[int, int]) -> int:
    converted_lines: List[str] = []
    object_count = 0

    with src_label.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            old_id = int(parts[0])
            if old_id not in class_map:
                raise ValueError(f"Clasa {old_id} nu exista in maparea pentru {src_label}")
            parts[0] = str(class_map[old_id])
            converted_lines.append(" ".join(parts))
            object_count += 1

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    with dst_label.open("w", encoding="utf-8") as f:
        if converted_lines:
            f.write("\n".join(converted_lines) + "\n")

    return object_count


def unique_target_stem(images_dst_dir: Path, candidate_stem: str, ext: str) -> str:
    stem = candidate_stem
    counter = 1
    while (images_dst_dir / f"{stem}{ext}").exists():
        stem = f"{candidate_stem}__{counter}"
        counter += 1
    return stem


def safe_target_stem(images_dst_dir: Path, candidate_stem: str, ext: str) -> str:
    """
    Returneaza un stem sigur pentru Windows (evita path-uri prea lungi).
    """
    # Lasam o marja pentru a evita erori pe sisteme fara long paths activate.
    max_full_path_len = 240
    full_candidate = images_dst_dir / f"{candidate_stem}{ext}"
    if len(str(full_candidate)) <= max_full_path_len:
        return unique_target_stem(images_dst_dir, candidate_stem, ext)

    digest = hashlib.sha1(candidate_stem.encode("utf-8")).hexdigest()[:12]

    reserve = len(str(images_dst_dir)) + len(ext) + len("__h_") + len(digest) + 2
    max_stem_len = max(24, max_full_path_len - reserve)
    truncated = candidate_stem[:max_stem_len]
    short_candidate = f"{truncated}__h_{digest}"

    return unique_target_stem(images_dst_dir, short_candidate, ext)


def merge_split(dataset: DatasetInfo, output_dir: Path, split: str) -> tuple[int, int, int]:
    images_src = dataset.source_dir / split / "images"
    labels_src = dataset.source_dir / split / "labels"

    images_dst = output_dir / split / "images"
    labels_dst = output_dir / split / "labels"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_labels = 0
    remapped_objects = 0

    for image_path in sorted(images_src.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        candidate = f"{dataset.slug}__{image_path.stem}"
        final_stem = safe_target_stem(images_dst, candidate, image_path.suffix)
        image_dst_path = images_dst / f"{final_stem}{image_path.suffix}"
        shutil.copy2(image_path, image_dst_path)
        copied_images += 1

        src_label_path = labels_src / f"{image_path.stem}.txt"
        dst_label_path = labels_dst / f"{final_stem}.txt"

        if src_label_path.exists():
            remapped_objects += remap_label_file(
                src_label=src_label_path,
                dst_label=dst_label_path,
                class_map=dataset.class_map,
            )
            copied_labels += 1
        else:
            # Imagine fara label -> cream fisier gol ca sa pastram perechea.
            dst_label_path.touch()
            copied_labels += 1

    return copied_images, copied_labels, remapped_objects


def write_output_yaml(output_dir: Path, class_names: List[str]) -> None:
    lines = [
        "train: ../train/images",
        "val: ../valid/images",
        "test: ../test/images",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
        "",
    ]
    (output_dir / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    dataset_paths = [Path(p).resolve() for p in args.datasets]
    output_root = Path(args.output_root).resolve()
    output_dir = output_root / args.output_name

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Folderul de iesire exista deja: {output_dir}\n"
                "Foloseste alt --output-name sau adauga --overwrite."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_info, global_class_names = build_datasets_info(dataset_paths)

    total_images = 0
    total_labels = 0
    total_objects = 0

    for dataset in datasets_info:
        print(f"[INFO] Procesez dataset: {dataset.source_dir}")
        for split in SPLITS:
            imgs, lbls, objs = merge_split(dataset, output_dir, split)
            total_images += imgs
            total_labels += lbls
            total_objects += objs
            print(
                f"  - {split}: imagini={imgs}, labels={lbls}, obiecte_remapate={objs}"
            )

    write_output_yaml(output_dir, global_class_names)

    print("\n[OK] Merge complet.")
    print(f"Output: {output_dir}")
    print(f"Clase finale ({len(global_class_names)}): {global_class_names}")
    print(
        f"Total copiate: imagini={total_images}, labels={total_labels}, obiecte_remapate={total_objects}"
    )


if __name__ == "__main__":
    main()
