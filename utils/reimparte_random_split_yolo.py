"""
Reimparte random un dataset YOLO in train/valid/test dupa procente dorite.

Caracteristici:
- muta perechile imagine + label impreuna
- random real (shuffle), nu alfabetic
- reproductibil cu --seed
- optional dry-run pentru verificare fara modificari

Exemplu:
    python utils/reimparte_random_split_yolo.py \
      --dataset date/ecology_dataset \
      --train 0.8 --valid 0.1 --test 0.1 \
      --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tqdm import tqdm


SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class PairItem:
    image_path: Path
    label_path: Path
    stem: str
    ext: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reimparte random un dataset YOLO in train/valid/test."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Folderul datasetului (care contine train/valid/test).",
    )
    parser.add_argument("--train", type=float, required=True, help="Procent train (ex: 0.8).")
    parser.add_argument("--valid", type=float, required=True, help="Procent valid (ex: 0.1).")
    parser.add_argument("--test", type=float, required=True, help="Procent test (ex: 0.1).")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pentru random shuffle (implicit 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afiseaza distributia fara mutari de fisiere.",
    )
    return parser.parse_args()


def validate_splits(dataset: Path) -> None:
    for split in SPLITS:
        images_dir = dataset / split / "images"
        labels_dir = dataset / split / "labels"
        if not images_dir.exists():
            raise FileNotFoundError(f"Lipseste folderul: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Lipseste folderul: {labels_dir}")


def collect_pairs(dataset: Path) -> List[PairItem]:
    items: List[PairItem] = []
    for split in SPLITS:
        images_dir = dataset / split / "images"
        labels_dir = dataset / split / "labels"

        for img in sorted(images_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = labels_dir / f"{img.stem}.txt"
            if not lbl.exists():
                # Daca lipseste label-ul, cream unul gol ca sa pastram perechea corecta.
                lbl.touch()
            items.append(
                PairItem(
                    image_path=img,
                    label_path=lbl,
                    stem=img.stem,
                    ext=img.suffix.lower(),
                )
            )
    if not items:
        raise ValueError("Nu am gasit imagini in dataset.")
    return items


def build_counts(total: int, p_train: float, p_valid: float, p_test: float) -> tuple[int, int, int]:
    s = p_train + p_valid + p_test
    if abs(s - 1.0) > 1e-9:
        raise ValueError(
            f"Procentele trebuie sa aiba suma 1.0 (acum {s:.10f})."
        )

    train_n = int(total * p_train)
    valid_n = int(total * p_valid)
    test_n = total - train_n - valid_n
    return train_n, valid_n, test_n


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def unique_dest_image(images_dir: Path, stem: str, ext: str) -> Path:
    candidate = images_dir / f"{stem}{ext}"
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        p = images_dir / f"{stem}__rs{idx}{ext}"
        if not p.exists():
            return p
        idx += 1


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()

    if not dataset.exists():
        raise FileNotFoundError(f"Dataset inexistent: {dataset}")
    validate_splits(dataset)

    items = collect_pairs(dataset)
    total = len(items)
    train_n, valid_n, test_n = build_counts(total, args.train, args.valid, args.test)

    rng = random.Random(args.seed)
    rng.shuffle(items)

    train_items = items[:train_n]
    valid_items = items[train_n : train_n + valid_n]
    test_items = items[train_n + valid_n :]

    print(f"Dataset: {dataset}")
    print(f"Total imagini: {total}")
    print(f"Target split: train={args.train:.4f}, valid={args.valid:.4f}, test={args.test:.4f}")
    print(f"Target count: train={len(train_items)}, valid={len(valid_items)}, test={len(test_items)}")
    print(f"Seed: {args.seed}")
    print(f"Mod: {'DRY-RUN' if args.dry_run else 'MUTARE'}")

    if args.dry_run:
        return

    # Mutam intai toate perechile intr-un pool temporar ca sa evitam conflicte la mutari.
    tmp_pool = dataset / "__tmp_resplit_pool__"
    ensure_clean_dir(tmp_pool)
    pool_images = tmp_pool / "images"
    pool_labels = tmp_pool / "labels"
    pool_images.mkdir(parents=True, exist_ok=True)
    pool_labels.mkdir(parents=True, exist_ok=True)

    for idx, it in enumerate(
        tqdm(items, desc="Mut in pool temporar", unit="fisier", leave=True), start=1
    ):
        tmp_stem = f"item_{idx:07d}"
        tmp_img = pool_images / f"{tmp_stem}{it.ext}"
        tmp_lbl = pool_labels / f"{tmp_stem}.txt"
        shutil.move(str(it.image_path), str(tmp_img))
        shutil.move(str(it.label_path), str(tmp_lbl))
        it.image_path = tmp_img
        it.label_path = tmp_lbl

    # Golim folderele split curente.
    for split in SPLITS:
        images_dir = dataset / split / "images"
        labels_dir = dataset / split / "labels"
        for p in images_dir.iterdir():
            if p.is_file():
                p.unlink()
        for p in labels_dir.iterdir():
            if p.is_file():
                p.unlink()

    def place_group(group: List[PairItem], split: str) -> None:
        images_dir = dataset / split / "images"
        labels_dir = dataset / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for it in tqdm(group, desc=f"Plasez {split}", unit="fisier", leave=True):
            dst_img = unique_dest_image(images_dir, it.stem, it.ext)
            dst_lbl = labels_dir / f"{dst_img.stem}.txt"
            shutil.move(str(it.image_path), str(dst_img))
            shutil.move(str(it.label_path), str(dst_lbl))

    place_group(train_items, "train")
    place_group(valid_items, "valid")
    place_group(test_items, "test")

    shutil.rmtree(tmp_pool, ignore_errors=True)

    print("\n[OK] Reimpartire finalizata.")
    print(f"train={len(train_items)}, valid={len(valid_items)}, test={len(test_items)}")


if __name__ == "__main__":
    main()

