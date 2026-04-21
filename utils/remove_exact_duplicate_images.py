"""
Curata un dataset YOLO de imagini duplicate exacte (la nivel de pixeli).

Reguli:
- considera duplicate doar imaginile cu pixeli identici dupa decodare
- NU elimina imagini "aproape identice" (augmentari usoare, compresie diferita etc.)
- cand sterge o imagine duplicata, sterge si label-ul asociat (.txt)

Structura asteptata:
    <dataset_root>/
      train/images, train/labels
      valid/images, valid/labels
      test/images,  test/labels

Exemplu:
    python utils/remove_exact_duplicate_images.py --dataset date/ecology_dataset
    python utils/remove_exact_duplicate_images.py --dataset date/ecology_dataset --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm


DEFAULT_SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DuplicateEntry:
    duplicate_image: Path
    kept_image: Path
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sterge duplicate exacte la nivel de pixeli dintr-un dataset YOLO."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Calea catre folderul dataset (care contine train/valid/test).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split-uri procesate (implicit: train valid test).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Doar afiseaza ce ar sterge, fara sa stearga efectiv fisiere.",
    )
    return parser.parse_args()


def iter_images(images_dir: Path) -> Iterable[Path]:
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def pixel_hash(image_path: Path) -> str:
    """
    Hash pe pixeli efectivi:
    - aplica EXIF orientation
    - converteste in RGB
    - include dimensiunea in hash
    """
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        payload = img.tobytes()

    h = hashlib.sha256()
    h.update(f"{img.width}x{img.height}|RGB|".encode("utf-8"))
    h.update(payload)
    return h.hexdigest()


def validate_structure(dataset_root: Path, splits: List[str]) -> None:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset inexistent: {dataset_root}")

    for split in splits:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        if not images_dir.exists():
            raise FileNotFoundError(f"Lipseste folderul: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Lipseste folderul: {labels_dir}")


def find_duplicates(
    dataset_root: Path, splits: List[str]
) -> Tuple[List[DuplicateEntry], Dict[str, int], int]:
    hash_to_kept_image: Dict[str, Path] = {}
    duplicates: List[DuplicateEntry] = []
    skipped_invalid = 0
    processed_per_split: Dict[str, int] = {s: 0 for s in splits}

    for split in splits:
        images_dir = dataset_root / split / "images"
        imagini = list(iter_images(images_dir))
        for img_path in tqdm(
            imagini,
            desc=f"Analizez pixeli [{split}]",
            unit="img",
            leave=True,
        ):
            processed_per_split[split] += 1
            try:
                ph = pixel_hash(img_path)
            except (UnidentifiedImageError, OSError):
                skipped_invalid += 1
                print(f"[WARN] Imagine invalida/corupta, sar peste: {img_path}")
                continue

            kept = hash_to_kept_image.get(ph)
            if kept is None:
                hash_to_kept_image[ph] = img_path
                continue

            duplicates.append(
                DuplicateEntry(
                    duplicate_image=img_path,
                    kept_image=kept,
                    split=split,
                )
            )

    return duplicates, processed_per_split, skipped_invalid


def delete_duplicate_pair(dataset_root: Path, dup_image_path: Path, dry_run: bool) -> Tuple[bool, bool]:
    split = dup_image_path.parent.parent.name
    labels_dir = dataset_root / split / "labels"
    label_path = labels_dir / f"{dup_image_path.stem}.txt"

    removed_image = False
    removed_label = False

    if dry_run:
        return dup_image_path.exists(), label_path.exists()

    if dup_image_path.exists():
        dup_image_path.unlink()
        removed_image = True

    if label_path.exists():
        label_path.unlink()
        removed_label = True

    return removed_image, removed_label


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset).resolve()
    splits = args.splits

    validate_structure(dataset_root, splits)

    print(f"[INFO] Dataset: {dataset_root}")
    print(f"[INFO] Split-uri: {splits}")
    print(f"[INFO] Mod: {'DRY-RUN' if args.dry_run else 'STERGERE'}")

    duplicates, processed_per_split, skipped_invalid = find_duplicates(dataset_root, splits)

    removed_images = 0
    removed_labels = 0
    removed_per_split: Dict[str, int] = {s: 0 for s in splits}

    for d in tqdm(
        duplicates,
        desc="Sterg duplicate (imagine + label)",
        unit="fisier",
        leave=True,
    ):
        img_removed, lbl_removed = delete_duplicate_pair(
            dataset_root=dataset_root,
            dup_image_path=d.duplicate_image,
            dry_run=args.dry_run,
        )
        if img_removed:
            removed_images += 1
            removed_per_split[d.split] += 1
        if lbl_removed:
            removed_labels += 1

    print("\n[REZULTAT]")
    for split in splits:
        print(
            f"  - {split}: procesate={processed_per_split[split]}, "
            f"duplicate_gasite={removed_per_split[split]}"
        )
    print(f"  - total_duplicate_imagini: {len(duplicates)}")
    print(f"  - imagini_{'de_sters' if args.dry_run else 'sterse'}: {removed_images}")
    print(f"  - labeluri_{'de_sters' if args.dry_run else 'sterse'}: {removed_labels}")
    print(f"  - imagini_invalide_sarite: {skipped_invalid}")

    if duplicates:
        print("\n[EXEMPLE DUPLICATE]")
        for d in duplicates[:20]:
            print(f"  DUP:  {d.duplicate_image}")
            print(f"  KEEP: {d.kept_image}")
    else:
        print("\nNu au fost gasite duplicate exacte la nivel de pixeli.")


if __name__ == "__main__":
    main()
