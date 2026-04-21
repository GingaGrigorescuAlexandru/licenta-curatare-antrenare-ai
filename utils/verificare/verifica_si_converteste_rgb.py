"""
Verifica si converteste imaginile la RGB.

Scenariu tipic (dataset YOLO):
    python utils/verificare/verifica_si_converteste_rgb.py --dataset date/ecology_dataset --dry-run
    python utils/verificare/verifica_si_converteste_rgb.py --dataset date/ecology_dataset
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm


SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifica si converteste imaginile la RGB.")
    parser.add_argument("--dataset", required=True, help="Calea dataset-ului (train/valid/test).")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), help="Split-uri procesate.")
    parser.add_argument("--dry-run", action="store_true", help="Doar raport, fara modificari.")
    return parser.parse_args()


def iter_images(images_dir: Path) -> Iterable[Path]:
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def convert_to_rgb_in_place(image_path: Path) -> bool:
    """
    Returneaza True daca imaginea a fost modificata, altfel False.
    """
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode == "RGB":
            return False

        if "A" in img.getbands():
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.getchannel("A"))
            out = bg
        else:
            out = img.convert("RGB")

        save_kwargs = {}
        ext = image_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 95
            save_kwargs["optimize"] = True
        out.save(image_path, **save_kwargs)
        return True


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset inexistent: {dataset}")

    total_files = 0
    total_converted = 0
    total_invalid = 0
    mode_counter_before: Counter[str] = Counter()

    print(f"Dataset: {dataset}")
    print(f"Split-uri: {args.splits}")
    print(f"Mod: {'DRY-RUN' if args.dry_run else 'CONVERSIE'}")

    for split in args.splits:
        images_dir = dataset / split / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Lipseste folderul: {images_dir}")

        split_files = list(iter_images(images_dir))
        split_total = len(split_files)
        split_converted = 0
        split_invalid = 0

        for img_path in tqdm(split_files, desc=f"RGB [{split}]", unit="img", leave=True):
            total_files += 1
            try:
                with Image.open(img_path) as img:
                    mode_counter_before[img.mode] += 1
                    needs_convert = img.mode != "RGB"
            except (UnidentifiedImageError, OSError):
                split_invalid += 1
                total_invalid += 1
                continue

            if needs_convert:
                if not args.dry_run and convert_to_rgb_in_place(img_path):
                    split_converted += 1
                elif args.dry_run:
                    split_converted += 1

        total_converted += split_converted
        print(
            f"[{split}] imagini={split_total}, non_rgb={'de_convertit' if args.dry_run else 'convertite'}={split_converted}, invalide={split_invalid}"
        )

    print("\n[REZULTAT]")
    print(f"total_imagini: {total_files}")
    print(f"total_non_rgb_{'de_convertit' if args.dry_run else 'convertite'}: {total_converted}")
    print(f"total_imagini_invalide: {total_invalid}")
    print(f"distributie_moduri_initiale: {dict(mode_counter_before)}")


if __name__ == "__main__":
    main()

