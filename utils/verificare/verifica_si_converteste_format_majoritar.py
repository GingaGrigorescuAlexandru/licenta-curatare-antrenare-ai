"""
Verifica formatul imaginilor si converteste totul in formatul majoritar.

Exemple:
    python utils/verificare/verifica_si_converteste_format_majoritar.py --dataset date/ecology_dataset --dry-run
    python utils/verificare/verifica_si_converteste_format_majoritar.py --dataset date/ecology_dataset
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm


SPLITS = ("train", "valid", "test")
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
NORM_EXT = {
    ".jpg": ".jpg",
    ".jpeg": ".jpg",
    ".png": ".png",
    ".bmp": ".bmp",
    ".tif": ".tif",
    ".tiff": ".tif",
    ".webp": ".webp",
}
PIL_FORMAT = {
    ".jpg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".webp": "WEBP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Converteste imaginile la formatul majoritar.")
    parser.add_argument("--dataset", required=True, help="Calea dataset-ului (train/valid/test).")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), help="Split-uri procesate.")
    parser.add_argument("--dry-run", action="store_true", help="Doar raport, fara modificari.")
    parser.add_argument(
        "--target-format",
        default="",
        help="Optional: forteaza formatul tinta (jpg/png/bmp/tif/webp).",
    )
    return parser.parse_args()


def iter_images(images_dir: Path) -> Iterable[Path]:
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def normalize_ext(ext: str) -> str:
    low = ext.lower()
    if low not in NORM_EXT:
        raise ValueError(f"Extensie nesuportata: {ext}")
    return NORM_EXT[low]


def choose_target_format(ext_counter: Counter[str], forced: str) -> str:
    if forced:
        f = forced.lower().lstrip(".")
        if f not in {"jpg", "png", "bmp", "tif", "webp"}:
            raise ValueError("--target-format trebuie sa fie unul dintre: jpg/png/bmp/tif/webp")
        return f".{f}"

    if not ext_counter:
        raise ValueError("Nu am gasit imagini pentru a determina formatul majoritar.")
    return ext_counter.most_common(1)[0][0]


def unique_path(candidate: Path) -> Path:
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        p = candidate.with_name(f"{candidate.stem}__fmt{idx}{candidate.suffix}")
        if not p.exists():
            return p
        idx += 1


def convert_image_format(src: Path, dst: Path, target_ext: str) -> None:
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)
        out = img

        # JPEG nu suporta alpha; compunem pe fundal alb.
        if target_ext == ".jpg":
            if "A" in out.getbands():
                bg = Image.new("RGB", out.size, (255, 255, 255))
                bg.paste(out, mask=out.getchannel("A"))
                out = bg
            else:
                out = out.convert("RGB")

        save_kwargs = {"format": PIL_FORMAT[target_ext]}
        if target_ext == ".jpg":
            save_kwargs["quality"] = 95
            save_kwargs["optimize"] = True
        out.save(dst, **save_kwargs)


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset inexistent: {dataset}")

    all_images: list[Path] = []
    ext_counter: Counter[str] = Counter()
    invalid_files = 0

    for split in args.splits:
        images_dir = dataset / split / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Lipseste folderul: {images_dir}")
        for img in iter_images(images_dir):
            all_images.append(img)
            ext_counter[normalize_ext(img.suffix)] += 1

    target_ext = choose_target_format(ext_counter, args.target_format)
    to_convert = [p for p in all_images if normalize_ext(p.suffix) != target_ext]

    print(f"Dataset: {dataset}")
    print(f"Split-uri: {args.splits}")
    print(f"Distributie formate initiale: {dict(ext_counter)}")
    print(f"Format tinta: {target_ext}")
    print(f"Imagini de {'convertit' if args.dry_run else 'convertite'}: {len(to_convert)}")
    print(f"Mod: {'DRY-RUN' if args.dry_run else 'CONVERSIE'}")

    converted = 0
    removed_old = 0

    for src in tqdm(to_convert, desc="Conversie format", unit="img", leave=True):
        if args.dry_run:
            converted += 1
            continue

        dst_candidate = src.with_suffix(target_ext)
        dst = unique_path(dst_candidate)
        try:
            convert_image_format(src=src, dst=dst, target_ext=target_ext)
        except (UnidentifiedImageError, OSError):
            invalid_files += 1
            if dst.exists():
                dst.unlink(missing_ok=True)
            continue

        if src.exists():
            src.unlink()
            removed_old += 1
        converted += 1

    print("\n[REZULTAT]")
    print(f"total_imagini: {len(all_images)}")
    print(f"format_tinta: {target_ext}")
    print(f"imagini_{'de_convertit' if args.dry_run else 'convertite'}: {converted}")
    if not args.dry_run:
        print(f"fisiere_vechi_sterse: {removed_old}")
    print(f"imagini_invalide_sarite: {invalid_files}")


if __name__ == "__main__":
    main()

