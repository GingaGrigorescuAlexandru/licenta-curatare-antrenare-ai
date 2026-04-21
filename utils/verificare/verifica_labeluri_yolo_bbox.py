"""
Verifica daca label-urile unui dataset sunt in format YOLO Bounding Box.

Raportul include:
- statistica pe split-uri (train/valid/test)
- tipuri de fisiere label (bbox_only, segmentation_only, mixed, invalid, empty)
- statistica pe linii (cate linii bbox/segmentare/invalide)
- exemple de erori

Utilizare:
    python data/verificare/verifica_labeluri_yolo_bbox.py --dataset date/ecology_dataset
    python data/verificare/verifica_labeluri_yolo_bbox.py --dataset date/ecology_dataset --json-out data/verificare/raport_labels.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

SPLITS = ("train", "valid", "test")


@dataclass
class LineParseResult:
    kind: str  # bbox | segmentation | invalid
    reason: str = ""


@dataclass
class SplitStats:
    split: str
    total_label_files: int = 0
    bbox_only_files: int = 0
    segmentation_only_files: int = 0
    mixed_bbox_segmentation_files: int = 0
    invalid_files: int = 0
    empty_files: int = 0
    total_lines: int = 0
    bbox_lines: int = 0
    segmentation_lines: int = 0
    invalid_lines: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit label-uri pentru format YOLO bounding box."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Folderul dataset (ex: date/ecology_dataset).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        help="Split-uri de analizat (implicit: train valid test).",
    )
    parser.add_argument(
        "--max-error-examples",
        type=int,
        default=20,
        help="Cate exemple de erori sa afiseze (implicit 20).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional: salveaza raportul in JSON la calea data.",
    )
    return parser.parse_args()


def is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def is_non_negative_int(token: str) -> bool:
    try:
        value = int(token)
        return value >= 0 and str(value) == token
    except ValueError:
        return False


def parse_label_line(line: str) -> LineParseResult:
    parts = line.strip().split()
    if not parts:
        return LineParseResult(kind="invalid", reason="linie goala")

    if not is_non_negative_int(parts[0]):
        return LineParseResult(kind="invalid", reason="class_id invalid (nu e int >= 0)")

    if len(parts) == 5:
        if not all(is_float(tok) for tok in parts[1:]):
            return LineParseResult(kind="invalid", reason="bbox cu coordonate non-numerice")

        x_center, y_center, width, height = (float(tok) for tok in parts[1:])
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            return LineParseResult(kind="invalid", reason="bbox centru in afara intervalului [0,1]")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            return LineParseResult(kind="invalid", reason="bbox width/height in afara (0,1]")
        return LineParseResult(kind="bbox")

    # Segmentare YOLO: class_id + perechi (x,y), deci numar total impar >= 7
    if len(parts) >= 7 and len(parts) % 2 == 1:
        if not all(is_float(tok) for tok in parts[1:]):
            return LineParseResult(kind="invalid", reason="segmentare cu coordonate non-numerice")
        coords = [float(tok) for tok in parts[1:]]
        if any(c < 0.0 or c > 1.0 for c in coords):
            return LineParseResult(kind="invalid", reason="segmentare cu coordonate in afara [0,1]")
        return LineParseResult(kind="segmentation")

    return LineParseResult(
        kind="invalid",
        reason="numar de valori incompatibil cu bbox(5) sau segmentare(>=7, impar)",
    )


def classify_file(lines: List[str]) -> str:
    if not lines:
        return "empty"

    has_bbox = False
    has_seg = False
    has_invalid = False

    for ln in lines:
        result = parse_label_line(ln)
        if result.kind == "bbox":
            has_bbox = True
        elif result.kind == "segmentation":
            has_seg = True
        else:
            has_invalid = True

    if has_invalid:
        return "invalid"
    if has_bbox and has_seg:
        return "mixed_bbox_segmentation"
    if has_bbox:
        return "bbox_only"
    if has_seg:
        return "segmentation_only"
    return "empty"


def read_non_empty_lines(label_path: Path) -> List[str]:
    raw = label_path.read_text(encoding="utf-8", errors="replace")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def analyze_split(
    dataset_root: Path, split: str, max_error_examples: int
) -> Tuple[SplitStats, List[Dict[str, str]]]:
    stats = SplitStats(split=split)
    error_examples: List[Dict[str, str]] = []

    labels_dir = dataset_root / split / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Lipseste folderul: {labels_dir}")

    label_files = sorted(labels_dir.glob("*.txt"))
    stats.total_label_files = len(label_files)

    for label_file in tqdm(
        label_files,
        desc=f"Verific label-uri [{split}]",
        unit="fisier",
        leave=True,
    ):
        lines = read_non_empty_lines(label_file)
        ftype = classify_file(lines)

        if ftype == "bbox_only":
            stats.bbox_only_files += 1
        elif ftype == "segmentation_only":
            stats.segmentation_only_files += 1
        elif ftype == "mixed_bbox_segmentation":
            stats.mixed_bbox_segmentation_files += 1
        elif ftype == "invalid":
            stats.invalid_files += 1
        else:
            stats.empty_files += 1

        for idx, line in enumerate(lines, start=1):
            parsed = parse_label_line(line)
            stats.total_lines += 1
            if parsed.kind == "bbox":
                stats.bbox_lines += 1
            elif parsed.kind == "segmentation":
                stats.segmentation_lines += 1
            else:
                stats.invalid_lines += 1
                if len(error_examples) < max_error_examples:
                    error_examples.append(
                        {
                            "split": split,
                            "file": str(label_file),
                            "line_no": str(idx),
                            "reason": parsed.reason,
                            "line": line[:200],
                        }
                    )

    return stats, error_examples


def print_split_stats(stats: SplitStats) -> None:
    print(f"\n[{stats.split}]")
    print(f"  label_files_total: {stats.total_label_files}")
    print(f"  bbox_only_files: {stats.bbox_only_files}")
    print(f"  segmentation_only_files: {stats.segmentation_only_files}")
    print(f"  mixed_bbox_segmentation_files: {stats.mixed_bbox_segmentation_files}")
    print(f"  invalid_files: {stats.invalid_files}")
    print(f"  empty_files: {stats.empty_files}")
    print(f"  total_lines: {stats.total_lines}")
    print(f"  bbox_lines: {stats.bbox_lines}")
    print(f"  segmentation_lines: {stats.segmentation_lines}")
    print(f"  invalid_lines: {stats.invalid_lines}")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset inexistent: {dataset_root}")

    print(f"Dataset: {dataset_root}")
    print(f"Split-uri: {args.splits}")

    split_stats: List[SplitStats] = []
    all_error_examples: List[Dict[str, str]] = []

    for split in args.splits:
        stats, errors = analyze_split(
            dataset_root=dataset_root,
            split=split,
            max_error_examples=max(0, args.max_error_examples - len(all_error_examples)),
        )
        split_stats.append(stats)
        all_error_examples.extend(errors)
        print_split_stats(stats)

    total = SplitStats(split="TOTAL")
    for s in split_stats:
        total.total_label_files += s.total_label_files
        total.bbox_only_files += s.bbox_only_files
        total.segmentation_only_files += s.segmentation_only_files
        total.mixed_bbox_segmentation_files += s.mixed_bbox_segmentation_files
        total.invalid_files += s.invalid_files
        total.empty_files += s.empty_files
        total.total_lines += s.total_lines
        total.bbox_lines += s.bbox_lines
        total.segmentation_lines += s.segmentation_lines
        total.invalid_lines += s.invalid_lines

    print("\n[TOTAL]")
    print(f"  label_files_total: {total.total_label_files}")
    print(f"  bbox_only_files: {total.bbox_only_files}")
    print(f"  segmentation_only_files: {total.segmentation_only_files}")
    print(f"  mixed_bbox_segmentation_files: {total.mixed_bbox_segmentation_files}")
    print(f"  invalid_files: {total.invalid_files}")
    print(f"  empty_files: {total.empty_files}")
    print(f"  total_lines: {total.total_lines}")
    print(f"  bbox_lines: {total.bbox_lines}")
    print(f"  segmentation_lines: {total.segmentation_lines}")
    print(f"  invalid_lines: {total.invalid_lines}")

    if all_error_examples:
        print("\n[EXEMPLE ERORI]")
        for err in all_error_examples:
            print(
                f"  - split={err['split']} file={err['file']} line={err['line_no']} "
                f"reason={err['reason']} text='{err['line']}'"
            )
    else:
        print("\nNu au fost detectate linii invalide.")

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "dataset": str(dataset_root),
            "splits": args.splits,
            "stats": [asdict(s) for s in split_stats],
            "total": asdict(total),
            "error_examples": all_error_examples,
        }
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nRaport JSON salvat: {out_path}")


if __name__ == "__main__":
    main()
