"""Move YOLO labels drawn on legacy PNGs (LAS, stretched to 512) onto a pipeline output's geometry.

Label files are ``<case_id>__slice_0013.txt`` with ``class cx cy w h`` lines. Each case's
target geometry is read from ``<png-dir>/cases/<case_id>/case.json``. Boxes are clipped to
the canvas; boxes left with no area are dropped and listed in ``conversion_report.csv``.

Example
-------
python -m modules.convert_labels \
    --labels-dir /Users/yschoi/code/09_FHV/yolo/FHV_bbox_full_v1/labels/train/images \
    --png-dir /Volumes/GURI_1/20261029_FLAIR_PAIK_yolo_png \
    --dest /Volumes/GURI_1/20261029_FLAIR_PAIK_yolo_png/labels_FHV_bbox_full_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from modules import preprocess as pp


def legacy_geometry(geometry: dict, size: int = 512) -> dict:
    """Geometry of the legacy PNGs (every slice stretched to ``size``) for the same scan."""
    return {**geometry, "resampled_xy": [size, size], "offset_xy": [0, 0], "canvas": size}


def convert_labels(labels_dir: Path, png_dir: Path, dest: Path, legacy_size: int = 512) -> dict:
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty; choose a new folder")
    dest.mkdir(parents=True, exist_ok=True)
    report, geometries = [], {}
    counts = {"files": 0, "boxes": 0, "clipped": 0, "dropped": 0, "missing_case": 0}
    for label_path in sorted(labels_dir.glob("*__slice_*.txt")):
        case_id = label_path.name.split("__slice_")[0]
        if case_id not in geometries:
            marker = png_dir / "cases" / case_id / "case.json"
            geometries[case_id] = json.loads(marker.read_text())["geometry"] if marker.exists() else None
        target = geometries[case_id]
        if target is None:
            counts["missing_case"] += 1
            report.append({"file": label_path.name, "line": "", "status": "case_not_in_png_dir"})
            continue
        source = legacy_geometry(target, legacy_size)
        lines = []
        for i, line in enumerate(label_path.read_text().split("\n")):
            if not line.strip():
                continue
            cls, *box = line.split()
            box = tuple(float(v) for v in box)
            counts["boxes"] += 1
            moved = pp.convert_box(box, source, target)
            if moved is None:
                counts["dropped"] += 1
                report.append({"file": label_path.name, "line": i + 1, "status": "dropped_outside_canvas"})
                continue
            # A box that lost area to the canvas edge (brain-centred crop) is kept but reported.
            exact_area = box[2] * box[3] * (target["resampled_xy"][0] / legacy_size) * (target["resampled_xy"][1] / legacy_size)
            if moved[2] * moved[3] < exact_area * 0.999:
                counts["clipped"] += 1
                report.append({"file": label_path.name, "line": i + 1, "status": "clipped_at_canvas_edge"})
            lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in moved))
        (dest / label_path.name).write_text("\n".join(lines) + ("\n" if lines else ""))
        counts["files"] += 1
    with (dest / "conversion_report.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "line", "status"])
        writer.writeheader()
        writer.writerows(report)
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels-dir", required=True, help="Folder of legacy YOLO label .txt files.")
    parser.add_argument("--png-dir", required=True, help="modules.pipeline --out-dir holding cases/*/case.json.")
    parser.add_argument("--dest", required=True, help="New, empty folder for converted labels.")
    parser.add_argument("--legacy-size", type=int, default=512)
    args = parser.parse_args(argv)
    counts = convert_labels(Path(args.labels_dir), Path(args.png_dir), Path(args.dest), args.legacy_size)
    print(f"[INFO] {counts} -> {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
