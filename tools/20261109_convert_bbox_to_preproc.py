"""Map FHV bbox annotations made on the legacy PAIK PNGs onto the current pipeline's PNG canvas, with QC overlays.

The legacy export resized each LAS slice straight to 512x512; the current pipeline resamples to 0.41 mm/px
and pads or crops to 512 (``case.json`` -> ``geometry``). Both keep the slice order and use
``flipud(slice.T)``, so a box maps per axis by a scale and a shift, in pixel-edge coordinates:

    col_new = col_old * rx / 512 + ox
    row_new = 512 - ((512 - row_old) * ry / 512 + oy)

Inputs are only read. The output folder must not exist yet; images are never copied, only labels and
overlays are written.

usage:
    python tools/20261109_convert_bbox_to_preproc.py \
        --legacy-png-dir <yolo/positive_slices_all/images> \
        --yolo-labels <yolo/FHV_bbox_full_v1/labels/train/images> --yolo-yaml <.../data.yaml> \
        --cvat-dir <yolo/FHV_bbox_full_v1_uhk100percent> \
        --preproc-dir <.../20261030_FLAIR_preproc> --out-dir <new folder>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

N = 512
CLASSES = ("FHV", "uncertain", "YHK_reject", "YHK_new")
COLORS_BGR = {"FHV": (40, 40, 230), "uncertain": (0, 215, 255), "YHK_reject": (170, 170, 170), "YHK_new": (60, 200, 60)}
HEADER = 54
LEGACY_STEM = re.compile(r"(?P<case>.+)__slice_(?P<slice>\d{4})")


def new_stem(legacy_stem: str) -> tuple[str, str]:
    m = LEGACY_STEM.fullmatch(legacy_stem)
    if not m:
        raise ValueError(f"unexpected legacy name: {legacy_stem}")
    return m["case"], f"{m['case']}_slice_{m['slice']}"


def edge_map(geometry: dict) -> tuple[float, float, float, float]:
    """(sc, tc, sr, tr) so that col_new = sc*col_old + tc and row_new = sr*row_old + tr (pixel edges)."""
    rx, ry = geometry["resampled_xy"]
    ox, oy = geometry["offset_xy"]
    return rx / N, float(ox), ry / N, float(N - ry - oy)


def map_box(box, m):
    sc, tc, sr, tr = m
    x0, y0, x1, y1 = box
    return sc * x0 + tc, sr * y0 + tr, sc * x1 + tc, sr * y1 + tr


def clip_box(box):
    x0, y0, x1, y1 = (float(np.clip(v, 0, N)) for v in box)
    return (x0, y0, x1, y1), (x0, y0, x1, y1) != tuple(box)


def same_geometry(geometry: dict) -> bool:
    return list(geometry["resampled_xy"]) == [N, N] and list(geometry["offset_xy"]) == [0, 0]


def warp_legacy(img: np.ndarray, m) -> np.ndarray:
    sc, tc, sr, tr = m
    # pixel centres: c_new = s * c_old + 0.5 * s + t - 0.5
    M = np.float32([[sc, 0, 0.5 * sc + tc - 0.5], [0, sr, 0.5 * sr + tr - 0.5]])
    return cv2.warpAffine(img, M, (N, N), flags=cv2.INTER_LINEAR)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- readers (legacy pixel-edge coordinates)

def read_yolo(label_dir: Path, names: list[str]) -> dict[str, list[tuple[str, tuple]]]:
    out = {}
    for f in sorted(label_dir.glob("*.txt")):
        if f.name.startswith("."):
            continue
        boxes = []
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = (float(v) * N for v in (cx, cy, w, h))
            boxes.append((names[int(c)], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        out[f.stem] = boxes
    return out


def read_cvat(cvat_dir: Path):
    lines = [json.loads(l) for l in (cvat_dir / "data" / "manifest.jsonl").read_text().splitlines() if l.strip()]
    frames = [l for l in lines if "name" in l]
    stems = [Path(fr["name"]).name for fr in frames]
    doc = json.loads((cvat_dir / "annotations.json").read_text())
    if len(doc) != 1 or doc[0]["tags"] or doc[0]["tracks"]:
        raise ValueError("expected one job with rectangle shapes only")
    out = {s: [] for s in stems}
    for s in doc[0]["shapes"]:
        if s["type"] != "rectangle" or s["rotation"]:
            raise ValueError(f"unsupported shape: {s['type']} rotation={s['rotation']}")
        out[stems[s["frame"]]].append((s["label"], tuple(s["points"])))
    return out, doc, lines


# ---------------------------------------------------------------- overlay

def draw_box(img, box, label):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    color = COLORS_BGR.get(label, (255, 0, 255))
    if label == "YHK_reject":  # dashed
        for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            length = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
            for t in range(0, length, 6):
                p = (a[0] + (b[0] - a[0]) * t // max(length, 1), a[1] + (b[1] - a[1]) * t // max(length, 1))
                q_t = min(t + 3, length)
                q = (a[0] + (b[0] - a[0]) * q_t // max(length, 1), a[1] + (b[1] - a[1]) * q_t // max(length, 1))
                cv2.line(img, p, q, color, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)


def overlay(legacy_img, new_img, legacy_boxes, new_boxes, title_lines):
    panels = []
    for img, boxes, caption in ((legacy_img, legacy_boxes, "legacy PNG + original boxes"),
                                (new_img, new_boxes, "current PNG + converted boxes")):
        p = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for label, box in boxes:
            draw_box(p, box, label)
        cv2.putText(p, caption, (6, N - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(p)
    body = np.hstack([panels[0], np.full((N, 4, 3), 255, np.uint8), panels[1]])
    head = np.zeros((HEADER, body.shape[1], 3), np.uint8)
    for i, text in enumerate(title_lines):
        cv2.putText(head, text, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    x = body.shape[1] - 360
    for label in CLASSES:
        cv2.rectangle(head, (x, 8), (x + 12, 20), COLORS_BGR[label], -1)
        cv2.putText(head, label, (x + 16, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        x += 16 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0] + 14
    return np.vstack([head, body])


def process_slice(job: dict) -> dict:
    """Build both overlays and QC numbers for one slice."""
    legacy = cv2.imread(job["legacy_png"], cv2.IMREAD_GRAYSCALE)
    new = cv2.imread(job["new_png"], cv2.IMREAD_GRAYSCALE)
    if legacy is None or new is None:
        raise FileNotFoundError(job["legacy_png"] if legacy is None else job["new_png"])
    m = tuple(job["map"])
    warped = warp_legacy(legacy, m)
    row = {"case_id": job["case_id"], "png": job["new_name"], "same_geometry": job["same"],
           "iou_as_is": round(iou(legacy > 0, new > 0), 4), "iou_mapped": round(iou(warped > 0, new > 0), 4)}
    g = job["geometry"]
    geo_text = "same as legacy" if job["same"] else f"scale {g['resampled_xy'][0]}/512, {g['resampled_xy'][1]}/512 offset {tuple(g['offset_xy'])}"
    for name, sets in job["sets"].items():
        legacy_boxes = [(l, tuple(b)) for l, b in sets["legacy"]]
        new_boxes = [(l, tuple(b)) for l, b in sets["new"]]
        shift = max((max(abs(a - b) for a, b in zip(lb, nb)) for (_, lb), (_, nb) in zip(legacy_boxes, new_boxes)), default=0.0)
        no_brain = 0
        for _, (x0, y0, x1, y1) in new_boxes:
            xa, xb, ya, yb = int(np.floor(x0)), int(np.ceil(x1)), int(np.floor(y0)), int(np.ceil(y1))
            no_brain += not (xb > xa and yb > ya and (new[ya:yb, xa:xb] > 0).any())
        row[f"{name}_boxes"] = len(new_boxes)
        row[f"{name}_max_shift_px"] = round(shift, 2)
        row[f"{name}_boxes_without_brain"] = no_brain
        title = [f"{job['new_name']}   [{name}]   boxes {len(new_boxes)}" + ("   (no boxes)" if not new_boxes else ""),
                 f"geometry: {geo_text}   brain IoU as-is {row['iou_as_is']:.3f} -> mapped {row['iou_mapped']:.3f}   max box shift {shift:.1f}px"]
        out = overlay(legacy, new, legacy_boxes, new_boxes, title)
        if not cv2.imwrite(str(Path(sets["overlay"])), out, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(sets["overlay"])
    return row


# ---------------------------------------------------------------- main

def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--legacy-png-dir", type=Path, required=True)
    ap.add_argument("--yolo-labels", type=Path, required=True)
    ap.add_argument("--yolo-yaml", type=Path, required=True)
    ap.add_argument("--cvat-dir", type=Path, required=True)
    ap.add_argument("--preproc-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit-cases", type=int, default=0, help="first N cases only (dry run)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if args.out_dir.exists():
        raise SystemExit(f"[ERROR] output exists, refusing to write: {args.out_dir}")

    names = []
    for line in args.yolo_yaml.read_text().splitlines():
        m = re.match(r"\s+(\d+):\s*(\S+)", line)
        if m:
            names.append(m[2])
    if tuple(names) != CLASSES:
        raise SystemExit(f"[ERROR] unexpected YOLO class names: {names}")

    yolo = read_yolo(args.yolo_labels, names)
    cvat, cvat_doc, cvat_lines = read_cvat(args.cvat_dir)
    if {s["label"] for s in cvat_doc[0]["shapes"]} - set(CLASSES):
        raise SystemExit("[ERROR] unexpected CVAT labels")
    legacy_pngs = {p.stem for p in args.legacy_png_dir.glob("*.png") if not p.name.startswith(".")}
    stems = sorted(legacy_pngs | set(yolo) | set(cvat))
    unknown = sorted((set(yolo) | set(cvat)) - legacy_pngs)
    if unknown:
        raise SystemExit(f"[ERROR] annotations without a legacy PNG: {unknown[:5]}")

    cases = sorted({new_stem(s)[0] for s in stems})
    if args.limit_cases:
        cases = cases[: args.limit_cases]
        stems = [s for s in stems if new_stem(s)[0] in cases]
    geometry = {}
    for c in cases:
        geometry[c] = json.loads((args.preproc_dir / "FLAIR_nii_p" / c / "case.json").read_text())["geometry"]

    sets = {"full_v1": yolo, "uhk100": cvat}
    for name in sets:
        (args.out_dir / name / "overlay").mkdir(parents=True)
    (args.out_dir / "full_v1" / "labels").mkdir()
    (args.out_dir / "uhk100" / "data").mkdir()

    jobs, box_rows, clipped = [], [], {n: 0 for n in sets}
    converted = {n: {} for n in sets}
    for s in stems:
        case, stem = new_stem(s)
        g = geometry[case]
        m = edge_map(g)
        job = {"case_id": case, "new_name": f"{stem}.png", "geometry": g, "same": same_geometry(g), "map": m,
               "legacy_png": str(args.legacy_png_dir / f"{s}.png"),
               "new_png": str(args.preproc_dir / "png" / "positive_slices" / f"{stem}.png"), "sets": {}}
        for name, source in sets.items():
            legacy_boxes = source.get(s, [])
            new_boxes = []
            for i, (label, box) in enumerate(legacy_boxes):
                nb, was_clipped = clip_box(map_box(box, m))
                clipped[name] += was_clipped
                new_boxes.append((label, nb))
                box_rows.append({"set": name, "case_id": case, "png": f"{stem}.png", "box_index": i, "label": label,
                                 **{f"legacy_{k}": round(v, 3) for k, v in zip(("x0", "y0", "x1", "y1"), box)},
                                 **{f"new_{k}": round(v, 3) for k, v in zip(("x0", "y0", "x1", "y1"), nb)},
                                 "clipped": was_clipped, "same_geometry": job["same"],
                                 "resampled_xy": "x".join(map(str, g["resampled_xy"])), "offset_xy": "x".join(map(str, g["offset_xy"]))})
            converted[name][s] = new_boxes
            job["sets"][name] = {"legacy": legacy_boxes, "new": new_boxes,
                                 "overlay": str(args.out_dir / name / "overlay" / f"{stem}.png")}
        jobs.append(job)

    # full_v1: YOLO labels for every slice that had a label file
    for s, boxes in converted["full_v1"].items():
        if s not in yolo:
            continue
        lines = []
        for label, (x0, y0, x1, y1) in boxes:
            lines.append(f"{CLASSES.index(label)} {(x0 + x1) / 2 / N:.6f} {(y0 + y1) / 2 / N:.6f} {(x1 - x0) / N:.6f} {(y1 - y0) / N:.6f}")
        (args.out_dir / "full_v1" / "labels" / f"{new_stem(s)[1]}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    (args.out_dir / "full_v1" / "data.yaml").write_text(args.yolo_yaml.read_text())

    # uhk100: CVAT 1.1 with renamed frames and converted points (frame order kept)
    keep = set(stems)
    frame_names = [Path(l["name"]).name for l in cvat_lines if "name" in l]
    doc = json.loads(json.dumps(cvat_doc))
    for shape in doc[0]["shapes"]:
        s = frame_names[shape["frame"]]
        shape["points"] = list(clip_box(map_box(tuple(shape["points"]), edge_map(geometry[new_stem(s)[0]])))[0]) if s in keep else shape["points"]
    if not args.limit_cases:
        (args.out_dir / "uhk100" / "annotations.json").write_text(json.dumps(doc))
        out_lines = []
        for l in cvat_lines:
            if "name" in l:
                l = {**l, "name": f"images/{new_stem(Path(l['name']).name)[1]}"}
            out_lines.append(json.dumps(l, separators=(",", ":")))
        (args.out_dir / "uhk100" / "data" / "manifest.jsonl").write_text("\n".join(out_lines) + "\n")
        (args.out_dir / "uhk100" / "task.json").write_text((args.cvat_dir / "task.json").read_text())

    with ProcessPoolExecutor(args.workers) as ex:
        qc = list(ex.map(process_slice, jobs, chunksize=8))

    for name in sets:
        rows = [r for r in box_rows if r["set"] == name]
        with open(args.out_dir / name / "labels_boxes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(box_rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(args.out_dir / "qc_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(qc[0]))
        w.writeheader()
        w.writerows(sorted(qc, key=lambda r: (r["same_geometry"], r["iou_mapped"])))

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name, "git_commit": git_commit(), "python": platform.python_version(),
        "transform": "col_new = col_old*rx/512 + ox; row_new = 512 - ((512-row_old)*ry/512 + oy); pixel-edge coords, (rx,ry)=resampled_xy, (ox,oy)=offset_xy from case.json; boxes clipped to [0,512]",
        "inputs": {
            "legacy_png_dir": str(args.legacy_png_dir), "preproc_dir": str(args.preproc_dir),
            "yolo_labels": str(args.yolo_labels), "yolo_yaml": str(args.yolo_yaml), "yolo_yaml_sha256": sha256(args.yolo_yaml),
            "cvat_dir": str(args.cvat_dir), "cvat_annotations_sha256": sha256(args.cvat_dir / "annotations.json"),
        },
        "limit_cases": args.limit_cases,
        "counts": {
            "cases": len(cases), "slices": len(stems),
            "same_geometry_cases": sum(same_geometry(geometry[c]) for c in cases),
            **{f"{n}_label_slices": sum(1 for s in stems if s in sets[n]) for n in sets},
            **{f"{n}_boxes": sum(len(sets[n].get(s, [])) for s in stems) for n in sets},
            **{f"{n}_boxes_clipped": clipped[n] for n in sets},
        },
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["counts"], indent=1))


if __name__ == "__main__":
    main()
