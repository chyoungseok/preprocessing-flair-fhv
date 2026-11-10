"""Map the UHK100 FHV bbox annotation (CVAT, legacy PAIK PNGs) onto the current pipeline's PNG canvas.

The legacy export resized each LAS slice straight to 512x512; the current pipeline resamples to 0.41 mm/px
and pads or crops to 512 (``case.json`` -> ``geometry``). Both keep the slice order and use
``flipud(slice.T)``, so a box maps per axis by a scale and a shift, in pixel-edge coordinates:

    col_new = col_old * rx / 512 + ox
    row_new = 512 - ((512 - row_old) * ry / 512 + oy)

Outputs the converted annotation as CVAT 1.1 and as YOLO txt (all four classes, CVAT label order), and a
QC overlay per slice whose converted boxes are read back from the YOLO files. Inputs are only read and images
are never copied. ``--labels-out-dir`` and ``--qc-out-dir`` must not exist yet.

usage:
    python tools/20261109_convert_bbox_to_preproc.py \
        --legacy-png-dir <yolo/positive_slices_all/images> \
        --cvat-dir <yolo/FHV_bbox_full_v1_uhk100percent> \
        --preproc-dir <.../20261030_FLAIR_preproc> \
        --labels-out-dir <new folder> --qc-out-dir <new folder>
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
GAP = 4
CHECKER = 32
NCC_PAD = 3
LEGACY_STEM = re.compile(r"(?P<case>.+)__slice_(?P<slice>\d{4})")
FONT = cv2.FONT_HERSHEY_SIMPLEX


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
    clipped = tuple(float(np.clip(v, 0, N)) for v in box)
    return clipped, clipped != tuple(box)


def same_geometry(geometry: dict) -> bool:
    return list(geometry["resampled_xy"]) == [N, N] and list(geometry["offset_xy"]) == [0, 0]


def warp_legacy(img: np.ndarray, m) -> np.ndarray:
    sc, tc, sr, tr = m
    # pixel centres: c_new = s * c_old + 0.5 * s + t - 0.5
    M = np.float32([[sc, 0, 0.5 * sc + tc - 0.5], [0, sr, 0.5 * sr + tr - 0.5]])
    return cv2.warpAffine(img, M, (N, N), flags=cv2.INTER_LINEAR)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def box_pixels(box, pad=0):
    x0, y0, x1, y1 = box
    return (max(0, int(np.floor(x0)) - pad), max(0, int(np.floor(y0)) - pad),
            min(N, int(np.ceil(x1)) + pad), min(N, int(np.ceil(y1)) + pad))


def box_ncc(a: np.ndarray, b: np.ndarray, box) -> float:
    """Normalized cross-correlation of two images inside a box (a few px of margin); NaN if flat."""
    xa, ya, xb, yb = box_pixels(box, NCC_PAD)
    pa, pb = a[ya:yb, xa:xb].astype(np.float64).ravel(), b[ya:yb, xa:xb].astype(np.float64).ravel()
    if pa.size < 4 or pa.std() == 0 or pb.std() == 0:
        return float("nan")
    return float(np.corrcoef(pa, pb)[0, 1])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- annotation I/O

def read_cvat(cvat_dir: Path):
    """CVAT 1.1 export -> ({legacy stem: [(label, (x0, y0, x1, y1)), ...]}, doc, manifest lines, frame stems)."""
    lines = [json.loads(l) for l in (cvat_dir / "data" / "manifest.jsonl").read_text().splitlines() if l.strip()]
    stems = [Path(l["name"]).name for l in lines if "name" in l]
    doc = json.loads((cvat_dir / "annotations.json").read_text())
    if len(doc) != 1 or doc[0]["tags"] or doc[0]["tracks"]:
        raise ValueError("expected one job with rectangle shapes only")
    out = {s: [] for s in stems}
    for s in doc[0]["shapes"]:
        if s["type"] != "rectangle" or s["rotation"]:
            raise ValueError(f"unsupported shape: {s['type']} rotation={s['rotation']}")
        out[stems[s["frame"]]].append((s["label"], tuple(s["points"])))
    return out, doc, lines, stems


def yolo_line(label: str, box) -> str:
    x0, y0, x1, y1 = box
    return f"{CLASSES.index(label)} {(x0 + x1) / 2 / N:.6f} {(y0 + y1) / 2 / N:.6f} {(x1 - x0) / N:.6f} {(y1 - y0) / N:.6f}"


def read_yolo_file(path: str) -> list[tuple[str, tuple]]:
    boxes = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = (float(v) * N for v in (cx, cy, w, h))
            boxes.append((CLASSES[int(c)], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return boxes


# ---------------------------------------------------------------- overlay

def draw_box(img, box, label):
    # pixel-edge box -> the pixels it covers: columns x0 .. x1-1
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x1, y1 = max(x0, x1 - 1), max(y0, y1 - 1)
    color = COLORS_BGR.get(label, (255, 0, 255))
    if label == "YHK_reject":  # dashed
        for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            length = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)
            for t in range(0, length, 6):
                u = min(t + 3, length)
                p = (a[0] + (b[0] - a[0]) * t // length, a[1] + (b[1] - a[1]) * t // length)
                q = (a[0] + (b[0] - a[0]) * u // length, a[1] + (b[1] - a[1]) * u // length)
                cv2.line(img, p, q, color, 1)
    else:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)


def checkerboard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:N, 0:N]
    return np.where(((yy // CHECKER) + (xx // CHECKER)) % 2 == 0, a, b)


def overlay(legacy_img, new_img, warped_img, legacy_boxes, new_boxes, title_lines):
    panels = []
    for img, boxes, caption in ((legacy_img, legacy_boxes, "legacy PNG + original CVAT boxes"),
                                (new_img, new_boxes, "current PNG + converted YOLO boxes"),
                                (checkerboard(warped_img, new_img), new_boxes, f"checker: legacy mapped / current ({CHECKER}px)")):
        p = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for label, box in boxes:
            draw_box(p, box, label)
        cv2.putText(p, caption, (6, N - 8), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(p)
    sep = np.full((N, GAP, 3), 255, np.uint8)
    body = np.hstack([panels[0], sep, panels[1], sep, panels[2]])
    head = np.zeros((HEADER, body.shape[1], 3), np.uint8)
    for i, text in enumerate(title_lines):
        cv2.putText(head, text, (6, 18 + 18 * i), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    x = body.shape[1] - 340
    for label in CLASSES:
        cv2.rectangle(head, (x, 8), (x + 12, 20), COLORS_BGR[label], -1)
        cv2.putText(head, label, (x + 16, 19), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        x += 16 + cv2.getTextSize(label, FONT, 0.4, 1)[0][0] + 14
    return np.vstack([head, body])


def process_slice(job: dict) -> dict:
    """Write the overlay for one slice (converted boxes read back from YOLO); return QC numbers."""
    legacy = cv2.imread(job["legacy_png"], cv2.IMREAD_GRAYSCALE)
    new = cv2.imread(job["new_png"], cv2.IMREAD_GRAYSCALE)
    if legacy is None or new is None:
        raise FileNotFoundError(job["legacy_png"] if legacy is None else job["new_png"])
    warped = warp_legacy(legacy, tuple(job["map"]))
    legacy_boxes = [(l, tuple(b)) for l, b in job["legacy"]]
    yolo_boxes = read_yolo_file(job["yolo"])
    if [l for l, _ in yolo_boxes] != [l for l, _ in legacy_boxes]:
        raise ValueError(f"YOLO classes differ from CVAT: {job['yolo']}")
    g = job["geometry"]
    row = {"case_id": job["case_id"], "png": job["new_name"], "same_geometry": job["same"],
           "resampled_xy": "x".join(map(str, g["resampled_xy"])), "offset_xy": "x".join(map(str, g["offset_xy"])),
           "iou_as_is": round(iou(legacy > 0, new > 0), 4), "iou_mapped": round(iou(warped > 0, new > 0), 4),
           "boxes": len(yolo_boxes)}
    shift = max((max(abs(a - b) for a, b in zip(lb, nb)) for (_, lb), (_, nb) in zip(legacy_boxes, yolo_boxes)), default=0.0)
    ncc = [box_ncc(warped, new, b) for _, b in yolo_boxes]
    no_brain = 0
    for _, b in yolo_boxes:
        xa, ya, xb, yb = box_pixels(b)
        no_brain += not (xb > xa and yb > ya and (new[ya:yb, xa:xb] > 0).any())
    valid = [v for v in ncc if v == v]
    row.update({"max_shift_px": round(shift, 2), "min_box_ncc": round(min(valid), 3) if valid else "",
                "boxes_without_brain": no_brain})
    geo_text = "same as legacy" if job["same"] else f"resampled {row['resampled_xy']} offset {row['offset_xy']}"
    counts = ", ".join(f"{c} {n}" for c in CLASSES if (n := sum(l == c for l, _ in yolo_boxes)))
    title = [f"{job['new_name']}   [uhk100]   " + (f"boxes {len(yolo_boxes)} ({counts})" if yolo_boxes else "no boxes"),
             f"geometry: {geo_text}   brain IoU as-is {row['iou_as_is']:.3f} -> mapped {row['iou_mapped']:.3f}"
             f"   max box shift {shift:.1f}px" + (f"   min box NCC {row['min_box_ncc']}" if valid else "")]
    if not cv2.imwrite(job["overlay"], overlay(legacy, new, warped, legacy_boxes, yolo_boxes, title), [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(job["overlay"])
    return {"row": row, "ncc": ncc, "stem": job["legacy_stem"]}


# ---------------------------------------------------------------- main

def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", Path(__file__).name], cwd=Path(__file__).parent,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + (" (script modified)" if dirty else "")
    except OSError:
        return ""


def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--legacy-png-dir", type=Path, required=True)
    ap.add_argument("--cvat-dir", type=Path, required=True)
    ap.add_argument("--preproc-dir", type=Path, required=True)
    ap.add_argument("--labels-out-dir", type=Path, required=True)
    ap.add_argument("--qc-out-dir", type=Path, required=True)
    ap.add_argument("--cases", nargs="*", default=[], help="only these case_ids (dry run)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    for d in (args.labels_out_dir, args.qc_out_dir):
        if d.exists():
            raise SystemExit(f"[ERROR] output exists, refusing to write: {d}")

    task = json.loads((args.cvat_dir / "task.json").read_text())
    if tuple(l["name"] for l in task["labels"]) != CLASSES:
        raise SystemExit(f"[ERROR] unexpected CVAT task labels: {[l['name'] for l in task['labels']]}")
    cvat, cvat_doc, cvat_lines, cvat_frames = read_cvat(args.cvat_dir)
    if {s["label"] for s in cvat_doc[0]["shapes"]} - set(CLASSES):
        raise SystemExit("[ERROR] unexpected CVAT labels")

    legacy_pngs = {p.stem for p in args.legacy_png_dir.glob("*.png") if not p.name.startswith(".")}
    if set(cvat_frames) != legacy_pngs:
        raise SystemExit(f"[ERROR] CVAT frames and legacy PNGs differ: {sorted(set(cvat_frames) ^ legacy_pngs)[:5]}")
    stems = sorted(cvat_frames)
    if args.cases:
        stems = [s for s in stems if new_stem(s)[0] in set(args.cases)]
    cases = sorted({new_stem(s)[0] for s in stems})
    geometry = {c: json.loads((args.preproc_dir / "FLAIR_nii_p" / c / "case.json").read_text())["geometry"] for c in cases}

    yolo_dir = args.labels_out_dir / "yolo" / "labels"
    yolo_dir.mkdir(parents=True)
    (args.labels_out_dir / "cvat" / "data").mkdir(parents=True)
    (args.qc_out_dir / "overlay").mkdir(parents=True)

    jobs, box_rows = [], []
    for s in stems:
        case, stem = new_stem(s)
        g = geometry[case]
        m = edge_map(g)
        new_png = args.preproc_dir / "png" / "positive_slices" / f"{stem}.png"
        if not new_png.exists():
            raise SystemExit(f"[ERROR] current PNG missing: {new_png}")
        lines = []
        for i, (label, box) in enumerate(cvat[s]):
            nb, was_clipped = clip_box(map_box(box, m))
            lines.append(yolo_line(label, nb))
            box_rows.append({"case_id": case, "png": f"{stem}.png", "box_index": i, "label": label,
                             **{f"legacy_{k}": round(v, 3) for k, v in zip(("x0", "y0", "x1", "y1"), box)},
                             **{f"new_{k}": round(v, 3) for k, v in zip(("x0", "y0", "x1", "y1"), nb)},
                             "clipped": was_clipped, "degenerate": nb[2] - nb[0] <= 0 or nb[3] - nb[1] <= 0,
                             "same_geometry": same_geometry(g),
                             "resampled_xy": "x".join(map(str, g["resampled_xy"])), "offset_xy": "x".join(map(str, g["offset_xy"])),
                             "box_ncc": None, "legacy_stem": s})
        yolo_path = yolo_dir / f"{stem}.txt"
        yolo_path.write_text("".join(f"{l}\n" for l in lines))
        jobs.append({"case_id": case, "legacy_stem": s, "new_name": f"{stem}.png", "geometry": g, "same": same_geometry(g),
                     "map": m, "legacy_png": str(args.legacy_png_dir / f"{s}.png"), "new_png": str(new_png),
                     "legacy": cvat[s], "yolo": str(yolo_path), "overlay": str(args.qc_out_dir / "overlay" / f"{stem}.png")})
    (args.labels_out_dir / "yolo" / "data.yaml").write_text(
        "# classes in CVAT task label order (uhk100)\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))

    # CVAT 1.1 with renamed frames and converted points; frame order and shape order kept
    doc = json.loads(json.dumps(cvat_doc))
    kept = set(stems)
    doc[0]["shapes"] = [sh for sh in doc[0]["shapes"] if cvat_frames[sh["frame"]] in kept]
    frame_index = {s: i for i, s in enumerate(f for f in cvat_frames if f in kept)}
    for sh in doc[0]["shapes"]:
        s = cvat_frames[sh["frame"]]
        sh["points"] = list(clip_box(map_box(tuple(sh["points"]), edge_map(geometry[new_stem(s)[0]])))[0])
        sh["frame"] = frame_index[s]
    (args.labels_out_dir / "cvat" / "annotations.json").write_text(json.dumps(doc))
    out_lines = [json.dumps(l, separators=(",", ":")) for l in cvat_lines if "name" not in l]
    out_lines += [json.dumps({**l, "name": f"images/{new_stem(Path(l['name']).name)[1]}"}, separators=(",", ":"))
                  for l in cvat_lines if "name" in l and Path(l["name"]).name in kept]
    (args.labels_out_dir / "cvat" / "data" / "manifest.jsonl").write_text("\n".join(out_lines) + "\n")
    (args.labels_out_dir / "cvat" / "task.json").write_text((args.cvat_dir / "task.json").read_text())

    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(process_slice, jobs, chunksize=8))

    ncc = {r["stem"]: r["ncc"] for r in results}
    for b in box_rows:
        v = ncc[b.pop("legacy_stem")][b["box_index"]]
        b["box_ncc"] = round(v, 4) if v == v else ""
    write_csv(args.labels_out_dir / "labels_boxes.csv", box_rows)
    write_csv(args.qc_out_dir / "qc_summary.csv", sorted((r["row"] for r in results), key=lambda r: (r["same_geometry"], r["iou_mapped"])))
    write_csv(args.qc_out_dir / "qc_boxes.csv", sorted(box_rows, key=lambda b: (b["box_ncc"] == "", b["box_ncc"] if b["box_ncc"] != "" else 0)))

    counts = {
        "cases": len(cases), "slices": len(stems), "same_geometry_cases": sum(same_geometry(geometry[c]) for c in cases),
        "slices_without_boxes": sum(not cvat[s] for s in stems), "boxes": len(box_rows),
        **{f"boxes_{c}": sum(b["label"] == c for b in box_rows) for c in CLASSES},
        "boxes_clipped": sum(b["clipped"] for b in box_rows), "boxes_degenerate": sum(b["degenerate"] for b in box_rows),
    }
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name, "git_commit": git_commit(), "python": platform.python_version(),
        "transform": "col_new = col_old*rx/512 + ox; row_new = 512 - ((512-row_old)*ry/512 + oy); pixel-edge coords, "
                     "(rx,ry)=resampled_xy, (ox,oy)=offset_xy from case.json; boxes clipped to [0,512]",
        "yolo_classes": dict(enumerate(CLASSES)),
        "inputs": {"legacy_png_dir": str(args.legacy_png_dir), "preproc_dir": str(args.preproc_dir), "cvat_dir": str(args.cvat_dir),
                   "cvat_annotations_sha256": sha256(args.cvat_dir / "annotations.json")},
        "labels_out_dir": str(args.labels_out_dir), "qc_out_dir": str(args.qc_out_dir),
        "cases_filter": args.cases, "counts": counts,
    }
    for d in (args.labels_out_dir, args.qc_out_dir):
        (d / "run_metadata.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
