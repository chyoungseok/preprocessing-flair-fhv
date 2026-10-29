"""FLAIR NIfTI -> YOLO PNG preprocessing pipeline.

Per case: SynthStrip -> N4 (in brain mask) -> mask -> reorient (LAS) ->
per-slice 1-99 percentile normalization -> 512x512 resize -> 8-bit PNG.

Output layout (``--out-dir``)::

    positive_slices_all/images/<case_id>__slice_0013.png, manifest.csv
    negative_slices_all/images/<case_id>__slice_0001.png, manifest.csv
    cases/<case_id>/   SynthStrip brain/mask NIfTI, QC PNG, case.json (done marker)
    cases.csv          per-case status (done / failed / unlabeled / missing_input)
    run_metadata.json  settings and package versions
    logs/errors.txt

Only cases present in the label table are exported. A slice listed in
``fhv_slice_num`` is positive and every other slice of that case is negative.

Example
-------
python -m modules.pipeline \
    --src-dir /Volumes/GURI_1/FLAIR_PAIK_nii \
    --out-dir /Volumes/GURI_1/20261029_FLAIR_PAIK_yolo_png \
    --label-xlsx /Users/yschoi/code/09_FHV/csv_files/df_flair_label_UHKinspection.xlsx
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np

from modules import preprocess as pp
from modules.labels import DEFAULT_LABEL_COLUMN, CaseLabel, load_label_table


PIPELINE_NAME = "modules.pipeline"
CLASS_DIRS = {1: "positive_slices_all", 0: "negative_slices_all"}
MANIFEST_COLUMNS = (
    "case_id", "label", "slice_index_1based", "png_relative_path", "label_table_row",
    "fhv_slice_num_source", "source_nifti", "shape_xyz", "zooms_xyz",
)
# Settings that change pixel values; resuming into an output made with other values is refused.
SETTING_KEYS = (
    "skullstrip_backend", "synthstrip_image", "orientation", "normalization",
    "size_policy", "target_size", "label_column",
)


def case_id_from_path(path: Path) -> str:
    name = path.name
    stem = name[:-7] if name.endswith(".nii.gz") else path.stem
    return stem.split("_FLAIR", 1)[0]


def find_inputs(src_dir: Path, pattern: str) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for path in sorted(src_dir.rglob(pattern)):
        if not path.is_file() or path.name.startswith("."):
            continue
        case_id = case_id_from_path(path)
        if case_id in inputs:
            raise ValueError(f"duplicate input for case {case_id}: {inputs[case_id]} and {path}")
        inputs[case_id] = path
    return inputs


def _display(slice_xy: np.ndarray) -> np.ndarray:
    return np.flipud(slice_xy.T)


def save_qc(path: Path, raw_xyz: np.ndarray, mask_xyz: np.ndarray, corrected_xyz: np.ndarray, pixels: np.ndarray, z: int, title: str) -> None:
    """Raw slice with brain-mask contour | N4-corrected masked slice | exported PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = _display(raw_xyz[:, :, z])
    vmax = np.percentile(raw, 99.5) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    axes[0].imshow(raw, cmap="gray", vmin=0, vmax=vmax)
    axes[0].contour(_display(mask_xyz[:, :, z]).astype(float), levels=[0.5], colors="lime", linewidths=0.8)
    axes[0].set_title("input + SynthStrip mask")
    corrected = _display(corrected_xyz[:, :, z])
    axes[1].imshow(corrected, cmap="gray", vmin=0, vmax=np.percentile(corrected, 99.5) or 1.0)
    axes[1].set_title("N4 + mask")
    axes[2].imshow(pixels, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"PNG {pixels.shape[1]}x{pixels.shape[0]}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def skullstrip_paths(case_dir: Path, case_id: str) -> tuple[Path, Path]:
    return case_dir / f"{case_id}_brain.nii.gz", case_dir / f"{case_id}_brainMask.nii.gz"


def needs_skullstrip(case_dir: Path, case_id: str, settings: dict) -> bool:
    """SynthStrip outputs are reused, even with --overwrite, unless missing or made by another image."""
    if not all(p.exists() for p in skullstrip_paths(case_dir, case_id)):
        return True
    marker = case_dir / "case.json"
    if marker.exists():
        previous = json.loads(marker.read_text())["settings"]
        return any(previous[k] != settings[k] for k in ("skullstrip_backend", "synthstrip_image"))
    return False


def process_case(input_path: Path, label: CaseLabel, out_dir: Path, settings: dict, overwrite: bool, qc: bool) -> dict:
    """Run one case end to end and return its ``case.json`` content."""
    import cv2
    import nibabel as nib
    from nibabel.orientations import aff2axcodes

    case_id = label.case_id
    case_dir = out_dir / "cases" / case_id
    done_marker = case_dir / "case.json"
    if done_marker.exists() and not overwrite:
        return json.loads(done_marker.read_text())
    case_dir.mkdir(parents=True, exist_ok=True)

    brain_path, mask_path = skullstrip_paths(case_dir, case_id)
    if needs_skullstrip(case_dir, case_id, settings):
        with _SKULLSTRIP_SLOTS or nullcontext():
            pp.SKULLSTRIP_BACKENDS[settings["skullstrip_backend"]](input_path, brain_path, mask_path, settings)

    corrected, mask = pp.n4_bias_correct(brain_path, mask_path)
    corrected *= mask
    source = nib.load(str(input_path))
    affine = source.affine
    volume, _ = pp.reorient(corrected, affine, settings["orientation"])
    n_slices = volume.shape[2]
    positives = set(label.positive_slices_1based)
    out_of_range = sorted(z for z in positives if z > n_slices)
    if out_of_range:
        raise ValueError(f"label slices {out_of_range} exceed slice count {n_slices}")

    normalize = pp.NORMALIZATION_POLICIES[settings["normalization"]]
    standardize_size = pp.SIZE_POLICIES[settings["size_policy"]]
    rows = []
    pixels_by_z = {}
    for z in range(n_slices):
        slice_1based = z + 1
        cls = int(slice_1based in positives)
        pixels = pp.to_png_pixels(standardize_size(normalize(volume[:, :, z]), settings["target_size"]))
        filename = f"{case_id}__slice_{slice_1based:04d}.png"
        relative = Path(CLASS_DIRS[cls]) / "images" / filename
        # A label edit between runs can move a slice to the other class folder.
        (out_dir / CLASS_DIRS[1 - cls] / "images" / filename).unlink(missing_ok=True)
        if not cv2.imwrite(str(out_dir / relative), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f"failed to write {relative}")
        pixels_by_z[z] = pixels
        rows.append({
            "case_id": case_id,
            "label": cls,
            "slice_index_1based": slice_1based,
            "png_relative_path": str(relative),
            "label_table_row": label.table_row,
            "fhv_slice_num_source": label.source_value,
            "source_nifti": str(input_path),
            "shape_xyz": "x".join(str(n) for n in source.shape[:3]),
            "zooms_xyz": "x".join(f"{float(n):g}" for n in source.header.get_zooms()[:3]),
        })

    if qc:
        qc_z = min(positives) - 1 if positives else n_slices // 2
        raw, _ = pp.reorient(np.asanyarray(source.dataobj, dtype=np.float32), affine, settings["orientation"])
        mask_oriented, _ = pp.reorient(mask, affine, settings["orientation"])
        save_qc(case_dir / f"{case_id}_qc.png", raw, mask_oriented, volume, pixels_by_z[qc_z], qc_z,
                f"{case_id}  slice {qc_z + 1}/{n_slices}  {'FHV+' if positives else ''}")

    record = {
        "case_id": case_id,
        "source_nifti": str(input_path),
        "axcodes_input": "".join(aff2axcodes(affine)),
        "axcodes_output": settings["orientation"],
        "n_slices": n_slices,
        "n_positive": len(positives),
        "settings": {key: settings[key] for key in SETTING_KEYS},
        "rows": rows,
    }
    done_marker.write_text(json.dumps(record, indent=1))
    return record


# Cross-process semaphore limiting concurrent SynthStrip runs; None means unlimited.
_SKULLSTRIP_SLOTS = None


def _init_worker(threads: int, skullstrip_slots=None) -> None:
    import cv2
    import SimpleITK as sitk

    global _SKULLSTRIP_SLOTS
    _SKULLSTRIP_SLOTS = skullstrip_slots
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    cv2.setNumThreads(1)


def _run_case(args: tuple) -> tuple[str, dict | None, str | None]:
    label = args[1]
    try:
        return label.case_id, process_case(*args), None
    except Exception:
        return label.case_id, None, traceback.format_exc()


def _write_csv(path: Path, columns, rows) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _package_versions() -> dict:
    import cv2
    import nibabel
    import SimpleITK

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "nibabel": nibabel.__version__,
        "SimpleITK": SimpleITK.Version_VersionString(),
        "opencv": cv2.__version__,
    }


def check_output_dir(out_dir: Path, settings: dict, overwrite: bool) -> None:
    """Refuse to mix outputs from a different tool or different pixel settings."""
    metadata_path = out_dir / "run_metadata.json"
    existing = [out_dir / d for d in CLASS_DIRS.values() if (out_dir / d).exists()]
    if not metadata_path.exists():
        if existing:
            raise FileExistsError(f"{out_dir} already has class folders not made by {PIPELINE_NAME}; use a new --out-dir")
        return
    previous = json.loads(metadata_path.read_text())
    changed = {k: (previous.get("settings", {}).get(k), settings[k]) for k in SETTING_KEYS
               if previous.get("settings", {}).get(k) != settings[k]}
    if changed and not overwrite:
        raise ValueError(f"{out_dir} was made with different settings {changed}; use a new --out-dir or --overwrite")


def run(args: argparse.Namespace) -> int:
    src_dir, out_dir = Path(args.src_dir).expanduser(), Path(args.out_dir).expanduser()
    settings = {
        "skullstrip_backend": args.skullstrip_backend,
        "synthstrip_image": args.synthstrip_image,
        "docker_platform": args.platform,
        "synthstrip_python": args.synthstrip_python,
        "orientation": args.orientation,
        "normalization": args.normalization,
        "size_policy": args.size_policy,
        "target_size": args.target_size,
        "label_column": args.label_column,
    }
    check_output_dir(out_dir, settings, args.overwrite)

    labels = load_label_table(args.label_xlsx, args.label_column)
    inputs = find_inputs(src_dir, args.pattern)
    status_rows = [{"case_id": c, "status": "unlabeled", "detail": str(p)} for c, p in inputs.items() if c not in labels]
    status_rows += [{"case_id": c, "status": "missing_input", "detail": ""} for c in labels if c not in inputs]

    case_ids = sorted(c for c in inputs if c in labels)
    if args.pid:
        case_ids = [c for c in case_ids if c in set(args.pid)]
    if args.limit is not None:
        case_ids = case_ids[: args.limit]
    print(f"[INFO] inputs={len(inputs)} labelled={len(labels)} to_process={len(case_ids)} workers={args.workers}")

    for class_dir in CLASS_DIRS.values():
        (out_dir / class_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    skullstrip_slots = args.workers
    if any(needs_skullstrip(out_dir / "cases" / c, c, settings) for c in case_ids):
        if args.skullstrip_backend == "docker":
            pp.check_docker_available()
            skullstrip_slots = min(args.workers, pp.docker_parallel_slots())
        else:
            if not settings["synthstrip_python"]:
                raise ValueError("native backend needs the fhv_synthstrip conda env (requirements-synthstrip.txt), "
                                 "--synthstrip-python or FHV_SYNTHSTRIP_PYTHON; or use --skullstrip-backend docker")
            pp.native_synthstrip_home(args.synthstrip_image, args.platform)  # extract once, before workers start
            skullstrip_slots = min(args.workers, pp.SYNTHSTRIP_NATIVE_SLOTS)
        print(f"[INFO] concurrent SynthStrip runs: {skullstrip_slots}")

    metadata = {
        "pipeline": PIPELINE_NAME,
        "started": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "src_dir": str(src_dir),
        "label_table": str(Path(args.label_xlsx).expanduser()),
        "settings": settings,
        "fixed_parameters": {
            "n4_shrink_xyz": list(pp.N4_SHRINK_XYZ),
            "n4_max_iterations": list(pp.N4_MAX_ITERATIONS),
            "display_transform": pp.DISPLAY_TRANSFORM,
            "png": "8-bit grayscale, round(clip(x, 0, 1) * 255)",
        },
        "versions": _package_versions(),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    tasks = [(inputs[c], labels[c], out_dir, settings, args.overwrite, not args.no_qc) for c in case_ids]
    records: dict[str, dict] = {}
    started = time.time()
    finished = 0

    def collect(case_id: str, record: dict | None, error: str | None) -> None:
        nonlocal finished
        finished += 1
        if record is not None:
            records[case_id] = record
            status_rows.append({"case_id": case_id, "status": "done", "detail": f"slices={record['n_slices']} positive={record['n_positive']}"})
        else:
            status_rows.append({"case_id": case_id, "status": "failed", "detail": error.strip().splitlines()[-1]})
            with (out_dir / "logs" / "errors.txt").open("a") as f:
                f.write(f"{'=' * 60}\n{datetime.now().isoformat(timespec='seconds')} {case_id} {inputs[case_id]}\n{error}\n")
        print(f"[{finished}/{len(tasks)}] {case_id}: {'done' if record else 'FAILED'} ({time.time() - started:.0f}s)")

    if args.workers <= 1:
        _init_worker(os.cpu_count() or 1)
        for task in tasks:
            collect(*_run_case(task))
    else:
        threads = max(1, (os.cpu_count() or 1) // args.workers)
        with multiprocessing.Manager() as manager:
            slots = manager.Semaphore(skullstrip_slots) if skullstrip_slots < args.workers else None
            with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(threads, slots)) as pool:
                for future in as_completed([pool.submit(_run_case, task) for task in tasks]):
                    collect(*future.result())

    # Manifests cover every completed case in this output, including earlier runs.
    reported = {row["case_id"] for row in status_rows}
    for marker in sorted((out_dir / "cases").glob("*/case.json")):
        case_id = marker.parent.name
        record = records.setdefault(case_id, json.loads(marker.read_text()))
        if case_id not in reported:
            status_rows.append({"case_id": case_id, "status": "done", "detail": f"slices={record['n_slices']} positive={record['n_positive']}"})
    for cls, class_dir in CLASS_DIRS.items():
        rows = [row for c in sorted(records) for row in records[c]["rows"] if row["label"] == cls]
        _write_csv(out_dir / class_dir / "manifest.csv", MANIFEST_COLUMNS, rows)
    _write_csv(out_dir / "cases.csv", ("case_id", "status", "detail"), sorted(status_rows, key=lambda r: r["case_id"]))

    metadata["finished"] = datetime.now().isoformat(timespec="seconds")
    metadata["counts"] = {status: sum(r["status"] == status for r in status_rows)
                          for status in ("done", "failed", "unlabeled", "missing_input")}
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[INFO] {metadata['counts']} -> {out_dir}")
    return 1 if metadata["counts"]["failed"] else 0


def default_synthstrip_python() -> str | None:
    """$FHV_SYNTHSTRIP_PYTHON, else <conda>/envs/fhv_synthstrip/bin/python next to the running env."""
    if os.environ.get("FHV_SYNTHSTRIP_PYTHON"):
        return os.environ["FHV_SYNTHSTRIP_PYTHON"]
    candidate = Path(sys.prefix).parent / "fhv_synthstrip" / "bin" / "python"
    return str(candidate) if candidate.exists() else None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src-dir", required=True, help="Folder searched recursively for input FLAIR NIfTI files.")
    parser.add_argument("--out-dir", required=True, help="New output folder (YYYYMMDD_ prefix recommended).")
    parser.add_argument("--label-xlsx", required=True, help="Label table (.xlsx or .csv) with id_subject[, id_scan] and the label column.")
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--pattern", default="*_FLAIR.nii*", help="Input filename glob; case_id is the part before '_FLAIR'.")
    parser.add_argument("--pid", action="append", help="Process only this case_id (repeatable).")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N labelled cases.")
    parser.add_argument("--workers", type=int, default=4, help="Cases processed in parallel.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run finished cases and allow changed settings.")
    parser.add_argument("--no-qc", action="store_true", help="Skip the per-case QC PNG.")
    parser.add_argument("--skullstrip-backend", choices=sorted(pp.SKULLSTRIP_BACKENDS), default="native",
                        help="native: SynthStrip on the Apple GPU (MPS); docker: the amd64 image under emulation.")
    parser.add_argument("--synthstrip-image", default=pp.DEFAULT_SYNTHSTRIP_IMAGE,
                        help="Docker image to run, or to copy mri_synthstrip and weights from (native).")
    parser.add_argument("--synthstrip-python", default=default_synthstrip_python(),
                        help="native backend: python with PyTorch (MPS Conv3d) and surfa. "
                             "Default: $FHV_SYNTHSTRIP_PYTHON, else the sibling conda env 'fhv_synthstrip'.")
    parser.add_argument("--platform", default="linux/amd64", help="Docker --platform; the SynthStrip image is amd64-only.")
    parser.add_argument("--orientation", choices=pp.ORIENTATIONS, default="LAS")
    parser.add_argument("--normalization", choices=sorted(pp.NORMALIZATION_POLICIES), default="slice_p1_p99")
    parser.add_argument("--size-policy", choices=sorted(pp.SIZE_POLICIES), default="resize")
    parser.add_argument("--target-size", type=int, default=512)
    return parser


def main(argv=None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
