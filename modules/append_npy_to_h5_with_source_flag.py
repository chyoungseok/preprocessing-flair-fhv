"""Append subject NPY slices to an existing adjacent-slice H5 dataset.

This is for the workflow where an existing H5 is treated as the base dataset
and a directory of per-PID NPY files is converted to the same adjacent-slice
schema, then appended row-wise.

Added provenance keys:
  - data_source: string tag for each row
  - is_new_data: int8 flag, 0 for base H5 rows and 1 for appended NPY rows

Examples
--------
python -m modules.append_npy_to_h5_with_source_flag --dry-run

python -m modules.append_npy_to_h5_with_source_flag \
  --base-h5 /path/base.h5 \
  --npy-root /path/FLAIR_2025_nii \
  --output-h5 /path/output_plus_2025.h5 \
  --image-suffix _FLAIR_brain_bfCorrected_masked_reorient_AxialSlices_resized.npy \
  --label-suffix _FLAIR_brain_bfCorrected_masked_reorient_label_sliceLevel.npy \
  --dicom-root /path/dicom_root \
  --range-only \
  --min-index 0 \
  --max-index 100 \
  --new-subject-filter negative-only \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

try:
    from .dicom_scanner_metadata import (
        MANUFACTURER_NORMALIZATION,
        UNKNOWN_MANUFACTURER,
        get_subject_manufacturer,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from dicom_scanner_metadata import (
        MANUFACTURER_NORMALIZATION,
        UNKNOWN_MANUFACTURER,
        get_subject_manufacturer,
    )


DEFAULT_BASE_H5 = (
    "/Volumes/GURI_2/h5/FLAIR_HVS_detection_nii_preproc_resize_scanner_h5_whole-pos-neg/"
    "flair_slice_dataset_adjacent_resize_scanner_pidForm_whole-pos-neg.h5"
)
DEFAULT_NPY_ROOT = "/Volumes/GURI_2/FLAIR_2025_nii"
DEFAULT_OUTPUT_H5 = (
    "/Volumes/GURI_2/h5/FLAIR_HVS_detection_nii_preproc_resize_scanner_h5_whole-pos-neg_plus_2025/"
    "flair_slice_dataset_adjacent_resize_scanner_pidForm_whole-pos-neg_plus_2025.h5"
)
DEFAULT_IMAGE_SUFFIX = "_FLAIR_brain_bfCorrected_masked_reorient_AxialSlices_resized.npy"
DEFAULT_LABEL_SUFFIX = "_FLAIR_brain_bfCorrected_masked_reorient_label_sliceLevel.npy"
SUBJECT_FILTER_CHOICES = ("all", "negative-only", "positive-only")

CORE_KEYS = [
    "images",
    "labels",
    "subject_ids",
    "local_slice_index",
    "boundary_flag",
    "scanner_manufacturer",
]
PROVENANCE_KEYS = ["data_source", "is_new_data"]


@dataclass(frozen=True)
class SubjectPlan:
    subject_id: str
    img_path: Path
    label_path: Path
    n_slices: int
    selected_center_positions: np.ndarray
    scanner_manufacturer: str


def _list_subject_dirs(npy_root: Path, limit: int | None = None, pid: str | None = None) -> list[Path]:
    subjects = sorted(
        p for p in npy_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if pid is not None:
        subjects = [p for p in subjects if p.name == pid]
        if not subjects:
            raise FileNotFoundError(f"Subject folder not found: {npy_root / pid}")
    if limit is not None:
        subjects = subjects[:limit]
    return subjects


def _subject_files(subject_dir: Path, image_suffix: str, label_suffix: str) -> tuple[Path, Path]:
    subject_id = subject_dir.name
    img_path = subject_dir / f"{subject_id}{image_suffix}"
    label_path = subject_dir / f"{subject_id}{label_suffix}"
    if not img_path.exists():
        raise FileNotFoundError(f"Missing image NPY: {img_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label NPY: {label_path}")
    return img_path, label_path


def _center_labels(labels: np.ndarray) -> np.ndarray:
    if labels.ndim == 1:
        return labels
    if labels.ndim == 2:
        return labels[:, labels.shape[1] // 2]
    raise ValueError(f"Unsupported label shape: {labels.shape}")


def _selection_mask(labels: np.ndarray, min_index: int, max_index: int, range_only: bool) -> np.ndarray:
    center_labels = _center_labels(labels)
    local_slice_index = np.arange(1, len(center_labels) + 1, dtype=np.int32)
    range_mask = (local_slice_index >= min_index) & (local_slice_index <= max_index)
    if range_only:
        return range_mask
    if np.any(center_labels == 1):
        return center_labels == 1
    return range_mask


def _subject_label_class(center_labels: np.ndarray) -> str:
    return "positive" if np.any(center_labels == 1) else "negative"


def _passes_subject_filter(subject_label_class: str, new_subject_filter: str) -> bool:
    if new_subject_filter == "all":
        return True
    if new_subject_filter == "negative-only":
        return subject_label_class == "negative"
    if new_subject_filter == "positive-only":
        return subject_label_class == "positive"
    raise ValueError(f"Unsupported new_subject_filter: {new_subject_filter}")


def _neighbor_positions(center_pos: int, n_slices: int) -> tuple[int, int, int]:
    prev_pos = center_pos - 1 if center_pos > 0 else center_pos
    next_pos = center_pos + 1 if center_pos < n_slices - 1 else center_pos
    return prev_pos, center_pos, next_pos


def _boundary_flag(center_pos: int, n_slices: int) -> int:
    if center_pos == 0:
        return 1
    if center_pos == n_slices - 1:
        return 2
    return 0


def _build_new_plan(
    npy_root: Path,
    image_suffix: str,
    label_suffix: str,
    expected_hw: tuple[int, int],
    min_index: int,
    max_index: int,
    range_only: bool,
    dicom_root: Path | None,
    strict_missing: bool,
    strict_dicom_meta: bool,
    new_subject_filter: str,
    limit: int | None,
    pid: str | None,
) -> tuple[list[SubjectPlan], list[dict[str, object]]]:
    plans: list[SubjectPlan] = []
    manifest_rows: list[dict[str, object]] = []

    for subject_dir in tqdm(_list_subject_dirs(npy_root, limit=limit, pid=pid), desc="Planning new NPY"):
        subject_id = subject_dir.name
        try:
            img_path, label_path = _subject_files(subject_dir, image_suffix, label_suffix)
        except FileNotFoundError as exc:
            if strict_missing:
                raise
            manifest_rows.append({
                "subject_id": subject_id,
                "status": "skipped_missing_npy",
                "subject_label_class": "unknown",
                "message": str(exc),
                "selected_slices": 0,
                "scanner_manufacturer": "",
            })
            continue

        images = np.load(img_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        center_labels = _center_labels(labels)
        subject_label_class = _subject_label_class(center_labels)

        if images.ndim != 3:
            raise ValueError(f"Expected image shape (N,H,W), got {images.shape}: {img_path}")
        if images.shape[0] != len(center_labels):
            raise ValueError(
                f"Image/label slice count mismatch for {subject_id}: {images.shape[0]} vs {len(center_labels)}"
            )
        current_hw = tuple(images.shape[1:])
        if current_hw != expected_hw:
            raise ValueError(f"Image shape mismatch for {subject_id}: {current_hw} != {expected_hw}")

        if not _passes_subject_filter(subject_label_class, new_subject_filter):
            manifest_rows.append({
                "subject_id": subject_id,
                "status": f"skipped_{subject_label_class}_subject",
                "subject_label_class": subject_label_class,
                "message": f"new_subject_filter={new_subject_filter}",
                "selected_slices": 0,
                "scanner_manufacturer": "",
            })
            continue

        selected = np.where(_selection_mask(labels, min_index, max_index, range_only))[0].astype(np.int32)
        if selected.size == 0:
            manifest_rows.append({
                "subject_id": subject_id,
                "status": "skipped_no_selected_slices",
                "subject_label_class": subject_label_class,
                "message": "",
                "selected_slices": 0,
                "scanner_manufacturer": "",
            })
            continue

        scanner_manufacturer = UNKNOWN_MANUFACTURER
        if dicom_root is not None:
            scanner_manufacturer = get_subject_manufacturer(
                dicom_root=dicom_root,
                pid=subject_id,
                strict=strict_dicom_meta,
            )

        plans.append(
            SubjectPlan(
                subject_id=subject_id,
                img_path=img_path,
                label_path=label_path,
                n_slices=images.shape[0],
                selected_center_positions=selected,
                scanner_manufacturer=scanner_manufacturer,
            )
        )
        manifest_rows.append({
            "subject_id": subject_id,
            "status": "selected",
            "subject_label_class": subject_label_class,
            "message": "",
            "selected_slices": int(selected.size),
            "scanner_manufacturer": scanner_manufacturer,
        })

    return plans, manifest_rows


def _write_manifest(path: Path | None, rows: list[dict[str, object]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "status",
        "subject_label_class",
        "message",
        "selected_slices",
        "scanner_manufacturer",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_direct_chunks(src_dset, dst_dset, dst_offset: int, desc: str) -> bool:
    """Copy compressed chunks without decompression when layouts allow it."""
    if src_dset.chunks is None or dst_dset.chunks != src_dset.chunks:
        return False
    if src_dset.chunks[0] != 1:
        return False

    for i in tqdm(range(src_dset.shape[0]), desc=desc):
        src_coord = (i,) + (0,) * (len(src_dset.shape) - 1)
        dst_coord = (dst_offset + i,) + (0,) * (len(dst_dset.shape) - 1)
        filter_mask, chunk = src_dset.id.read_direct_chunk(src_coord)
        dst_dset.id.write_direct_chunk(dst_coord, chunk, filter_mask=filter_mask)
    return True


def _copy_rows(src_dset, dst_dset, dst_offset: int, chunk_rows: int, desc: str) -> None:
    for start in tqdm(range(0, src_dset.shape[0], chunk_rows), desc=desc):
        stop = min(start + chunk_rows, src_dset.shape[0])
        dst_dset[dst_offset + start:dst_offset + stop] = src_dset[start:stop]


def _create_output_datasets(base_h5, out_h5, total_rows: int, string_dtype):
    datasets = {}
    for key in CORE_KEYS:
        if key not in base_h5:
            raise KeyError(f"Base H5 is missing required key: {key}")
        src = base_h5[key]
        dtype = string_dtype if src.dtype.kind == "O" else src.dtype
        kwargs = {"shape": (total_rows, *src.shape[1:]), "dtype": dtype}
        if src.chunks is not None:
            kwargs["chunks"] = src.chunks
        if src.compression is not None:
            kwargs["compression"] = src.compression
            kwargs["compression_opts"] = src.compression_opts
        datasets[key] = out_h5.create_dataset(key, **kwargs)

    datasets["data_source"] = out_h5.create_dataset("data_source", shape=(total_rows,), dtype=string_dtype)
    datasets["is_new_data"] = out_h5.create_dataset("is_new_data", shape=(total_rows,), dtype=np.int8)
    return datasets


def append_npy_to_h5_with_source_flag(
    base_h5_path: str | Path = DEFAULT_BASE_H5,
    npy_root: str | Path = DEFAULT_NPY_ROOT,
    output_h5_path: str | Path = DEFAULT_OUTPUT_H5,
    image_suffix: str = DEFAULT_IMAGE_SUFFIX,
    label_suffix: str = DEFAULT_LABEL_SUFFIX,
    dicom_root: str | Path | None = None,
    base_source_tag: str = "whole-pos-neg",
    new_source_tag: str = "FLAIR_2025_nii",
    min_index: int = 0,
    max_index: int = 100,
    range_only: bool = False,
    overwrite: bool = False,
    strict_missing: bool = False,
    strict_dicom_meta: bool = False,
    new_subject_filter: str = "all",
    limit: int | None = None,
    pid: str | None = None,
    chunk_rows: int = 4096,
    manifest_csv: str | Path | None = None,
    dry_run: bool = False,
) -> str:
    base_h5_path = Path(base_h5_path)
    npy_root = Path(npy_root)
    output_h5_path = Path(output_h5_path)
    dicom_root = Path(dicom_root) if dicom_root is not None else None
    manifest_csv = Path(manifest_csv) if manifest_csv is not None else None

    if not base_h5_path.exists():
        raise FileNotFoundError(f"Base H5 not found: {base_h5_path}")
    if not npy_root.exists():
        raise FileNotFoundError(f"NPY root not found: {npy_root}")
    if output_h5_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output already exists. Use --overwrite to replace it: {output_h5_path}")
    if new_subject_filter not in SUBJECT_FILTER_CHOICES:
        raise ValueError(f"--new-subject-filter must be one of {SUBJECT_FILTER_CHOICES}, got: {new_subject_filter}")

    with h5py.File(base_h5_path, "r") as base:
        n_base = base["images"].shape[0]
        expected_hw = tuple(base["images"].shape[2:])
        plans, manifest_rows = _build_new_plan(
            npy_root=npy_root,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            expected_hw=expected_hw,
            min_index=min_index,
            max_index=max_index,
            range_only=range_only,
            dicom_root=dicom_root,
            strict_missing=strict_missing,
            strict_dicom_meta=strict_dicom_meta,
            new_subject_filter=new_subject_filter,
            limit=limit,
            pid=pid,
        )

        n_new = int(sum(len(plan.selected_center_positions) for plan in plans))
        n_total = n_base + n_new
        print(f"[INFO] Base rows: {n_base}")
        print(f"[INFO] New subjects selected: {len(plans)}")
        print(f"[INFO] New rows: {n_new}")
        print(f"[INFO] Output rows: {n_total}")
        print(f"[INFO] Expected image HW: {expected_hw}")

        _write_manifest(manifest_csv, manifest_rows)
        if manifest_csv is not None:
            print(f"[INFO] Wrote manifest: {manifest_csv}")

        if dry_run:
            print("[DRY-RUN] No H5 file was written.")
            return str(output_h5_path)

        output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(output_h5_path, "w") as out:
            for key, value in base.attrs.items():
                out.attrs[key] = value
            out.attrs["base_h5_path"] = str(base_h5_path)
            out.attrs["n_base_rows"] = int(n_base)
            out.attrs["n_new_rows"] = int(n_new)
            out.attrs["base_source_tag"] = base_source_tag
            out.attrs["new_source_tag"] = new_source_tag
            out.attrs["npy_root"] = str(npy_root)
            out.attrs["image_suffix"] = image_suffix
            out.attrs["label_suffix"] = label_suffix
            out.attrs["new_subject_filter"] = new_subject_filter
            out.attrs["scanner_manufacturer_normalization"] = MANUFACTURER_NORMALIZATION
            if dicom_root is not None:
                out.attrs["scanner_metadata_source"] = "dicom"
                out.attrs["dicom_root"] = str(dicom_root)

            dsets = _create_output_datasets(base, out, n_total, string_dtype)

            for key in CORE_KEYS:
                copied_direct = _copy_direct_chunks(
                    base[key],
                    dsets[key],
                    dst_offset=0,
                    desc=f"Copy base {key} direct",
                )
                if not copied_direct:
                    _copy_rows(base[key], dsets[key], 0, chunk_rows, f"Copy base {key}")

            for start in tqdm(range(0, n_base, chunk_rows), desc="Fill base data_source"):
                stop = min(start + chunk_rows, n_base)
                dsets["data_source"][start:stop] = [base_source_tag] * (stop - start)
                dsets["is_new_data"][start:stop] = np.zeros((stop - start,), dtype=np.int8)

            out_idx = n_base
            for plan in tqdm(plans, desc="Append new NPY"):
                images = np.load(plan.img_path, mmap_mode="r")
                labels = _center_labels(np.load(plan.label_path, mmap_mode="r"))

                for center_pos in plan.selected_center_positions:
                    triplet_pos = _neighbor_positions(int(center_pos), plan.n_slices)
                    dsets["images"][out_idx] = images[list(triplet_pos)].astype(np.float32, copy=False)
                    dsets["labels"][out_idx] = labels[list(triplet_pos)].astype(np.int8, copy=False)
                    dsets["subject_ids"][out_idx] = plan.subject_id
                    dsets["local_slice_index"][out_idx] = int(center_pos) + 1
                    dsets["boundary_flag"][out_idx] = _boundary_flag(int(center_pos), plan.n_slices)
                    dsets["scanner_manufacturer"][out_idx] = plan.scanner_manufacturer
                    dsets["data_source"][out_idx] = new_source_tag
                    dsets["is_new_data"][out_idx] = 1
                    out_idx += 1

            if out_idx != n_total:
                raise RuntimeError(f"Internal row count mismatch: wrote {out_idx}, expected {n_total}")

    print(f"[DONE] Saved appended H5 to: {output_h5_path}")
    return str(output_h5_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append per-subject NPY slices to an existing adjacent-slice H5.")
    parser.add_argument("--base-h5", default=DEFAULT_BASE_H5)
    parser.add_argument("--npy-root", default=DEFAULT_NPY_ROOT)
    parser.add_argument("--output-h5", default=DEFAULT_OUTPUT_H5)
    parser.add_argument("--image-suffix", default=DEFAULT_IMAGE_SUFFIX)
    parser.add_argument("--label-suffix", default=DEFAULT_LABEL_SUFFIX)
    parser.add_argument("--dicom-root", default=None)
    parser.add_argument("--base-source-tag", default="whole-pos-neg")
    parser.add_argument("--new-source-tag", default="FLAIR_2025_nii")
    parser.add_argument("--min-index", type=int, default=0)
    parser.add_argument("--max-index", type=int, default=100)
    parser.add_argument("--range-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-missing", action="store_true")
    parser.add_argument("--strict-dicom-meta", action="store_true")
    parser.add_argument(
        "--new-subject-filter",
        choices=SUBJECT_FILTER_CHOICES,
        default="all",
        help=(
            "Filter appended NPY subjects by subject-level labels: "
            "all keeps all subjects, negative-only keeps subjects with no positive labels, "
            "positive-only keeps subjects with at least one positive label."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pid", default=None)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    append_npy_to_h5_with_source_flag(
        base_h5_path=args.base_h5,
        npy_root=args.npy_root,
        output_h5_path=args.output_h5,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        dicom_root=args.dicom_root,
        base_source_tag=args.base_source_tag,
        new_source_tag=args.new_source_tag,
        min_index=args.min_index,
        max_index=args.max_index,
        range_only=args.range_only,
        overwrite=args.overwrite,
        strict_missing=args.strict_missing,
        strict_dicom_meta=args.strict_dicom_meta,
        new_subject_filter=args.new_subject_filter,
        limit=args.limit,
        pid=args.pid,
        chunk_rows=args.chunk_rows,
        manifest_csv=args.manifest_csv,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
