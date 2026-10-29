"""Step 06: create one filtered adjacent-slice HDF5 dataset from slice NPY files.

The default selection is label-independent relative-index selection:

- keep center slices whose 1-based local slice index divided by the subject's
  total slice count is within 45..75 percent
- labels are saved as supervision signals only; they do not control which
  slices are selected

Use ``--selection-mode local`` with ``--min-index``/``--max-index`` for legacy
1-based local slice index selection.

For each kept center slice, the saved image and label are adjacent triplets:
``[previous, center, next]``. Subject boundaries use self-reference padding.

Examples
--------
python -m modules.step6_create_filtered_h5
python -m modules.step6_create_filtered_h5 --src-dir /path/slices --output-h5 /path/flair_slice_dataset_adjacent_1215.h5
python -m modules.step6_create_filtered_h5 --image-suffix _AxialSlices_padded.npy --label-suffix _label_sliceLevel.npy
python -m modules.step6_create_filtered_h5 --selection-mode local --min-index 12 --max-index 15
"""

import argparse
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


DEFAULT_SRC_DIR = "/Users/yschoi/data/FLAIR/flair_preproc_v02_slices_whole"
DEFAULT_OUTPUT_H5 = (
    "/Users/yschoi/data/FLAIR/flair_preproc_v02_slices_whole_h5_uhkInspection/"
    "flair_slice_dataset_adjacent_1215.h5"
)
DEFAULT_IMAGE_SUFFIX = "_AxialSlices_padded.npy"
DEFAULT_LABEL_SUFFIX = "_label_sliceLevel.npy"
DEFAULT_SELECTION_MODE = "relative"
DEFAULT_MIN_REL_INDEX = 45.0
DEFAULT_MAX_REL_INDEX = 75.0
SELECTION_MODES = {"relative", "local"}


@dataclass(frozen=True)
class SubjectPlan:
    subject_id: str
    h5_subject_id: str
    h5_scan_id: str
    h5_case_id: str
    img_path: Path
    label_path: Path
    n_slices: int
    selected_center_positions: np.ndarray
    scanner_manufacturer: str | None = None


def _list_subject_dirs(src_dir: Path, limit: int | None = None, pid: str | None = None) -> list[Path]:
    subjects = sorted(
        p for p in src_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "logs"
    )
    if pid is not None:
        subjects = [p for p in subjects if p.name == pid]
        if not subjects:
            raise FileNotFoundError(f"Subject folder not found: {src_dir / pid}")
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


def _load_h5_subject_id_map(subject_id_table: Path | None) -> dict[str, str]:
    """Load folder-name -> H5 subject_id labels.

    Supported table schemas:
      - id_subject + id_session: map id_session folder names to id_subject_id_session.
      - id_subject only: map id_subject folder names to id_subject.
    """
    if subject_id_table is None:
        return {}
    if not subject_id_table.exists():
        raise FileNotFoundError(f"Subject ID table not found: {subject_id_table}")

    import pandas as pd

    df = pd.read_excel(subject_id_table)
    if "id_subject" not in df.columns:
        raise KeyError("Subject ID table is missing required column: id_subject")

    id_map: dict[str, str] = {}
    duplicates = []

    if "id_session" in df.columns:
        for _, row in df[["id_subject", "id_session"]].dropna().iterrows():
            id_subject = str(row["id_subject"]).strip()
            id_session = str(row["id_session"]).strip()
            h5_subject_id = f"{id_subject}_{id_session}"
            if id_session in id_map and id_map[id_session] != h5_subject_id:
                duplicates.append((id_session, id_map[id_session], h5_subject_id))
                continue
            id_map[id_session] = h5_subject_id
    else:
        for _, row in df[["id_subject"]].dropna().iterrows():
            id_subject = str(row["id_subject"]).strip()
            if id_subject in id_map and id_map[id_subject] != id_subject:
                duplicates.append((id_subject, id_map[id_subject], id_subject))
                continue
            id_map[id_subject] = id_subject

    if duplicates:
        preview = ", ".join(f"{folder_id}: {old} vs {new}" for folder_id, old, new in duplicates[:5])
        raise ValueError(f"Duplicate subject ID mappings in {subject_id_table}: {preview}")
    return id_map


def _split_subject_scan_id(folder_id: str) -> tuple[str, str]:
    """Split a folder ID like BP00001_P14163 into subject and scan IDs."""
    if "_" not in folder_id:
        return folder_id, ""
    subject_id, scan_id = folder_id.split("_", 1)
    return subject_id, scan_id


def _center_labels(labels: np.ndarray) -> np.ndarray:
    """Return scalar center labels from either (N,) or adjacent-like (N, C) labels."""
    if labels.ndim == 1:
        return labels
    if labels.ndim == 2:
        center_col = labels.shape[1] // 2
        return labels[:, center_col]
    raise ValueError(f"Unsupported label shape: {labels.shape}")


def _relative_indices(n_slices: int) -> np.ndarray:
    """Return 1-based local slice index as percent of subject slice count."""
    if n_slices <= 0:
        raise ValueError(f"n_slices must be positive, got {n_slices}")
    local_slice_index = np.arange(1, n_slices + 1, dtype=np.float32)
    return local_slice_index / float(n_slices) * 100.0


def _selection_mask(
    labels: np.ndarray,
    min_index: int,
    max_index: int,
    selection_mode: str = DEFAULT_SELECTION_MODE,
    min_rel_index: float = DEFAULT_MIN_REL_INDEX,
    max_rel_index: float = DEFAULT_MAX_REL_INDEX,
    selection_invert: bool = False,
    range_only: bool = False,
) -> np.ndarray:
    """Return selected center slices without using label positivity.

    ``range_only`` is kept for CLI/API compatibility. Since selection no longer
    gives positive-label slices priority, it has no effect.
    """
    center_labels = _center_labels(labels)
    n_slices = len(center_labels)

    if selection_mode == "relative":
        rel = _relative_indices(n_slices)
        range_mask = (rel >= min_rel_index) & (rel <= max_rel_index)
        return ~range_mask if selection_invert else range_mask

    if selection_mode == "local":
        local_slice_index = np.arange(1, n_slices + 1, dtype=np.int32)
        range_mask = (local_slice_index >= min_index) & (local_slice_index <= max_index)
        return ~range_mask if selection_invert else range_mask

    raise ValueError(f"Unsupported selection_mode: {selection_mode}")


def _validate_selection_args(
    selection_mode: str,
    min_index: int,
    max_index: int,
    min_rel_index: float,
    max_rel_index: float,
) -> None:
    """Validate local/relative slice selection arguments."""
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {sorted(SELECTION_MODES)}, got {selection_mode}")
    if min_index > max_index:
        raise ValueError(f"min_index must be <= max_index, got {min_index}, {max_index}")
    if not (0.0 <= min_rel_index < max_rel_index <= 100.0):
        raise ValueError(
            "Relative index bounds must satisfy 0 <= min_rel_index < max_rel_index <= 100, "
            f"got {min_rel_index}, {max_rel_index}"
        )


def _build_plan(
    src_dir: Path,
    min_index: int,
    max_index: int,
    selection_mode: str,
    min_rel_index: float,
    max_rel_index: float,
    selection_invert: bool,
    image_suffix: str,
    label_suffix: str,
    limit: int | None = None,
    pid: str | None = None,
    strict_missing: bool = False,
    range_only: bool = False,
    include_scanner_manufacturer: bool = False,
    dicom_root: Path | None = None,
    strict_dicom_meta: bool = False,
    subject_id_table: Path | None = None,
    split_subject_scan_ids: bool = False,
) -> tuple[list[SubjectPlan], tuple[int, int]]:
    plans: list[SubjectPlan] = []
    image_hw = None
    skipped_missing = []
    h5_subject_id_map = _load_h5_subject_id_map(subject_id_table)

    for subject_dir in tqdm(_list_subject_dirs(src_dir, limit=limit, pid=pid), desc="Planning"):
        try:
            img_path, label_path = _subject_files(subject_dir, image_suffix, label_suffix)
        except FileNotFoundError as exc:
            if strict_missing:
                raise
            skipped_missing.append((subject_dir.name, str(exc)))
            continue

        images = np.load(img_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        center_labels = _center_labels(labels)

        if images.ndim != 3:
            raise ValueError(f"Expected image shape (N,H,W), got {images.shape}: {img_path}")
        if images.shape[0] != len(center_labels):
            raise ValueError(
                f"Image/label slice count mismatch for {subject_dir.name}: "
                f"{images.shape[0]} vs {len(center_labels)}"
            )

        current_hw = tuple(images.shape[1:])
        if image_hw is None:
            image_hw = current_hw
        elif current_hw != image_hw:
            raise ValueError(f"Image shape mismatch for {subject_dir.name}: {current_hw} != {image_hw}")

        selected_center_positions = np.where(
            _selection_mask(
                labels=labels,
                min_index=min_index,
                max_index=max_index,
                selection_mode=selection_mode,
                min_rel_index=min_rel_index,
                max_rel_index=max_rel_index,
                selection_invert=selection_invert,
                range_only=range_only,
            )
        )[0]
        if selected_center_positions.size == 0:
            continue

        scanner_manufacturer = None
        if include_scanner_manufacturer:
            if dicom_root is None:
                raise ValueError("--dicom-root is required when --include-scanner-manufacturer is used.")
            scanner_manufacturer = get_subject_manufacturer(
                dicom_root=dicom_root,
                pid=subject_dir.name,
                strict=strict_dicom_meta,
            )

        if h5_subject_id_map and subject_dir.name not in h5_subject_id_map:
            raise KeyError(f"No id_session match for subject folder {subject_dir.name} in {subject_id_table}")

        h5_subject_id = h5_subject_id_map.get(subject_dir.name, subject_dir.name)
        h5_scan_id = ""
        h5_case_id = subject_dir.name
        if split_subject_scan_ids:
            h5_subject_id, h5_scan_id = _split_subject_scan_id(subject_dir.name)

        plans.append(
            SubjectPlan(
                subject_id=subject_dir.name,
                h5_subject_id=h5_subject_id,
                h5_scan_id=h5_scan_id,
                h5_case_id=h5_case_id,
                img_path=img_path,
                label_path=label_path,
                n_slices=images.shape[0],
                selected_center_positions=selected_center_positions.astype(np.int32),
                scanner_manufacturer=scanner_manufacturer,
            )
        )

    if not plans or image_hw is None:
        raise ValueError("No selected slices found. Check source data and slice-index range.")

    if skipped_missing:
        print(f"[INFO] Skipped folders without expected NPY files: {len(skipped_missing)}")
        for subject_id, message in skipped_missing[:10]:
            print(f"[SKIP] {subject_id}: {message}")
        if len(skipped_missing) > 10:
            print(f"[SKIP] ... {len(skipped_missing) - 10} more")

    return plans, image_hw


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


def create_filtered_adjacent_h5_from_npy(
    src_dir: str | Path = DEFAULT_SRC_DIR,
    output_h5: str | Path = DEFAULT_OUTPUT_H5,
    min_index: int = 12,
    max_index: int = 15,
    selection_mode: str = DEFAULT_SELECTION_MODE,
    min_rel_index: float = DEFAULT_MIN_REL_INDEX,
    max_rel_index: float = DEFAULT_MAX_REL_INDEX,
    selection_invert: bool = False,
    compression: str = "gzip",
    compression_level: int = 4,
    image_suffix: str = DEFAULT_IMAGE_SUFFIX,
    label_suffix: str = DEFAULT_LABEL_SUFFIX,
    overwrite: bool = False,
    limit: int | None = None,
    pid: str | None = None,
    strict_missing: bool = False,
    range_only: bool = False,
    include_scanner_manufacturer: bool = False,
    dicom_root: str | Path | None = None,
    strict_dicom_meta: bool = False,
    subject_id_table: str | Path | None = None,
    split_subject_scan_ids: bool = False,
) -> str:
    """Create the final filtered adjacent HDF5 file directly from subject NPY files."""
    src_dir = Path(src_dir)
    output_h5 = Path(output_h5)
    dicom_root = Path(dicom_root) if dicom_root is not None else None
    subject_id_table = Path(subject_id_table) if subject_id_table is not None else None
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if output_h5.exists() and not overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite to replace it: {output_h5}")
    _validate_selection_args(selection_mode, min_index, max_index, min_rel_index, max_rel_index)
    if range_only:
        print("[WARN] --range-only is deprecated and has no effect; selection is label-independent.")

    plans, (height, width) = _build_plan(
        src_dir=src_dir,
        min_index=min_index,
        max_index=max_index,
        selection_mode=selection_mode,
        min_rel_index=min_rel_index,
        max_rel_index=max_rel_index,
        selection_invert=selection_invert,
        image_suffix=image_suffix,
        label_suffix=label_suffix,
        limit=limit,
        pid=pid,
        strict_missing=strict_missing,
        range_only=range_only,
        include_scanner_manufacturer=include_scanner_manufacturer,
        dicom_root=dicom_root,
        strict_dicom_meta=strict_dicom_meta,
        subject_id_table=subject_id_table,
        split_subject_scan_ids=split_subject_scan_ids,
    )
    total_selected = int(sum(len(plan.selected_center_positions) for plan in plans))

    print(f"[INFO] Selection mode: {selection_mode}")
    if selection_mode == "relative":
        print(f"[INFO] Relative index range: {min_rel_index:g}..{max_rel_index:g}%")
    else:
        print(f"[INFO] Local slice index range: {min_index}..{max_index}")
    if selection_invert:
        print("[INFO] Selection range is inverted.")
    print(f"[INFO] Subjects with selected slices: {len(plans)}")
    print(f"[INFO] Selected center slices: {total_selected}")
    print(f"[INFO] Output image shape: ({total_selected}, 3, {height}, {width})")

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_h5, "w") as f_out:
        f_out.attrs["selection_mode"] = selection_mode
        f_out.attrs["min_index"] = int(min_index)
        f_out.attrs["max_index"] = int(max_index)
        f_out.attrs["min_rel_index"] = float(min_rel_index)
        f_out.attrs["max_rel_index"] = float(max_rel_index)
        f_out.attrs["selection_invert"] = bool(selection_invert)
        f_out.attrs["range_only_deprecated"] = bool(range_only)
        f_out.attrs["split_subject_scan_ids"] = bool(split_subject_scan_ids)
        # images: selected center slices as adjacent triplets [previous, center, next].
        dset_images = f_out.create_dataset(
            "images",
            shape=(total_selected, 3, height, width),
            dtype=np.float32,
            chunks=(1, 3, height, width),
            compression=compression,
            compression_opts=compression_level,
        )
        # labels: slice-level labels for the same [previous, center, next] triplets.
        dset_labels = f_out.create_dataset(
            "labels",
            shape=(total_selected, 3),
            dtype=np.int8,
            chunks=(1, 3),
            compression=compression,
            compression_opts=compression_level,
        )
        # subject_ids: subject/case ID for each selected center slice.
        dset_subject_ids = f_out.create_dataset(
            "subject_ids",
            shape=(total_selected,),
            dtype=string_dtype,
        )
        # scan_ids: scan/session component split from folder IDs such as BP00001_P14163.
        dset_scan_ids = f_out.create_dataset(
            "scan_ids",
            shape=(total_selected,),
            dtype=string_dtype,
        )
        # case_ids: original folder/case ID, kept so the split IDs can be traced back exactly.
        dset_case_ids = f_out.create_dataset(
            "case_ids",
            shape=(total_selected,),
            dtype=string_dtype,
        )
        # local_slice_index: 1-based center-slice index within each subject.
        dset_local_slice_index = f_out.create_dataset(
            "local_slice_index",
            shape=(total_selected,),
            dtype=np.int32,
        )
        # local_slice_index_rel: center-slice position as percent of subject slice count.
        dset_local_slice_index_rel = f_out.create_dataset(
            "local_slice_index_rel",
            shape=(total_selected,),
            dtype=np.float32,
        )
        # boundary_flag: center-slice boundary marker; 0=normal, 1=first, 2=last.
        # First/last center slices use self-reference padding for missing previous/next slices.
        dset_boundary_flag = f_out.create_dataset(
            "boundary_flag",
            shape=(total_selected,),
            dtype=np.int8,
        )
        dset_scanner_manufacturer = None
        if include_scanner_manufacturer:
            f_out.attrs["scanner_metadata_source"] = "dicom"
            f_out.attrs["scanner_manufacturer_level"] = "subject"
            f_out.attrs["scanner_manufacturer_normalization"] = MANUFACTURER_NORMALIZATION
            dset_scanner_manufacturer = f_out.create_dataset(
                "scanner_manufacturer",
                shape=(total_selected,),
                dtype=string_dtype,
            )

        out_idx = 0
        for plan in tqdm(plans, desc="Writing H5"):
            images = np.load(plan.img_path, mmap_mode="r")
            labels = _center_labels(np.load(plan.label_path, mmap_mode="r"))

            for center_pos in plan.selected_center_positions:
                triplet_pos = _neighbor_positions(int(center_pos), plan.n_slices)
                local_index = int(center_pos) + 1

                dset_images[out_idx] = images[list(triplet_pos)]
                dset_labels[out_idx] = labels[list(triplet_pos)]
                dset_subject_ids[out_idx] = plan.h5_subject_id
                dset_scan_ids[out_idx] = plan.h5_scan_id
                dset_case_ids[out_idx] = plan.h5_case_id
                dset_local_slice_index[out_idx] = local_index
                dset_local_slice_index_rel[out_idx] = local_index / float(plan.n_slices) * 100.0
                dset_boundary_flag[out_idx] = _boundary_flag(int(center_pos), plan.n_slices)
                if dset_scanner_manufacturer is not None:
                    dset_scanner_manufacturer[out_idx] = plan.scanner_manufacturer or UNKNOWN_MANUFACTURER
                out_idx += 1

    print(f"[DONE] Saved filtered adjacent H5 to: {output_h5}")
    return str(output_h5)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create filtered adjacent-slice HDF5 directly from subject NPY files."
    )
    parser.add_argument("--src-dir", default=DEFAULT_SRC_DIR)
    parser.add_argument("--output-h5", default=DEFAULT_OUTPUT_H5)
    parser.add_argument("--min-index", type=int, default=12)
    parser.add_argument("--max-index", type=int, default=15)
    parser.add_argument(
        "--selection-mode",
        choices=sorted(SELECTION_MODES),
        default=DEFAULT_SELECTION_MODE,
        help=(
            "Slice selection basis. Default 'relative' selects by local_slice_index_rel; "
            "'local' selects by 1-based local_slice_index."
        ),
    )
    parser.add_argument(
        "--min-rel-index",
        type=float,
        default=DEFAULT_MIN_REL_INDEX,
        help="Minimum local_slice_index_rel percentage for --selection-mode relative. Default: 45.",
    )
    parser.add_argument(
        "--max-rel-index",
        type=float,
        default=DEFAULT_MAX_REL_INDEX,
        help="Maximum local_slice_index_rel percentage for --selection-mode relative. Default: 75.",
    )
    parser.add_argument(
        "--selection-invert",
        action="store_true",
        help=(
            "Invert the selected range. For relative mode, selects local_slice_index_rel outside "
            "[--min-rel-index, --max-rel-index]. For local mode, selects slices outside "
            "[--min-index, --max-index]."
        ),
    )
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--compression-level", type=int, default=4)
    parser.add_argument("--image-suffix", default=DEFAULT_IMAGE_SUFFIX)
    parser.add_argument("--label-suffix", default=DEFAULT_LABEL_SUFFIX)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pid", default=None, help="Run only one subject folder.")
    parser.add_argument(
        "--subject-id-table",
        default=None,
        help=(
            "Optional Excel table with an id_subject column. "
            "If id_session is also present, folder names are matched to id_session and H5 subject_ids are written as "
            "{id_subject}_{id_session}. If only id_subject is present, folder names are matched to id_subject."
        ),
    )
    parser.add_argument(
        "--split-subject-scan-ids",
        action="store_true",
        help=(
            "Split folder IDs like BP00001_P14163 into subject_ids=BP00001 and scan_ids=P14163. "
            "The original folder ID is also saved in case_ids."
        ),
    )
    parser.add_argument(
        "--dicom-root",
        default=None,
        help="Root DICOM directory with group/PID folders. Required with --include-scanner-manufacturer.",
    )
    parser.add_argument(
        "--include-scanner-manufacturer",
        action="store_true",
        help="Add subject-level normalized scanner_manufacturer metadata from one DICOM per PID.",
    )
    parser.add_argument(
        "--strict-dicom-meta",
        action="store_true",
        help="Fail when DICOM manufacturer metadata is missing, unreadable, or not Philips/GE/Siemens.",
    )
    parser.add_argument(
        "--range-only",
        action="store_true",
        help="Deprecated no-op. Selection is now label-independent for every subject.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any folder under src-dir is missing the expected image/label NPY files.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    create_filtered_adjacent_h5_from_npy(
        src_dir=args.src_dir,
        output_h5=args.output_h5,
        min_index=args.min_index,
        max_index=args.max_index,
        selection_mode=args.selection_mode,
        min_rel_index=args.min_rel_index,
        max_rel_index=args.max_rel_index,
        selection_invert=args.selection_invert,
        compression=args.compression,
        compression_level=args.compression_level,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        overwrite=args.overwrite,
        limit=args.limit,
        pid=args.pid,
        strict_missing=args.strict_missing,
        range_only=args.range_only,
        include_scanner_manufacturer=args.include_scanner_manufacturer,
        dicom_root=args.dicom_root,
        strict_dicom_meta=args.strict_dicom_meta,
        subject_id_table=args.subject_id_table,
        split_subject_scan_ids=args.split_subject_scan_ids,
    )


if __name__ == "__main__":
    main()
