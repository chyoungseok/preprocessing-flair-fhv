"""Step 05: export RAS NIfTI volumes as padded/normalized axial slices.

Examples
--------
python -m modules.step5_resize_normalization single --input-path in.nii.gz --dst-dir /path/slices
python -m modules.step5_resize_normalization batch --src-dir /path/preproc --dst-dir /path/slices
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from modules.logging_utils import append_error_log, resolve_log_file

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")


DEFAULT_TARGET_SIZE = 672
TARGET_HW = (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE)
PREPROCESS_MODES = {"pad", "resize"}
DEFAULT_LOG_FILE = "/Volumes/X31/FLAIR_preprocessing_logs/err_resize.txt"
DEFAULT_SRC_DIR = "/Volumes/X31/FLAIR_HVS_detection_nii_preproc"
DEFAULT_DST_DIR = "/Volumes/X31/FLAIR_HVS_detection_slices"
DEFAULT_LABEL_XLSX = "/Users/yschoi/code/09_FHV/modules_v02/df_flair_label_UHKinspection.xlsx"
DEFAULT_LABEL_COLUMN = "fhv_slice_num"


def is_disabled_label_path(label_xlsx) -> bool:
    """Return True when the CLI value explicitly disables label loading."""
    if label_xlsx is None:
        return True
    return str(label_xlsx).strip().lower() in {"", "none", "null", "false", "no"}


def load_label_table(label_xlsx):
    """Load the optional slice-level label table.

    Missing or disabled label files are not fatal: Step 5 still saves image
    slices and metadata, while ``*_label_sliceLevel.npy`` is skipped.
    """
    if is_disabled_label_path(label_xlsx):
        print("[WARN] label xlsx disabled; label npy will not be saved.")
        return None

    label_path = Path(label_xlsx).expanduser()
    if not label_path.exists():
        print(f"[WARN] label xlsx not found: {label_path}; label npy will not be saved.")
        return None

    if label_path.suffix.lower() == ".csv":
        df_label = pd.read_csv(label_path, index_col=0)
    else:
        df_label = pd.read_excel(label_path, index_col=0)

    if df_label.index.name and df_label.index.name not in df_label.columns:
        df_label = df_label.reset_index()
    return df_label


def write_error(log_file, input_path, output_dir, error):
    """Append one failed case to the resize/normalization log."""
    append_error_log(log_file, input_path, output_dir, error)


def output_suffix_for_mode(preprocess_mode: str) -> str:
    """Return the image npy suffix for the selected preprocessing mode."""
    if preprocess_mode == "pad":
        return "padded"
    if preprocess_mode == "resize":
        return "resized"
    raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")


def expected_slice_output(dst_dir, nifti_path, preprocess_mode="pad"):
    """Return expected slice npy path for one input NIfTI."""
    output_prefix = output_prefix_from_path(nifti_path)
    subject_id = default_id_from_path(nifti_path)
    suffix = output_suffix_for_mode(preprocess_mode)
    return Path(dst_dir) / subject_id / f"{output_prefix}_AxialSlices_{suffix}.npy"


def get_value_from_dataframe(df: pd.DataFrame, col_name: str, conditional_pair: dict):
    """Return one DataFrame value using an exact single-row condition."""
    if col_name not in df.columns:
        raise KeyError(f"Label column not found: {col_name}")

    con = pd.Series(True, index=df.index)
    for key, val in conditional_pair.items():
        if key not in df.columns:
            raise KeyError(f"Label column not found: {key}")
        con &= df[key].astype(str).isin([str(val)])

    n_match = int(con.sum())
    if n_match == 0:
        raise ValueError(f"No label row found for condition: {conditional_pair}")
    if n_match > 1:
        raise ValueError(f"Multiple label rows found for condition: {conditional_pair}")
    return df.loc[con, col_name].values[0]


def parse_list_value(value):
    """Parse list-like label values loaded from Excel."""
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "nan":
            return []
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, tuple):
                return list(parsed)
            return [parsed]
        except (SyntaxError, ValueError):
            tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", stripped)
            if tokens:
                parsed = []
                for token in tokens:
                    number = float(token)
                    parsed.append(int(number) if number.is_integer() else number)
                return parsed
        raise ValueError(f"Could not parse list-like label value: {value}")
    return [value]


def label_condition_for_subject(df_label: pd.DataFrame, subject_id: str):
    """Build a unique label lookup condition for subject-only or subject_session IDs."""
    session_cols = ("id_session", "id_scan")
    available_session_cols = [col for col in session_cols if col in df_label.columns]

    if "_" in subject_id and "id_subject" in df_label.columns and available_session_cols:
        id_subject, id_session = subject_id.split("_", 1)
        return {"id_subject": id_subject, available_session_cols[0]: id_session}

    match_counts = {}
    for col in ("id_subject", *session_cols):
        if col not in df_label.columns:
            continue
        n_match = int((df_label[col].astype(str) == str(subject_id)).sum())
        match_counts[col] = n_match
        if n_match == 1:
            return {col: subject_id}

    if match_counts:
        raise ValueError(f"No unique label row for {subject_id}: {match_counts}")
    raise KeyError("Label table must contain 'id_subject', 'id_session', or 'id_scan'.")


def target_hw_from_size(target_size: int):
    """Return square target H/W from a positive CLI size."""
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    return (int(target_size), int(target_size))


def pad_to_target(slice2d: np.ndarray, target_hw=TARGET_HW) -> np.ndarray:
    """Zero-pad a 2D slice to ``target_hw`` with center alignment."""
    h, w = slice2d.shape
    th, tw = target_hw
    if h > th or w > tw:
        raise ValueError(f"Cannot zero-pad slice shape {(h, w)} into smaller target {target_hw}; use resize mode.")
    out = np.zeros((th, tw), dtype=np.float32)
    y0 = (th - h) // 2
    x0 = (tw - w) // 2
    out[y0:y0 + h, x0:x0 + w] = slice2d
    return out


def resize_to_target(slice2d: np.ndarray, target_hw=TARGET_HW) -> np.ndarray:
    """Resize a 2D slice to ``target_hw`` using anti-aliased linear interpolation."""
    if slice2d.shape == tuple(target_hw):
        return slice2d.astype(np.float32, copy=False)

    from skimage.transform import resize

    return resize(
        slice2d,
        output_shape=target_hw,
        order=1,
        mode="constant",
        cval=0,
        clip=True,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32, copy=False)


def preprocess_slice(slice2d: np.ndarray, preprocess_mode="pad", target_hw=TARGET_HW) -> np.ndarray:
    """Apply the selected slice-size standardization policy."""
    if preprocess_mode == "pad":
        return pad_to_target(slice2d, target_hw=target_hw)
    if preprocess_mode == "resize":
        return resize_to_target(slice2d, target_hw=target_hw)
    raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")


def default_id_from_path(nifti_path: str) -> str:
    """Extract subject/case ID from a NIfTI filename."""
    name = output_prefix_from_path(nifti_path)
    if "_FLAIR" in name:
        return name.split("_FLAIR")[0]
    return name


def output_prefix_from_path(nifti_path: str) -> str:
    """Return the NIfTI filename stem while preserving preprocessing suffixes."""
    name = Path(nifti_path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def choose_middle_z(len_z: int) -> int:
    """Return a reproducible middle-slice index."""
    return (len_z - 1) // 2


def show_mid_slice_before_after(vol3d_before_pad, arr2d_after_pad, path_save=None, preprocess_mode="pad"):
    """Save a simple mid-slice QC image comparing before and after preprocessing."""
    if path_save is None:
        return

    import matplotlib.pyplot as plt

    z = choose_middle_z(vol3d_before_pad.shape[2])
    sl_before = vol3d_before_pad[:, :, z]
    sl_after = arr2d_after_pad[z]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sl_before.T, cmap="gray", origin="lower")
    axes[0].set_title(f"Original | z={z} | shape=({sl_before.shape[0]},{sl_before.shape[1]})")
    axes[0].axis("off")
    axes[1].imshow(sl_after.T, cmap="gray", origin="lower")
    axes[1].set_title(f"{preprocess_mode.title()} | z={z} | shape=({sl_after.shape[0]},{sl_after.shape[1]})")
    axes[1].axis("off")
    os.makedirs(Path(path_save).parent, exist_ok=True)
    print(f"save qc figure --> {path_save}")
    fig.savefig(path_save, bbox_inches="tight", dpi=150)
    plt.close(fig)


def find_existing_label_npy(nifti_path: Path) -> Path | None:
    """Return one existing slice-level label npy in the input case directory."""
    label_paths = sorted(
        p for p in nifti_path.parent.glob("*label_sliceLevel.npy")
        if p.is_file() and not p.name.startswith(".")
    )
    if not label_paths:
        return None
    if len(label_paths) > 1:
        preferred = [p for p in label_paths if output_prefix_from_path(nifti_path) in p.name]
        if len(preferred) == 1:
            return preferred[0]
        raise ValueError(f"Multiple label npy files found in {nifti_path.parent}: {[p.name for p in label_paths]}")
    return label_paths[0]


def load_labels_from_existing_npy(nifti_path: Path, z_count: int):
    """Load an existing per-slice label vector for original-volume resizing workflows."""
    label_path = find_existing_label_npy(nifti_path)
    if label_path is None:
        return None, None

    labels = np.load(label_path).astype(np.int32, copy=False)
    if labels.ndim != 1:
        raise ValueError(f"Existing label npy must be 1D, got shape={labels.shape}: {label_path}")
    if labels.shape[0] != z_count:
        raise ValueError(f"Existing label length {labels.shape[0]} does not match NIfTI Z={z_count}: {label_path}")
    return labels.copy(), label_path


def save_positive_slice_qc(
    vol3d_before: np.ndarray,
    arr2d_after: np.ndarray,
    positive_indices: np.ndarray,
    path_dir: Path,
    output_prefix: str,
    preprocess_mode: str,
) -> int:
    """Save before/after QC PNGs only for positive slice indices."""
    if positive_indices.size == 0:
        return 0

    import matplotlib.pyplot as plt

    def display_limits(slice2d: np.ndarray):
        values = slice2d[np.isfinite(slice2d)]
        if values.size == 0:
            return None, None
        vmin, vmax = np.percentile(values, (1, 99))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return None, None
        return vmin, vmax

    path_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for z in positive_indices:
        z = int(z)
        sl_before = vol3d_before[:, :, z]
        sl_after = arr2d_after[z]
        before_vmin, before_vmax = display_limits(sl_before)
        after_vmin, after_vmax = display_limits(sl_after)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(sl_before.T, cmap="gray", origin="lower", vmin=before_vmin, vmax=before_vmax)
        axes[0].set_title(f"Original | slice={z + 1}")
        axes[0].axis("off")
        axes[1].imshow(sl_after.T, cmap="gray", origin="lower", vmin=after_vmin, vmax=after_vmax)
        axes[1].set_title(f"{preprocess_mode.title()} | slice={z + 1}")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(path_dir / f"{output_prefix}_slice{z + 1:03d}_{preprocess_mode}_qc.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        saved += 1
    return saved


def pad_axial_slice(
    nifti_path: str,
    out_root: str,
    df_label: pd.DataFrame = None,
    normalize=True,
    label_column=DEFAULT_LABEL_COLUMN,
    preprocess_mode="pad",
    target_size=DEFAULT_TARGET_SIZE,
):
    """Convert one NIfTI volume into standardized axial slices and optional labels."""
    import nibabel as nib

    if preprocess_mode not in PREPROCESS_MODES:
        raise ValueError(f"preprocess_mode must be one of {sorted(PREPROCESS_MODES)}, got {preprocess_mode}")
    target_hw = target_hw_from_size(target_size)

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    img = nib.load(nifti_path)
    vol = img.get_fdata()
    z_count = vol.shape[2]

    subject_id = default_id_from_path(nifti_path)
    output_prefix = output_prefix_from_path(nifti_path)
    subj_dir = out_root / subject_id
    subj_dir.mkdir(parents=True, exist_ok=True)

    images = np.zeros((z_count, target_hw[0], target_hw[1]), dtype=np.float32)
    labels = np.zeros((z_count,), dtype=np.int32) if df_label is not None else None
    label_slices = None
    label_source = ""
    if labels is not None:
        try:
            label_condition = label_condition_for_subject(df_label, subject_id)
            label_slices = get_value_from_dataframe(
                df=df_label.copy(),
                col_name=label_column,
                conditional_pair=label_condition,
            )
            label_slices = np.array(parse_list_value(label_slices), dtype=int) - 1
            label_source = f"table:{label_column}"
        except (KeyError, ValueError) as exc:
            print(f"[WARN] label unavailable for {subject_id}; image/meta will be saved without label npy. {exc}")
            labels = None
    if labels is None:
        try:
            labels, existing_label_path = load_labels_from_existing_npy(Path(nifti_path), z_count)
            if labels is not None:
                label_slices = np.flatnonzero(labels > 0)
                label_source = f"existing_npy:{existing_label_path}"
                print(f"[INFO] loaded existing labels --> {existing_label_path}")
        except ValueError as exc:
            print(f"[WARN] existing label npy unavailable for {subject_id}; image/meta will be saved without label npy. {exc}")
            labels = None

    for z in range(z_count):
        sl = vol[:, :, z].astype(np.float32)
        if normalize:
            vmin, vmax = np.percentile(sl, (1, 99))
            if vmax > vmin:
                sl = np.clip((sl - vmin) / (vmax - vmin), 0, 1)
            else:
                sl = np.zeros_like(sl, dtype=np.float32)
        images[z] = preprocess_slice(sl, preprocess_mode=preprocess_mode, target_hw=target_hw)
        if labels is not None and label_slices is not None and z in label_slices:
            labels[z] = 1

    positive_indices = np.flatnonzero(labels > 0) if labels is not None else np.array([], dtype=int)
    positive_qc_count = save_positive_slice_qc(
        vol3d_before=vol,
        arr2d_after=images,
        positive_indices=positive_indices,
        path_dir=subj_dir / f"qc_{preprocess_mode}_positive",
        output_prefix=output_prefix,
        preprocess_mode=preprocess_mode,
    )

    show_mid_slice_before_after(
        vol3d_before_pad=vol,
        arr2d_after_pad=images,
        path_save=subj_dir / f"qc_{preprocess_mode}" / f"{output_prefix}.png",
        preprocess_mode=preprocess_mode,
    )

    output_suffix = output_suffix_for_mode(preprocess_mode)
    path_save_arr = subj_dir / f"{output_prefix}_AxialSlices_{output_suffix}.npy"
    np.save(path_save_arr, images)
    print(f"save slices --> {path_save_arr}")
    if labels is not None:
        path_save_label = subj_dir / f"{output_prefix}_label_sliceLevel.npy"
        np.save(path_save_label, labels)
        print(f"save labels --> {path_save_label}")

    hdr = img.header
    meta = {
        "nifti_path": str(nifti_path),
        "shape_xyz": list(vol.shape),
        "zooms": [float(x) for x in hdr.get_zooms()[:3]],
        "affine": img.affine.tolist(),
        "dtype_saved": "float32",
        "preprocess_mode": preprocess_mode,
        "target_hw": list(target_hw),
        "Z": int(z_count),
        "label_column": label_column if df_label is not None else "",
        "label_source": label_source,
        "label_saved": labels is not None,
        "positive_slice_indices_0based": [int(x) for x in positive_indices],
        "positive_slice_indices_onebased": [int(x) + 1 for x in positive_indices],
        "positive_qc_png_count": int(positive_qc_count),
    }
    with open(subj_dir / f"{output_prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return str(subj_dir)


def run_one_case(
    input_path,
    dst_dir=DEFAULT_DST_DIR,
    label_xlsx=DEFAULT_LABEL_XLSX,
    log_file=DEFAULT_LOG_FILE,
    normalize=True,
    overwrite=False,
    label_column=DEFAULT_LABEL_COLUMN,
    df_label: pd.DataFrame = None,
    label_xlsx_checked=False,
    preprocess_mode="pad",
    target_size=DEFAULT_TARGET_SIZE,
):
    """Export normalized axial slices for one NIfTI file."""
    input_path = Path(input_path)
    log_file = resolve_log_file(log_file, DEFAULT_LOG_FILE, dst_dir)
    out_path = expected_slice_output(dst_dir, input_path, preprocess_mode=preprocess_mode)
    if out_path.exists() and not overwrite:
        print(f"[SKIP] already done: {out_path}")
        return str(out_path.parent)

    if df_label is None and not label_xlsx_checked:
        df_label = load_label_table(label_xlsx)

    try:
        print("=" * 20)
        print(f"input: {input_path}")
        out_dir = pad_axial_slice(
            nifti_path=str(input_path),
            out_root=dst_dir,
            normalize=normalize,
            df_label=df_label.copy() if df_label is not None else None,
            label_column=label_column,
            preprocess_mode=preprocess_mode,
            target_size=target_size,
        )
        print()
        return out_dir
    except Exception as e:
        write_error(log_file, input_path, dst_dir, e)
        print(f"[ERROR] 변환 실패 -> 로그 저장: {input_path}")
        raise


def matches_include_keywords(filename, include_keywords=None):
    """Return True when filename contains at least one requested include keyword."""
    if not include_keywords:
        return True
    return any(keyword in filename for keyword in include_keywords)


def run_batch(
    src_dir=DEFAULT_SRC_DIR,
    dst_dir=DEFAULT_DST_DIR,
    label_xlsx=DEFAULT_LABEL_XLSX,
    log_file=DEFAULT_LOG_FILE,
    input_suffix="_reorient.nii.gz",
    normalize=True,
    overwrite=False,
    limit=None,
    pid=None,
    include_keywords=None,
    label_column=DEFAULT_LABEL_COLUMN,
    preprocess_mode="pad",
    target_size=DEFAULT_TARGET_SIZE,
):
    """Export normalized axial slices for every subject folder."""
    log_file = resolve_log_file(log_file, DEFAULT_LOG_FILE, dst_dir)
    df_label = load_label_table(label_xlsx)
    subjects = sorted([
        p for p in Path(src_dir).iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ])
    if pid is not None:
        subjects = [p for p in subjects if p.name == pid]
        if not subjects:
            raise FileNotFoundError(f"Subject folder not found: {Path(src_dir) / pid}")
    if limit is not None:
        subjects = subjects[:limit]

    for subject_dir in tqdm(subjects):
        input_files = sorted([
            p for p in subject_dir.iterdir()
            if not p.name.startswith(".")
            and p.name.endswith(input_suffix)
            and matches_include_keywords(p.name, include_keywords)
        ])
        if not input_files:
            print(f"[SKIP] resize input 없음: {subject_dir.name}")
            continue

        for input_path in input_files:
            try:
                run_one_case(
                    input_path=input_path,
                    dst_dir=dst_dir,
                    label_xlsx=label_xlsx,
                    log_file=log_file,
                    normalize=normalize,
                    overwrite=overwrite,
                    label_column=label_column,
                    df_label=df_label,
                    label_xlsx_checked=True,
                    preprocess_mode=preprocess_mode,
                    target_size=target_size,
                )
            except Exception as e:
                print(f"[ERROR] 변환 실패 -> 로그 저장: {input_path}")


def build_arg_parser():
    """Build CLI parser for step5 resize/normalization."""
    parser = argparse.ArgumentParser(description="Export RAS NIfTI volumes as padded/normalized npy slices.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(p):
        p.add_argument("--dst-dir", default=DEFAULT_DST_DIR)
        p.add_argument(
            "--label-xlsx",
            default=DEFAULT_LABEL_XLSX,
            help=(
                "Optional slice-level label table. If missing, or set to None, "
                "image npy/meta are saved and label npy is skipped."
            ),
        )
        p.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
        p.add_argument("--log-file", default=None, help="Default: <dst volume>/logs/err_resize.txt.")
        p.add_argument("--no-normalize", action="store_true")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument(
            "--preprocess-mode",
            choices=sorted(PREPROCESS_MODES),
            default="pad",
            help="Slice size standardization policy. Default preserves legacy zero-padding behavior.",
        )
        p.add_argument(
            "--target-size",
            type=int,
            default=DEFAULT_TARGET_SIZE,
            help="Square output H/W size. Default preserves legacy 672x672 outputs.",
        )

    single = subparsers.add_parser("single", help="Export one reoriented NIfTI file.")
    single.add_argument("--input-path", required=True)
    add_common_options(single)

    batch = subparsers.add_parser("batch", help="Run subject-folder batch.")
    batch.add_argument("--src-dir", default=DEFAULT_SRC_DIR)
    batch.add_argument("--input-suffix", default="_reorient.nii.gz")
    batch.add_argument("--limit", type=int, default=None)
    batch.add_argument("--pid", default=None, help="Run only one subject folder by PID.")
    batch.add_argument(
        "--include-keyword",
        action="append",
        dest="include_keywords",
        default=None,
        help=(
            "Only process candidate input files whose filename contains this keyword. "
            "Can be specified multiple times; a file is included if it matches any keyword."
        ),
    )
    add_common_options(batch)
    return parser


def cli_main(argv=None):
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    if args.command == "single":
        run_one_case(
            input_path=args.input_path,
            dst_dir=args.dst_dir,
            label_xlsx=args.label_xlsx,
            log_file=args.log_file,
            normalize=not args.no_normalize,
            overwrite=args.overwrite,
            label_column=args.label_column,
            preprocess_mode=args.preprocess_mode,
            target_size=args.target_size,
        )
        return 0

    run_batch(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        label_xlsx=args.label_xlsx,
        log_file=args.log_file,
        input_suffix=args.input_suffix,
        normalize=not args.no_normalize,
        overwrite=args.overwrite,
        limit=args.limit,
        pid=args.pid,
        include_keywords=args.include_keywords,
        label_column=args.label_column,
        preprocess_mode=args.preprocess_mode,
        target_size=args.target_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
