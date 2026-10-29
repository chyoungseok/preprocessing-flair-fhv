"""Step 07: convert GradCAM NPY volumes back to original FLAIR space.

Use this after training/inference on Step5 NPY volumes when the GradCAM result
has one heatmap per original axial slice, shaped ``(Z, 672, 672)``. The script
removes Step5 center padding, reconstructs the reoriented NIfTI volume shape,
then applies the inverse orientation transform so the heatmap can be overlaid
on the unprocessed original FLAIR NIfTI.

Examples
--------
Single case:

    python -m modules.step7_gradcam_to_original_space single \
        --gradcam-npy /path/P00006_gradcam.npy \
        --meta-json /Volumes/X31/FLAIR_HVS_detection_nii_preproc/P00006/P00006_FLAIR_brain_bfCorrected_masked_reorient_meta.json \
        --original-nii /Volumes/X31/org/FLAIR_nii/P00006/P00006_FLAIR.nii \
        --out-nii /Volumes/X31/FLAIR_HVS_detection_nii_preproc/P00006/P00006_gradcam_original_space.nii.gz

Batch:

    python -m modules.step7_gradcam_to_original_space batch \
        --gradcam-root /path/gradcam_npy \
        --preproc-root /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
        --original-root /Volumes/X31/org/FLAIR_nii \
        --out-root /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
        --include-keyword gradcam
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from modules.logging_utils import append_error_log, resolve_log_file


TARGET_HW = (672, 672)
DEFAULT_PREPROC_ROOT = "/Volumes/X31/FLAIR_HVS_detection_nii_preproc"
DEFAULT_ORIGINAL_ROOT = "/Volumes/X31/org/FLAIR_nii"
DEFAULT_LOG_FILE = "/Volumes/X31/FLAIR_preprocessing_logs/err_gradcam_to_original_space.txt"


def output_prefix_from_path(path):
    """Return filename stem while handling .nii.gz cleanly."""
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def subject_id_from_name(path):
    """Extract PID from common FLAIR-derived filenames."""
    name = output_prefix_from_path(path)
    if "_FLAIR" in name:
        return name.split("_FLAIR")[0]
    return name.split("_")[0]


def matches_include_keywords(filename, include_keywords=None):
    """Return True when filename contains at least one requested include keyword."""
    if not include_keywords:
        return True
    return any(keyword in filename for keyword in include_keywords)


def is_visible_file(path):
    """Exclude macOS AppleDouble and other dot files."""
    return not Path(path).name.startswith(".")


def write_error(log_file, input_path, output_path, error):
    """Append one failed case to the batch log."""
    append_error_log(log_file, input_path, output_path, error)


def load_meta(meta_json):
    """Load Step5 meta JSON and validate required keys."""
    with open(meta_json) as f:
        meta = json.load(f)
    missing = [key for key in ("nifti_path", "shape_xyz", "affine") if key not in meta]
    if missing:
        raise KeyError(f"Missing key(s) in meta JSON {meta_json}: {missing}")
    return meta


def unpad_672_volume(padded_volume, shape_xyz, target_hw=TARGET_HW):
    """Undo Step5 center padding from (Z, 672, 672) to (X, Y, Z)."""
    arr = np.asarray(padded_volume)
    if arr.ndim != 3:
        raise ValueError(f"Expected GradCAM shape (Z,H,W), got {arr.shape}")

    x, y, z = [int(v) for v in shape_xyz]
    target_h, target_w = target_hw
    if arr.shape[0] != z:
        raise ValueError(f"Slice count mismatch: GradCAM Z={arr.shape[0]} vs meta Z={z}")
    if arr.shape[1:] != (target_h, target_w):
        raise ValueError(f"Expected padded HW {(target_h, target_w)}, got {arr.shape[1:]}")
    if x > target_h or y > target_w:
        raise ValueError(f"Cannot unpad shape {(x, y)} from target {(target_h, target_w)}")

    y0 = (target_h - x) // 2
    x0 = (target_w - y) // 2
    cropped = arr[:, y0:y0 + x, x0:x0 + y]
    return np.transpose(cropped, (1, 2, 0)).astype(np.float32, copy=False)


def reorient_volume_to_original_space(volume_reoriented, reoriented_nii, original_nii):
    """Apply inverse orientation from reoriented NIfTI space to original FLAIR space."""
    import nibabel as nib
    from nibabel.orientations import apply_orientation, io_orientation, ornt_transform

    reor_img = nib.load(str(reoriented_nii))
    orig_img = nib.load(str(original_nii))

    reor_ornt = io_orientation(reor_img.affine)
    orig_ornt = io_orientation(orig_img.affine)
    transform = ornt_transform(reor_ornt, orig_ornt)
    volume_original = apply_orientation(volume_reoriented, transform)

    if volume_original.shape != orig_img.shape[:3]:
        raise ValueError(
            f"Original-space shape mismatch: transformed {volume_original.shape} "
            f"vs original {orig_img.shape[:3]}"
        )
    return volume_original.astype(np.float32, copy=False), orig_img


def save_overlay_qc(original_nii, heatmap_nii, out_png, alpha=0.45):
    """Save a mid-slice overlay QC image."""
    import nibabel as nib
    import matplotlib.pyplot as plt

    orig = nib.load(str(original_nii)).get_fdata()
    heat = nib.load(str(heatmap_nii)).get_fdata()
    z = heat.shape[2] // 2
    heat_slice = heat[:, :, z]
    heat_masked = np.ma.masked_where(heat_slice <= 0, heat_slice)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(orig[:, :, z].T, cmap="gray", origin="lower")
    ax.imshow(heat_masked.T, cmap="jet", origin="lower", alpha=alpha)
    ax.set_title(f"GradCAM in original FLAIR space | z={z}")
    ax.axis("off")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[QC] saved overlay: {out_png}")


def convert_one_case(
    gradcam_npy,
    meta_json,
    original_nii,
    out_nii,
    overwrite=False,
    qc=True,
    qc_png=None,
):
    """Convert one GradCAM NPY volume into original FLAIR NIfTI space."""
    import nibabel as nib

    out_nii = Path(out_nii)
    if out_nii.exists() and not overwrite:
        print(f"[SKIP] already done: {out_nii}")
        return str(out_nii)

    meta = load_meta(meta_json)
    reoriented_nii = Path(meta["nifti_path"])
    if not reoriented_nii.exists():
        raise FileNotFoundError(f"Reoriented NIfTI from meta not found: {reoriented_nii}")
    if not Path(original_nii).exists():
        raise FileNotFoundError(f"Original FLAIR NIfTI not found: {original_nii}")

    gradcam = np.load(gradcam_npy)
    volume_reoriented = unpad_672_volume(gradcam, meta["shape_xyz"])
    volume_original, orig_img = reorient_volume_to_original_space(
        volume_reoriented=volume_reoriented,
        reoriented_nii=reoriented_nii,
        original_nii=original_nii,
    )

    out_nii.parent.mkdir(parents=True, exist_ok=True)
    header = orig_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(volume_original, orig_img.affine, header), str(out_nii))
    print(f"[OK] saved: {out_nii}")

    if qc:
        if qc_png is None:
            qc_png = out_nii.with_name(f"{output_prefix_from_path(out_nii)}_overlay_qc.png")
        save_overlay_qc(original_nii=original_nii, heatmap_nii=out_nii, out_png=qc_png)
    return str(out_nii)


def find_one_file(directory, pattern, include_keywords=None):
    """Find exactly one visible file matching pattern and optional include keywords."""
    candidates = sorted(
        p for p in Path(directory).glob(pattern)
        if p.is_file() and is_visible_file(p) and matches_include_keywords(p.name, include_keywords)
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"Multiple candidates in {directory} for {pattern}: {candidates}")
    return candidates[0]


def default_original_nii(original_root, pid):
    """Return the expected original FLAIR NIfTI path for one PID."""
    return Path(original_root) / pid / f"{pid}_FLAIR.nii"


def default_out_nii(out_root, pid, gradcam_path):
    """Build default original-space GradCAM output path."""
    prefix = output_prefix_from_path(gradcam_path)
    return Path(out_root) / pid / f"{prefix}_original_FLAIR_space.nii.gz"


def run_batch(
    gradcam_root,
    preproc_root=DEFAULT_PREPROC_ROOT,
    original_root=DEFAULT_ORIGINAL_ROOT,
    out_root=DEFAULT_PREPROC_ROOT,
    gradcam_pattern="*.npy",
    meta_pattern="*_meta.json",
    include_keywords=None,
    log_file=DEFAULT_LOG_FILE,
    overwrite=False,
    limit=None,
    pid=None,
    qc=True,
):
    """Convert GradCAM NPY files for every PID folder under gradcam_root."""
    log_file = resolve_log_file(log_file, DEFAULT_LOG_FILE, out_root)
    gradcam_root = Path(gradcam_root)
    subjects = sorted(p for p in gradcam_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if pid is not None:
        subjects = [p for p in subjects if p.name == pid]
        if not subjects:
            raise FileNotFoundError(f"Subject folder not found: {gradcam_root / pid}")
    if limit is not None:
        subjects = subjects[:limit]

    for subject_dir in tqdm(subjects):
        pid_current = subject_dir.name
        try:
            gradcam_path = find_one_file(subject_dir, gradcam_pattern, include_keywords=include_keywords)
            if gradcam_path is None:
                print(f"[SKIP] GradCAM NPY 없음: {pid_current}")
                continue

            meta_path = find_one_file(Path(preproc_root) / pid_current, meta_pattern)
            if meta_path is None:
                print(f"[SKIP] meta JSON 없음: {pid_current}")
                continue

            original_nii = default_original_nii(original_root, pid_current)
            out_nii = default_out_nii(out_root, pid_current, gradcam_path)
            convert_one_case(
                gradcam_npy=gradcam_path,
                meta_json=meta_path,
                original_nii=original_nii,
                out_nii=out_nii,
                overwrite=overwrite,
                qc=qc,
            )
        except Exception as exc:
            out_dir = Path(out_root) / pid_current
            write_error(log_file, subject_dir, out_dir, exc)
            print(f"[ERROR] 변환 실패 -> 로그 저장: {subject_dir}")


def build_arg_parser():
    """Build CLI parser for GradCAM original-space conversion."""
    parser = argparse.ArgumentParser(
        description="Convert (Z,672,672) GradCAM NPY volumes back to original FLAIR NIfTI space."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Convert one GradCAM NPY file.")
    single.add_argument("--gradcam-npy", required=True, help="Input GradCAM NPY shaped (Z,672,672).")
    single.add_argument("--meta-json", required=True, help="Step5 meta JSON for the matching NPY input.")
    single.add_argument("--original-nii", required=True, help="Original unprocessed FLAIR NIfTI.")
    single.add_argument("--out-nii", required=True, help="Output GradCAM NIfTI in original FLAIR space.")
    single.add_argument("--overwrite", action="store_true", help="Overwrite existing output NIfTI.")
    single.add_argument("--no-qc", action="store_true", help="Do not save overlay QC PNG.")
    single.add_argument("--qc-png", default=None, help="Optional output overlay QC PNG path.")

    batch = subparsers.add_parser("batch", help="Convert subject-folder batch.")
    batch.add_argument("--gradcam-root", required=True, help="Root with PID folders containing GradCAM NPY files.")
    batch.add_argument("--preproc-root", default=DEFAULT_PREPROC_ROOT, help="Root with PID folders containing Step5 meta JSON.")
    batch.add_argument("--original-root", default=DEFAULT_ORIGINAL_ROOT, help="Root with original PID/PID_FLAIR.nii files.")
    batch.add_argument("--out-root", default=DEFAULT_PREPROC_ROOT, help="Root where output NIfTI files will be saved.")
    batch.add_argument("--gradcam-pattern", default="*.npy", help="Glob pattern for GradCAM NPY within each PID folder.")
    batch.add_argument("--meta-pattern", default="*_meta.json", help="Glob pattern for Step5 meta JSON within each PID folder.")
    batch.add_argument(
        "--include-keyword",
        action="append",
        dest="include_keywords",
        default=None,
        help="Only process GradCAM NPY files whose filename contains this keyword. Can be used multiple times.",
    )
    batch.add_argument("--log-file", default=None, help="Batch error log path. Default: <out volume>/logs/err_gradcam_to_original_space.txt.")
    batch.add_argument("--overwrite", action="store_true", help="Overwrite existing output NIfTI files.")
    batch.add_argument("--limit", type=int, default=None, help="Process only the first N sorted PID folders.")
    batch.add_argument("--pid", default=None, help="Run only one PID folder.")
    batch.add_argument("--no-qc", action="store_true", help="Do not save overlay QC PNG.")
    return parser


def cli_main(argv=None):
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    if args.command == "single":
        convert_one_case(
            gradcam_npy=args.gradcam_npy,
            meta_json=args.meta_json,
            original_nii=args.original_nii,
            out_nii=args.out_nii,
            overwrite=args.overwrite,
            qc=not args.no_qc,
            qc_png=args.qc_png,
        )
        return 0

    run_batch(
        gradcam_root=args.gradcam_root,
        preproc_root=args.preproc_root,
        original_root=args.original_root,
        out_root=args.out_root,
        gradcam_pattern=args.gradcam_pattern,
        meta_pattern=args.meta_pattern,
        include_keywords=args.include_keywords,
        log_file=args.log_file,
        overwrite=args.overwrite,
        limit=args.limit,
        pid=args.pid,
        qc=not args.no_qc,
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
