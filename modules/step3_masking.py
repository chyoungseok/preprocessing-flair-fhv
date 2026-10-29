"""Apply a brain mask to preprocessed FLAIR NIfTI images.

CLI examples
------------
python -m modules.step3_masking batch
python -m modules.step3_masking single --brain-path brain.nii.gz --mask-path mask.nii.gz --out-path masked.nii.gz
"""

import argparse
import glob
import os
import sys
import numpy as np
from pathlib import Path

from modules.logging_utils import append_error_log, resolve_log_file


DEFAULT_SRC_DIR = "/Volumes/X31/FLAIR_HVS_detection_nii_preproc"
DEFAULT_LOG_FILE = "/Volumes/X31/FLAIR_preprocessing_logs/err_masking.txt"


def middle_slice(arr):
    """Return the middle axial slice from a 3D array."""
    return arr[:, :, arr.shape[2] // 2]


def save_masking_qc(brain_path, mask_path, out_path, qc_path):
    """Save a mid-slice QC PNG for bias-corrected brain, mask, and masked output."""
    import nibabel as nib
    import matplotlib.pyplot as plt

    brain_arr = nib.load(str(brain_path)).get_fdata()
    mask_arr = nib.load(str(mask_path)).get_fdata()
    out_arr = nib.load(str(out_path)).get_fdata()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("Brain", middle_slice(brain_arr), "gray"),
        ("Mask", middle_slice(mask_arr), "gray"),
        ("Masked", middle_slice(out_arr), "gray"),
    ]
    for ax, (title, data, cmap) in zip(axes, panels):
        ax.imshow(data.T, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.axis("off")

    qc_path = Path(qc_path)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(qc_path, dpi=150)
    plt.close(fig)
    print(f"[QC] saved: {qc_path}")


def apply_mask_and_save(brain_path: str, mask_path: str, out_path: str):
    """
    Brain NIfTI와 mask NIfTI를 로드하여 mask를 적용한 결과를 저장합니다.

    Args:
        brain_path (str): brain NIfTI 파일 경로 (.nii / .nii.gz).
        mask_path (str): brain mask NIfTI 파일 경로 (.nii / .nii.gz).
        out_path (str): 결과를 저장할 경로 (.nii / .nii.gz).

    Returns:
        str: 저장된 결과 NIfTI 파일 경로.
    """
    import nibabel as nib

    # --- 1) Brain과 Mask 로드 ---
    brain_nii = nib.load(brain_path)
    mask_nii = nib.load(mask_path)

    brain_data = brain_nii.get_fdata()  # (X,Y,Z)
    mask_data = mask_nii.get_fdata()    # (X,Y,Z)

    # --- 2) Mask 적용 ---
    # mask 값이 0인 부분은 제거 (즉, 곱하면 해당 voxel=0)
    masked_data = brain_data * (mask_data > 0)

    # --- 3) 결과 NIfTI 생성 ---
    masked_nii = nib.Nifti1Image(masked_data, affine=brain_nii.affine, header=brain_nii.header)

    # --- 4) 저장 ---
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(masked_nii, out_path)

    print(f"[INFO] Masked brain saved to {out_path}")
    return str(out_path)


def write_error(log_file, brain_path, out_path, error):
    """Append one failed case to the masking log."""
    append_error_log(log_file, brain_path, out_path, error)


def default_masked_output(subject_dir, brain_path):
    """Build the default ``*_masked.nii.gz`` output path."""
    name = Path(brain_path).name
    base = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
    return str(Path(subject_dir) / f"{base}_masked.nii.gz")


def run_one_case(brain_path, mask_path, out_path, overwrite=False, qc=True):
    """Apply one mask, skipping existing output unless ``overwrite`` is set."""
    if not overwrite and Path(out_path).exists():
        print(f"[SKIP] already done: {out_path}")
        if qc:
            qc_path = Path(out_path).parent / "qc_masking" / f"{Path(out_path).stem}_masking_qc.png"
            if not qc_path.exists():
                save_masking_qc(brain_path, mask_path, out_path, qc_path)
        return out_path
    out_path = apply_mask_and_save(brain_path=brain_path, mask_path=mask_path, out_path=out_path)
    if qc:
        qc_path = Path(out_path).parent / "qc_masking" / f"{Path(out_path).stem}_masking_qc.png"
        save_masking_qc(brain_path, mask_path, out_path, qc_path)
    return out_path


def matches_include_keywords(filename, include_keywords=None):
    """Return True when filename contains at least one requested include keyword."""
    if not include_keywords:
        return True
    return any(keyword in filename for keyword in include_keywords)


def is_visible_file(path):
    """Return True for regular input files, excluding macOS AppleDouble dot files."""
    return not Path(path).name.startswith(".")


def run_batch(
    src_dir=DEFAULT_SRC_DIR,
    log_file=DEFAULT_LOG_FILE,
    overwrite=False,
    limit=None,
    pid=None,
    qc=True,
    include_keywords=None,
):
    """Apply brain masks for every subject folder under ``src_dir``."""
    from tqdm.auto import tqdm
    log_file = resolve_log_file(log_file, DEFAULT_LOG_FILE, src_dir)

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
        try:
            brain_candidates = sorted([
                p for p in subject_dir.iterdir()
                if is_visible_file(p)
                and p.name.endswith("_bfCorrected.nii.gz")
                and "_masked" not in p.name
                and matches_include_keywords(p.name, include_keywords)
            ])
            mask_candidates = sorted(
                p for p in glob.glob(str(subject_dir / "*brainMask.nii.gz"))
                if is_visible_file(p)
            )
            if not brain_candidates:
                print(f"[SKIP] bfCorrected image 없음: {subject_dir.name}")
                continue
            if not mask_candidates:
                write_error(log_file, subject_dir, subject_dir, "brainMask file not found")
                print(f"[SKIP] brainMask 없음 -> 로그 저장: {subject_dir.name}")
                continue

            for brain_path in brain_candidates:
                out_path = default_masked_output(subject_dir, brain_path)
                run_one_case(
                    brain_path=str(brain_path),
                    mask_path=mask_candidates[0],
                    out_path=out_path,
                    overwrite=overwrite,
                    qc=qc,
                )
        except Exception as e:
            write_error(log_file, subject_dir, subject_dir, e)
            print(f"[ERROR] masking 실패 -> 로그 저장: {subject_dir}")


def build_arg_parser():
    """Build CLI parser for masking."""
    parser = argparse.ArgumentParser(description="Apply brain masks to NIfTI files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Apply one mask.")
    single.add_argument("--brain-path", required=True)
    single.add_argument("--mask-path", required=True)
    single.add_argument("--out-path", required=True)
    single.add_argument("--overwrite", action="store_true")
    single.add_argument("--no-qc", action="store_true")

    batch = subparsers.add_parser("batch", help="Run subject-folder batch.")
    batch.add_argument("--src-dir", default=DEFAULT_SRC_DIR)
    batch.add_argument("--log-file", default=None, help="Default: <src volume>/logs/err_masking.txt.")
    batch.add_argument("--overwrite", action="store_true")
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
    batch.add_argument("--no-qc", action="store_true")
    return parser


def cli_main(argv=None):
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    if args.command == "single":
        run_one_case(args.brain_path, args.mask_path, args.out_path, overwrite=args.overwrite, qc=not args.no_qc)
        return 0

    run_batch(
        src_dir=args.src_dir,
        log_file=args.log_file,
        overwrite=args.overwrite,
        limit=args.limit,
        pid=args.pid,
        qc=not args.no_qc,
        include_keywords=args.include_keywords,
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
