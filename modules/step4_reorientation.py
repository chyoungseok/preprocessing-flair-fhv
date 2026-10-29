"""Step 04: reorient masked FLAIR NIfTI files to a target orientation.

This step scans subject folders such as ``P00006``, finds ``*_masked.nii.gz``
files, and saves ``*_masked_reorient.nii.gz``. It does not modify label tables;
slice-level labels are kept in their original axial index order.

Examples
--------
python -m modules.step4_reorientation single --input-path in.nii.gz --output-path out.nii.gz
python -m modules.step4_reorientation batch --src-dir /path/preproc --dst-dir /path/preproc
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm.auto import tqdm

from modules.logging_utils import append_error_log, resolve_log_file


DEFAULT_LOG_FILE = "/Volumes/X31/FLAIR_preprocessing_logs/err_reorientation.txt"
DEFAULT_SRC_DIR = "/Volumes/X31/FLAIR_HVS_detection_nii_preproc"
DEFAULT_DST_DIR = "/Volumes/X31/FLAIR_HVS_detection_nii_preproc"
SUPPORTED_TARGET_ORIENTATIONS = {"RAS", "LAS"}


def write_error(log_file, input_path, output_path, error):
    """Append a case-level error message to the reorientation log."""
    append_error_log(log_file, input_path, output_path, error)


def normalize_target_orientation(target_orientation: str):
    """Return validated 3-letter target orientation tuple."""
    target = str(target_orientation).upper()
    if target not in SUPPORTED_TARGET_ORIENTATIONS:
        raise ValueError(f"target_orientation must be one of {sorted(SUPPORTED_TARGET_ORIENTATIONS)}, got {target_orientation}")
    return tuple(target)


def reorient_img(img, path_save=None, verbose=False, target_orientation="RAS"):
    """Reorient an image to target orientation without interpolation."""
    import nibabel as nib
    import numpy as np
    from nibabel.orientations import aff2axcodes, apply_orientation, axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform

    target_axcodes = normalize_target_orientation(target_orientation)
    ax_before = aff2axcodes(img.affine)
    if ax_before == target_axcodes:
        img_after = img
    elif target_axcodes == ("R", "A", "S"):
        img_after = nib.as_closest_canonical(img)
    else:
        original_ornt = io_orientation(img.affine)
        target_ornt = axcodes2ornt(target_axcodes)
        transform = ornt_transform(original_ornt, target_ornt)
        reoriented_data = apply_orientation(np.asanyarray(img.dataobj), transform)
        reoriented_affine = img.affine @ inv_ornt_aff(transform, img.shape)
        img_after = img.__class__(reoriented_data, reoriented_affine, header=img.header.copy())
    ax_after = aff2axcodes(img_after.affine)

    if verbose:
        print(f"Axcodes: {ax_before} -> {ax_after}")

    if path_save is not None:
        nib.save(img_after, str(path_save))
    return img_after


def reorient_case(in_path, out_path, verbose=True, target_orientation="RAS"):
    """Save one NIfTI in target orientation."""
    import nibabel as nib

    target_orientation = "".join(normalize_target_orientation(target_orientation))
    print(f"=== Reorientation to {target_orientation} ===")
    img = nib.load(str(in_path))
    reorient_img(img, path_save=out_path, verbose=verbose, target_orientation=target_orientation)
    print(f"Input : {in_path}")
    print(f"Output: {out_path}")


def save_reorientation_qc(input_path, output_path, qc_dir, target_orientation="RAS"):
    """Save orientation metadata and a before/after mid-slice QC PNG."""
    import nibabel as nib
    import matplotlib.pyplot as plt
    from nibabel.orientations import aff2axcodes

    input_img = nib.load(str(input_path))
    output_img = nib.load(str(output_path))
    target_axcodes = normalize_target_orientation(target_orientation)

    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    base = Path(output_path).name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    else:
        base = Path(base).stem

    qc_json = qc_dir / f"{base}_reorientation_qc.json"
    qc_png = qc_dir / f"{base}_reorientation_qc.png"
    with open(qc_json, "w") as f:
        json.dump(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "target_orientation": list(target_axcodes),
                "axcodes_before": list(aff2axcodes(input_img.affine)),
                "axcodes_after": list(aff2axcodes(output_img.affine)),
                "shape_before": list(input_img.shape[:3]),
                "shape_after": list(output_img.shape[:3]),
            },
            f,
            indent=2,
        )

    input_arr = input_img.get_fdata()
    output_arr = output_img.get_fdata()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(input_arr[:, :, input_arr.shape[2] // 2].T, cmap="gray", origin="lower")
    axes[0].set_title("Before")
    axes[0].axis("off")
    axes[1].imshow(output_arr[:, :, output_arr.shape[2] // 2].T, cmap="gray", origin="lower")
    axes[1].set_title("".join(target_axcodes))
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(qc_png, dpi=150)
    plt.close(fig)
    print(f"[QC] saved: {qc_json}, {qc_png}")


def reorient_output_path(subject_out_dir, input_name):
    """Build the ``*_reorient.nii.gz`` path for one input NIfTI file."""
    base = input_name[:-7] if input_name.endswith(".nii.gz") else Path(input_name).stem
    return os.path.join(subject_out_dir, f"{base}_reorient.nii.gz")


def run_one_case(input_path, output_path, overwrite=False, qc=True, qc_dir=None, target_orientation="RAS"):
    """Run target-orientation reorientation for one NIfTI file."""
    target_orientation = "".join(normalize_target_orientation(target_orientation))
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"[SKIP] already done: {output_path}")
        if qc:
            save_reorientation_qc(
                input_path=input_path,
                output_path=output_path,
                qc_dir=Path(qc_dir) if qc_dir is not None else output_path.parent / "qc_reorientation",
                target_orientation=target_orientation,
            )
        return str(output_path)

    reorient_case(
        in_path=input_path,
        out_path=output_path,
        verbose=True,
        target_orientation=target_orientation,
    )
    if qc:
        save_reorientation_qc(
            input_path=input_path,
            output_path=output_path,
            qc_dir=Path(qc_dir) if qc_dir is not None else output_path.parent / "qc_reorientation",
            target_orientation=target_orientation,
        )
    return str(output_path)


def matches_include_keywords(filename, include_keywords=None):
    """Return True when filename contains at least one requested include keyword."""
    if not include_keywords:
        return True
    return any(keyword in filename for keyword in include_keywords)


def run_batch(
    src_dir=DEFAULT_SRC_DIR,
    dst_dir=DEFAULT_DST_DIR,
    log_file=DEFAULT_LOG_FILE,
    input_suffix="_masked.nii.gz",
    overwrite=False,
    limit=None,
    pid=None,
    qc=True,
    include_keywords=None,
    target_orientation="RAS",
):
    """Run target-orientation reorientation for a batch."""
    target_orientation = "".join(normalize_target_orientation(target_orientation))
    log_file = resolve_log_file(log_file, DEFAULT_LOG_FILE, dst_dir)

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
        subject_out_dir = Path(dst_dir) / subject_dir.name
        subject_out_dir.mkdir(parents=True, exist_ok=True)

        input_files = sorted([
            p.name for p in subject_dir.iterdir()
            if not p.name.startswith(".")
            and p.name.endswith(input_suffix)
            and "_reorient" not in p.name
            and matches_include_keywords(p.name, include_keywords)
        ])
        if not input_files:
            print(f"[SKIP] reorientation input 없음: {subject_dir.name}")
            continue

        for input_name in input_files:
            input_path = subject_dir / input_name
            output_path = reorient_output_path(subject_out_dir, input_name)
            try:
                run_one_case(
                    input_path=input_path,
                    output_path=output_path,
                    overwrite=overwrite,
                    qc=qc,
                    qc_dir=subject_out_dir / "qc_reorientation",
                    target_orientation=target_orientation,
                )
            except Exception as e:
                write_error(log_file, input_path, output_path, e)
                print(f"[ERROR] 변환 실패 -> 로그 저장: {input_path}")


def build_arg_parser():
    """Build CLI parser for step4 reorientation."""
    parser = argparse.ArgumentParser(description="Reorient NIfTI files to a target orientation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(p):
        p.add_argument(
            "--target-orientation",
            choices=sorted(SUPPORTED_TARGET_ORIENTATIONS),
            default="RAS",
            help="Target orientation for voxel axis reorder/flip. No interpolation is performed.",
        )

    single = subparsers.add_parser("single", help="Reorient one NIfTI file.")
    single.add_argument("--input-path", required=True)
    single.add_argument("--output-path", required=True)
    single.add_argument("--overwrite", action="store_true")
    single.add_argument("--no-qc", action="store_true")
    add_common_options(single)

    batch = subparsers.add_parser("batch", help="Run subject-folder batch.")
    batch.add_argument("--src-dir", default=DEFAULT_SRC_DIR)
    batch.add_argument("--dst-dir", default=DEFAULT_DST_DIR)
    batch.add_argument("--log-file", default=None, help="Default: <dst volume>/logs/err_reorientation.txt.")
    batch.add_argument("--input-suffix", default="_masked.nii.gz")
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
    add_common_options(batch)
    return parser


def cli_main(argv=None):
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    if args.command == "single":
        run_one_case(
            input_path=args.input_path,
            output_path=args.output_path,
            overwrite=args.overwrite,
            qc=not args.no_qc,
            target_orientation=args.target_orientation,
        )
        return 0

    run_batch(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        log_file=args.log_file,
        input_suffix=args.input_suffix,
        overwrite=args.overwrite,
        limit=args.limit,
        pid=args.pid,
        qc=not args.no_qc,
        include_keywords=args.include_keywords,
        target_orientation=args.target_orientation,
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
