"""Run FLAIR preprocessing steps 1 through 5 as one pipeline.

Examples
--------
Run a whole subject-folder batch:

    python -m modules.step1_to_step5_pipeline batch \
        --src-dir /Volumes/X31/org/FLAIR_nii \
        --dst-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
        --platform linux/amd64

Run one NIfTI file:

    python -m modules.step1_to_step5_pipeline single \
        --input-path /Volumes/X31/org/FLAIR_nii/P00006/P00006_FLAIR.nii \
        --dst-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
        --platform linux/amd64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from modules import step1_skull_stripping as step1
from modules import step2_bias_field_correction as step2
from modules import step3_masking as step3
from modules import step4_reorientation as step4
from modules import step5_resize_normalization as step5


DEFAULT_ORIGINAL_SRC_DIR = "/Volumes/X31/org/FLAIR_nii"
DEFAULT_PREPROC_DIR = "/Volumes/X31/FLAIR_HVS_detection_nii_preproc"
DEFAULT_LABEL_XLSX = "/Users/yschoi/code/09_FHV/modules_v02/df_flair_label_UHKinspection.xlsx"
DEFAULT_LABEL_COLUMN = "fhv_slice_num"
STEP2_INCLUDE_KEYWORD = "_FLAIR_brain.nii.gz"
STEP3_INCLUDE_KEYWORD = "_FLAIR_brain_bfCorrected.nii.gz"
STEP4_INCLUDE_KEYWORD = "_FLAIR_brain_bfCorrected_masked.nii.gz"
STEP5_INCLUDE_KEYWORD = "_FLAIR_brain_bfCorrected_masked_reorient.nii.gz"


def _subject_id_from_input(input_path: Path, explicit_pid: str | None = None) -> str:
    if explicit_pid:
        return explicit_pid
    name = input_path.name
    stem = name[:-7] if name.endswith(".nii.gz") else input_path.stem
    if "_FLAIR" in stem:
        return stem.split("_FLAIR", 1)[0]
    if input_path.parent.name:
        return input_path.parent.name
    return stem


def _print_step(step_name: str) -> None:
    print()
    print("=" * 80)
    print(step_name)
    print("=" * 80)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{seconds:.2f}s"


def _run_timed(step_name: str, timings: list[tuple[str, float]], func, *args, **kwargs):
    _print_step(step_name)
    start = time.perf_counter()
    try:
        return func(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        timings.append((step_name, elapsed))
        print(f"[TIME] {step_name}: {_format_duration(elapsed)}")


def _print_timing_summary(title: str, timings: list[tuple[str, float]], total_seconds: float) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    for step_name, elapsed in timings:
        print(f"{step_name}: {_format_duration(elapsed)}")
    print("-" * 80)
    print(f"Total: {_format_duration(total_seconds)} ({total_seconds:.2f} sec)")


def run_batch_pipeline(args: argparse.Namespace) -> None:
    """Run steps 1-5 for subject folders under ``args.src_dir``."""
    step5_dst_dir = args.step5_dst_dir or args.dst_dir
    gpu = step1._parse_gpu(args.gpu)
    timings: list[tuple[str, float]] = []
    total_start = time.perf_counter()

    try:
        if not args.skip_docker_check:
            step1.check_docker_available()

        _run_timed(
            "Step 1/5: Skull stripping with SynthStrip",
            timings,
            step1.run_batch,
            src_dir=args.src_dir,
            dst_dir=args.dst_dir,
            input_pattern=args.input_pattern,
            log_file=args.step1_log_file,
            image=args.image,
            use_gpu=args.use_gpu,
            gpu=gpu,
            threads=args.threads,
            platform=args.platform,
            overwrite=args.overwrite,
            limit=args.limit,
            pid=args.pid,
            qc=not args.no_qc,
        )

        _run_timed(
            "Step 2/5: N4 bias field correction",
            timings,
            step2.run_batch,
            src_dir=args.dst_dir,
            dst_dir=args.dst_dir,
            log_file=args.step2_log_file,
            overwrite=args.overwrite,
            limit=args.limit,
            pid=args.pid,
            include_keywords=[args.step2_include_keyword],
        )

        _run_timed(
            "Step 3/5: Apply brain mask",
            timings,
            step3.run_batch,
            src_dir=args.dst_dir,
            log_file=args.step3_log_file,
            overwrite=args.overwrite,
            limit=args.limit,
            pid=args.pid,
            qc=not args.no_qc,
            include_keywords=[args.step3_include_keyword],
        )

        _run_timed(
            f"Step 4/5: Reorientation to {args.step4_target_orientation}",
            timings,
            step4.run_batch,
            src_dir=args.dst_dir,
            dst_dir=args.dst_dir,
            log_file=args.step4_log_file,
            input_suffix=args.step4_input_suffix,
            overwrite=args.overwrite,
            limit=args.limit,
            pid=args.pid,
            qc=not args.no_qc,
            include_keywords=[args.step4_include_keyword],
            target_orientation=args.step4_target_orientation,
        )

        _run_timed(
            "Step 5/5: Resize, pad, and normalize",
            timings,
            step5.run_batch,
            src_dir=args.dst_dir,
            dst_dir=step5_dst_dir,
            label_xlsx=args.label_xlsx,
            log_file=args.step5_log_file,
            input_suffix=args.step5_input_suffix,
            normalize=not args.no_normalize,
            overwrite=args.overwrite,
            limit=args.limit,
            pid=args.pid,
            include_keywords=[args.step5_include_keyword],
            label_column=args.label_column,
            preprocess_mode=args.step5_preprocess_mode,
            target_size=args.step5_target_size,
        )
    finally:
        _print_timing_summary("Pipeline timing summary", timings, time.perf_counter() - total_start)


def run_single_pipeline(args: argparse.Namespace) -> None:
    """Run steps 1-5 for one NIfTI file."""
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input NIfTI not found: {input_path}")

    pid = _subject_id_from_input(input_path, args.pid)
    subject_out_dir = Path(args.dst_dir).expanduser().resolve() / pid
    step5_dst_dir = args.step5_dst_dir or args.dst_dir
    gpu = step1._parse_gpu(args.gpu)
    timings: list[tuple[str, float]] = []
    total_start = time.perf_counter()

    try:
        if not args.skip_docker_check:
            step1.check_docker_available()

        brain_path, mask_path = _run_timed(
            "Step 1/5: Skull stripping with SynthStrip",
            timings,
            step1.run_one_case,
            input_path=str(input_path),
            output_dir=str(subject_out_dir),
            image=args.image,
            use_gpu=args.use_gpu,
            gpu=gpu,
            threads=args.threads,
            platform=args.platform,
            overwrite=args.overwrite,
            qc=not args.no_qc,
        )

        corrected_path = _run_timed(
            "Step 2/5: N4 bias field correction",
            timings,
            step2.run_one_case,
            in_path=str(brain_path),
            out_dir=str(subject_out_dir),
            out_name=pid,
            mask_path=str(mask_path),
            overwrite=args.overwrite,
        )

        masked_path = step3.default_masked_output(subject_out_dir, corrected_path)
        _run_timed(
            "Step 3/5: Apply brain mask",
            timings,
            step3.run_one_case,
            brain_path=str(corrected_path),
            mask_path=str(mask_path),
            out_path=str(masked_path),
            overwrite=args.overwrite,
            qc=not args.no_qc,
        )

        reoriented_path = step4.reorient_output_path(subject_out_dir, Path(masked_path).name)
        _run_timed(
            f"Step 4/5: Reorientation to {args.step4_target_orientation}",
            timings,
            step4.run_one_case,
            input_path=str(masked_path),
            output_path=str(reoriented_path),
            overwrite=args.overwrite,
            qc=not args.no_qc,
            target_orientation=args.step4_target_orientation,
        )

        _run_timed(
            "Step 5/5: Resize, pad, and normalize",
            timings,
            step5.run_one_case,
            input_path=str(reoriented_path),
            dst_dir=step5_dst_dir,
            label_xlsx=args.label_xlsx,
            log_file=args.step5_log_file,
            normalize=not args.no_normalize,
            overwrite=args.overwrite,
            label_column=args.label_column,
            preprocess_mode=args.step5_preprocess_mode,
            target_size=args.step5_target_size,
        )
    finally:
        _print_timing_summary(f"Pipeline timing summary for {pid}", timings, time.perf_counter() - total_start)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dst-dir", default=DEFAULT_PREPROC_DIR, help="Root directory for step 1-4 outputs.")
    parser.add_argument(
        "--step5-dst-dir",
        default=None,
        help="Root directory for step 5 npy outputs. Defaults to --dst-dir.",
    )
    parser.add_argument(
        "--label-xlsx",
        default=DEFAULT_LABEL_XLSX,
        help=(
            "Optional original slice-level label Excel/CSV file for Step 5. "
            "If missing, or set to None, image npy/meta are saved and label npy is skipped."
        ),
    )
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN, help="Step 5 label column to read from --label-xlsx.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run steps even when expected outputs exist.")
    parser.add_argument("--pid", default=None, help="Run one subject ID only.")
    parser.add_argument("--no-qc", action="store_true", help="Disable QC PNG/JSON outputs where supported.")
    parser.add_argument("--no-normalize", action="store_true", help="Step 5: save padded slices without normalization.")
    parser.add_argument(
        "--step5-preprocess-mode",
        choices=sorted(step5.PREPROCESS_MODES),
        default="pad",
        help="Step 5 slice size standardization policy. Default preserves legacy zero-padding behavior.",
    )
    parser.add_argument(
        "--step5-target-size",
        type=int,
        default=step5.DEFAULT_TARGET_SIZE,
        help="Step 5 square output H/W size. Default preserves legacy 672x672 outputs.",
    )

    parser.add_argument("--image", default=step1.DEFAULT_IMAGE, help="Docker SynthStrip image tag.")
    parser.add_argument("--use-gpu", action="store_true", help="Use Docker GPU and SynthStrip -g.")
    parser.add_argument("--gpu", default=None, help="GPU spec: all, device=0, or 0,1.")
    parser.add_argument("--threads", type=int, default=None, help="SynthStrip CPU thread count.")
    parser.add_argument("--platform", default=None, help='Docker platform, e.g. "linux/amd64".')
    parser.add_argument("--skip-docker-check", action="store_true", help="Skip Docker daemon availability check.")

    parser.add_argument("--step1-log-file", default=None, help="Default: <dst volume>/logs/err_skullstrip.txt.")
    parser.add_argument("--step2-log-file", default=None, help="Default: <dst volume>/logs/err_bias_field_correction.txt.")
    parser.add_argument("--step3-log-file", default=None, help="Default: <dst volume>/logs/err_masking.txt.")
    parser.add_argument("--step4-log-file", default=None, help="Default: <dst volume>/logs/err_reorientation.txt.")
    parser.add_argument("--step5-log-file", default=None, help="Default: <step5 dst volume>/logs/err_resize.txt.")

    parser.add_argument("--step2-include-keyword", default=STEP2_INCLUDE_KEYWORD)
    parser.add_argument("--step3-include-keyword", default=STEP3_INCLUDE_KEYWORD)
    parser.add_argument("--step4-include-keyword", default=STEP4_INCLUDE_KEYWORD)
    parser.add_argument("--step5-include-keyword", default=STEP5_INCLUDE_KEYWORD)
    parser.add_argument("--step4-input-suffix", default="_masked.nii.gz")
    parser.add_argument("--step5-input-suffix", default="_reorient.nii.gz")
    parser.add_argument(
        "--step4-target-orientation",
        choices=sorted(step4.SUPPORTED_TARGET_ORIENTATIONS),
        default="RAS",
        help="Step 4 target orientation. Uses axis reorder/flip only; no interpolation.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FLAIR preprocessing steps 1 through 5.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser("batch", help="Run all subject folders under --src-dir.")
    batch.add_argument("--src-dir", default=DEFAULT_ORIGINAL_SRC_DIR, help="Root directory with original subject folders.")
    batch.add_argument("--input-pattern", default=step1.DEFAULT_BATCH_PATTERN, help="Step 1 input glob pattern per subject.")
    batch.add_argument("--limit", type=int, default=None, help="Process only the first N sorted subject folders.")
    _add_common_options(batch)

    single = subparsers.add_parser("single", help="Run one input NIfTI file.")
    single.add_argument("--input-path", required=True, help="Input .nii or .nii.gz path.")
    _add_common_options(single)

    return parser


def cli_main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "batch":
        run_batch_pipeline(args)
        return 0
    if args.command == "single":
        run_single_pipeline(args)
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(cli_main())
