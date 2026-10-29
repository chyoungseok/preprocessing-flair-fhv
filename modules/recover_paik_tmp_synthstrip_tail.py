"""Recover PAIK preprocessing cases when Docker cannot mount /Volumes directly.

This keeps the normal Step 2-5 implementation and only routes SynthStrip
through /private/tmp, which Docker Desktop can still mount when /Volumes is
stale inside the VM.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from modules import step1_skull_stripping as step1
from modules import step2_bias_field_correction as step2
from modules import step3_masking as step3
from modules import step4_reorientation as step4
from modules import step5_resize_normalization as step5
from modules.run_flat_step1_to_step5_batch import find_inputs, subject_id_from_input


DEFAULT_SRC_DIR = "/Volumes/GURI_1/FLAIR_PAIK_nii"
DEFAULT_DST_DIR = "/Volumes/GURI_1/FLAIR_PAIK_nii_preproc"
DEFAULT_LABEL_XLSX = "/Users/yschoi/code/09_FHV/csv_files/df_flair_label_UHKinspection.xlsx"
DEFAULT_STATUS = "/Volumes/GURI_1/FLAIR_PAIK_nii_preproc/logs/paik_tmp_synthstrip_recovery_status.jsonl"
DEFAULT_TMP_ROOT = "/private/tmp/paik_tmp_synthstrip_recovery"


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_synthstrip_via_tmp(
    input_path: Path,
    subject_dir: Path,
    tmp_root: Path,
    image: str,
    platform: str | None,
    overwrite: bool,
    qc: bool,
) -> tuple[Path, Path]:
    subject_dir.mkdir(parents=True, exist_ok=True)
    step1.copy_original_input(input_path, subject_dir, overwrite=overwrite)
    final_brain, final_mask = step1.output_paths_for_input(input_path, subject_dir)
    if final_brain.exists() and final_mask.exists() and not overwrite:
        print(f"[SKIP] SynthStrip already done: {final_brain}, {final_mask}")
    else:
        work_dir = tmp_root / subject_dir.name
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp_input = work_dir / input_path.name
        shutil.copy2(input_path, tmp_input)
        tmp_brain, tmp_mask = step1.output_paths_for_input(tmp_input, work_dir)

        cmd = ["docker", "run", "--rm"]
        if platform:
            cmd += ["--platform", platform]
        cmd += [
            "-v",
            f"{work_dir}:/work",
            image,
            "-i",
            f"/work/{tmp_input.name}",
            "-o",
            f"/work/{tmp_brain.name}",
            "-m",
            f"/work/{tmp_mask.name}",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        if not tmp_brain.exists() or not tmp_mask.exists():
            raise FileNotFoundError(f"SynthStrip temporary outputs missing: {tmp_brain}, {tmp_mask}")
        shutil.copy2(tmp_brain, final_brain)
        shutil.copy2(tmp_mask, final_mask)
        shutil.rmtree(work_dir, ignore_errors=True)

    if qc:
        qc_path = subject_dir / "qc_skullstrip" / f"{subject_dir.name}_skullstrip_qc.png"
        step1.save_skullstrip_qc(input_path, final_brain, final_mask, qc_path)
    return final_brain, final_mask


def run_case(args: argparse.Namespace, input_path: Path) -> None:
    pid = subject_id_from_input(input_path)
    subject_dir = args.dst_dir / pid

    brain_path, mask_path = run_synthstrip_via_tmp(
        input_path=input_path,
        subject_dir=subject_dir,
        tmp_root=args.tmp_root,
        image=args.image,
        platform=args.platform,
        overwrite=args.overwrite_step1,
        qc=not args.no_qc,
    )
    corrected_path = Path(
        step2.run_one_case(
            in_path=str(brain_path),
            out_dir=str(subject_dir),
            out_name=pid,
            mask_path=str(mask_path),
            overwrite=args.overwrite,
        )
    )
    masked_path = Path(step3.default_masked_output(subject_dir, corrected_path))
    step3.run_one_case(
        brain_path=str(corrected_path),
        mask_path=str(mask_path),
        out_path=str(masked_path),
        overwrite=args.overwrite,
        qc=not args.no_qc,
    )
    reoriented_path = Path(step4.reorient_output_path(subject_dir, masked_path.name))
    step4.run_one_case(
        input_path=str(masked_path),
        output_path=str(reoriented_path),
        overwrite=args.overwrite,
        qc=not args.no_qc,
        target_orientation=args.step4_target_orientation,
    )
    step5.run_one_case(
        input_path=str(reoriented_path),
        dst_dir=str(args.dst_dir),
        label_xlsx=str(args.label_xlsx),
        normalize=True,
        overwrite=args.overwrite,
        label_column=args.label_column,
        preprocess_mode=args.step5_preprocess_mode,
        target_size=args.step5_target_size,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", default=DEFAULT_SRC_DIR)
    parser.add_argument("--dst-dir", default=DEFAULT_DST_DIR)
    parser.add_argument("--pattern", default="*_FLAIR.nii")
    parser.add_argument("--label-xlsx", default=DEFAULT_LABEL_XLSX)
    parser.add_argument("--label-column", default="fhv_slice_num")
    parser.add_argument("--start-index", type=int, default=2507)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pid", action="append", default=None, help="Run selected PID(s) instead of a sorted index range.")
    parser.add_argument("--status-jsonl", default=DEFAULT_STATUS)
    parser.add_argument("--tmp-root", default=DEFAULT_TMP_ROOT)
    parser.add_argument("--image", default=step1.DEFAULT_IMAGE)
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--step4-target-orientation", choices=sorted(step4.SUPPORTED_TARGET_ORIENTATIONS), default="LAS")
    parser.add_argument("--step5-preprocess-mode", choices=sorted(step5.PREPROCESS_MODES), default="resize")
    parser.add_argument("--step5-target-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite Step 2-5 outputs.")
    parser.add_argument("--overwrite-step1", action="store_true", help="Overwrite existing SynthStrip outputs.")
    parser.add_argument("--no-qc", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.src_dir = Path(args.src_dir).expanduser().resolve()
    args.dst_dir = Path(args.dst_dir).expanduser().resolve()
    args.label_xlsx = Path(args.label_xlsx).expanduser().resolve()
    args.status_jsonl = Path(args.status_jsonl).expanduser().resolve()
    args.tmp_root = Path(args.tmp_root).expanduser().resolve()

    inputs = find_inputs(args.src_dir, args.pattern)
    if args.pid:
        wanted = set(args.pid)
        inputs = [p for p in inputs if subject_id_from_input(p) in wanted]
    else:
        inputs = inputs[args.start_index :]
        if args.limit is not None:
            inputs = inputs[: args.limit]

    print(f"[RECOVERY] inputs={len(inputs)} dst={args.dst_dir}")
    print(f"[RECOVERY] status={args.status_jsonl}")
    failed = 0
    for input_path in inputs:
        pid = subject_id_from_input(input_path)
        index = find_inputs(args.src_dir, args.pattern).index(input_path)
        started_at = datetime.now().isoformat(timespec="seconds")
        start = time.perf_counter()
        print(f"[CASE {index + 1}] START {pid} {input_path}")
        try:
            run_case(args, input_path)
        except Exception as exc:
            failed += 1
            append_jsonl(
                args.status_jsonl,
                {
                    "pid": pid,
                    "input_path": str(input_path),
                    "index": index,
                    "status": "failed",
                    "started_at": started_at,
                    "elapsed_sec": round(time.perf_counter() - start, 3),
                    "error": repr(exc),
                },
            )
            print(f"[CASE {index + 1}] FAILED {pid}: {exc}")
            continue
        append_jsonl(
            args.status_jsonl,
            {
                "pid": pid,
                "input_path": str(input_path),
                "index": index,
                "status": "ok",
                "started_at": started_at,
                "elapsed_sec": round(time.perf_counter() - start, 3),
            },
        )
        print(f"[CASE {index + 1}] OK {pid}")
    print(f"[RECOVERY] done total={len(inputs)} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
