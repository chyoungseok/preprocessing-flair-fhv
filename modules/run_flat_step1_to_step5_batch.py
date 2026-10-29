"""Run the step 1-5 pipeline over a flat directory of FLAIR NIfTI files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_SRC_DIR = "/Volumes/GURI_1/FLAIR_PAIK_nii"
DEFAULT_DST_DIR = "/Volumes/GURI_1/FLAIR_PAIK_nii_preproc"
DEFAULT_LABEL_XLSX = "/Users/yschoi/code/09_FHV/csv_files/df_flair_label_UHKinspection.xlsx"
DEFAULT_LABEL_COLUMN = "fhv_slice_num"


def subject_id_from_input(input_path: Path) -> str:
    name = input_path.name
    stem = name[:-7] if name.endswith(".nii.gz") else input_path.stem
    if "_FLAIR" in stem:
        return stem.split("_FLAIR", 1)[0]
    return stem


def find_inputs(src_dir: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in src_dir.glob(pattern)
        if path.is_file() and not path.name.startswith(".")
    )


def latest_step5_meta(dst_dir: Path, pid: str) -> Path | None:
    subject_dir = dst_dir / pid
    metas = sorted(subject_dir.glob("*_reorient_meta.json"))
    if metas:
        return metas[-1]
    metas = sorted(subject_dir.glob("*_meta.json"))
    return metas[-1] if metas else None


def is_completed(dst_dir: Path, pid: str, preprocess_mode: str, target_size: int, target_orientation: str) -> bool:
    meta_path = latest_step5_meta(dst_dir, pid)
    if meta_path is None:
        return False

    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False

    if meta.get("preprocess_mode") != preprocess_mode:
        return False
    if meta.get("target_hw") != [target_size, target_size]:
        return False

    subject_dir = dst_dir / pid
    resized = list(subject_dir.glob(f"*_AxialSlices_{'resized' if preprocess_mode == 'resize' else 'padded'}.npy"))
    reorient_json = list((subject_dir / "qc_reorientation").glob("*_reorient_qc.json"))
    if target_orientation and reorient_json:
        try:
            reorient_meta = json.loads(reorient_json[-1].read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if reorient_meta.get("target_orientation") != target_orientation:
            return False
    return bool(resized)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_one(args: argparse.Namespace, input_path: Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "modules.step1_to_step5_pipeline",
        "single",
        "--input-path",
        str(input_path),
        "--dst-dir",
        str(args.dst_dir),
        "--label-xlsx",
        str(args.label_xlsx),
        "--label-column",
        args.label_column,
        "--step4-target-orientation",
        args.step4_target_orientation,
        "--step5-preprocess-mode",
        args.step5_preprocess_mode,
        "--step5-target-size",
        str(args.step5_target_size),
    ]
    if args.platform:
        cmd.extend(["--platform", args.platform])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.no_qc:
        cmd.append("--no-qc")
    if args.skip_docker_check:
        cmd.append("--skip-docker-check")

    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", default=DEFAULT_SRC_DIR)
    parser.add_argument("--dst-dir", default=DEFAULT_DST_DIR)
    parser.add_argument("--pattern", default="*_FLAIR.nii")
    parser.add_argument("--label-xlsx", default=DEFAULT_LABEL_XLSX)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--step4-target-orientation", choices=["LAS", "RAS"], default="LAS")
    parser.add_argument("--step5-preprocess-mode", choices=["pad", "resize"], default="resize")
    parser.add_argument("--step5-target-size", type=int, default=512)
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based index into the sorted input list.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip cases already matching the requested output settings.")
    parser.add_argument("--no-qc", action="store_true")
    parser.add_argument("--skip-docker-check", action="store_true")
    parser.add_argument(
        "--status-jsonl",
        default=None,
        help="Default: <dst-dir>/logs/flat_step1_to_step5_status_<timestamp>.jsonl",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.src_dir = Path(args.src_dir).expanduser().resolve()
    args.dst_dir = Path(args.dst_dir).expanduser().resolve()
    args.label_xlsx = Path(args.label_xlsx).expanduser().resolve()

    inputs = find_inputs(args.src_dir, args.pattern)
    if args.start_index:
        inputs = inputs[args.start_index :]
    if args.limit is not None:
        inputs = inputs[: args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    status_jsonl = Path(args.status_jsonl) if args.status_jsonl else args.dst_dir / "logs" / f"flat_step1_to_step5_status_{timestamp}.jsonl"
    status_jsonl = status_jsonl.expanduser().resolve()

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BATCH] inputs={len(inputs)} src={args.src_dir} dst={args.dst_dir}")
    print(f"[BATCH] status={status_jsonl}")
    print(
        "[BATCH] settings="
        f"orientation={args.step4_target_orientation} "
        f"mode={args.step5_preprocess_mode} "
        f"target={args.step5_target_size} "
        f"overwrite={args.overwrite} resume={args.resume}"
    )

    failed = 0
    skipped = 0
    for index, input_path in enumerate(inputs, start=args.start_index):
        pid = subject_id_from_input(input_path)
        started_at = datetime.now().isoformat(timespec="seconds")
        case_start = time.perf_counter()
        print(f"[CASE {index + 1}] START {pid} {input_path}")

        if args.resume and not args.overwrite and is_completed(
            args.dst_dir,
            pid,
            args.step5_preprocess_mode,
            args.step5_target_size,
            args.step4_target_orientation,
        ):
            skipped += 1
            append_jsonl(
                status_jsonl,
                {
                    "pid": pid,
                    "input_path": str(input_path),
                    "index": index,
                    "status": "skipped",
                    "started_at": started_at,
                    "elapsed_sec": 0,
                },
            )
            print(f"[CASE {index + 1}] SKIP {pid}")
            continue

        returncode = run_one(args, input_path)
        elapsed = round(time.perf_counter() - case_start, 3)
        status = "ok" if returncode == 0 else "failed"
        if returncode != 0:
            failed += 1

        append_jsonl(
            status_jsonl,
            {
                "pid": pid,
                "input_path": str(input_path),
                "index": index,
                "status": status,
                "returncode": returncode,
                "started_at": started_at,
                "elapsed_sec": elapsed,
            },
        )
        print(f"[CASE {index + 1}] {status.upper()} {pid} elapsed={elapsed}s")

    print(f"[BATCH] done total={len(inputs)} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
