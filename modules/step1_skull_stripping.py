"""Run SynthStrip skull stripping through Docker.

The module can be imported as a normal Python API or executed as a CLI.
It supports one-case execution and subject-folder batch execution.

CLI examples
------------
Run one NIfTI file:

    python -m modules.step1_skull_stripping single \
        --input-path /Volumes/X31/FLAIR_HVS_detection_nii_preproc/P00006/P00006_FLAIR.nii \
        --output-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc/P00006

Run every subject folder under a root directory:

    python -m modules.step1_skull_stripping batch \
        --src-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
        --dst-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc

On Apple Silicon, add this if Docker image compatibility fails:

    --platform linux/amd64
"""

import argparse
import glob
import os
import shutil
import subprocess as sp
import sys
from pathlib import Path
from typing import Optional, List, Union

from tqdm.auto import tqdm

from modules.logging_utils import append_error_log, resolve_log_file


DEFAULT_IMAGE = "freesurfer/synthstrip:latest"
DEFAULT_SRC_DIR = "/Volumes/X31/FLAIR_HVS_detection_nii_preproc"
DEFAULT_DST_DIR = "/Volumes/X31/FLAIR_HVS_detection_nii_preproc"
DEFAULT_LOG_FILE = "/Volumes/X31/FLAIR_preprocessing_logs/err_skullstrip.txt"
DEFAULT_BATCH_PATTERN = "*_FLAIR.nii*"
EXCLUDE_INPUT_KEYWORDS = (
    "_brain",
    "brainMask",
    "bfCorrected",
    "reorient",
    "masked",
)


def middle_slice(arr):
    """Return the middle axial slice from a 3D array."""
    return arr[:, :, arr.shape[2] // 2]


def save_skullstrip_qc(input_path, brain_path, mask_path, qc_path):
    """Save a mid-slice QC PNG for input, stripped brain, and mask."""
    import nibabel as nib
    import matplotlib.pyplot as plt

    input_arr = nib.load(str(input_path)).get_fdata()
    brain_arr = nib.load(str(brain_path)).get_fdata()
    mask_arr = nib.load(str(mask_path)).get_fdata()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("Input", middle_slice(input_arr), "gray"),
        ("Brain", middle_slice(brain_arr), "gray"),
        ("Mask", middle_slice(mask_arr), "gray"),
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


def check_docker_available() -> None:
    """Docker daemon에 연결 가능한지 확인한다."""
    try:
        sp.run(
            ["docker", "info"],
            check=True,
            stdout=sp.DEVNULL,
            stderr=sp.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Docker command not found. Docker Desktop을 설치하고 실행하세요.") from exc
    except sp.CalledProcessError as exc:
        msg = exc.stderr.strip()
        raise RuntimeError(
            "Docker daemon에 연결할 수 없습니다. Docker Desktop을 먼저 실행한 뒤 다시 시도하세요."
            f"\nDocker error: {msg}"
        ) from exc


def output_paths_for_input(input_path: Union[str, Path], output_dir: Union[str, Path]):
    """Return expected SynthStrip brain and mask output paths."""
    in_path = Path(input_path)
    out_dir = Path(output_dir)
    fname = in_path.name
    base = fname[:-7] if fname.endswith(".nii.gz") else in_path.stem
    return out_dir / f"{base}_brain.nii.gz", out_dir / f"{base}_brainMask.nii.gz"


def copy_original_input(input_path: Union[str, Path], output_dir: Union[str, Path], overwrite: bool = False):
    """Copy the original input NIfTI into the subject output directory."""
    in_path = Path(input_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    copied_path = out_dir / in_path.name

    if in_path == copied_path.resolve():
        return copied_path
    if copied_path.exists() and not overwrite:
        print(f"[SKIP] original already copied: {copied_path}")
        return copied_path

    shutil.copy2(in_path, copied_path)
    print(f"[COPY] original NIfTI: {in_path} -> {copied_path}")
    return copied_path


def run_synthstrip(
    input_path: str,
    output_dir: str,
    image: str = DEFAULT_IMAGE,
    use_gpu: bool = False,
    gpu: Optional[Union[str, List[int]]] = None,  # "all" 또는 [0], [1], [0,1] 등
    threads: Optional[int] = None,
    platform: Optional[str] = None,
):
    """
    Run SynthStrip via Docker, with optional GPU pinning.

    Parameters
    ----------
    input_path : str
        입력 NIfTI 경로 (.nii / .nii.gz).
    output_dir : str
        출력 폴더.
    image : str
        Docker 이미지 태그.
    use_gpu : bool
        True면 GPU 사용(-g) 및 --gpus 옵션 전달.
    gpu : "all" 또는 리스트
        GPU 선택. "all"이면 모든 GPU, [0]이면 0번, [1]이면 1번, [0,1]이면 0/1번.
        use_gpu=False면 무시됨.
    threads : Optional[int]
        CPU 스레드 수(-t). GPU 모드에서도 프리/후처리에 도움될 수 있음.
    platform : Optional[str]
        Docker platform 옵션. Apple Silicon에서 이미지 호환 문제가 있으면 "linux/amd64" 사용.
    """
    in_path = Path(input_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input NIfTI not found: {in_path}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    brain_out, mask_out = output_paths_for_input(in_path, out_dir)

    host_dir = str(in_path.parent)
    out_dir_host = str(out_dir)
    file_name = in_path.name

    cmd = ["docker", "run", "--rm"]
    if platform:
        cmd += ["--platform", platform]

    # GPU 전달: --gpus <spec>
    if use_gpu:
        if gpu is None or gpu == "all":
            cmd += ["--gpus", "all"]
        elif isinstance(gpu, list) and gpu and all(isinstance(x, int) for x in gpu):
            device_str = ",".join(str(x) for x in gpu)
            cmd += ["--gpus", f"device={device_str}"]
        elif isinstance(gpu, str):
            # 임의의 문자열을 직접 전달하고 싶을 때 (예: "device=0")
            cmd += ["--gpus", gpu]
        else:
            raise ValueError("gpu 파라미터는 'all', List[int], 또는 'device=...' 문자열이어야 합니다.")

    # 볼륨 마운트
    cmd += [
        "-v", f"{host_dir}:/data",
        "-v", f"{out_dir_host}:/out",
        image,                      # ENTRYPOINT=mri_synthstrip
        "-i", f"/data/{file_name}",
        "-o", f"/out/{brain_out.name}",
        "-m", f"/out/{mask_out.name}",
    ]

    # 스레드 옵션
    if threads is not None and threads > 0:
        cmd += ["-t", str(threads)]

    # GPU 스위치(-g)
    if use_gpu:
        cmd += ["-g"]

    print("Running:", " ".join(cmd))
    sp.run(cmd, check=True)
    if not brain_out.exists() or not mask_out.exists():
        missing = [str(p) for p in (brain_out, mask_out) if not p.exists()]
        raise FileNotFoundError(f"SynthStrip finished but output file(s) are missing: {missing}")

    return brain_out, mask_out


def _parse_gpu(gpu_value: Optional[str]):
    """Parse CLI GPU values into Docker GPU specifications."""
    if gpu_value is None:
        return None
    if gpu_value == "all" or gpu_value.startswith("device="):
        return gpu_value
    try:
        return [int(x.strip()) for x in gpu_value.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "gpu must be 'all', 'device=0', or comma-separated integers such as '0,1'."
        ) from exc


def _is_candidate_input(path: Union[str, Path]):
    """Keep original FLAIR inputs and avoid derived preprocessing outputs."""
    name = Path(path).name
    if name.startswith("."):
        return False
    return not any(keyword in name for keyword in EXCLUDE_INPUT_KEYWORDS)


def find_subject_input(subject_dir: Union[str, Path], pattern: str):
    """Find one input NIfTI under a subject folder."""
    subject_dir = Path(subject_dir)
    candidates = sorted(
        Path(p) for p in glob.glob(str(subject_dir / pattern))
        if _is_candidate_input(p)
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"Multiple input candidates found in {subject_dir}: {candidates}")
    return candidates[0]


def write_error(log_file, input_path, output_dir, error):
    """Append one failed case to the batch log."""
    append_error_log(log_file, input_path, output_dir, error)


def run_one_case(
    input_path,
    output_dir,
    image=DEFAULT_IMAGE,
    use_gpu=False,
    gpu=None,
    threads=None,
    platform=None,
    overwrite=False,
    qc=True,
):
    """Run one SynthStrip case, skipping existing outputs unless overwritten."""
    copy_original_input(input_path, output_dir, overwrite=overwrite)
    brain_out, mask_out = output_paths_for_input(input_path, output_dir)
    if not overwrite and brain_out.exists() and mask_out.exists():
        print(f"[SKIP] already done: {brain_out}, {mask_out}")
        if qc:
            qc_path = Path(output_dir) / "qc_skullstrip" / f"{Path(input_path).stem}_skullstrip_qc.png"
            if not qc_path.exists():
                save_skullstrip_qc(input_path, brain_out, mask_out, qc_path)
        return brain_out, mask_out

    brain_out, mask_out = run_synthstrip(
        input_path=str(input_path),
        output_dir=str(output_dir),
        image=image,
        use_gpu=use_gpu,
        gpu=gpu,
        threads=threads,
        platform=platform,
    )
    if qc:
        qc_path = Path(output_dir) / "qc_skullstrip" / f"{Path(input_path).stem}_skullstrip_qc.png"
        save_skullstrip_qc(input_path, brain_out, mask_out, qc_path)
    return brain_out, mask_out


def run_batch(
    src_dir,
    dst_dir,
    input_pattern=DEFAULT_BATCH_PATTERN,
    log_file=DEFAULT_LOG_FILE,
    image=DEFAULT_IMAGE,
    use_gpu=False,
    gpu=None,
    threads=None,
    platform=None,
    overwrite=False,
    limit=None,
    pid=None,
    qc=True,
):
    """Run SynthStrip for every subject folder under ``src_dir``."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    log_file = resolve_log_file(log_file, DEFAULT_LOG_FILE, dst_dir)
    subjects = sorted([
        p for p in src_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ])
    if pid is not None:
        subjects = [p for p in subjects if p.name == pid]
        if not subjects:
            raise FileNotFoundError(f"Subject folder not found: {src_dir / pid}")
    if limit is not None:
        subjects = subjects[:limit]

    for subject_dir in tqdm(subjects):
        output_dir = dst_dir / subject_dir.name
        try:
            input_path = find_subject_input(subject_dir, input_pattern)
            if input_path is None:
                print(f"[SKIP] NIfTI 파일 없음: {subject_dir.name}")
                continue

            run_one_case(
                input_path=input_path,
                output_dir=output_dir,
                image=image,
                use_gpu=use_gpu,
                gpu=gpu,
                threads=threads,
                platform=platform,
                overwrite=overwrite,
                qc=qc,
            )
        except Exception as e:
            write_error(log_file, subject_dir, output_dir, e)
            print(f"[ERROR] 변환 실패 -> 로그 저장: {subject_dir}")


def build_arg_parser():
    """Build the command-line parser for single and batch SynthStrip runs."""
    parser = argparse.ArgumentParser(
        description="Run Docker SynthStrip for one NIfTI or a subject-folder batch."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(p):
        p.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image tag.")
        p.add_argument("--use-gpu", action="store_true", help="Use Docker GPU and SynthStrip -g.")
        p.add_argument("--gpu", default=None, help="GPU spec: all, device=0, or 0,1.")
        p.add_argument("--threads", type=int, default=None, help="SynthStrip CPU thread count.")
        p.add_argument("--platform", default=None, help='Docker platform, e.g. "linux/amd64".')
        p.add_argument("--overwrite", action="store_true", help="Re-run even when outputs exist.")
        p.add_argument("--no-qc", action="store_true", help="Do not generate QC PNG.")
        p.add_argument(
            "--skip-docker-check",
            action="store_true",
            help="Skip the initial Docker daemon availability check.",
        )

    single = subparsers.add_parser("single", help="Run one NIfTI file.")
    single.add_argument("--input-path", required=True, help="Input .nii or .nii.gz path.")
    single.add_argument("--output-dir", required=True, help="Output directory.")
    add_common_options(single)

    batch = subparsers.add_parser("batch", help="Run all subject folders under a root directory.")
    batch.add_argument("--src-dir", default=DEFAULT_SRC_DIR, help="Root directory with subject folders.")
    batch.add_argument("--dst-dir", default=DEFAULT_DST_DIR, help="Root output directory.")
    batch.add_argument("--input-pattern", default=DEFAULT_BATCH_PATTERN, help="Input glob pattern per subject.")
    batch.add_argument(
        "--log-file",
        default=None,
        help="Batch error log path. Default: <dst volume>/logs/err_skullstrip.txt.",
    )
    batch.add_argument("--limit", type=int, default=None, help="Process only the first N subjects.")
    batch.add_argument("--pid", default=None, help="Run only one subject folder by PID.")
    add_common_options(batch)

    return parser


def cli_main(argv=None):
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    gpu = _parse_gpu(args.gpu)

    if not args.skip_docker_check:
        check_docker_available()

    if args.command == "single":
        run_one_case(
            input_path=args.input_path,
            output_dir=args.output_dir,
            image=args.image,
            use_gpu=args.use_gpu,
            gpu=gpu,
            threads=args.threads,
            platform=args.platform,
            overwrite=args.overwrite,
            qc=not args.no_qc,
        )
        return 0

    if args.command == "batch":
        run_batch(
            src_dir=args.src_dir,
            dst_dir=args.dst_dir,
            input_pattern=args.input_pattern,
            log_file=args.log_file,
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
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(cli_main())
