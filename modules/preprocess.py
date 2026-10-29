"""Volume- and slice-level operations of the FLAIR preprocessing pipeline.

Each step keeps exactly one implemented policy. Alternatives (another
orientation, normalization or size policy, a non-Docker SynthStrip backend) are
added by registering a function in the matching ``*_POLICIES`` / ``*_BACKENDS``
dict and exposing the new key as a CLI choice in ``modules.pipeline``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess as sp
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Step 1. Skull stripping (SynthStrip)
# ---------------------------------------------------------------------------

DEFAULT_SYNTHSTRIP_IMAGE = "freesurfer/synthstrip:latest"
# Measured peak of one SynthStrip container (~5.6 GiB on 512x512x24 FLAIR) plus margin.
# Containers beyond MemTotal // this value get OOM-killed (exit 137).
SYNTHSTRIP_PEAK_BYTES = 6 * 1024**3


def check_docker_available() -> None:
    """Raise a readable error when the Docker daemon is not reachable."""
    try:
        sp.run(["docker", "info"], check=True, stdout=sp.DEVNULL, stderr=sp.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Docker command not found. Install and start Docker Desktop.") from exc
    except sp.CalledProcessError as exc:
        raise RuntimeError(f"Docker daemon is not reachable. Start Docker Desktop first.\n{exc.stderr.strip()}") from exc


def docker_parallel_slots() -> int:
    """Number of SynthStrip containers that fit in the Docker VM memory at once."""
    out = sp.run(["docker", "info", "--format", "{{.MemTotal}}"], check=True, stdout=sp.PIPE, text=True).stdout
    return max(1, int(out.strip()) // SYNTHSTRIP_PEAK_BYTES)


def _run_synthstrip(cmd: list[str], brain_path: Path, mask_path: Path, env: dict | None = None) -> None:
    result = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"SynthStrip failed ({result.returncode}):\n{result.stdout[-2000:]}")
    missing = [str(p) for p in (brain_path, mask_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"SynthStrip finished but outputs are missing: {missing}")


def synthstrip_docker(input_path: Path, brain_path: Path, mask_path: Path, settings: dict) -> None:
    """Run SynthStrip in a throwaway Docker container (default mask, CSF included)."""
    input_path = input_path.resolve()
    cmd = ["docker", "run", "--rm"]
    if settings["docker_platform"]:
        cmd += ["--platform", settings["docker_platform"]]
    cmd += [
        "-v", f"{input_path.parent}:/data:ro",
        "-v", f"{brain_path.parent.resolve()}:/out",
        settings["synthstrip_image"],
        "-i", f"/data/{input_path.name}",
        "-o", f"/out/{brain_path.name}",
        "-m", f"/out/{mask_path.name}",
    ]
    _run_synthstrip(cmd, brain_path, mask_path)


# The native backend runs the same mri_synthstrip script and weights as the Docker
# image, copied out of it once and switched from CPU to the Apple GPU (MPS):
# arm64 CPU PyTorch needs >40 GB for SynthStrip's Conv3d, MPS ~1 GB.
# On 6 PAIK cases the MPS masks matched Docker in all but 1 voxel.
SYNTHSTRIP_CACHE = Path.home() / ".cache" / "fhv_preproc" / "synthstrip"
SYNTHSTRIP_CPU_LINE = "device = torch.device('cpu')"
SYNTHSTRIP_NATIVE_SLOTS = 2  # concurrent MPS runs


def native_synthstrip_home(image: str, platform: str | None) -> Path:
    """Return a FREESURFER_HOME holding the image's mri_synthstrip (MPS) and models; extract once."""
    home = SYNTHSTRIP_CACHE / re.sub(r"[^A-Za-z0-9._-]", "_", image)
    if (home / "mri_synthstrip").exists():
        return home
    check_docker_available()
    staging = home.with_name(home.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    create = ["docker", "create"] + (["--platform", platform] if platform else []) + [image]
    container = sp.run(create, check=True, stdout=sp.PIPE, text=True).stdout.strip()
    try:
        for path in ("/freesurfer/mri_synthstrip", "/freesurfer/models"):
            sp.run(["docker", "cp", f"{container}:{path}", str(staging)], check=True, stdout=sp.DEVNULL)
    finally:
        sp.run(["docker", "rm", container], check=True, stdout=sp.DEVNULL)
    script = staging / "mri_synthstrip"
    source = script.read_text()
    if source.count(SYNTHSTRIP_CPU_LINE) != 1:
        raise RuntimeError(f"unexpected mri_synthstrip in {image}; cannot switch it to MPS")
    script.write_text(source.replace(SYNTHSTRIP_CPU_LINE, "device = torch.device('mps')"))
    digest = sp.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], check=True, stdout=sp.PIPE, text=True)
    (staging / "image_id.txt").write_text(digest.stdout)
    staging.rename(home)
    return home


def synthstrip_native(input_path: Path, brain_path: Path, mask_path: Path, settings: dict) -> None:
    """Run the cached mri_synthstrip on MPS with ``settings['synthstrip_python']`` (torch + surfa)."""
    home = native_synthstrip_home(settings["synthstrip_image"], settings["docker_platform"])
    cmd = [settings["synthstrip_python"], "-I", str(home / "mri_synthstrip"),
           "-i", str(input_path), "-o", str(brain_path), "-m", str(mask_path)]
    _run_synthstrip(cmd, brain_path, mask_path, env={**os.environ, "FREESURFER_HOME": str(home)})


SKULLSTRIP_BACKENDS = {"docker": synthstrip_docker, "native": synthstrip_native}


# ---------------------------------------------------------------------------
# Step 2. N4 bias field correction
# ---------------------------------------------------------------------------

# Bias is estimated on an in-plane shrunk copy and evaluated on the full grid.
# z is never shrunk because thick-slice FLAIR has only ~20-30 slices.
N4_SHRINK_XYZ = (4, 4, 1)
N4_MAX_ITERATIONS = (50, 50, 30, 20)


def n4_bias_correct(brain_path: Path, mask_path: Path):
    """Return (corrected brain, binary mask) as (X, Y, Z) float32 / bool arrays."""
    import SimpleITK as sitk

    image = sitk.Cast(sitk.ReadImage(str(brain_path)), sitk.sitkFloat32)
    mask = sitk.Cast(sitk.ReadImage(str(mask_path)) > 0, sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(N4_MAX_ITERATIONS))
    corrector.SetConvergenceThreshold(0.0)
    corrector.Execute(sitk.Shrink(image, list(N4_SHRINK_XYZ)), sitk.Shrink(mask, list(N4_SHRINK_XYZ)))
    log_bias = corrector.GetLogBiasFieldAsImage(image)
    corrected = image / sitk.Exp(log_bias)

    # SimpleITK arrays are (Z, Y, X); transpose to nibabel voxel order (X, Y, Z).
    corrected_xyz = sitk.GetArrayFromImage(corrected).T.astype(np.float32, copy=False)
    mask_xyz = sitk.GetArrayFromImage(mask).T > 0
    return corrected_xyz, mask_xyz


# ---------------------------------------------------------------------------
# Step 4. Reorientation (axis flips/permutations only, no interpolation)
# ---------------------------------------------------------------------------

def reorient(volume_xyz: np.ndarray, affine: np.ndarray, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Reorder voxel axes of ``volume_xyz`` so the affine matches ``target`` axcodes."""
    from nibabel.orientations import apply_orientation, axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform

    transform = ornt_transform(io_orientation(affine), axcodes2ornt(tuple(target)))
    return apply_orientation(volume_xyz, transform), affine @ inv_ornt_aff(transform, volume_xyz.shape)


ORIENTATIONS = ("LAS",)


# ---------------------------------------------------------------------------
# Step 5. Slice normalization, in-plane size, PNG pixels
# ---------------------------------------------------------------------------

def normalize_slice_p1_p99(slice2d: np.ndarray) -> np.ndarray:
    """Per-slice min-max between the 1st/99th percentile of all pixels, clipped to [0, 1]."""
    slice2d = slice2d.astype(np.float32, copy=False)
    vmin, vmax = np.percentile(slice2d, (1, 99))
    if vmax <= vmin:
        return np.zeros_like(slice2d, dtype=np.float32)
    return np.clip((slice2d - vmin) / (vmax - vmin), 0, 1).astype(np.float32, copy=False)


NORMALIZATION_POLICIES = {"slice_p1_p99": normalize_slice_p1_p99}


def resize_square(slice2d: np.ndarray, size: int) -> np.ndarray:
    """Resize to ``size x size``: bilinear when enlarging, area averaging when shrinking."""
    import cv2

    if slice2d.shape == (size, size):
        return slice2d
    shrinking = slice2d.shape[0] > size or slice2d.shape[1] > size
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(slice2d, (size, size), interpolation=interpolation)


SIZE_POLICIES = {"resize": resize_square}


def to_png_pixels(slice_xy: np.ndarray) -> np.ndarray:
    """Map a [0, 1] (X, Y) slice to display-oriented 8-bit pixels: ``flipud(slice.T)``."""
    display = np.flipud(slice_xy.T)
    return np.rint(np.clip(display, 0.0, 1.0) * 255.0).astype(np.uint8)


DISPLAY_TRANSFORM = "flipud(transpose(slice_xy))"
