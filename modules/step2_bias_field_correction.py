# -*- coding: utf-8 -*-
"""
N4 bias correction + QC (bias field 시각화 + 통계적 QC)
- 목적:
  (1) N4가 '저주파 곱셈 편향(multiplicative low-frequency field)'만 제거했는지 검증
  (2) 뇌 내부의 신호 균일도(uniformity)가 향상되었는지 정량화
- 모델 가정:
  I_obs(x) = I_true(x) * B(x) + ε,  여기서 B(x)는 매우 매끈한(저주파) bias field
  → log(I_obs) = log(I_true) + log(B)  (N4는 log(B)를 저차 B-spline으로 추정)
  # 공학적 해석:
  # - B(x)는 RF 송/수신 불균일(B1+/B1-), coil 감도, 안테나 배열 등에 의해 생기는 cm-스케일의 저주파 성분.
  # - multiplicative 가정 때문에 log-domain으로 가면 additive로 단순화되어(summation), 저주파 회복 문제가 안정화됨.
  # - 저차 B-spline은 밴드제한된(low-pass) 함수군으로, 고주파 조직 구조(경계/혈관/병변)를 따라갈 수 없도록 규제(regularization)의 역할.

- 출력:
  1) N4 보정본(.nii.gz)
  2) log-bias/bias field(.nii.gz)  → B(x)와 log B(x)의 공간적 매끈함을 직접 확인
  3) bias field 미리보기 PNG (mid-slice)
  4) slice-wise mean & CoV 곡선 PNG  → z-방향 저주파 기울기 감소 여부 확인
  5) 원본/보정 histogram 비교 PNG   → 조직 대비(고주파 구조) 보존 + 스케일 일관성
  6) 요약 지표 .txt
"""

import argparse
import glob
import os
import sys
from pathlib import Path
import numpy as np

from modules.logging_utils import append_error_log, resolve_log_file


DEFAULT_SRC_DIR = "/Volumes/X31/FLAIR_HVS_detection_nii_preproc"
DEFAULT_DST_DIR = "/Volumes/X31/FLAIR_HVS_detection_nii_preproc"
DEFAULT_LOG_FILE = "/Volumes/X31/FLAIR_preprocessing_logs/err_bias_field_correction.txt"


def run_n4_bias_correction(
    in_path: str,
    out_dir: str,
    out_name: str,
    external_mask_path: str = None,
    use_otsu_mask_if_none: bool = True,
    shrink_factor: int = 2,                      # (구버전 SimpleITK에선 사용되지 않지만, 인터페이스만 유지)
    max_iters: tuple = (50, 50, 30, 20),         # multi-resolution 반복 횟수(저주파부터 점진적 수렴)
    hist_bins: int = 256,
    qc_png_dpi: int = 160,
):
    import SimpleITK as sitk
    import matplotlib.pyplot as plt

    # =========================================
    # 준비: 출력 디렉토리 생성
    # =========================================
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    qc_dir          = os.path.join(out_dir, f"{out_name}_bias_field_correction_QC")
    out_n4_path     = os.path.join(out_dir, f"{Path(in_path).name.split('.')[0]}_bfCorrected.nii.gz")
    out_logbias_nii = os.path.join(qc_dir, f"{out_name}_logbias.nii.gz")
    out_bias_nii    = os.path.join(qc_dir, f"{out_name}_bias.nii.gz")
    Path(qc_dir).mkdir(parents=True, exist_ok=True)

    # =========================================
    # 1) 로드 & 마스크
    # =========================================
    img = sitk.ReadImage(in_path)
    img = sitk.Cast(img, sitk.sitkFloat32)     # 연산 안정성
    arr = sitk.GetArrayFromImage(img)          # [z, y, x]

    # 마스크 준비: 외부(정교) 마스크 > Otsu(러프)
    # - 이유: bias 추정은 뇌 내부 분포에 기반해야 함. 배경/두개골은 outlier → B(x) 추정을 왜곡.
    # - 수학적으로는 log-likelihood를 뇌 내부 샘플에 한정해 fitting하는 효과 → 추정분산↓, 오차편향↓.
    if external_mask_path and Path(external_mask_path).exists():
        mask = sitk.ReadImage(str(external_mask_path))
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    else:
        if use_otsu_mask_if_none:
            mask = sitk.OtsuThreshold(img, 0, 1, 200)   # 전역 임계 기반 대략적 두개골 제거
            # closing으로 작은 홀/틈 메움 → 연결성↑, outlier leakage↓
            try:
                mask = sitk.BinaryMorphologicalClosing(mask, (1, 1, 1))
            except Exception:
                mask = sitk.BinaryMorphologicalClosing(mask, 1)
        else:
            mask = sitk.OtsuThreshold(img, 0, 1, 200)
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

    # Save mask image to QC directory
    out_mask_nii = os.path.join(qc_dir, f"{out_name}_mask.nii.gz")
    sitk.WriteImage(mask, out_mask_nii)

    msk_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
    brain = (msk_arr > 0)  # 이후 모든 통계는 brain mask 내부에서만 수행(공학적 타당성)

    # =========================================
    # 2) N4 실행 (구버전: Execute(image, mask) 시그니처)
    # =========================================
    # - N4는 log(I) 공간에서 저차 B-spline으로 log(B)를 적합.
    # - max_iters는 multiresolution 단계별 수렴 제약(저주파 → 비교적 중간 주파).
    # - Multires는 coarse-to-fine 전략으로 로컬 최적화의 안정성↑(non-convexity 완화).
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(max_iters))
    try:
        corrector.SetConvergenceThreshold(0.0)  # 미세 수렴 기준(버전에 따라 옵션 부재 가능)
    except AttributeError:
        pass

    # shrink_factor 전달 불가한 구버전이므로 Execute(img, mask) 사용
    # (연산량이 크면 사전 crop로 ROI 축소 권장: B(x)가 저주파일수록 bbox crop의 경계효과가 적음)
    corrected = corrector.Execute(img, mask)

    # 저장: NIfTI(.nii.gz) 권장(헤더/호환성, 용량 절감)
    sitk.WriteImage(corrected, out_n4_path)

    # =========================================
    # 3) Bias field(log-bias) 추출 & 저장
    # =========================================
    # - 검증 핵심: 추정된 log-bias가 '저주파적'인지(=매끈한 필드인지) 직접 시각화
    # - 고주파 조직 경계/혈관 모양이 보이면 N4가 실제 구조를 bias로 오적합한 것(과보정 위험)
    # - 공학적 관점: 추정된 logB의 파워스펙트럼이 저주파 대역에 대부분 집중되어야 함(비정식 확인).
    log_bias_img = corrector.GetLogBiasFieldAsImage(img)  # log(B)
    bias_img     = sitk.Exp(log_bias_img)                 # B = exp(log B)

    sitk.WriteImage(log_bias_img, out_logbias_nii)
    sitk.WriteImage(bias_img,     out_bias_nii)

    # 미리보기 PNG: mid-slice
    def save_mid_slice_png(vol_img: sitk.Image, title: str, out_png: str, cmap="viridis"):
        arr_ = sitk.GetArrayFromImage(vol_img)  # [z,y,x] 또는 [y,x]
        # - mid-slice 시각화는 저주파 구조의 smoothness를 직관적으로 보는데 유용
        # - 실제론 전 슬라이스를 훑는 것이 좋으나, QC 비용/속도 절충을 위해 중앙 단면을 대표로 사용
        if arr_.ndim == 2:
            sl = arr_
        else:
            zc = arr_.shape[0] // 2
            sl = arr_[zc]
        # robust windowing: 극단치 영향↓ → 패턴 가독성↑ (2~98 백분위)
        finite = np.isfinite(sl)
        if not finite.any():
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = np.percentile(sl[finite], [2, 98])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = float(np.nanmin(sl)), float(np.nanmax(sl))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = 0.0, 1.0
        plt.figure(figsize=(6,5))
        plt.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(title); plt.axis('off'); plt.tight_layout()
        plt.savefig(out_png, dpi=qc_png_dpi); plt.close()

    save_mid_slice_png(log_bias_img, "log-bias (mid-slice)", os.path.join(qc_dir, "logbias_mid.png"), cmap="jet")
    save_mid_slice_png(bias_img,     "bias (mid-slice)",     os.path.join(qc_dir, "bias_mid.png"),     cmap="jet")

    # =========================================
    # 4) 통계적 QC (mask 내부 기준)
    # =========================================
    corr_arr = sitk.GetArrayFromImage(corrected).astype(np.float32)

    def finite_mask(a):
        return np.isfinite(a) & (a != 0)
        # 이유: 0/Inf/NaN은 배경/아티팩트이며, 분산 및 히스토그램에 비물리적 영향(heavy tail)을 유발.

    def slice_stats(a, brain_mask):
        """
        z-슬라이스별 평균/CoV 계산.
        - mean(z): 저주파 편향이 존재하면 z축을 따라 완만한 기울기/굴곡이 나타남
          · 공학적 관점: mean(z)의 저주파 성분(저차 다항/저주파 필터 통과 성분)이 작아져야 함.
        - CoV(z)=std/mean: 동일 조직 내 intensity 균일도 지표(스케일 불변)
          → 보정 후 CoV 감소가 '균일도 향상'의 정량적 근거
          · 스케일 불변성(scale-invariance) 덕분에 절대 스케일 변화에 둔감 → multiplicative 보정 검증에 적합.
        """
        Z = a.shape[0]
        means = np.zeros(Z, dtype=np.float32)
        covs  = np.zeros(Z, dtype=np.float32)
        for z in range(Z):
            m = brain_mask[z] & finite_mask(a[z])
            if m.sum() > 0:
                vals = a[z][m]
                mu = float(vals.mean())
                sd = float(vals.std())
                means[z] = mu
                covs[z]  = (sd / (mu + 1e-6))
            else:
                means[z] = np.nan
                covs[z]  = np.nan
        return means, covs

    mean0, cov0 = slice_stats(arr,      brain)   # 보정 전
    mean1, cov1 = slice_stats(corr_arr, brain)   # 보정 후

    def save_slice_curves(mean0, mean1, cov0, cov1, out_png):
        # 시각적 의의:
        # - mean 곡선의 평탄화(DC/저주파 성분 감소)는 B(x) 제거의 간접 증거
        # - CoV 감소는 조직 내 동질성(Uniformity ↑)과 site/세션 간 정규화 용이성 ↑를 의미
        z = np.arange(len(mean0))
        plt.figure(figsize=(10,6))
        ax1 = plt.subplot(2,1,1)
        ax1.plot(z, mean0, label="Before mean")
        ax1.plot(z, mean1, label="After mean")
        ax1.set_title("Slice-wise mean (brain mask inside)")
        ax1.set_xlabel("slice (z)"); ax1.set_ylabel("mean intensity")
        ax1.legend(); ax1.grid(True, alpha=.3)

        ax2 = plt.subplot(2,1,2)
        ax2.plot(z, cov0, label="Before CoV")
        ax2.plot(z, cov1, label="After CoV")
        ax2.set_title("Slice-wise CoV (std/mean)")
        ax2.set_xlabel("slice (z)"); ax2.set_ylabel("CoV")
        ax2.legend(); ax2.grid(True, alpha=.3)

        plt.tight_layout()
        plt.savefig(out_png, dpi=qc_png_dpi); plt.close()

    save_slice_curves(mean0, mean1, cov0, cov1, os.path.join(qc_dir, "slice_mean_cov.png"))

    # 히스토그램 비교:
    # - 전역 분포 폭이 줄면(=분산↓) bias로 인한 스케일/셰이딩 변화가 줄었음을 시사
    # - 다만 병변/high-intensity tail은 보존되어야 함(고주파 구조 보존성 검증)
    # - 수학적 관점: 보정 전후 분포의 엔트로피/피크 날카로움(sharpness) 비교도 가능(여기선 단순화하여 시각 비교).
    vals0 = arr[brain & finite_mask(arr)].ravel()
    vals1 = corr_arr[brain & finite_mask(corr_arr)].ravel()

    def robust_range(v):
        # 공통 bins 범위 산정(극단치 영향 완화) → 공정 비교
        # - 같은 range를 쓰지 않으면, 히스토그램 모양 비교가 범위 의존적으로 왜곡될 수 있음.
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0.0, 1.0
        lo = np.percentile(v, 0.5)
        hi = np.percentile(v, 99.5)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.min(v)), float(np.max(v))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = 0.0, 1.0
        return lo, hi

    lo0, hi0 = robust_range(vals0)
    lo1, hi1 = robust_range(vals1)
    lo, hi = min(lo0, lo1), max(hi0, hi1)

    plt.figure(figsize=(7,5))
    plt.hist(vals0, bins=hist_bins, range=(lo,hi), density=True, alpha=0.5, label="Before")
    plt.hist(vals1, bins=hist_bins, range=(lo,hi), density=True, alpha=0.5, label="After")
    plt.title("Global histogram (brain mask inside)")
    plt.xlabel("intensity"); plt.ylabel("density")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "hist_before_after.png"), dpi=qc_png_dpi)
    plt.close()

    # =========================================
    # 5) 요약 지표 저장
    # =========================================
    def finite_summary(x):
        # 곡선의 중앙경향/변동성 요약 → 평탄화/변동성 감소를 수치로 보고
        # - mean(z)의 표준편차↓는 저주파 기울기 제거의 정량 지표
        # - CoV(z)의 평균↓는 조직 단면 내 균질성↑ (스케일에 둔감하므로 robust)
        x = x[np.isfinite(x)]
        if x.size == 0: return np.nan, np.nan
        return float(np.nanmean(x)), float(np.nanstd(x))

    mean_mu0, mean_sd0 = finite_summary(mean0)
    mean_mu1, mean_sd1 = finite_summary(mean1)
    cov_mu0,  cov_sd0  = finite_summary(cov0)
    cov_mu1,  cov_sd1  = finite_summary(cov1)

    with open(os.path.join(qc_dir, "summary.txt"), "w") as f:
        f.write("=== N4 Bias Correction QC Summary ===\n")
        f.write(f"Input : {in_path}\n")
        f.write(f"Output: {out_n4_path}\n\n")
        f.write(f"Slice-wise mean (mask inside):\n")
        f.write(f"  Before: mean={mean_mu0:.4f}, std={mean_sd0:.4f}\n")
        f.write(f"  After : mean={mean_mu1:.4f}, std={mean_sd1:.4f}\n\n")
        f.write(f"Slice-wise CoV (mask inside):\n")
        f.write(f"  Before: mean={cov_mu0:.4f}, std={cov_sd0:.4f}\n")
        f.write(f"  After : mean={cov_mu1:.4f}, std={cov_sd1:.4f}\n\n")
        f.write("Interpretation:\n")
        f.write("- mean(z) 평탄화는 z-방향 저주파 스케일링(bias)의 제거를 의미합니다.\n")
        f.write("- CoV(z) 감소는 조직 내 균질도(스케일 불변 지표)가 향상되었음을 의미합니다.\n")
        f.write("- histogram 폭이 좁아지고(high-frequency tail 유지) log-bias가 매끈하면,\n")
        f.write("  N4가 고주파 구조를 보존한 채 저주파 bias만 제거했을 가능성이 큽니다.\n")
        f.write("- 반대로 병변 경계가 무뎌지거나, log-bias에 고주파 패턴이 보이면 과보정/오적합을 의심하세요.\n")

    # 간단 출력 (slice, cov 지표 포함)
    print(
        f"[OK] N4 Bias Correction 완료 → mean(z) sd: {mean_sd0:.4f}→{mean_sd1:.4f}, "
        f"CoV(z) mean: {cov_mu0:.4f}→{cov_mu1:.4f},\n"
        f"Input: {in_path},\n"
        f"Output: {out_n4_path},\n"
    )


def expected_n4_output(in_path, out_dir):
    """Return the expected N4 output path for one input image."""
    return os.path.join(out_dir, f"{Path(in_path).name.split('.')[0]}_bfCorrected.nii.gz")


def write_error(log_file, input_path, output_dir, error):
    """Append one failed case to the batch log."""
    append_error_log(log_file, input_path, output_dir, error)


def run_one_case(in_path, out_dir, out_name=None, mask_path=None, overwrite=False):
    """Run one N4 bias-correction case with optional skip-on-existing-output."""
    out_name = out_name or Path(out_dir).name
    expected_out = expected_n4_output(in_path, out_dir)
    if not overwrite and os.path.exists(expected_out):
        print(f"[SKIP] already done: {expected_out}")
        return expected_out

    run_n4_bias_correction(
        in_path=in_path,
        out_dir=out_dir,
        out_name=out_name,
        external_mask_path=mask_path,
    )
    return expected_out


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
    dst_dir=DEFAULT_DST_DIR,
    log_file=DEFAULT_LOG_FILE,
    overwrite=False,
    limit=None,
    pid=None,
    include_keywords=None,
):
    """Run N4 bias correction for every subject folder under ``src_dir``."""
    from tqdm.auto import tqdm
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
        out_dir = Path(dst_dir) / subject_dir.name
        input_files = sorted([
            p for p in subject_dir.iterdir()
            if is_visible_file(p)
            and p.name.endswith("_brain.nii.gz")
            and "_bfCorrected" not in p.name
            and matches_include_keywords(p.name, include_keywords)
        ])
        if not input_files:
            print(f"[SKIP] brain image 없음: {subject_dir.name}")
            continue

        mask_candidates = sorted(
            p for p in glob.glob(str(subject_dir / "*brainMask.nii.gz"))
            if is_visible_file(p)
        )
        if not mask_candidates:
            write_error(log_file, subject_dir, out_dir, "brainMask file not found")
            print(f"[SKIP] brainMask 없음 -> 로그 저장: {subject_dir.name}")
            continue

        for in_path in input_files:
            try:
                run_one_case(
                    in_path=str(in_path),
                    out_dir=str(out_dir),
                    out_name=subject_dir.name,
                    mask_path=mask_candidates[0],
                    overwrite=overwrite,
                )
            except Exception as e:
                write_error(log_file, in_path, out_dir, e)
                print(f"[ERROR] 변환 실패 -> 로그 저장: {in_path}")


def build_arg_parser():
    """Build CLI parser for N4 bias correction."""
    parser = argparse.ArgumentParser(
        description=(
            "Step 02: run N4 bias field correction on SynthStrip brain images. "
            "Batch mode scans subject folders for '*_brain.nii.gz' and uses the matching "
            "'*brainMask.nii.gz' as the external mask. Existing '*_bfCorrected.nii.gz' "
            "outputs are skipped unless --overwrite is set."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "single",
        help="Run N4 bias correction for one brain NIfTI file.",
    )
    single.add_argument(
        "--input-path",
        required=True,
        help="Input skull-stripped brain NIfTI path, usually '*_brain.nii.gz'.",
    )
    single.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory where the corrected NIfTI and QC folder will be saved. "
            "The corrected file name is derived from --input-path with '_bfCorrected.nii.gz'."
        ),
    )
    single.add_argument(
        "--out-name",
        default=None,
        help=(
            "Name prefix for the QC directory and QC NIfTI files. "
            "Defaults to the output directory name, typically the PID."
        ),
    )
    single.add_argument(
        "--mask-path",
        default=None,
        help=(
            "Optional external brain mask NIfTI, usually '*brainMask.nii.gz'. "
            "If omitted, an Otsu mask is generated internally."
        ),
    )
    single.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run correction even if the expected '*_bfCorrected.nii.gz' output already exists.",
    )

    batch = subparsers.add_parser(
        "batch",
        help="Run N4 bias correction for all subject folders under --src-dir.",
    )
    batch.add_argument(
        "--src-dir",
        default=DEFAULT_SRC_DIR,
        help=(
            "Root directory containing subject folders, e.g. P00006/. "
            "Each subject folder must contain '*_brain.nii.gz' and '*brainMask.nii.gz'."
        ),
    )
    batch.add_argument(
        "--dst-dir",
        default=DEFAULT_DST_DIR,
        help=(
            "Root output directory. Outputs are written to '<dst-dir>/<PID>/'. "
            "Use the same path as --src-dir to save results beside the skull-stripped files."
        ),
    )
    batch.add_argument(
        "--log-file",
        default=None,
        help="Path to the text log where failed or mask-missing cases are appended. Default: <dst volume>/logs/err_bias_field_correction.txt.",
    )
    batch.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run completed cases instead of skipping existing '*_bfCorrected.nii.gz' files.",
    )
    batch.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N sorted subject folders; useful for a small test run.",
    )
    batch.add_argument(
        "--pid",
        default=None,
        help="Run only one subject folder by PID, e.g. P00006. Default processes all subjects.",
    )
    batch.add_argument(
        "--include-keyword",
        action="append",
        dest="include_keywords",
        default=None,
        help=(
            "Only process candidate input files whose filename contains this keyword. "
            "Can be specified multiple times; a file is included if it matches any keyword. "
            "Example: --include-keyword FLAIR"
        ),
    )

    return parser


def cli_main(argv=None):
    """CLI entry point.

    Examples
    --------
    python -m modules.step2_bias_field_correction batch
    python -m modules.step2_bias_field_correction single --input-path in.nii.gz --output-dir out
    """
    args = build_arg_parser().parse_args(argv)
    if args.command == "single":
        run_one_case(
            in_path=args.input_path,
            out_dir=args.output_dir,
            out_name=args.out_name,
            mask_path=args.mask_path,
            overwrite=args.overwrite,
        )
        return 0

    run_batch(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        log_file=args.log_file,
        overwrite=args.overwrite,
        limit=args.limit,
        pid=args.pid,
        include_keywords=args.include_keywords,
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
