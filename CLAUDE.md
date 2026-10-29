# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

축상(axial) FLAIR MRI에서 FHV(FLAIR hyperintense vessel)를 탐지하는 연구(09_FHV)의 **전처리 코드**만 떼어 낸 저장소다. 탐지 모델(YOLO26n-P2 계열)과 MIL 모델의 학습·평가 코드는 여기 없다. 모델은 임상의의 검토를 돕는 용도이며, 단독으로 진단하지 않는다.

- 데이터 흐름: DICOM → NIfTI → 전처리(이 저장소) → NumPy `(Z, H, W)` → 슬라이스별 PNG(YOLO 입력, 512×512) 또는 HDF5(MIL 입력)
- 코호트: 내부 UHK(GT 버전 full-v1, UHK70, UHK100 — 현재 주 연구는 UHK100), 외부 Guri2025, PAIK
- DICOM → NIfTI 변환과 NPY → PNG 변환 스크립트는 이 저장소에 없다(위치 미확인).

## 환경과 실행

- 빌드 설정, 의존성 파일, 테스트, 린터가 없다. 필요한 패키지는 `numpy`, `pandas`, `nibabel`, `SimpleITK`, `scikit-image`, `h5py`, `matplotlib`, `tqdm`이다. Step 1은 Docker와 `freesurfer/synthstrip` 이미지를 쓴다. Apple Silicon에서는 `--platform linux/amd64`를 붙인다.
- 모든 스크립트는 저장소 루트에서 모듈로 실행한다. 대부분 `single` / `batch` 하위 명령을 갖고, `batch`는 `--pid`, `--limit`, `--overwrite`, `--no-qc`를 지원한다.

```bash
python -m modules.step1_to_step5_pipeline single --input-path <PID>_FLAIR.nii --dst-dir <out> --platform linux/amd64
python -m modules.step1_to_step5_pipeline batch --pid P00006 --src-dir <org> --dst-dir <out> --platform linux/amd64
python -m modules.step6_create_filtered_h5 --src-dir <npy_root> --output-h5 <out.h5>
python -m modules.compare_h5_files --left old.h5 --right new.h5
```

- 각 단계별 명령 예시와 옵션은 `modules/README.md`에 있다.
- 기본 경로가 `/Volumes/X31`, `/Volumes/GURI_1`, `/Users/yschoi/...`로 하드코딩되어 있다. 인자를 생략하면 그 경로를 읽고 쓰므로 `--src-dir`/`--dst-dir`를 명시한다.

## 아키텍처

### 단계와 파일 이름 규칙

각 단계는 독립 CLI이며, 앞 단계 출력 파일의 **접미사**로 입력을 찾는다(`--include-keyword`, `--step*-input-suffix`). 파일 이름을 바꾸면 다음 단계가 입력을 찾지 못한다. 출력은 `<dst>/<PID>/` 폴더에 쌓인다.

| 단계 | 모듈 | 처리 | 출력 접미사 |
|---|---|---|---|
| 1 | `step1_skull_stripping` | Docker SynthStrip | `_FLAIR_brain.nii.gz`, `_brainMask.nii.gz` |
| 2 | `step2_bias_field_correction` | SimpleITK N4(brain mask 사용) | `_bfCorrected.nii.gz` |
| 3 | `step3_masking` | (선택) mask 재적용, 뇌 밖 0 | `_masked.nii.gz` |
| 4 | `step4_reorientation` | NiBabel로 RAS/LAS 정렬. 보간·정합·리샘플링 없음, 라벨 재매핑 없음 | `_reorient.nii.gz` |
| 5 | `step5_resize_normalization` | axial NPY 추출, 슬라이스별 1–99 백분위수 min-max(clip [0,1]), in-plane 크기 맞춤, 슬라이스 라벨 | `_AxialSlices_{padded,resized}.npy`, `_label_sliceLevel.npy`, `_meta.json` |
| 6 | `step6_create_filtered_h5` | 슬라이스 선택 후 `[이전, 중앙, 다음]` triplet HDF5 | `.h5` |
| 7 | `step7_gradcam_to_original_space` | `(Z,672,672)` GradCAM을 원본 FLAIR 공간으로 역변환 | `.nii.gz` |

- `step1_to_step5_pipeline`은 Step 1–5 모듈의 함수를 직접 import해서 순서대로 호출한다. `run_flat_step1_to_step5_batch`는 PID 폴더 없이 파일이 한 폴더에 모인 코호트(PAIK)용이고, `recover_paik_tmp_synthstrip_tail`은 Docker가 `/Volumes`를 마운트하지 못할 때 SynthStrip만 `/private/tmp`를 거쳐 돌리는 복구용이다.
- `modules/__init__.py`는 일부러 하위 모듈을 import하지 않는다. 무거운 의존성 때문에 `--help` 같은 가벼운 CLI가 막히지 않게 하려는 것이므로, 여기에 import를 추가하지 않는다.
- 공통 로깅은 `logging_utils`가 맡는다. 에러 로그는 대상 볼륨의 `logs/`에 쌓이고, 쓸 수 없으면 `./logs`로 대체한다.
- 각 단계는 QC 산출물(`qc_skullstrip/`, `*_bias_field_correction_QC/`, `qc_masking/`, `qc_reorientation/`, `qc_pad|qc_resize/`, `qc_*_positive/`)을 만든다. 전처리를 바꿨다면 QC를 확인한 뒤 다음 단계로 넘어간다.

### Step 5 크기·라벨 정책

- `--preprocess-mode pad`(기본): 중앙 zero-pad로 672×672를 만든다. 보간하지 않는다.
- `--preprocess-mode resize`: `skimage.transform.resize(order=1, anti_aliasing=True, preserve_range=True)`로 `--target-size`(예: 512)에 맞춘다. in-plane spacing만 바뀐다.
- z축은 리샘플링하지 않는다. 출력 슬라이스 수는 항상 원본 NIfTI의 Z와 같다. voxel spacing과 affine은 `*_meta.json`의 `zooms`·`affine`에 남는다.
- 어떤 데이터가 어떤 모드로 만들어졌는지는 `*_meta.json`의 `preprocess_mode`, `target_hw`로 확인한다. README와 UHK 경로의 기본값은 `pad 672` + RAS이고, PAIK용 실행기(`run_flat_*`, `recover_*`)의 기본값은 `resize 512` + LAS다.
- 라벨은 `--label-xlsx`의 `--label-column fhv_slice_num`에서 읽는다. 없으면 같은 PID 폴더의 기존 `*label_sliceLevel.npy`로 대체한다. 이는 **슬라이스 단위 양성 여부**이며 YOLO의 bounding box GT와 다르다.

### Step 6 / HDF5

- 기본 선택은 `--selection-mode relative`, 상대 위치 45–75%이며 라벨과 무관하다. 라벨은 지도 신호로만 저장된다. 예전 데이터셋을 재현할 때만 `--selection-mode local --min-index 12 --max-index 15`를 쓴다.
- 경계 슬라이스는 자기 자신으로 채우고 `boundary_flag`로 표시한다. 주요 키: `images`, `labels`, `subject_ids`, `scan_ids`, `case_ids`, `local_slice_index`, `local_slice_index_rel`, `boundary_flag`, (`--include-scanner-manufacturer --dicom-root`를 주면) `scanner_manufacturer`.
- `dicom_scanner_metadata`는 pydicom 없이 DICOM Manufacturer 태그를 직접 읽어 Philips/GE/Siemens/UNKNOWN으로 정규화한다.
- `append_npy_to_h5_with_source_flag`는 기존 H5에 새 코호트 NPY를 같은 스키마로 덧붙이고 `data_source`, `is_new_data`를 기록한다. `getDataset.py`는 Step 6 이전의 단순 변환기다.
- Step 6·7은 주로 MIL 연구용이다. **MIL 결과를 YOLO 결과와 섞지 마라.**

### 기타

- `step0_slice_idx2zcoord`는 방향 변환으로 슬라이스 순서가 바뀔 때만 쓰는 선택 유틸리티다(슬라이스 번호 → 물리 z좌표). 현재 LAS/RAS 데이터의 기본 파이프라인에는 포함되지 않는다.
- 스캔 간·템플릿 정합(registration)은 파이프라인에 없다.

## 작업 규칙

- 답변은 한국어로 한다. 결론을 먼저 말하고 관찰 결과, 해석, 가설, 권고를 구분한다.
- 데이터 개수와 수치는 실제 결과 파일에서 확인한다. 자료에 없는 값을 추정해서 채우지 않는다.
- **원본 DICOM, NIfTI, GT 라벨, 기존 전처리 결과를 수정하거나 덮어쓰지 않는다.** 새 산출물은 새 경로에 저장하고, `--overwrite`는 사용자가 명시적으로 요청했을 때만 쓴다.
- 새로 만드는 파일과 폴더 이름은 `YYYYMMDD_`로 시작한다.
- 파일 삭제, 대량 이동, 외부 볼륨(`/Volumes/X31`, `/Volumes/GURI_1` 등)에 쓰기 전에는 먼저 확인을 받는다.
- 영상 데이터(`*.nii`, `*.nii.gz`, `*.dcm`, `data/`, `outputs/`)는 커밋하지 않는다(`.gitignore`).
- 데이터는 환자 단위로 분할한다. 같은 환자의 스캔이나 슬라이스가 서로 다른 split에 들어가지 않게 한다.
- 라벨 파일이 없다고 음성 GT로 간주하지 않는다. FHV, uncertain, YHK_new, rejected의 포함·제외 여부는 실험별 manifest와 protocol을 따른다.

## 확인이 필요한 항목 (TODO)

- 코호트별 스캐너 제조사, 자장 세기, 슬라이스 두께, in-plane 해상도
- DICOM → NIfTI 변환 도구와 옵션(예: dcm2niix)
- NPY → PNG 변환 스크립트의 위치, 비트 깊이, 스케일링 방식
- YOLO용 512×512가 `resize` 모드로 바로 만들어졌는지, 672 pad 후 다시 변환했는지
- 코호트별 방향(RAS 또는 LAS) 설정
