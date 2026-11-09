# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

축상(axial) FLAIR MRI에서 FHV(FLAIR hyperintense vessel)를 연구하는 프로젝트(09_FHV)의 **전처리 코드** 저장소다. 원본 FLAIR NIfTI를 전처리하고, 단계마다 NIfTI를 저장한 뒤, 512×512 axial 슬라이스 PNG로 내보낸다. **범위는 PNG 저장까지다.** 모델과 그 학습 데이터 형식은 다루지 않는다.

- 코호트: 내부 UHK(GT 버전 full-v1, UHK70, UHK100 — 현재 주 연구는 UHK100), 외부 Guri2025, PAIK
- DICOM → NIfTI 변환은 이 저장소에 없다(도구 미확인).

## 슬라이스 라벨의 출처

임상의가 슬라이스마다 FHV가 있는지 보고 기록한 annotation이다. 환자는 보통 MRI를 2회(초기 init, 추적 fu) 촬영했다. 파이프라인은 2차 표만 읽는다(`--label-xlsx`).

| | 1차 (초기) | 2차 (재검토, 현재 사용) |
|---|---|---|
| 파일 | `data/labels/FLAIR_slice_qc_20240102.xlsx` (시트 `label_common_V2`) | `data/labels/df_flair_label_UHKinspection.xlsx` (`Sheet1`) |
| 단위 | 환자 1명 = 1행 (1,258행) | 촬영 1회 = 1행 (2,516행) |
| 환자 ID | `Reg_No` (`BP_00001`, 밑줄 있음) | `id_subject` (`BP00001`) |
| 촬영 ID | `PKFs` 목록 (예: `['P14147', 'P14163']`), `initial_PK`, `fu_PK` | `id_scan` |
| init/fu | 열 위치로 구분 | `is_fu` (True/False) |
| FHV 슬라이스 | `FHV_init_slice`, `FHV_fu_slice` (예: `13-15` 같은 범위 표기) | `fhv_slice_num` (예: `[13, 14, 15]`) |

- `data/labels/`는 원본 `/Users/yschoi/code/09_FHV/csv_files/`의 사본이다(SHA-256 동일). 환자 ID가 들어 있어 `data/`는 git에서 제외되어 있고, 커밋하지 않는다.
- 슬라이스 번호는 1-based이며 원본 NIfTI의 axial 순서를 따른다.
- 1차 표의 나머지 열은 거의 쓰지 않는다.
- **init/fu 대응:** init 촬영은 `initial_PK`, 곧 `PKFs[0]`이다.
  - 촬영이 3회 이상인 환자가 215명 있다. 이들의 fu 촬영은 `PKFs[1]`이 아니라 **`fu_PK`, 곧 `PKFs`의 마지막**이다.
  - 2차 표는 이 두 촬영만 행으로 가진다. 2,516행은 1,258명 × 2회다.
- **1차와 2차의 차이** (2026-10-30, scan 단위 실측):
  - 양성 scan: 1차 271개 → 2차 273개. 양성 환자: 206명 → 212명.
  - 양쪽 모두 양성인 227개 scan은 슬라이스 번호가 완전히 같다.
  - 바뀐 것은 양성 여부뿐이다. 1차에서만 양성 44개, 2차에서만 양성 46개이고, 라벨이 바뀐 환자는 76명이다. init과 fu가 서로 바뀐 경우는 없다.
- 2차 표에서 `fhv_slice_num`이 빈 행은 "검토했고 FHV 슬라이스 없음"이다. 표에 없는 scan은 라벨이 없는 것이며 음성이 아니다.

## 실행

- 환경: conda `flair_preproc`(Python 3.10, nibabel, SimpleITK, opencv, pandas, openpyxl, matplotlib).
  - Skull stripping의 기본 백엔드인 `native`는 같은 conda 아래의 `fhv_synthstrip` 환경(`requirements-synthstrip.txt`)이 필요하다.
  - 처음 한 번은 Docker Desktop이 켜져 있어야 `freesurfer/synthstrip` 이미지에서 스크립트와 가중치를 복사할 수 있다.
  - `--skullstrip-backend docker`는 amd64 이미지를 에뮬레이션으로 돌리며 약 4배 느리다.
- 빌드 설정, 테스트, 린터는 없다.

```bash
python -m modules.pipeline --src-dir <nii_dir> --out-dir <new_out_dir> --label-xlsx data/labels/df_flair_label_UHKinspection.xlsx
python -m modules.pipeline ... --pid P17914 --workers 1   # 한 케이스만, 순차 실행 (촬영 ID 또는 case_id)
```

- PAIK 원본은 `/Volumes/One Touch/01_raw_data/01_stroke/PAIK_hospital/FLAIR_nii/<scan>/<scan>_FLAIR.nii`이다. 전처리 결과는 `/Volumes/One Touch/02_processed_data/1_stroke/PAIK_hospital/` 아래에 만든다.
- `--src-dir`은 재귀 검색한다(`*_FLAIR.nii*`, `.`으로 시작하는 파일 제외). 파일명의 `_FLAIR` 앞부분이 `case_id`이거나 고유한 `id_scan`이면 라벨과 매칭된다.
- 상세 옵션과 출력 구조는 `modules/README.md`를 본다.

## 아키텍처

- `modules/pipeline.py`: CLI, 케이스 탐색, 케이스 단위 병렬 처리(`ProcessPoolExecutor`), 단계별 NIfTI/PNG/manifest/QC/메타데이터 쓰기.
- `modules/preprocess.py`: 단계별 연산.
  - 순서: SynthStrip(native MPS 또는 Docker) → N4 → mask → 방향 정렬 → 볼륨 정규화 → 크기 맞춤 → PNG 픽셀
  - 크기 맞춤은 geometry를 반환하고, `canvas_affine`이 캔버스 격자의 affine을 만든다.
- `tools/20261109_convert_bbox_to_preproc.py`: 파이프라인 밖의 일회성 도구. 기존 YOLO용 PNG(512로 바로 resize)에 그린 PAIK bbox(YOLO `FHV_bbox_full_v1`, CVAT `uhk100percent`)를 `case.json` geometry로 현재 PNG 좌표에 옮기고 비교 overlay를 만든다. 결과는 `PAIK_hospital/20261109_FLAIR_preproc_bbox/`. 273 scan 중 54건만 좌표가 바뀐다.
- `modules/labels.py`: 라벨 표를 `{case_id: CaseLabel}`로 읽는다(`case_id` = `id_subject_id_scan`). `match_inputs`가 입력 파일과 라벨을 연결한다.
- **출력 구조:**
  - PNG: `<out>/png/{positive,negative}_slices/<case_id>_slice_0001.png` (케이스 하위 폴더 없음)
  - manifest: `<out>/png/manifest.csv` (클래스를 합친 한 파일)
  - 단계별 NIfTI 7개, QC, `case.json`: `<out>/FLAIR_nii_p/<case_id>/`
  - NIfTI 이름은 접미사를 이어 붙인다(`_FLAIR_brain` → `_bfCorrected` → `_masked` → `_reorient` → `_norm` → `_resampled`, 그리고 `_FLAIR_brainMask`).

핵심 설계:

- **단계마다 정책이 하나뿐이다.** 선택지는 `preprocess.py`의 레지스트리(`ORIENTATIONS`, `NORMALIZATION_POLICIES`, `SIZE_POLICIES`, `SKULLSTRIP_BACKENDS`)에 있다. 기본값 외에는 `docker` 백엔드만 있다. 사용자가 원할 때만 항목을 추가하고, 기본값을 바꾸거나 추가하기 전에 사용자 확인을 받는다.
- **볼륨 단위 처리:** 정규화와 크기 정책은 볼륨 전체를 받는다. 크기 정책은 `geometry`(`shape_xy`, `zooms_xy`, `resampled_xy`, `offset_xy`, `canvas`, `brain_cropped`)를 함께 돌려주고, 이 값은 `case.json`에 기록된다. PNG 좌표를 바꾸는 변경은 geometry와 `canvas_affine`에도 반영한다.
- **케이스 하나를 메모리에서 끝까지 처리한다.** 단계마다 NIfTI를 저장하고, 재실행할 때는 SynthStrip 결과(brain, mask)를 재사용한다.
- **완료 표시와 manifest:** `FLAIR_nii_p/<case_id>/case.json`이 완료 표시다. manifest는 매 실행 끝에 모든 `case.json`에서 다시 만든다.
- **폴더 보호:** `run_metadata.json`의 `settings`가 출력 폴더의 설정이다. 다른 설정이나 다른 도구가 만든 폴더에는 쓰기를 거부한다(`check_output_dir`).
- **배열 축 순서:** SimpleITK 배열은 `(Z, Y, X)`이고, nibabel 순서 `(X, Y, Z)`로 맞추려고 `.T`를 쓴다. PNG는 `(X, Y)` 슬라이스에 `flipud(slice.T)`를 적용한다.
- **spawn 주의:** `ProcessPoolExecutor`는 macOS에서 spawn 방식이다. 파이프라인을 import해서 돌리는 스크립트는 `if __name__ == "__main__":` 보호가 필요하다.

현재 기본값과 검증된 사실:

- 기본값:
  - SynthStrip native(MPS)
  - N4 in-plane 4배 축소
  - LAS
  - 뇌 마스크 기준 볼륨 0.5–99.5 백분위수 정규화(`volume_brain_p05_p995`)
  - 0.41 mm/px 리샘플링 후 512 pad, 512보다 크면 뇌 중심 crop(`fixed_spacing_pad`)
  - 8-bit PNG
- **SynthStrip 동시 실행:** 프로세스 간 semaphore로 제한한다. Docker는 메모리 / 6 GiB(`docker_parallel_slots`), native는 2개다.
- **`native` 백엔드:**
  - Docker 이미지의 `mri_synthstrip`과 가중치를 `~/.cache/fhv_preproc/synthstrip/`에 꺼내고 장치만 MPS로 바꿔 실행한다.
  - 마스크는 Docker와 voxel 1개 이내로 같고 약 4배 빠르다.
  - arm64 CPU PyTorch로 돌리면 Conv3d가 40 GB 이상을 써서 죽는다.
  - FreeSurfer 코드는 저장소에 커밋하지 않는다.
  - PyPI의 surfa 0.6.x는 numpy 2에서 실패하므로 surfa는 master 커밋으로 고정했다.
- **PAIK 원본 형태:** 대부분 이미 LAS이고, FOV는 200–220 mm다.
- **남은 위험(미적용):** SynthStrip 기본 경계(`-b 1`)가 피질 표면의 FHV 신호를 자를 수 있다. 경계를 넓히는 것(`-b 3`)은 검토만 했다.
- **이전 스크립트:** 예전 단계별 스크립트(step0–7, H5/MIL, GradCAM 역변환 등)는 커밋 `e3cabae`에 있다.

## 작업 규칙

- 답변은 한국어로 한다. 결론을 먼저 말하고 관찰 결과, 해석, 가설, 권고를 구분한다.
- 데이터 개수와 수치는 실제 결과 파일에서 확인한다. 자료에 없는 값을 추정해서 채우지 않는다.
- **원본 DICOM, NIfTI, GT 라벨, 기존 전처리 결과를 수정하거나 덮어쓰지 않는다.** 새 산출물은 새 경로에 저장하고, `--overwrite`는 사용자가 명시적으로 요청했을 때만 쓴다.
- 새로 만드는 파일과 폴더 이름은 `YYYYMMDD_`로 시작한다.
- 파일명과 폴더명에 언더바 두 개(`__`)를 연속으로 쓰지 않는다.
- 파일 삭제, 대량 이동, 외부 볼륨(`/Volumes/One Touch`, `/Volumes/GURI_1` 등)에 쓰기 전에는 먼저 확인을 받는다. 외부 볼륨은 접근이 오래 걸리거나 멈출 수 있으므로 명령에 timeout을 둔다.
- 영상 데이터(`*.nii`, `*.nii.gz`, `*.dcm`, `data/`, `outputs/`)는 커밋하지 않는다(`.gitignore`).
- 데이터는 환자 단위로 분할한다. 같은 환자의 스캔이나 슬라이스가 서로 다른 split에 들어가지 않게 한다.
- 라벨 표에 없다고 음성으로 간주하지 않는다.

## 확인이 필요한 항목 (TODO)

- 코호트별 스캐너 제조사, 자장 세기, 슬라이스 두께, in-plane 해상도
- DICOM → NIfTI 변환 도구와 옵션(예: dcm2niix)
- 전체 코호트에서 `native`와 `docker` 마스크를 표본 비교(현재는 PAIK 6 case만 검증).
