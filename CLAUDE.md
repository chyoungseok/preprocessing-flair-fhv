# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

축상(axial) FLAIR MRI에서 FHV(FLAIR hyperintense vessel)를 탐지하는 연구(09_FHV)의 **전처리 코드** 저장소다. 원본 FLAIR NIfTI를 받아 YOLO(YOLO26n-P2 계열) 학습용 512×512 슬라이스 PNG를 바로 만든다. 모델 학습·평가 코드는 여기 없다. 모델은 임상의의 검토를 돕는 용도이며, 단독으로 진단하지 않는다.

- 코호트: 내부 UHK(GT 버전 full-v1, UHK70, UHK100 — 현재 주 연구는 UHK100), 외부 Guri2025, PAIK
- DICOM → NIfTI 변환은 이 저장소에 없다(도구 미확인).
- 기존 YOLO 학습 데이터는 `/Users/yschoi/code/09_FHV/yolo/00_preproc_pipeline`(2026-08-13 PAIK 실행)이 만들었다. 이 저장소의 기본값은 그 데이터와 같은 결과를 내도록 맞춰져 있다.

## 실행

- 환경: conda `flair_preproc`(Python 3.10, nibabel, SimpleITK, opencv, pandas, openpyxl, matplotlib). Skull stripping에 Docker Desktop과 `freesurfer/synthstrip` 이미지가 필요하다. 이 이미지는 amd64 전용이라 Apple Silicon에서는 에뮬레이션으로 돈다.
- 빌드 설정, 테스트, 린터는 없다.

```bash
python -m modules.pipeline --src-dir <nii_dir> --out-dir <new_out_dir> --label-xlsx <labels.xlsx>
python -m modules.pipeline ... --pid BP00013_P17914 --workers 1   # 한 케이스만, 순차 실행
```

- `--src-dir`은 재귀 검색한다(`*_FLAIR.nii*`, `.`으로 시작하는 파일 제외). `case_id`는 파일명의 `_FLAIR` 앞부분이다.
- 상세 옵션과 출력 구조는 `modules/README.md`를 본다.

## 아키텍처

- `modules/pipeline.py`: CLI, 케이스 탐색, 라벨 매칭, 케이스 단위 병렬 처리(`ProcessPoolExecutor`), PNG/manifest/QC/메타데이터 쓰기.
- `modules/preprocess.py`: 단계별 연산. SynthStrip(Docker) → N4 → mask → 방향 정렬 → 슬라이스 정규화 → 크기 맞춤 → PNG 픽셀.
- `modules/labels.py`: 라벨 표를 `{case_id: CaseLabel}`로 읽는다. 표에 scan 열이 있으면 `case_id`는 `id_subject_id_scan`(또는 `id_session`)이고, 없으면 `id_subject`다.

핵심 설계:

- **단계마다 정책이 하나뿐이다.** 선택지는 `preprocess.py`의 레지스트리(`ORIENTATIONS`, `NORMALIZATION_POLICIES`, `SIZE_POLICIES`, `SKULLSTRIP_BACKENDS`)에 하나씩만 등록되어 있다. 사용자가 원할 때만 항목을 추가한다. 기본값을 바꾸거나 추가하기 전에 사용자 확인을 받는다.
- **케이스 하나를 메모리에서 끝까지 처리한다.** 디스크에 남는 중간 파일은 SynthStrip의 brain/mask뿐이고, 재실행할 때 이 파일을 재사용한다.
- `cases/<case_id>/case.json`이 완료 표시다. manifest는 매 실행 끝에 모든 `case.json`에서 다시 만든다.
- `run_metadata.json`의 `settings`가 출력 폴더의 설정이다. 다른 설정이나 다른 도구가 만든 폴더에는 쓰기를 거부한다(`check_output_dir`).
- SimpleITK 배열은 `(Z, Y, X)`이고, nibabel 순서 `(X, Y, Z)`로 맞추려고 `.T`를 쓴다. PNG는 `(X, Y)` 슬라이스에 `flipud(slice.T)`를 적용한다.
- `ProcessPoolExecutor`는 macOS에서 spawn 방식이다. 파이프라인을 import해서 돌리는 스크립트는 `if __name__ == "__main__":` 보호가 필요하다.

기존 데이터와의 관계:

- 기본값은 LAS, 슬라이스별 전체 화소 1–99 백분위수 정규화, 512 resize, 8-bit PNG다.
- 새 코드로 Docker SynthStrip을 돌린 마스크는 기존 마스크와 voxel 단위로 같다.
- N4 `N4_SHRINK_XYZ=(1,1,1)`로 돌리면 기존 PNG와 픽셀이 완전히 같다(PAIK 6 case 실측). 기본값 `(4,4,1)`은 약 10배 빠르고 평균 0.05–0.09 계조 차이가 난다.
- SynthStrip 동시 실행 수는 Docker 메모리 / 6 GiB로 제한된다(`docker_parallel_slots`, 프로세스 간 semaphore).
- 라벨은 **슬라이스 단위 양성 여부**(1-based `fhv_slice_num`)이며 YOLO bounding box GT와 다르다.
- 알려진 위험(선행연구 검토, 미적용):
  - 정규화에 마스크 밖의 0이 섞이고, 슬라이스 단위라 밝은 FHV가 99%에서 포화될 수 있다.
  - in-plane spacing을 통일하지 않아 FHV 크기가 스캐너마다 다르게 보인다.
  - SynthStrip 마스크가 피질 표면의 FHV를 자를 수 있다.
  - 바꾸면 기존 학습 데이터와 분포가 달라진다.
- 이전 단계별 스크립트(step0–7, H5/MIL, GradCAM 역변환 등)는 커밋 `e3cabae`에 있다.

## 작업 규칙

- 답변은 한국어로 한다. 결론을 먼저 말하고 관찰 결과, 해석, 가설, 권고를 구분한다.
- 데이터 개수와 수치는 실제 결과 파일에서 확인한다. 자료에 없는 값을 추정해서 채우지 않는다.
- **원본 DICOM, NIfTI, GT 라벨, 기존 전처리 결과를 수정하거나 덮어쓰지 않는다.** 새 산출물은 새 경로에 저장하고, `--overwrite`는 사용자가 명시적으로 요청했을 때만 쓴다.
- 새로 만드는 파일과 폴더 이름은 `YYYYMMDD_`로 시작한다.
- 파일 삭제, 대량 이동, 외부 볼륨(`/Volumes/X31`, `/Volumes/GURI_1` 등)에 쓰기 전에는 먼저 확인을 받는다. 외부 볼륨은 접근이 오래 걸리거나 멈출 수 있으므로 명령에 timeout을 둔다.
- 영상 데이터(`*.nii`, `*.nii.gz`, `*.dcm`, `data/`, `outputs/`)는 커밋하지 않는다(`.gitignore`).
- 데이터는 환자 단위로 분할한다. 같은 환자의 스캔이나 슬라이스가 서로 다른 split에 들어가지 않게 한다.
- 라벨 표에 없다고 음성 GT로 간주하지 않는다. FHV, uncertain, YHK_new, rejected의 포함·제외 여부는 실험별 manifest와 protocol을 따른다.

## 확인이 필요한 항목 (TODO)

- 코호트별 스캐너 제조사, 자장 세기, 슬라이스 두께, in-plane 해상도
- DICOM → NIfTI 변환 도구와 옵션(예: dcm2niix)
- 네이티브(Docker 없는) SynthStrip이 Docker 결과와 같은 마스크를 내는지 검증 → 같으면 `SKULLSTRIP_BACKENDS`에 추가. 현재 병목이다(에뮬레이션, 컨테이너당 최대 약 5.6 GiB라 동시에 2개까지만 실행).
