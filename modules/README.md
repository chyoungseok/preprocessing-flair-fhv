# FLAIR → YOLO PNG 전처리

원본 FLAIR NIfTI를 받아 YOLO 학습용 axial 슬라이스 PNG를 바로 만든다. 중간 NPY/HDF5는 만들지 않는다.

```bash
conda activate flair_preproc   # nibabel, SimpleITK, opencv, pandas, openpyxl, matplotlib
# Docker Desktop 실행 필요 (SynthStrip)

python -m modules.pipeline \
  --src-dir /Volumes/GURI_1/FLAIR_PAIK_nii \
  --out-dir /Volumes/GURI_1/YYYYMMDD_FLAIR_PAIK_yolo_png \
  --label-xlsx /Users/yschoi/code/09_FHV/csv_files/df_flair_label_UHKinspection.xlsx

# 일부만: --pid BP00013_P17914 (반복 가능), --limit 10
```

## 처리 순서 (케이스마다 메모리에서 연속 처리)

| 단계 | 구현 | 기본값 |
|---|---|---|
| Skull stripping | Docker SynthStrip, 기본 마스크(뇌 주변 CSF 포함) | `freesurfer/synthstrip:latest`, `--platform linux/amd64` |
| Bias correction | SimpleITK N4, brain mask 안에서 추정 | in-plane 4배 축소 `(4,4,1)`에서 추정 후 원해상도에 적용, 반복 `(50,50,30,20)` |
| Masking | 보정 영상 × brain mask | — |
| Orientation | nibabel 축 재배열(보간 없음) | LAS |
| Normalization | 슬라이스별 전체 화소 1–99 백분위수 min-max, [0,1] clip | `slice_p1_p99` |
| Size | 확대는 `cv2.INTER_LINEAR`, 축소는 `cv2.INTER_AREA`. z축은 그대로 | `resize`, 512×512 |
| PNG | `flipud(slice.T)`, `round(x*255)` 8-bit grayscale | `<case_id>__slice_0001.png` (1-based) |

이 기본값은 기존 YOLO 학습 데이터(2026-08-13 PAIK 실행, `09_FHV/yolo/00_preproc_pipeline`)와 같다. PAIK 6 case로 검증한 결과는 다음과 같다.

- SynthStrip 마스크는 기존 마스크와 voxel 단위로 완전히 같다.
- N4를 원해상도로 돌리면 PNG도 기존과 픽셀이 완전히 같다.
- 기본값인 `(4,4,1)` 축소에서는 평균 0.05–0.09 계조, 최대 5 계조(0–255) 차이가 난다.

## 속도와 병렬 처리

- `--workers`(기본 4)개 케이스를 병렬로 처리한다.
- SynthStrip 컨테이너 하나는 메모리를 최대 약 5.6 GiB 쓴다. 그래서 동시에 띄우는 컨테이너 수는 Docker 메모리 / 6 GiB로 자동 제한된다(16 GB이면 2개). 이를 넘으면 컨테이너가 OOM으로 죽는다(exit 137).
- SynthStrip은 amd64 이미지를 에뮬레이션으로 돌리므로 시간 대부분을 차지한다. M4 Pro, Docker 16 GB 기준 측정값:
  - 순차 실행: 케이스당 약 14초
  - `--workers 4`: 케이스당 약 10초
  - SynthStrip 결과가 이미 있을 때(N4 이후만): 케이스당 약 1.7초

## 출력

```
<out-dir>/
  positive_slices_all/images/*.png, manifest.csv   # fhv_slice_num에 있는 슬라이스
  negative_slices_all/images/*.png, manifest.csv   # 같은 scan의 나머지 슬라이스
  cases/<case_id>/  *_brain.nii.gz, *_brainMask.nii.gz, *_qc.png, case.json
  cases.csv          # done / failed / unlabeled / missing_input
  run_metadata.json  # 설정, 고정 파라미터, 패키지 버전
  logs/errors.txt
```

- 라벨 표에 없는 입력(`unlabeled`)은 처리하지 않는다. 음성으로 간주하지 않는다.
- `case_id`는 입력 파일명에서 `_FLAIR` 앞부분이다. 라벨 표에서는 `<id_subject>_<id_scan|id_session>` 또는 `id_subject`로 만든다.
- `cases/<case_id>/case.json`이 있으면 끝난 케이스로 보고 건너뛴다. SynthStrip 결과가 남아 있으면 Docker를 다시 돌리지 않는다.
- manifest에는 끝난 케이스만 들어간다. 실패한 케이스가 남긴 PNG가 `images/`에 있을 수 있으므로 학습 목록은 manifest를 기준으로 쓴다.
- 다른 도구가 만든 폴더나 다른 설정으로 만든 폴더에는 쓰지 않는다. 덮어쓰려면 `--overwrite`를 쓴다.

## 다른 설정을 추가하는 방법

현재는 각 단계에 정책이 하나씩만 구현되어 있다. 정책을 추가하려면 `modules/preprocess.py`에서 해당 레지스트리에 함수를 등록한다. 그러면 `pipeline.py`의 CLI 선택지에 자동으로 나타난다.

- 방향: `ORIENTATIONS`에 `"RAS"` 추가(`reorient()`는 임의 축 코드를 처리한다)
- 정규화: `NORMALIZATION_POLICIES`
- 크기: `SIZE_POLICIES` (예: 고정 spacing으로 리샘플링한 뒤 pad)
- Skull stripping: `SKULLSTRIP_BACKENDS` (예: 네이티브 PyTorch SynthStrip)
