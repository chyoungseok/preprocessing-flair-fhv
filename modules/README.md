# FLAIR → 슬라이스 PNG 전처리

원본 FLAIR NIfTI를 전처리하고, 단계마다 NIfTI를 저장한 뒤, axial 슬라이스 PNG로 내보낸다. 이 파이프라인은 PNG 저장까지만 한다.

```bash
conda activate flair_preproc   # nibabel, SimpleITK, opencv, pandas, openpyxl, matplotlib
# SynthStrip(native): 같은 conda 아래 fhv_synthstrip 환경이 필요하다 (아래 "SynthStrip 백엔드").
# 처음 한 번은 Docker Desktop이 켜져 있어야 한다 (이미지에서 스크립트·가중치를 복사).

python -m modules.pipeline \
  --src-dir "/Volumes/One Touch/01_raw_data/01_stroke/PAIK_hospital/FLAIR_nii" \
  --out-dir "/Volumes/One Touch/02_processed_data/1_stroke/PAIK_hospital/YYYYMMDD_FLAIR_preproc" \
  --label-xlsx data/labels/df_flair_label_UHKinspection.xlsx

# 일부만: --pid P17914 또는 --pid BP00013_P17914 (반복 가능), --limit 10
```

## 처리 순서 (케이스마다 메모리에서 연속 처리)

| 단계 | 구현 | 기본값 |
|---|---|---|
| 1 Skull stripping | SynthStrip v1 가중치, 기본 마스크(뇌 주변 CSF 포함) | `native`(Apple GPU), `freesurfer/synthstrip:latest`에서 복사 |
| 2 Bias correction | SimpleITK N4, brain mask 안에서 추정 | in-plane 4배 축소 `(4,4,1)`에서 추정 후 원해상도에 적용, 반복 `(50,50,30,20)` |
| 3 Masking | 보정 영상 × brain mask | — |
| 4 Orientation | nibabel 축 재배열(보간 없음) | LAS |
| 5 Normalization | 뇌 마스크 안 voxel의 **볼륨 전체** 0.5–99.5 백분위수로 min-max, [0,1] clip, 마스크 밖 0 | `volume_brain_p05_p995` |
| 6 Size | in-plane을 **0.41 mm/px로 리샘플링**(확대 `INTER_LINEAR`, 축소 `INTER_AREA`)한 뒤 512에 맞춤. 512보다 작으면 중앙 pad, 크면 뇌 중심 기준 crop. z축은 그대로 | `fixed_spacing_pad`, 512×512 |
| PNG | `flipud(slice.T)`, `round(x*255)` 8-bit grayscale | `<case_id>_slice_0001.png` (1-based) |

- **정규화:** 볼륨 전체에서 범위를 하나만 정하므로 슬라이스 사이 대비가 일정하다. 배경 0이 통계에 섞이지 않는다. 상한을 99.5%로 두어 가장 밝은 구조(FHV)가 거의 포화되지 않는다.
- **크기:** FOV와 matrix에 관계없이 1 px이 0.41 mm다. 0.41 mm는 PAIK FOV 중앙값 210 mm ÷ 512이고, 512 px는 210 mm를 덮어 성인 뇌가 들어간다. 정사각형이 아닌 matrix도 가로세로 비율이 유지된다.

## 출력

```
<out-dir>/
  png/
    positive_slices/<case_id>_slice_0013.png   # fhv_slice_num에 있는 슬라이스
    negative_slices/<case_id>_slice_0001.png   # 같은 scan의 나머지 슬라이스
    manifest.csv                                          # PNG 1장 = 1행, label 0/1
  FLAIR_nii_p/<case_id>/
    <case_id>_FLAIR_brain.nii.gz                                    # 1 SynthStrip
    <case_id>_FLAIR_brainMask.nii.gz                                # 1 SynthStrip 마스크
    <case_id>_FLAIR_brain_bfCorrected.nii.gz                        # 2 N4
    <case_id>_FLAIR_brain_bfCorrected_masked.nii.gz                 # 3 마스크 재적용
    <case_id>_FLAIR_brain_bfCorrected_masked_reorient.nii.gz        # 4 LAS
    <case_id>_FLAIR_brain_bfCorrected_masked_reorient_norm.nii.gz   # 5 [0,1], 원래 voxel 간격
    <case_id>_FLAIR_brain_bfCorrected_masked_reorient_norm_resampled.nii.gz  # 6 PNG와 같은 격자(0.41 mm, 512×512×Z)
    <case_id>_qc.png
    case.json
  cases.csv          # done / failed / unlabeled / missing_input
  run_metadata.json  # 설정, 고정 파라미터, 패키지 버전
  logs/errors.txt
```

- **입력 매칭:** 입력은 파일명의 `_FLAIR` 앞부분으로 찾는다. 이 값이 `case_id`(`BP00001_P14147_FLAIR.nii`)이거나 촬영 ID(`P14147/P14147_FLAIR.nii`, 표에서 `id_scan`이 고유할 때)이면 매칭된다. `--pid`도 둘 다 받는다.
- **case_id:** 출력 `case_id`는 라벨 표에서 만든 `<id_subject>_<id_scan|id_session>`(또는 `id_subject`)이다.
- **상태 구분:**
  - `unlabeled`: 입력은 있지만 라벨 표에 없다. 처리하지 않으며, 음성으로 간주하지 않는다.
  - `missing_input`: 라벨 표에는 있지만 `--src-dir`에서 원본을 찾지 못했다. `--pid`나 `--limit`과 관계없이 표 전체를 기준으로 매긴다.
- **단계별 NIfTI:**
  - 모두 float32로 저장한다. 마스크만 SynthStrip이 저장한 형식을 따른다.
  - affine은 단계 1–3이 원본, 단계 4–5가 LAS로 정렬한 것, 단계 6이 캔버스 격자다(`preprocess.canvas_affine`).
  - 단계 6 볼륨에 `round(x*255)`와 `flipud(T)`를 적용하면 PNG와 같다.
- **재실행:**
  - `FLAIR_nii_p/<case_id>/case.json`이 있으면 끝난 케이스로 보고 건너뛴다.
  - SynthStrip 결과가 남아 있으면 SynthStrip은 다시 돌리지 않는다. 단, 백엔드나 이미지가 바뀌면 다시 돌린다.
- **manifest:** 끝난 케이스만 들어간다. 실패한 케이스가 남긴 PNG가 있을 수 있으므로 목록은 manifest를 기준으로 쓴다.
- **geometry:** `case.json`의 `geometry`는 voxel 좌표와 PNG 캔버스 좌표의 관계다.
  - `resampled_xy`: 리샘플링한 크기
  - `offset_xy`: 캔버스에 놓은 위치(음수는 crop)
  - `brain_cropped`: crop하다가 뇌가 잘렸는지
- **폴더 보호:** 다른 도구가 만든 폴더나 다른 설정으로 만든 폴더에는 쓰지 않는다. 덮어쓰려면 `--overwrite`를 쓴다.
- **AppleDouble 파일:** exFAT 외장 드라이브에서는 macOS가 파일마다 `._*` 파일을 만든다. Python `glob`은 이 파일을 건너뛴다. 옮기거나 압축하기 전에는 `dot_clean <폴더>`로 정리한다.

## SynthStrip 백엔드

| `--skullstrip-backend` | 방식 | SynthStrip 시간 |
|---|---|---|
| `native` (기본) | Docker 이미지에서 꺼낸 `mri_synthstrip`과 가중치를 Apple GPU(MPS)에서 실행 | 케이스당 약 3초 |
| `docker` | amd64 이미지를 에뮬레이션으로 실행 | 약 4배 느림 |

`native` 백엔드는 다음과 같이 동작한다.

- 처음 실행할 때 한 번만 Docker 이미지에서 스크립트와 가중치를 `~/.cache/fhv_preproc/synthstrip/<image>/`로 복사한다. 이때 장치 한 줄만 `cpu`에서 `mps`로 바꾼다. 그 이후에는 Docker가 필요 없다.
- PyTorch(MPS에서 Conv3d 지원, 2.1.2는 미지원)와 surfa가 설치된 Python이 필요하다. 기본값은 같은 conda 아래의 `fhv_synthstrip` 환경이고, `--synthstrip-python` 또는 `FHV_SYNTHSTRIP_PYTHON`으로 바꿀 수 있다. 환경은 다음과 같이 만든다.

  ```bash
  conda create -n fhv_synthstrip python=3.12
  /opt/anaconda3/envs/fhv_synthstrip/bin/pip install -r requirements-synthstrip.txt
  ```
- arm64 CPU로 돌리면 PyTorch가 Conv3d에 40 GB 이상을 써서 프로세스가 죽는다. 그래서 MPS만 쓴다.
- PAIK 6 case에서 마스크가 Docker와 비교해 5 case는 완전히 같았고, 1 case는 voxel 1개만 달랐다.

## 속도와 병렬 처리

- `--workers`(기본 4)개 케이스를 병렬로 처리한다. SynthStrip만 동시 실행 수를 따로 제한한다.
  - `native`: 2개. 4개로 늘려도 거의 빨라지지 않았다.
  - `docker`: Docker 메모리 / 6 GiB개(16 GB이면 2개). 컨테이너 하나가 최대 약 5.6 GiB를 쓰므로, 이를 넘으면 OOM으로 죽는다(exit 137).
- 단계별 시간(워커 하나, 순차 실행): SynthStrip 약 3초, N4 1–5초(가장 김), 정규화·크기 0.1초 이하, PNG 0.2초.
- PAIK 10 case(`--workers 4`, One Touch): 케이스당 약 2.9초, 용량은 케이스당 약 41 MB(NIfTI 29 MB, PNG 13 MB)다. 2,515 scan 전체면 약 2시간, 약 105 GB로 추정한다.
- `run_metadata.json`의 `layout`이 출력 구조 버전이다. 구조가 다른 폴더에는 이어 쓰지 않는다.
- 실행할 때마다 `--src-dir` 재귀 검색에 고정 비용이 든다. One Touch의 2,515개 폴더에서 약 17초였다.

## 다른 설정을 추가하는 방법

각 단계에는 정책이 하나씩만 있다. 정책을 추가하려면 `modules/preprocess.py`에서 해당 레지스트리에 함수를 등록한다. 그러면 `pipeline.py`의 CLI 선택지에 자동으로 나타난다.

- 방향: `ORIENTATIONS`에 `"RAS"` 추가(`reorient()`는 임의 축 코드를 처리한다)
- 정규화: `NORMALIZATION_POLICIES`. `(volume, mask) -> [0,1] volume`
- 크기: `SIZE_POLICIES`. `(volume, mask, zooms_xy, size) -> (canvas volume, geometry)`
- Skull stripping: `SKULLSTRIP_BACKENDS`
