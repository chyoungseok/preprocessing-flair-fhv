# FLAIR → YOLO PNG 전처리

원본 FLAIR NIfTI를 받아 YOLO 학습용 axial 슬라이스 PNG를 바로 만든다. 중간 NPY/HDF5는 만들지 않는다.

```bash
conda activate flair_preproc   # nibabel, SimpleITK, opencv, pandas, openpyxl, matplotlib
# SynthStrip(native): 같은 conda 아래 fhv_synthstrip 환경이 필요하다 (아래 "SynthStrip 백엔드").
# 처음 한 번은 Docker Desktop이 켜져 있어야 한다 (이미지에서 스크립트·가중치를 복사).

python -m modules.pipeline \
  --src-dir /Volumes/GURI_1/FLAIR_PAIK_nii \
  --out-dir /Volumes/GURI_1/YYYYMMDD_FLAIR_PAIK_yolo_png \
  --label-xlsx /Users/yschoi/code/09_FHV/csv_files/df_flair_label_UHKinspection.xlsx

# 일부만: --pid BP00013_P17914 (반복 가능), --limit 10
```

## 처리 순서 (케이스마다 메모리에서 연속 처리)

| 단계 | 구현 | 기본값 |
|---|---|---|
| Skull stripping | SynthStrip v1 가중치, 기본 마스크(뇌 주변 CSF 포함) | `native`(Apple GPU), `freesurfer/synthstrip:latest`에서 복사 |
| Bias correction | SimpleITK N4, brain mask 안에서 추정 | in-plane 4배 축소 `(4,4,1)`에서 추정 후 원해상도에 적용, 반복 `(50,50,30,20)` |
| Masking | 보정 영상 × brain mask | — |
| Orientation | nibabel 축 재배열(보간 없음) | LAS |
| Normalization | 뇌 마스크 안 voxel의 **볼륨 전체** 0.5–99.5 백분위수로 min-max, [0,1] clip, 마스크 밖 0 | `volume_brain_p05_p995` |
| Size | in-plane을 **0.41 mm/px로 리샘플링**(확대 `INTER_LINEAR`, 축소 `INTER_AREA`)한 뒤 512에 맞춤. 512보다 작으면 중앙 pad, 크면 뇌 중심 기준 crop. z축은 그대로 | `fixed_spacing_pad`, 512×512 |
| PNG | `flipud(slice.T)`, `round(x*255)` 8-bit grayscale | `<case_id>__slice_0001.png` (1-based) |

### 2026-10-29 기본값 변경 (정규화·크기)

기존 YOLO 학습 데이터(2026-08-13 PAIK 실행)와 **PNG가 다르다.** 이 파이프라인은 PNG 저장까지만 한다. 박스 GT는 다루지 않는다.

- **정규화:** 예전 방식은 슬라이스마다 배경 0까지 섞어 백분위수를 계산했다. 그래서 뇌가 작은 슬라이스는 대비가 과하게 커지고, 밝은 FHV가 포화됐다.
  - 예시 scan의 FHV 박스 안 255 화소 비율: 예전 0.81%(최대 4.5%), 새 방식 0.03%(최대 0.25%).
  - 상한을 99.9%로 올리면 포화는 0이 되지만 박스 평균 밝기가 108에서 91로 떨어져 대비가 줄어든다.
- **크기:** 예전 방식은 모든 슬라이스를 512로 늘렸으므로 픽셀 간격이 FOV/512였다.
  - PAIK은 FOV가 200–220 mm로 거의 같아서 차이가 작았다(1–99% 구간 1.10배, 최대 1.26배).
  - 대신 정사각형이 아닌 matrix(372×384, 58건)는 가로가 3% 늘어났다.
  - 새 방식은 FOV와 matrix에 관계없이 같은 mm가 같은 px이다. 0.41 mm는 PAIK FOV 중앙값 210 mm ÷ 512이고, 512 px는 210 mm를 덮는다.
  - FOV 210 mm인 scan(PAIK의 대부분)은 geometry가 예전과 같다.
- **예전 데이터 재현:** `--normalization legacy_slice_p1_p99 --size-policy legacy_resize`. 이렇게 돌리면 예전 기본값 결과와 PNG가 완전히 같다(6 case, 144장 확인). SynthStrip 마스크는 예전 마스크와 voxel 단위로 같고, 예전 PNG와의 차이는 N4 `(4,4,1)` 축소에서 오는 평균 0.05–0.09 계조뿐이다.

### PNG geometry

각 케이스의 `case.json`에 있는 `geometry`가 voxel 좌표와 PNG 캔버스 좌표의 관계를 기록한다.

- `resampled_xy`: 리샘플링한 크기
- `offset_xy`: 캔버스에 놓은 위치(음수는 crop)
- `brain_cropped`: crop하다가 뇌가 잘렸는지

## SynthStrip 백엔드

| `--skullstrip-backend` | 방식 | 케이스당 시간 (6 case, `--workers 4`) |
|---|---|---|
| `native` (기본) | Docker 이미지에서 꺼낸 `mri_synthstrip`과 가중치를 Apple GPU(MPS)에서 실행 | 약 2.5초 |
| `docker` | amd64 이미지를 에뮬레이션으로 실행 | 약 10초 |

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
- M4 Pro 기준 측정값(6 case, `--workers 4`):
  - `native`: 케이스당 약 2.5–3초
  - `docker`: 케이스당 약 10초
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
- 출력 `case_id`는 라벨 표에서 만든 `<id_subject>_<id_scan|id_session>`(또는 `id_subject`)이다.
- 입력은 파일명의 `_FLAIR` 앞부분으로 찾는다. 이 값이 `case_id`(`BP00001_P14147_FLAIR.nii`)이거나 촬영 ID(`P14147/P14147_FLAIR.nii`, 표에서 `id_scan`이 고유할 때)이면 매칭된다. `--pid`도 둘 다 받는다.
- `cases/<case_id>/case.json`이 있으면 끝난 케이스로 보고 건너뛴다. SynthStrip 결과가 남아 있으면 SynthStrip을 다시 돌리지 않는다. 단, 백엔드나 이미지가 바뀌면 다시 돌린다.
- manifest에는 끝난 케이스만 들어간다. 실패한 케이스가 남긴 PNG가 `images/`에 있을 수 있으므로 학습 목록은 manifest를 기준으로 쓴다.
- 다른 도구가 만든 폴더나 다른 설정으로 만든 폴더에는 쓰지 않는다. 덮어쓰려면 `--overwrite`를 쓴다.

## 다른 설정을 추가하는 방법

정책을 추가하려면 `modules/preprocess.py`에서 해당 레지스트리에 함수를 등록한다. 그러면 `pipeline.py`의 CLI 선택지에 자동으로 나타난다. 정규화와 크기의 `legacy_*` 정책은 예전 데이터를 재현하는 용도다.

- 방향: `ORIENTATIONS`에 `"RAS"` 추가(`reorient()`는 임의 축 코드를 처리한다)
- 정규화: `NORMALIZATION_POLICIES`. `(volume, mask) -> [0,1] volume`
- 크기: `SIZE_POLICIES`. `(volume, mask, zooms_xy, size) -> (canvas volume, geometry)`
- Skull stripping: `SKULLSTRIP_BACKENDS`
