# preprocessing-flair-fhv

축상(axial) FLAIR MRI에서 FHV(FLAIR hyperintense vessel)를 연구하기 위한 전처리 코드다. 원본 FLAIR NIfTI를 전처리하고, 단계마다 NIfTI를 저장한 뒤, 512×512 axial 슬라이스 PNG로 내보낸다. 슬라이스는 FHV 라벨에 따라 양성과 음성으로 나눈다.

범위는 PNG 저장까지다. 모델 학습과 학습 데이터 형식은 다루지 않는다.

## 처리 순서

```
FLAIR NIfTI
 → 1 Skull stripping   SynthStrip (Apple GPU/MPS 또는 Docker)
 → 2 Bias correction   SimpleITK N4 (brain mask 안에서 추정)
 → 3 Masking           보정 영상 × brain mask
 → 4 Orientation       LAS (보간 없는 축 재배열)
 → 5 Normalization     뇌 마스크 voxel의 볼륨 전체 0.5–99.5 백분위수 → [0,1]
 → 6 Size              in-plane 0.41 mm/px 리샘플링 → 512×512 pad/crop (z는 그대로)
 → PNG                 8-bit grayscale, 슬라이스마다 1장
```

단계별 구현, 기본값을 정한 이유, 출력 형식의 세부 사항은 [modules/README.md](modules/README.md)에 있다.

## 설치

```bash
# 메인 환경 (Python 3.10)
conda create -n flair_preproc python=3.10
conda activate flair_preproc
pip install nibabel SimpleITK opencv-python pandas openpyxl matplotlib

# SynthStrip native 백엔드용 환경 (PyTorch MPS + surfa)
conda create -n fhv_synthstrip python=3.12
/opt/anaconda3/envs/fhv_synthstrip/bin/pip install -r requirements-synthstrip.txt
```

- 처음 한 번은 Docker Desktop이 켜져 있어야 한다. `freesurfer/synthstrip` 이미지에서 스크립트와 가중치를 `~/.cache/fhv_preproc/synthstrip/`로 복사한다. 그 뒤로는 Docker가 필요 없다.
- `fhv_synthstrip` 환경 위치는 `--synthstrip-python` 또는 `FHV_SYNTHSTRIP_PYTHON`으로 바꿀 수 있다.
- `--skullstrip-backend docker`로 Docker 이미지를 직접 돌릴 수도 있다. amd64 에뮬레이션이라 약 4배 느리다.

## 실행

```bash
python -m modules.pipeline \
  --src-dir <원본 FLAIR NIfTI 폴더> \
  --out-dir <새 출력 폴더> \
  --label-xlsx data/labels/df_flair_label_UHKinspection.xlsx

# 일부만: --pid <촬영 ID 또는 case_id> (반복 가능), --limit 10, --workers 1
```

- **입력 찾기:** `--src-dir`을 재귀 검색해 `*_FLAIR.nii*` 파일을 찾는다.
- **라벨 표:** 촬영 1회가 한 행이고, `id_subject`, `id_scan`, `fhv_slice_num`(1-based 슬라이스 목록) 열을 쓴다.
  - 표에 없는 scan은 처리하지 않고 `unlabeled`로 기록한다. 이런 scan을 음성으로 간주하지 않는다.
- **재실행:** 같은 명령을 다시 실행하면 끝난 케이스는 건너뛴다.
- **폴더 보호:** 다른 설정으로 만든 폴더에는 쓰지 않는다. 덮어쓰려면 `--overwrite`를 쓴다.

## 출력

```
<out-dir>/
  png/
    positive_slices/<case_id>_slice_0013.png
    negative_slices/<case_id>_slice_0001.png
    manifest.csv                 # PNG 1장 = 1행, label 0/1
  FLAIR_nii_p/<case_id>/         # 단계별 NIfTI 7개, QC 이미지, case.json
  cases.csv                      # done / failed / unlabeled / missing_input
  run_metadata.json              # 설정, 고정 파라미터, 패키지 버전
```

## 성능 (PAIK, Apple Silicon, 외장 SSD)

| 항목 | 값 |
|---|---|
| 처리 결과 | 2,515 scan 모두 완료, 실패 0 |
| PNG | 60,537장 (양성 1,175) |
| 소요 시간 | 약 1시간 43분 (`--workers 4`, 케이스당 약 2.5초) |
| 용량 | 약 111 GB (케이스당 약 44 MB) |

## 저장소 구성

| 파일 | 역할 |
|---|---|
| `modules/pipeline.py` | CLI, 케이스 탐색, 병렬 처리, NIfTI/PNG/manifest 쓰기 |
| `modules/preprocess.py` | 단계별 연산과 정책 레지스트리 |
| `modules/labels.py` | 라벨 표 읽기, 입력 파일과 라벨 매칭 |
| `requirements-synthstrip.txt` | SynthStrip native 환경 버전 고정 |

- **커밋하지 않는 데이터:** 라벨 표(`data/`)와 영상 데이터는 환자 정보를 담고 있어 저장소에 포함하지 않는다.
- **FreeSurfer 코드:** SynthStrip 코드와 가중치도 저장소에 없다. 첫 실행 때 Docker 이미지에서 받는다.
