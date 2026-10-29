# CLAUDE.md — FHV 탐지 연구 (09_FHV)

## 1. 프로젝트 목적

축상 FLAIR MRI에서 FHV(FLAIR hyperintense vessel)의 의심 위치를 박스로 제시하여, 임상의의 검토를 돕는 탐지 모델(YOLO26n-P2 계열)을 개발한다. 목표는 놓치는 소견과 불필요한 검토 부담을 함께 줄이는 것이며, 모델이 단독으로 진단하거나 치료를 결정하는 시스템은 아니다.

## 2. FLAIR 데이터 사양

| 항목 | 내용 |
|---|---|
| 획득 | 2D axial 다중 슬라이스 FLAIR |
| 원본 형식 | DICOM (`.dcm`) |
| 변환 흐름 | DICOM → NIfTI → 전처리 → NumPy `(Z, H, W)` → 슬라이스별 PNG (PNG 1장 = 2D axial slice 1장) |
| 모델 입력 크기 | 512×512 |
| 스캐너 | **TODO: 조사 필요** (제조사, 자장 세기, 슬라이스 두께, in-plane 해상도). `modules/dicom_scanner_metadata.py`는 DICOM의 Manufacturer 태그를 읽어 제조사를 Philips/GE/Siemens/UNKNOWN으로 정규화한다. |
| 코호트·버전 | 내부 UHK(GT 버전: full-v1, UHK70, UHK100 — 현재 주 연구는 UHK100), 외부 Guri2025 |

- 전처리 후 영상의 voxel spacing은 원본과 같다. Step 5의 `resize` 모드를 쓰면 in-plane spacing만 달라진다. 원본 정보는 `*_meta.json`의 `zooms`·`affine`에 남는다.
- z축은 리샘플링하지 않는다. 출력 슬라이스 수는 항상 원본 NIfTI의 Z와 같다.

## 3. 전처리 파이프라인 (`modules/`, 자세한 내용은 `modules/README.md`)

| 단계 | 스크립트 | 내용 | 주요 출력 |
|---|---|---|---|
| 0 | (TODO: 도구 확인) | DICOM → NIfTI 변환 | `*_FLAIR.nii` |
| 1 | `step1_skull_stripping.py` | Docker로 SynthStrip을 실행한다 | `*_brain.nii.gz`, `*_brainMask.nii.gz` |
| 2 | `step2_bias_field_correction.py` | Brain mask를 이용해 N4 bias field 보정을 한다 | `*_bfCorrected.nii.gz` |
| 3 | `step3_masking.py` | (선택) mask를 다시 적용해 뇌 바깥을 0으로 만든다 | `*_masked.nii.gz` |
| 4 | `step4_reorientation.py` | NiBabel로 RAS 또는 LAS 방향을 맞춘다. 보간·정합·리샘플링은 하지 않는다 | `*_reorient.nii.gz` |
| 5 | `step5_resize_normalization.py` | axial 슬라이스를 NPY로 내보낸다. 슬라이스별 1–99 백분위수 min-max 정규화(clip [0, 1])를 적용하고 in-plane 크기를 맞춘다 | `*_AxialSlices_{padded\|resized}.npy`, `*_label_sliceLevel.npy`, `*_meta.json` |
| — | (TODO: 스크립트 위치 확인, `09_YOLO` 쪽으로 추정) | NPY → 슬라이스별 PNG로 변환 | YOLO 입력 이미지 |

- 1–5단계를 한 번에 실행하려면 `step1_to_step5_pipeline.py`를 쓴다(`single` / `batch`, `--pid`로 PID 하나만 처리할 수 있다).
- Step 5 크기 정책: 기본값은 `pad`(중앙 zero-pad, 672×672, 보간 없음)이다. `resize`는 bilinear 보간과 anti-aliasing을 적용하며 `--target-size 512`로 512×512를 만든다. **YOLO용 512×512 입력이 어느 모드로 만들어졌는지는 해당 데이터의 `*_meta.json`에서 `preprocess_mode`와 `target_hw`를 확인하라.**
- 슬라이스 라벨: Excel/CSV(`--label-xlsx`, `--label-column fhv_slice_num`)에서 읽고, 없으면 같은 폴더의 기존 `*label_sliceLevel.npy`로 대체한다. 이는 **슬라이스 단위 양성 여부**이며 YOLO의 bounding box GT와는 다르다.
- Step 6(`step6_create_filtered_h5.py`, 인접 슬라이스 triplet HDF5)과 Step 7(GradCAM을 원본 공간으로 역변환)은 주로 MIL 연구용이다. **MIL 결과를 YOLO 결과와 섞지 마라.**
- `step0_slice_idx2zcoord.py`는 방향 변환으로 슬라이스 순서가 바뀔 때만 쓰는 선택 유틸리티다.
- 각 단계는 QC 산출물(`qc_skullstrip/`, `*_bias_field_correction_QC/`, `qc_masking/`, `qc_reorientation/`, `qc_pad|qc_resize/`)을 만든다. 전처리를 바꿨다면 QC를 확인한 뒤 다음 단계로 넘어간다.
- 스캔 간·템플릿 정합(registration)은 **현재 파이프라인에 포함되어 있지 않다**.

## 4. 원하는 최종 출력

**전처리가 완료된 데이터**를 최종 출력으로 본다.

- Skull-stripped, N4 보정, masking, 방향 정렬을 마친 NIfTI (`*_FLAIR_brain_bfCorrected_masked_reorient.nii.gz`)
- 이를 정규화하고 크기를 맞춘 axial NPY와 슬라이스 라벨, meta JSON
- YOLO 학습·평가에 쓰는 512×512 슬라이스별 PNG
- 각 단계의 QC 이미지와 로그

## 5. 작업 규칙

### 기본
- 답변은 한국어로 한다. 논문 문장은 요청이 있을 때 학술 영어로 쓴다.
- 결론을 먼저 말하고 관찰 결과, 해석, 가설, 권고를 구분한다.
- 성능 수치와 데이터 개수는 실제 결과 파일에서 확인한다. 자료에 없는 값을 추정해서 채우지 않는다.
- 모델 이름(E10, D0, ADD-B, D1–D3, C12/C13/C23/C123 등)만 보고 구조나 설정을 추정하지 말고, 해당 run의 설정 파일을 확인한다.

### 데이터 보존
- **원본 DICOM, NIfTI, GT 라벨, 기존 전처리 결과, 기존 학습 run을 수정하거나 덮어쓰지 않는다.** 새 산출물은 새 경로에 저장한다. `--overwrite`는 사용자가 명시적으로 요청했을 때만 쓴다.
- 새로 만드는 파일과 폴더 이름은 `YYYYMMDD_`로 시작한다.
- 파일 삭제, 대량 이동, 외부 볼륨(`/Volumes/X31`, `/Volumes/GURI_1` 등)에 쓰기 전에는 먼저 확인을 받는다.

### 데이터 처리
- 데이터는 환자 단위로 분할한다. 같은 환자의 스캔이나 슬라이스가 서로 다른 split에 들어가지 않게 한다.
- 라벨 파일이 없다고 음성 GT로 간주하지 않는다.
- FHV, uncertain, YHK_new, rejected의 포함·제외 여부는 실험별 manifest와 protocol을 따른다.
- 예측지도 기반 학습에서는 환자 단위 out-of-fold 생성 여부를 확인한다.
- 보고하는 "병변 수"는 2D 슬라이스별 GT 박스 수이며, 3D 병변 수가 아니다.

### 평가와 해석
- A(mAP, IoU ≥ 0.5), B(IoU ≥ 0.2/0.1, Center), C(Any-overlap, 슬라이스 적중, zero-hit, Top-K, 표시량) 평가 체계의 성공 기준을 섞지 않는다. 지표를 쓸 때는 단위(박스, 슬라이스, 스캔)를 반드시 밝힌다.
- `best_map`과 `best_assist`(`center-sign-recall-first-v1`)를 구분한다. Checkpoint와 cutoff는 validation에서 고정한 뒤 test와 외부 평가에 적용한다.
- 내부 test를 완전히 독립된 검증으로 표현하지 않는다. Guri2025의 specificity를 과장하지 않는다. 기하학적 적중을 임상적 유효성과 동일시하지 않는다.

### 시각화
- GT는 초록, Pred는 주황으로 그린다. Confidence 값 대신 #1, #2 같은 순위를 표시한다. 비교 사례는 같은 환자·스캔·슬라이스·확대율로 맞춘다.

## 6. 확인이 필요한 항목 (TODO)

- [ ] 스캐너 제조사, 자장 세기, 슬라이스 두께, in-plane 해상도 (코호트별)
- [ ] DICOM → NIfTI 변환 도구와 옵션 (예: dcm2niix)
- [ ] NPY → PNG 변환 스크립트의 위치, 비트 깊이, 스케일링 방식
- [ ] YOLO용 512×512가 `resize` 모드로 직접 만들어졌는지, 672 pad 후 다시 변환했는지
- [ ] 코호트별 방향(RAS 또는 LAS) 설정
