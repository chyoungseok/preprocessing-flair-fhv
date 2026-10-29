# FLAIR Preprocessing Order

Recommended execution order for this project:

1. `step1_skull_stripping.py`
   Run Docker SynthStrip and save `*_brain.nii.gz` plus `*_brainMask.nii.gz`.

2. `step2_bias_field_correction.py`
   Run N4 bias field correction on the skull-stripped brain image using the
   SynthStrip brain mask.

3. `step3_masking.py`
   Optional cleanup step. Re-apply the SynthStrip brain mask to the
   N4-corrected image so mask-outside voxels are forced to zero.

4. `step4_reorientation.py`
   Reorient the final NIfTI image to RAS or LAS. This step uses NiBabel orientation
   utilities and does not perform interpolation, registration, or voxel-spacing
   resampling. Labels are not remapped. Use `--target-orientation` in Step 4 or
   `--step4-target-orientation` in the combined Step 1-5 pipeline.

5. `step5_resize_normalization.py`
   Export axial slices from NIfTI `(H, W, Z)` to NumPy `(Z, H, W)`, optionally
   normalize intensities, standardize in-plane size, and save slice-level labels.
   The default policy is legacy center zero-padding to `672 x 672`. The alternate
   `resize` mode applies in-plane 2D bilinear resizing to `--target-size`.

6. `step6_create_filtered_h5.py`
   Create one filtered adjacent-slice HDF5 dataset directly from the subject
   NumPy arrays. Current defaults select center slices by relative stack position
   from 45% through 75%, independent of label positivity, and store
   `[previous, center, next]` triplets. Labels are supervision signals only.

7. `step7_gradcam_to_original_space.py`
   Convert `(Z, 672, 672)` GradCAM NPY volumes back to original FLAIR NIfTI
   space for overlay.

`step0_slice_idx2zcoord.py` remains available as an optional utility for data
where reorientation may change axial slice order. It is not part of the default
pipeline for the current LAS/RAS-only dataset.

## Step 5 Size and Label Policy

`step5_resize_normalization.py` supports two in-plane size standardization modes:

- `--preprocess-mode pad`:
  Center zero-pad each axial slice to `--target-size`. This is the default and
  preserves legacy `672 x 672` outputs. It does not interpolate.
- `--preprocess-mode resize`:
  Resize each axial slice to `--target-size x --target-size` using
  `skimage.transform.resize(order=1, anti_aliasing=True, preserve_range=True)`.

Step 5 never resamples the z-axis. The number of output slices is always the
input NIfTI `Z`.

Labels are loaded from the Excel/CSV table specified by `--label-xlsx` and
`--label-column` when available. If label loading is disabled or unavailable,
the script can fall back to one existing `*label_sliceLevel.npy` in the same
input PID directory, which is useful when resizing already labeled external
validation data.

## Step 6 H5 Selection Policy

Current Step 6 defaults:

- `--selection-mode relative`
- `--min-rel-index 45`
- `--max-rel-index 75`
- `--image-suffix _AxialSlices_padded.npy`
- `--label-suffix _label_sliceLevel.npy`

This means H5 sample count is controlled by the selected relative slice window,
not by the number of positive labels. Use
`--selection-mode local --min-index 12 --max-index 15` only when reproducing the
legacy local-slice-index dataset.

Example commands:

```bash
python -m modules.step1_skull_stripping batch \
  --src-dir /Volumes/X31/org/FLAIR_nii \
  --dst-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --platform linux/amd64

python -m modules.step2_bias_field_correction batch \
  --src-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --dst-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --include-keyword _FLAIR_brain.nii.gz

python -m modules.step3_masking batch \
  --src-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --include-keyword _FLAIR_brain_bfCorrected.nii.gz

python -m modules.step4_reorientation batch \
  --src-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --dst-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --target-orientation RAS \
  --include-keyword _FLAIR_brain_bfCorrected_masked.nii.gz

python -m modules.step5_resize_normalization batch \
  --src-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --dst-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --label-xlsx /Users/yschoi/code/09_FHV/modules_v02/df_flair_label_UHKinspection.xlsx \
  --label-column fhv_slice_num \
  --preprocess-mode pad \
  --target-size 672 \
  --include-keyword _FLAIR_brain_bfCorrected_masked_reorient.nii.gz

python -m modules.step6_create_filtered_h5
```

Example: in-plane resize to `512 x 512` NPY instead of legacy padding:

```bash
python -m modules.step5_resize_normalization batch \
  --src-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --dst-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --label-xlsx /Users/yschoi/code/09_FHV/modules_v02/df_flair_label_UHKinspection.xlsx \
  --label-column fhv_slice_num \
  --preprocess-mode resize \
  --target-size 512 \
  --include-keyword _FLAIR_brain_bfCorrected_masked_reorient.nii.gz
```

Example: reproduce legacy local slice 12-15 H5 selection:

```bash
python -m modules.step6_create_filtered_h5 \
  --selection-mode local \
  --min-index 12 \
  --max-index 15
```

Run steps 1-5 as one pipeline:

```bash
python -m modules.step1_to_step5_pipeline batch \
  --src-dir /Volumes/X31/org/FLAIR_nii \
  --dst-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --label-xlsx /Users/yschoi/code/09_FHV/modules_v02/df_flair_label_UHKinspection.xlsx \
  --label-column fhv_slice_num \
  --step4-target-orientation RAS \
  --step5-preprocess-mode pad \
  --step5-target-size 672 \
  --platform linux/amd64
```

Example: run Step 1-5 with LAS orientation and `512 x 512` in-plane resizing:

```bash
python -m modules.step1_to_step5_pipeline single \
  --input-path /Volumes/GURI_1/FLAIR_PAIK_nii/BP00001_P14147_FLAIR.nii \
  --dst-dir /Volumes/GURI_1/FLAIR_PAIK_nii_preproc \
  --label-xlsx /Users/yschoi/code/09_FHV/csv_files/df_flair_label_UHKinspection.xlsx \
  --label-column fhv_slice_num \
  --step4-target-orientation LAS \
  --step5-preprocess-mode resize \
  --step5-target-size 512 \
  --platform linux/amd64
```

Run one PID:

```bash
python -m modules.step1_to_step5_pipeline batch \
  --pid P00006 \
  --src-dir /Volumes/X31/org/FLAIR_nii \
  --dst-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --platform linux/amd64

python -m modules.step1_to_step5_pipeline single \
  --input-path /Volumes/X31/org/FLAIR_nii/P00006/P00006_FLAIR.nii \
  --dst-dir /Volumes/X31/FLAIR_HVS_detection_nii_preproc \
  --platform linux/amd64
```

QC outputs:

- Step 1 writes `qc_skullstrip/*_skullstrip_qc.png` per subject.
- Step 2 writes `*_bias_field_correction_QC/` with bias maps, plots, and `summary.txt`.
- Step 3 writes `qc_masking/*_masking_qc.png` per subject.
- Step 4 writes `qc_reorientation/*_reorientation_qc.json` and `.png` per subject.
- Step 5 writes `qc_pad/*.png` or `qc_resize/*.png`, per-subject `*_meta.json`,
  and positive-label slice QC images under `qc_pad_positive/` or
  `qc_resize_positive/` when labels are available.
- Step 6 writes one HDF5 file with `images`, `labels`, `subject_ids`,
  `local_slice_index`, `local_slice_index_rel`, and `boundary_flag`.
