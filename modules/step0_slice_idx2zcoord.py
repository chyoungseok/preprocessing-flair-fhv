"""Step 00: convert original slice-level labels to physical z-coordinates.

Run this before RAS reorientation if slice-level labels were annotated on the
original image orientation. The generated ``fhv_z_center`` column is later used
to recover anatomically consistent slice indices after reorientation.

Example
-------
python -m modules.step0_slice_idx2zcoord \
    --label-xlsx modules_v02/df_flair_label_UHKinspection.xlsx \
    --out-xlsx modules_v02/df_flair_label_UHKinspection_zcoord.xlsx \
    --nii-root /Volumes/X31/FLAIR_HVS_detection_nii_preproc
"""

import argparse
import ast
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def parse_list_cell(cell):
    """Parse FHV slice labels from Excel cells into a list of integer indices."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, list):
        return [int(x) for x in cell]
    s = str(cell).strip()
    if s == "" or s.lower() == "nan":
        return []
    try:
        value = ast.literal_eval(s)
        if isinstance(value, (list, tuple)):
            return [int(x) for x in value]
        return [int(value)]
    except Exception:
        out = []
        for part in s.strip("[]").split(","):
            part = part.strip()
            if part:
                try:
                    out.append(int(part))
                except ValueError:
                    pass
        return out


def parse_fhv_slice_label(cell, missing_value=np.nan):
    """Parse original FHV slice labels such as '13', '13,15', or '13-15'.

    Use this for raw annotation columns before building ``fhv_slice_num``.
    Missing values stay as NaN so positive/negative rows remain easy to filter.
    """
    if cell is None or pd.isna(cell):
        return missing_value
    if isinstance(cell, list):
        return [int(x) for x in cell]

    out = []
    for part in str(cell).strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(float(x.strip())) for x in part.split("-", 1)]
            out.extend(range(start, end + 1))
        else:
            out.append(int(float(part)))
    return out if out else missing_value


def resolve_nii_path(row, nii_root=None, fname_pattern=None):
    """Resolve the source NIfTI path for one label row."""
    if "path_nii" in row and pd.notna(row["path_nii"]):
        path = str(row["path_nii"])
        if os.path.exists(path):
            return path

    if nii_root and fname_pattern and pd.notna(row.get("id_subject")) and pd.notna(row.get("id_session")):
        try:
            fname = fname_pattern.format(id_subject=row["id_subject"], id_session=row["id_session"])
            path = os.path.join(nii_root, fname)
            if os.path.exists(path):
                return path
        except Exception:
            pass

    if nii_root:
        subj = str(row.get("id_subject", "")).strip()
        sess = str(row.get("id_session", "")).strip()
        patterns = [
            f"**/*{subj}*{sess}*FLAIR*.nii*",
            f"**/*{subj}*{sess}*.nii*",
            f"**/*{sess}*FLAIR*.nii*",
        ]
        candidates = []
        for pattern in patterns:
            candidates += glob.glob(os.path.join(nii_root, pattern), recursive=True)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return sorted(candidates, key=len)[0]
    return None


def voxel_spacings_mm(affine):
    """Return voxel spacings from an affine matrix."""
    axes = affine[:3, :3]
    return np.sqrt((axes * axes).sum(axis=0))


def slice_centers_world_z(nx, ny, nz, affine):
    """Return world z-coordinate at the in-plane center for every slice."""
    i_mid = (nx - 1) / 2.0
    j_mid = (ny - 1) / 2.0
    ks = np.arange(nz, dtype=float)
    ijk = np.stack([np.full_like(ks, i_mid), np.full_like(ks, j_mid), ks], axis=1)
    ijk1 = np.hstack([ijk, np.ones((nz, 1))])
    xyz = ijk1 @ affine.T
    return xyz[:, 2]


def compute_center_slab_for_row(row, slice_num_one_based=True, round_digits=3, nii_root=None, fname_pattern=None):
    """Compute physical z-center and z-slab values for one label row."""
    import nibabel as nib

    idx_list = parse_list_cell(row.get("fhv_slice_num"))
    idx0 = [idx - 1 for idx in idx_list] if slice_num_one_based else idx_list[:]

    nii_path = resolve_nii_path(row, nii_root=nii_root, fname_pattern=fname_pattern)
    if not nii_path or not os.path.exists(nii_path) or len(idx0) == 0:
        return pd.Series({"fhv_z_center": [], "fhv_z_slab": [], "path_nii_resolved": nii_path or ""})

    img = nib.load(nii_path)
    nx, ny, nz = img.shape[:3]
    affine = img.affine
    zspacing = float(voxel_spacings_mm(affine)[2])
    zc_all = slice_centers_world_z(nx, ny, nz, affine)
    half = zspacing / 2.0

    centers = []
    slabs = []
    for idx in idx0:
        if 0 <= idx < nz:
            zc = float(zc_all[int(idx)])
            centers.append(round(zc, round_digits))
            slabs.append([round(zc - half, round_digits), round(zc + half, round_digits)])

    return pd.Series({"fhv_z_center": centers, "fhv_z_slab": slabs, "path_nii_resolved": nii_path})


def filter_label_df_by_pid(df: pd.DataFrame, pid: str):
    """Filter a label table to one PID using id_session or id_subject."""
    matches = []
    for col in ("id_session", "id_subject"):
        if col in df.columns:
            con = df[col].astype(str) == str(pid)
            if con.any():
                matches.append((col, con))
    if not matches:
        raise ValueError(f"No label row found for pid={pid}")
    if len(matches) > 1:
        counts = {col: int(con.sum()) for col, con in matches}
        raise ValueError(f"Ambiguous PID match for pid={pid}: {counts}")
    return df.loc[matches[0][1]].copy()


def write_qc_summary(df_z: pd.DataFrame, out_xlsx, qc_json=None):
    """Write a compact QC summary for z-coordinate label conversion."""
    if qc_json is None:
        qc_json = str(Path(out_xlsx).with_suffix("")) + "_qc.json"
    summary = {
        "output_xlsx": str(out_xlsx),
        "n_rows": int(len(df_z)),
        "n_path_resolved": int(df_z["path_nii_resolved"].astype(bool).sum()) if "path_nii_resolved" in df_z else 0,
        "n_with_fhv_z_center": int(df_z["fhv_z_center"].apply(lambda x: len(x) > 0 if isinstance(x, list) else str(x) not in ("", "[]", "nan")).sum()) if "fhv_z_center" in df_z else 0,
    }
    qc_json = Path(qc_json)
    qc_json.parent.mkdir(parents=True, exist_ok=True)
    with open(qc_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[QC] saved: {qc_json}")


def build_arg_parser():
    """Build CLI parser for converting slice labels to z-coordinates."""
    parser = argparse.ArgumentParser(
        description="Add fhv_z_center/fhv_z_slab columns to a label Excel file."
    )
    parser.add_argument("--label-xlsx", required=True, help="Input label Excel file.")
    parser.add_argument("--out-xlsx", required=True, help="Output Excel file.")
    parser.add_argument("--nii-root", required=True, help="Root directory for original NIfTI files.")
    parser.add_argument("--pid", default=None, help="Process only one PID; default processes all rows.")
    parser.add_argument("--qc-json", default=None, help="Optional QC summary JSON path.")
    parser.add_argument(
        "--fname-pattern",
        default=None,
        help="Optional pattern, e.g. '{id_session}/{id_session}_FLAIR.nii'.",
    )
    parser.add_argument(
        "--zero-based",
        action="store_true",
        help="Use if fhv_slice_num is already 0-based. Default assumes 1-based labels.",
    )
    return parser


def cli_main(argv=None):
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)

    df = pd.read_excel(args.label_xlsx, index_col=0)
    if args.pid is not None:
        df = filter_label_df_by_pid(df, args.pid)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rows.append(
            compute_center_slab_for_row(
                row,
                slice_num_one_based=not args.zero_based,
                nii_root=args.nii_root,
                fname_pattern=args.fname_pattern,
            )
        )

    df_z = pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)
    df_z.to_excel(args.out_xlsx)
    print(f"[OK] saved: {args.out_xlsx}")
    write_qc_summary(df_z, out_xlsx=args.out_xlsx, qc_json=args.qc_json)
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
