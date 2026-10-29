"""Compare two HDF5 files at schema and dataset-value level.

Examples
--------
python -m modules.compare_h5_files
python -m modules.compare_h5_files --left /path/old.h5 --right /path/new.h5
python -m modules.compare_h5_files --left /path/old.h5 --right /path/new.h5 --rtol 1e-5 --atol 1e-8
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_LEFT_H5 = (
    "/Users/yschoi/data/FLAIR/flair_preproc_v02_slices_whole_h5_uhkInspection/"
    "flair_slice_dataset_adjacent_1215.h5"
)
DEFAULT_RIGHT_H5 = (
    "/Volumes/X31/FLAIR_HVS_detection_nii_preproc_h5/"
    "flair_slice_dataset_adjacent_1215.h5"
)


def _decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return np.char.decode(value, "utf-8")
    return value


def _attrs_equal(left_attrs, right_attrs) -> tuple[bool, list[str]]:
    messages = []
    left_keys = set(left_attrs.keys())
    right_keys = set(right_attrs.keys())

    if left_keys != right_keys:
        messages.append(f"attribute keys differ: left_only={sorted(left_keys - right_keys)}, right_only={sorted(right_keys - left_keys)}")
        return False, messages

    ok = True
    for key in sorted(left_keys):
        left_value = _decode_if_bytes(left_attrs[key])
        right_value = _decode_if_bytes(right_attrs[key])
        if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
            same = np.array_equal(left_value, right_value)
        else:
            same = left_value == right_value
        if not bool(same):
            ok = False
            messages.append(f"attribute differs: {key}: {left_value!r} != {right_value!r}")
    return ok, messages


def _dataset_values_equal(
    left_dset,
    right_dset,
    chunk_rows: int,
    rtol: float,
    atol: float,
) -> tuple[bool, str]:
    n_rows = left_dset.shape[0] if left_dset.shape else 1
    numeric = np.issubdtype(left_dset.dtype, np.number) and np.issubdtype(right_dset.dtype, np.number)

    if left_dset.shape == ():
        left_value = left_dset[()]
        right_value = right_dset[()]
        if numeric and (rtol > 0 or atol > 0):
            same = np.allclose(left_value, right_value, rtol=rtol, atol=atol, equal_nan=True)
        else:
            same = np.array_equal(left_value, right_value)
        return bool(same), "" if same else "scalar value differs"

    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        left_arr = left_dset[start:stop]
        right_arr = right_dset[start:stop]

        if numeric and (rtol > 0 or atol > 0):
            same = np.allclose(left_arr, right_arr, rtol=rtol, atol=atol, equal_nan=True)
        else:
            same = np.array_equal(left_arr, right_arr)

        if not bool(same):
            return False, f"values differ in rows [{start}:{stop}]"

    return True, ""


def compare_h5_files(
    left_h5: str | Path = DEFAULT_LEFT_H5,
    right_h5: str | Path = DEFAULT_RIGHT_H5,
    chunk_rows: int = 32,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> bool:
    """Return True when both HDF5 files have identical keys, attrs, schema, and values."""
    left_h5 = Path(left_h5)
    right_h5 = Path(right_h5)
    if not left_h5.exists():
        raise FileNotFoundError(f"Left H5 not found: {left_h5}")
    if not right_h5.exists():
        raise FileNotFoundError(f"Right H5 not found: {right_h5}")

    all_ok = True
    with h5py.File(left_h5, "r") as left, h5py.File(right_h5, "r") as right:
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        print(f"[INFO] Left keys : {sorted(left_keys)}")
        print(f"[INFO] Right keys: {sorted(right_keys)}")

        if left_keys != right_keys:
            all_ok = False
            print(f"[FAIL] key mismatch: left_only={sorted(left_keys - right_keys)}, right_only={sorted(right_keys - left_keys)}")

        attrs_ok, attr_messages = _attrs_equal(left.attrs, right.attrs)
        if attrs_ok:
            print("[OK] file attributes match")
        else:
            all_ok = False
            print("[FAIL] file attributes differ")
            for message in attr_messages:
                print(f"       {message}")

        for key in sorted(left_keys & right_keys):
            left_obj = left[key]
            right_obj = right[key]

            if not isinstance(left_obj, h5py.Dataset) or not isinstance(right_obj, h5py.Dataset):
                same_type = type(left_obj) is type(right_obj)
                all_ok &= same_type
                print(f"[{'OK' if same_type else 'FAIL'}] {key}: object type {type(left_obj).__name__} vs {type(right_obj).__name__}")
                continue

            schema_ok = left_obj.shape == right_obj.shape and left_obj.dtype == right_obj.dtype
            print(
                f"[{'OK' if schema_ok else 'FAIL'}] {key}: "
                f"left shape={left_obj.shape}, dtype={left_obj.dtype}; "
                f"right shape={right_obj.shape}, dtype={right_obj.dtype}"
            )
            if not schema_ok:
                all_ok = False
                continue

            values_ok, message = _dataset_values_equal(
                left_obj,
                right_obj,
                chunk_rows=chunk_rows,
                rtol=rtol,
                atol=atol,
            )
            if values_ok:
                print(f"[OK] {key}: values match")
            else:
                all_ok = False
                print(f"[FAIL] {key}: {message}")

    print("[RESULT] H5 files match" if all_ok else "[RESULT] H5 files differ")
    return all_ok


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two HDF5 files.")
    parser.add_argument("--left", default=DEFAULT_LEFT_H5, help="Original/reference H5 path.")
    parser.add_argument("--right", default=DEFAULT_RIGHT_H5, help="New H5 path.")
    parser.add_argument("--chunk-rows", type=int, default=32)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    ok = compare_h5_files(
        left_h5=args.left,
        right_h5=args.right,
        chunk_rows=args.chunk_rows,
        rtol=args.rtol,
        atol=args.atol,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
