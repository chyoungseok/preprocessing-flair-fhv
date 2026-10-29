"""Slice-level FHV label table (Excel/CSV) keyed by case ID.

``case_id`` is ``<id_subject>_<id_scan>`` (or ``id_session``) when the table has
a scan column, otherwise ``id_subject``. ``fhv_slice_num`` holds 1-based slice
numbers such as ``[13, 14, 15]``; an empty cell means "labelled, no FHV slice".
Cases missing from the table are unlabelled, not negative.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LABEL_COLUMN = "fhv_slice_num"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SCAN_COLUMNS = ("id_scan", "id_session")


@dataclass(frozen=True)
class CaseLabel:
    case_id: str
    table_row: int
    source_value: str
    positive_slices_1based: tuple[int, ...]


def _clean_id(value: object, column: str, table_row: int) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not SAFE_ID.fullmatch(text):
        raise ValueError(f"label row {table_row}: invalid {column} value {value!r}")
    return text


def _positive_int(value: object, table_row: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"label row {table_row}: boolean is not a slice number")
    if isinstance(value, (float, np.floating)) and (not math.isfinite(value) or not float(value).is_integer()):
        raise ValueError(f"label row {table_row}: non-integer slice number {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"label row {table_row}: invalid slice number {value!r}") from exc
    if number <= 0:
        raise ValueError(f"label row {table_row}: slice numbers are 1-based, got {number}")
    return number


def parse_slice_numbers(value: object, table_row: int) -> tuple[int, ...]:
    """Parse a cell such as ``[13, 14, 15]``, ``13`` or ``13, 14`` into sorted unique 1-based indices."""
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "nan", "none", "null", "[]"}:
            return ()
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            value = re.findall(r"[+-]?\d+", text)
            if not value:
                raise ValueError(f"label row {table_row}: cannot parse slice list {text!r}")
    elif pd.isna(value):
        return ()
    items = value if isinstance(value, (list, tuple, set)) else [value]
    numbers = [_positive_int(item, table_row) for item in items]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"label row {table_row}: duplicate slice numbers {value!r}")
    return tuple(sorted(numbers))


def load_label_table(path: str | Path, label_column: str = DEFAULT_LABEL_COLUMN) -> dict[str, CaseLabel]:
    """Return ``{case_id: CaseLabel}``; duplicate case IDs are an error."""
    path = Path(path).expanduser()
    df = pd.read_csv(path, dtype=str) if path.suffix.lower() == ".csv" else pd.read_excel(path, dtype=str)
    if "id_subject" not in df.columns or label_column not in df.columns:
        raise KeyError(f"label table needs 'id_subject' and '{label_column}' columns: {path}")
    scan_column = next((c for c in SCAN_COLUMNS if c in df.columns), None)

    labels: dict[str, CaseLabel] = {}
    for index, row in df.iterrows():
        table_row = int(index) + 2  # header is row 1
        case_id = _clean_id(row["id_subject"], "id_subject", table_row)
        if scan_column is not None:
            case_id = f"{case_id}_{_clean_id(row[scan_column], scan_column, table_row)}"
        if case_id in labels:
            raise ValueError(f"label row {table_row}: duplicate case {case_id}")
        raw = row[label_column]
        labels[case_id] = CaseLabel(
            case_id=case_id,
            table_row=table_row,
            source_value="" if pd.isna(raw) else str(raw),
            positive_slices_1based=parse_slice_numbers(raw, table_row),
        )
    return labels
