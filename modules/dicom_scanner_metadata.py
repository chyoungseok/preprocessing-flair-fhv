"""Helpers for reading subject-level scanner manufacturer metadata from DICOM.

The H5 pipeline only needs a subject-level manufacturer label, so this module
reads the first usable DICOM file for each PID instead of scanning every slice.
"""

from __future__ import annotations

import struct
from pathlib import Path


UNKNOWN_MANUFACTURER = "UNKNOWN"
MANUFACTURER_NORMALIZATION = "Philips|GE|Siemens|UNKNOWN"
EXPLICIT_VR_32BIT = {"OB", "OD", "OF", "OL", "OW", "SQ", "UC", "UR", "UT", "UN"}
TRANSFER_SYNTAX_UID_TAG = (0x0002, 0x0010)
MANUFACTURER_TAG = (0x0008, 0x0070)
PIXEL_DATA_TAG = (0x7FE0, 0x0010)


def normalize_manufacturer(raw: str | None) -> str:
    """Normalize raw DICOM Manufacturer values to the scanner labels used in H5."""
    if not raw:
        return UNKNOWN_MANUFACTURER

    text = raw.strip().replace("\x00", "")
    lowered = text.lower()
    if "philips" in lowered:
        return "Philips"
    if "siemens" in lowered:
        return "Siemens"
    if lowered == "ge" or "ge medical" in lowered or "general electric" in lowered:
        return "GE"
    return UNKNOWN_MANUFACTURER


def find_pid_dicom_dir(dicom_root: str | Path, pid: str) -> Path | None:
    """Find the DICOM directory for a PID under the expected ``group/PID`` layout."""
    root = Path(dicom_root)
    if not root.exists():
        return None

    direct_pid_dir = root / pid
    if direct_pid_dir.is_dir():
        return direct_pid_dir

    direct_matches = sorted(p for p in root.glob(f"*/{pid}") if p.is_dir())
    if direct_matches:
        return direct_matches[0]

    return None


def _clean_text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()


def _read_element(data: bytes, pos: int, explicit_vr: bool) -> tuple[tuple[int, int], bytes, int] | None:
    if pos + 8 > len(data):
        return None

    group, elem = struct.unpack("<HH", data[pos : pos + 4])
    if explicit_vr:
        if pos + 8 > len(data):
            return None
        vr = data[pos + 4 : pos + 6].decode("ascii", errors="ignore")
        if vr in EXPLICIT_VR_32BIT:
            if pos + 12 > len(data):
                return None
            length = struct.unpack("<I", data[pos + 8 : pos + 12])[0]
            value_pos = pos + 12
        else:
            length = struct.unpack("<H", data[pos + 6 : pos + 8])[0]
            value_pos = pos + 8
    else:
        length = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        value_pos = pos + 8

    if length == 0xFFFFFFFF or value_pos + length > len(data):
        return None

    next_pos = value_pos + length + (length % 2)
    return (group, elem), data[value_pos : value_pos + length], next_pos


def read_manufacturer_from_dicom(dcm_path: str | Path) -> str | None:
    """Read raw Manufacturer from one DICOM file using a small header parser."""
    data = Path(dcm_path).read_bytes()
    pos = 132 if len(data) >= 132 and data[128:132] == b"DICM" else 0
    transfer_syntax = "1.2.840.10008.1.2.1"
    has_meta = pos == 132

    while has_meta and pos + 8 <= len(data):
        element = _read_element(data, pos, explicit_vr=True)
        if element is None:
            break
        tag, value, next_pos = element
        if tag[0] != 0x0002:
            break
        if tag == TRANSFER_SYNTAX_UID_TAG:
            transfer_syntax = _clean_text(value)
        pos = next_pos

    explicit_vr = has_meta and transfer_syntax != "1.2.840.10008.1.2"
    while pos + 8 <= len(data):
        tag_preview = struct.unpack("<HH", data[pos : pos + 4])
        if tag_preview == PIXEL_DATA_TAG or tag_preview[0] == 0xFFFE:
            break

        element = _read_element(data, pos, explicit_vr=explicit_vr)
        if element is None:
            break
        tag, value, pos = element
        if tag == MANUFACTURER_TAG:
            return _clean_text(value)

    return None


def get_subject_manufacturer(
    dicom_root: str | Path,
    pid: str,
    strict: bool = False,
) -> str:
    """Return normalized subject-level manufacturer by reading one DICOM file."""
    dicom_dir = find_pid_dicom_dir(dicom_root, pid)
    if dicom_dir is None:
        if strict:
            raise FileNotFoundError(f"DICOM folder not found for {pid} under {dicom_root}")
        print(f"[WARN] DICOM folder not found for {pid}; scanner_manufacturer=UNKNOWN")
        return UNKNOWN_MANUFACTURER

    dcm_path = next(
        (
            p
            for p in sorted(dicom_dir.rglob("*.dcm"))
            if p.is_file() and not p.name.startswith(".")
        ),
        None,
    )
    if dcm_path is None:
        if strict:
            raise FileNotFoundError(f"No .dcm files found for {pid}: {dicom_dir}")
        print(f"[WARN] No .dcm files found for {pid}; scanner_manufacturer=UNKNOWN")
        return UNKNOWN_MANUFACTURER

    try:
        raw = read_manufacturer_from_dicom(dcm_path)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Failed to read DICOM manufacturer for {pid}: {dcm_path}") from exc
        print(f"[WARN] Failed to read DICOM manufacturer for {pid}: {exc}; scanner_manufacturer=UNKNOWN")
        return UNKNOWN_MANUFACTURER

    normalized = normalize_manufacturer(raw)
    if normalized == UNKNOWN_MANUFACTURER:
        message = f"Unknown DICOM manufacturer for {pid}: {raw!r}"
        if strict:
            raise ValueError(message)
        print(f"[WARN] {message}; scanner_manufacturer=UNKNOWN")
    return normalized
