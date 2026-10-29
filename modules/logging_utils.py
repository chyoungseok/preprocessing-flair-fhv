"""Shared logging path helpers for preprocessing CLIs."""

from pathlib import Path


def log_dir_for_root(root_path) -> Path:
    """Return a logs directory tied to the mounted volume of ``root_path``.

    Examples
    --------
    /Volumes/GURI_2/FLAIR_nii -> /Volumes/GURI_2/logs
    /Volumes/X31/FLAIR_nii    -> /Volumes/X31/logs
    """
    path = Path(root_path).expanduser()
    parts = path.parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return Path("/Volumes") / parts[2] / "logs"
    return path.parent / "logs"


def resolve_log_file(log_file, default_log_file, root_path) -> Path:
    """Use a volume-local default log path unless the user supplied one."""
    if log_file is not None and Path(log_file) != Path(default_log_file):
        return Path(log_file).expanduser()
    return log_dir_for_root(root_path) / Path(default_log_file).name


def append_error_log(log_file, input_path, output_path, error) -> Path:
    """Append one error block, falling back to ./logs if the target is unwritable."""
    log_path = Path(log_file).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        log_path = Path.cwd() / "logs" / log_path.name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[WARN] log path is not writable; using fallback log: {log_path}")

    with open(log_path, "a") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Input : {input_path}\n")
        f.write(f"Output: {output_path}\n")
        f.write(f"Error : {error}\n\n")
    return log_path
