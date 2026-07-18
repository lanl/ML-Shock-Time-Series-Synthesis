"""Download and load the public SRS benchmark datasets.

The default dataset release is hosted at:
https://github.com/lanl/ML-Shock-Time-Series-Synthesis/releases/tag/datasets-v1.0.0

Each public NPZ release asset is a ZIP archive such as
``dataset-a-npz-v1.0.0.zip``. This module downloads the archive, extracts the
corresponding ``Dataset_A.npz`` file and JSON metadata, and caches the result.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import numpy as np
from numpy.lib.npyio import NpzFile

# ----------------------------
# GitHub dataset release config
# ----------------------------
OWNER = os.environ.get("GEN_SRS_RELEASE_OWNER", "lanl")
REPO = os.environ.get(
    "GEN_SRS_RELEASE_REPO",
    "ML-Shock-Time-Series-Synthesis",
)
TAG = os.environ.get("GEN_SRS_RELEASE_TAG", "datasets-v1.0.0")
DATASET_VERSION = os.environ.get("GEN_SRS_DATASET_VERSION", "v1.0.0")

DATASETS = ("A", "B", "C", "D")
USER_AGENT = "gen-srs-public-dataset-downloader"

# ----------------------------
# Local dataset cache
# ----------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def default_cache_data_dir() -> Path:
    """Return the platform-appropriate user cache directory."""
    if os.name == "nt":
        base = Path(
            os.environ.get(
                "LOCALAPPDATA",
                Path.home() / "AppData" / "Local",
            )
        )
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    return base / "gen_srs" / "Datasets"


def normalize_dataset_id(ds: str) -> str:
    """Normalize and validate a dataset identifier."""
    normalized = str(ds).strip().upper()
    if normalized not in DATASETS:
        raise ValueError(f"ds must be one of {DATASETS}; got {ds!r}")
    return normalized


def dataset_file_name(ds: str) -> str:
    """Return the canonical extracted NPZ filename."""
    return f"Dataset_{normalize_dataset_id(ds)}.npz"


def dataset_archive_name(ds: str) -> str:
    """Return the GitHub release asset filename for a dataset's NPZ ZIP.

    The default naming convention is ``dataset-a-npz-v1.0.0.zip``. A specific
    asset can be overridden with an environment variable such as
    ``GEN_SRS_DATASET_A_ASSET``.
    """
    ds = normalize_dataset_id(ds)
    override = os.environ.get(f"GEN_SRS_DATASET_{ds}_ASSET")
    if override:
        return override
    return f"dataset-{ds.lower()}-npz-{DATASET_VERSION}.zip"


def _is_repo_root(path: Path) -> bool:
    return (path / ".git").exists() and (path / "gen_srs_public").exists()


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen: set[Path] = set()

    for path in paths:
        resolved = path.expanduser().resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)

    return unique_paths


def candidate_data_dirs(data_dir: str | Path | None = None) -> list[Path]:
    """Return directories searched for extracted dataset files."""
    candidates: list[Path] = []

    if data_dir is not None:
        path = Path(data_dir).expanduser()
        if path.suffix.lower() not in {".npz", ".zip"}:
            candidates.append(path)
    else:
        override = os.environ.get("GEN_SRS_DATA_DIR")
        if override:
            path = Path(override).expanduser()
            if path.suffix.lower() not in {".npz", ".zip"}:
                candidates.append(path)

    # Use the repository-level Datasets directory only for a real checkout.
    # Installed packages should default to the user-writable cache instead of
    # attempting to write into site-packages.
    if _is_repo_root(REPO_ROOT):
        candidates.append(REPO_ROOT / "Datasets")

    # Support notebooks launched from the repository root or one directory
    # below it, even when the imported package resolves elsewhere.
    cwd = Path.cwd().resolve()
    if _is_repo_root(cwd):
        candidates.append(cwd / "Datasets")
    if _is_repo_root(cwd.parent):
        candidates.append(cwd.parent / "Datasets")

    candidates.append(default_cache_data_dir())
    return _deduplicate_paths(candidates)


def candidate_dataset_paths(
    ds: str,
    data_dir: str | Path | None = None,
) -> list[Path]:
    """Return candidate locations for an extracted NPZ file."""
    filename = dataset_file_name(ds)
    candidates: list[Path] = []

    if data_dir is not None:
        path = Path(data_dir).expanduser()
        if path.suffix.lower() == ".npz":
            candidates.append(path)
        elif path.suffix.lower() != ".zip":
            candidates.append(path / filename)
    else:
        override = os.environ.get("GEN_SRS_DATA_DIR")
        if override:
            path = Path(override).expanduser()
            if path.suffix.lower() == ".npz":
                candidates.append(path)

    for directory in candidate_data_dirs(data_dir=data_dir):
        candidates.append(directory / filename)

    return _deduplicate_paths(candidates)


def candidate_archive_paths(
    ds: str,
    data_dir: str | Path | None = None,
) -> list[Path]:
    """Return candidate locations for a previously downloaded ZIP archive."""
    archive_name = dataset_archive_name(ds)
    candidates: list[Path] = []

    if data_dir is not None:
        path = Path(data_dir).expanduser()
        if path.suffix.lower() == ".zip":
            candidates.append(path)
        elif path.suffix.lower() != ".npz":
            candidates.append(path / archive_name)
    else:
        override = os.environ.get("GEN_SRS_DATA_DIR")
        if override:
            path = Path(override).expanduser()
            if path.suffix.lower() == ".zip":
                candidates.append(path)

    for directory in candidate_data_dirs(data_dir=data_dir):
        candidates.append(directory / archive_name)
        candidates.append(directory / ".archives" / archive_name)

    return _deduplicate_paths(candidates)


def get_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return the directory where a downloaded dataset should be extracted."""
    if data_dir is not None:
        path = Path(data_dir).expanduser()
        if path.suffix.lower() in {".npz", ".zip"}:
            return path.parent.resolve(strict=False)
        return path.resolve(strict=False)

    override = os.environ.get("GEN_SRS_DATA_DIR")
    if override:
        path = Path(override).expanduser()
        if path.suffix.lower() in {".npz", ".zip"}:
            return path.parent.resolve(strict=False)
        return path.resolve(strict=False)

    return candidate_data_dirs()[0]


def find_existing_dataset(
    ds: str,
    data_dir: str | Path | None = None,
) -> Path | None:
    """Return the first existing local NPZ path, if available."""
    for candidate in candidate_dataset_paths(ds, data_dir=data_dir):
        if candidate.is_file():
            return candidate
    return None


DATA_DIR = get_data_dir()


def validate_release_config(owner: str, repo: str, tag: str) -> None:
    """Validate the GitHub release coordinates."""
    if not owner or not repo or not tag:
        raise RuntimeError(
            "Release asset configuration requires non-empty owner, repo, "
            "and tag values."
        )


def release_asset_url(asset_name: str) -> str:
    """Return the public browser-download URL for a release asset."""
    validate_release_config(OWNER, REPO, TAG)
    owner = urllib.parse.quote(OWNER, safe="")
    repo = urllib.parse.quote(REPO, safe="")
    tag = urllib.parse.quote(TAG, safe="")
    asset = urllib.parse.quote(asset_name, safe="")
    return f"https://github.com/{owner}/{repo}/releases/download/{tag}/{asset}"


def _download_to_path(url: str, dst: Path) -> None:
    """Download ``url`` atomically to ``dst``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part_path = dst.with_name(f"{dst.name}.part")
    part_path.unlink(missing_ok=True)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )

    try:
        with urllib.request.urlopen(request) as response, part_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        part_path.replace(dst)
    except urllib.error.HTTPError as exc:
        part_path.unlink(missing_ok=True)
        if exc.code == 404:
            raise RuntimeError(
                "GitHub release asset was not found. Confirm that the release "
                f"tag and asset filename are correct:\n  {url}"
            ) from exc
        raise
    except Exception:
        part_path.unlink(missing_ok=True)
        raise


def download_release_archive(ds: str, dst: Path) -> Path:
    """Download a dataset's public NPZ ZIP release asset."""
    archive_name = dataset_archive_name(ds)
    url = release_asset_url(archive_name)

    if dst.is_file() and zipfile.is_zipfile(dst):
        return dst
    if dst.exists():
        dst.unlink()

    print(f"Downloading {archive_name} ...")
    _download_to_path(url, dst)

    if not zipfile.is_zipfile(dst):
        dst.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded asset is not a valid ZIP file: {archive_name}"
        )

    print(f"Saved archive to {dst}")
    return dst


def _select_npz_member(archive: zipfile.ZipFile, ds: str) -> zipfile.ZipInfo:
    """Select the intended NPZ member from a release archive."""
    expected = dataset_file_name(ds).casefold()
    members = [
        member
        for member in archive.infolist()
        if not member.is_dir() and member.filename.lower().endswith(".npz")
    ]

    exact = [
        member
        for member in members
        if Path(member.filename).name.casefold() == expected
    ]
    if len(exact) == 1:
        return exact[0]
    if len(members) == 1:
        return members[0]

    names = "\n".join(f"  - {member.filename}" for member in members) or "  - none"
    raise RuntimeError(
        f"The archive for Dataset {normalize_dataset_id(ds)} must contain "
        f"exactly one NPZ file or a file named {dataset_file_name(ds)!r}.\n"
        f"NPZ members:\n{names}"
    )


def _copy_zip_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    dst: Path,
) -> None:
    """Copy one ZIP member atomically without trusting its stored path."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part_path = dst.with_name(f"{dst.name}.part")
    part_path.unlink(missing_ok=True)

    try:
        with archive.open(member) as source, part_path.open("wb") as output:
            shutil.copyfileobj(source, output)
        part_path.replace(dst)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise


def _validate_npz(path: Path) -> None:
    """Validate the arrays required by the public dataset loader."""
    required = {"ts", "t", "dataset", "t_units", "data_units", "sample_rate"}

    try:
        with np.load(path, allow_pickle=False) as dataset:
            missing = required.difference(dataset.files)
    except Exception as exc:
        raise RuntimeError(f"Could not open extracted dataset {path}: {exc}") from exc

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Extracted {path.name} is missing required arrays: {missing_text}"
        )


def extract_dataset_archive(ds: str, archive_path: Path, dst: Path) -> Path:
    """Extract the NPZ and accompanying JSON metadata from a release ZIP."""
    ds = normalize_dataset_id(ds)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            npz_member = _select_npz_member(archive, ds)
            _copy_zip_member(archive, npz_member, dst)

            # Copy accompanying JSON metadata using only its basename. This
            # also works when the ZIP stores files inside a top-level folder.
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".json"):
                    continue
                metadata_dst = dst.parent / Path(member.filename).name
                _copy_zip_member(archive, member, metadata_dst)

        _validate_npz(dst)
    except Exception:
        dst.unlink(missing_ok=True)
        raise

    print(f"Extracted dataset to {dst}")
    return dst


def dataset_resolution_error(
    ds: str,
    data_dir: str | Path | None = None,
    reason: str | None = None,
) -> RuntimeError:
    """Build a detailed dataset resolution error."""
    filename = dataset_file_name(ds)
    candidates = candidate_dataset_paths(ds, data_dir=data_dir)
    search_list = "\n".join(f"  - {candidate}" for candidate in candidates)
    message = (
        f"Could not locate {filename} locally.\n"
        f"Searched:\n{search_list}\n"
        "Place the NPZ or its release ZIP in one of those locations, pass "
        "data_dir=..., or set GEN_SRS_DATA_DIR."
    )
    if reason:
        message = f"{message}\n{reason}"
    return RuntimeError(message)


def _keep_downloaded_archives() -> bool:
    value = os.environ.get("GEN_SRS_KEEP_ARCHIVES", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def ensure_dataset(
    ds: str,
    data_dir: str | Path | None = None,
) -> Path:
    """Return a local NPZ path, downloading and extracting it if necessary."""
    ds = normalize_dataset_id(ds)

    existing = find_existing_dataset(ds, data_dir=data_dir)
    if existing is not None:
        return existing

    # Support a release ZIP that the user downloaded manually.
    for local_archive in candidate_archive_paths(ds, data_dir=data_dir):
        if not local_archive.is_file():
            continue

        output_dir = (
            local_archive.parent.parent
            if local_archive.parent.name == ".archives"
            else local_archive.parent
        )
        dst = output_dir / dataset_file_name(ds)
        try:
            return extract_dataset_archive(ds, local_archive, dst)
        except Exception as exc:
            raise dataset_resolution_error(
                ds,
                data_dir=data_dir,
                reason=f"Could not extract {local_archive}: {exc}",
            ) from exc

    data_directory = get_data_dir(data_dir)
    dst = data_directory / dataset_file_name(ds)
    archive_dst = data_directory / ".archives" / dataset_archive_name(ds)

    try:
        download_release_archive(ds, archive_dst)
        extracted = extract_dataset_archive(ds, archive_dst, dst)
        if not _keep_downloaded_archives():
            archive_dst.unlink(missing_ok=True)
        return extracted
    except Exception as exc:
        raise dataset_resolution_error(
            ds,
            data_dir=data_dir,
            reason=f"GitHub release download failed: {exc}",
        ) from exc


def load_dataset(
    ds: str,
    verbose: bool = True,
    data_dir: str | Path | None = None,
) -> NpzFile:
    """Load one benchmark dataset from its local cache.

    The returned ``NpzFile`` should be closed by the caller, preferably by
    using it as a context manager.
    """
    dst = ensure_dataset(ds, data_dir=data_dir)
    real_data = np.load(dst, allow_pickle=False)

    if verbose:
        dataset_summary(real_data, ds, data_dir=data_dir, data_path=dst)

    return real_data


def _scalar_value(value: np.ndarray) -> object:
    return value.item() if value.ndim == 0 else value


def dataset_summary(
    real_data: NpzFile,
    ds: str,
    data_dir: str | Path | None = None,
    data_path: str | Path | None = None,
) -> None:
    """Print a compact summary of an opened dataset."""
    ds = normalize_dataset_id(ds)
    y = real_data["ts"]
    t = real_data["t"]
    resolved_path = (
        Path(data_path).expanduser().resolve(strict=False)
        if data_path is not None
        else ensure_dataset(ds, data_dir=data_dir)
    )

    print(f"\n=== Dataset {ds} ===")
    print(f"path:         {resolved_path}")
    print(f"dataset name: {_scalar_value(real_data['dataset'])}")
    print(f"ts shape:     {y.shape}")
    print(f"ts max:       {y.max()}")
    print(f"t units:      {_scalar_value(real_data['t_units'])}")
    print(f"data units:   {_scalar_value(real_data['data_units'])}")
    print(f"t start:      {t[0]}")
    print(f"t end:        {t[-1]}")
    print(f"t count:      {len(t)}")
    print(f"sample rate:  {_scalar_value(real_data['sample_rate'])}")


def main() -> None:
    """Download all benchmark datasets into the selected data directory."""
    data_dir = get_data_dir()
    print(f"Using dataset folder: {data_dir}")
    for ds in DATASETS:
        ensure_dataset(ds, data_dir=data_dir)


if __name__ == "__main__":
    main()