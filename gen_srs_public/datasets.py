import os
from pathlib import Path
import urllib.request

import numpy as np
from numpy.lib.npyio import NpzFile

# ----------------------------
# GitHub Release config (public)
# ----------------------------
OWNER = os.environ.get("GEN_SRS_RELEASE_OWNER", "lanl")
REPO = os.environ.get("GEN_SRS_RELEASE_REPO", "your-repo")
TAG = os.environ.get("GEN_SRS_RELEASE_TAG", "v1.0.0")

DATASETS = ["A", "B", "C", "D"]  # Dataset_A.npz ... Dataset_D.npz

# ----------------------------
# Local dataset cache
# ----------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def default_cache_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    return base / "gen_srs" / "Datasets"


def normalize_dataset_id(ds: str) -> str:
    ds = str(ds).strip().upper()
    assert ds in DATASETS, f"ds must be one of {DATASETS}; got {ds!r}"
    return ds


def dataset_asset_name(ds: str) -> str:
    return f"Dataset_{normalize_dataset_id(ds)}.npz"


def candidate_data_dirs(data_dir: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []

    if data_dir is not None:
        data_dir_path = Path(data_dir).expanduser()
        if data_dir_path.suffix.lower() != ".npz":
            candidates.append(data_dir_path)
    else:
        override = os.environ.get("GEN_SRS_DATA_DIR")
        if override:
            override_path = Path(override).expanduser()
            if override_path.suffix.lower() != ".npz":
                candidates.append(override_path)

    candidates.append(REPO_ROOT / "Datasets")

    # Support notebooks run from a cloned repo even when the package import
    # resolves to an installed environment elsewhere.
    cwd = Path.cwd().resolve()
    candidates.append(cwd / "Datasets")
    if cwd != REPO_ROOT and ((cwd.parent / ".git").exists() or (cwd.parent / "gen_srs").exists()):
        candidates.append(cwd.parent / "Datasets")

    candidates.append(default_cache_data_dir())

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    return unique_candidates


def candidate_dataset_paths(ds: str, data_dir: str | Path | None = None) -> list[Path]:
    asset = dataset_asset_name(ds)
    candidates: list[Path] = []

    if data_dir is not None:
        data_dir_path = Path(data_dir).expanduser()
        if data_dir_path.suffix.lower() == ".npz":
            candidates.append(data_dir_path.resolve(strict=False))
        else:
            candidates.append((data_dir_path / asset).resolve(strict=False))
    else:
        override = os.environ.get("GEN_SRS_DATA_DIR")
        if override:
            override_path = Path(override).expanduser()
            if override_path.suffix.lower() == ".npz":
                candidates.append(override_path.resolve(strict=False))

    for directory in candidate_data_dirs(data_dir=data_dir):
        candidates.append(directory / asset)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    return unique_candidates


def get_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is not None:
        data_dir_path = Path(data_dir).expanduser()
        if data_dir_path.suffix.lower() == ".npz":
            return data_dir_path.parent.resolve(strict=False)
        return data_dir_path.resolve(strict=False)

    return candidate_data_dirs(data_dir=data_dir)[0]


def find_existing_dataset(ds: str, data_dir: str | Path | None = None) -> Path | None:
    for candidate in candidate_dataset_paths(ds, data_dir=data_dir):
        if candidate.exists():
            return candidate
    return None


DATA_DIR = get_data_dir()


def validate_release_config(owner: str, repo: str, tag: str) -> None:
    if repo == "your-repo":
        raise RuntimeError(
            "Set GEN_SRS_RELEASE_REPO (and optionally GEN_SRS_RELEASE_OWNER / "
            "GEN_SRS_RELEASE_TAG) before downloading release assets."
        )
    if not owner or not repo or not tag:
        raise RuntimeError("Release asset config requires non-empty owner, repo, and tag values.")


def release_asset_url(asset_name: str) -> str:
    validate_release_config(OWNER, REPO, TAG)
    return f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{asset_name}"


def download_if_missing(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    print(f"Downloading {dst.name} ...")
    urllib.request.urlretrieve(url, dst)
    print(f"Saved to {dst}")


def dataset_resolution_error(ds: str, data_dir: str | Path | None = None, reason: str | None = None) -> RuntimeError:
    asset = dataset_asset_name(ds)
    candidates = candidate_dataset_paths(ds, data_dir=data_dir)
    search_list = "\n".join(f"  - {candidate}" for candidate in candidates)
    message = (
        f"Could not locate {asset} locally.\n"
        f"Searched:\n{search_list}\n"
        "Place the file in one of those locations, pass `data_dir=...`, "
        "or set `GEN_SRS_DATA_DIR`."
    )
    if reason:
        message = f"{message}\n{reason}"
    return RuntimeError(message)


def ensure_dataset(ds: str, data_dir: str | Path | None = None) -> Path:
    ds = normalize_dataset_id(ds)
    existing = find_existing_dataset(ds, data_dir=data_dir)
    if existing is not None:
        return existing

    asset = dataset_asset_name(ds)
    dst = get_data_dir(data_dir) / asset

    try:
        url = release_asset_url(asset)
    except RuntimeError as exc:
        raise dataset_resolution_error(
            ds,
            data_dir=data_dir,
            reason=(
                "Release asset download is not configured yet. "
                "If you already downloaded the dataset manually, point "
                "`data_dir` or `GEN_SRS_DATA_DIR` at that folder."
            ),
        ) from exc

    try:
        download_if_missing(url, dst)
    except Exception as exc:
        raise dataset_resolution_error(
            ds,
            data_dir=data_dir,
            reason=f"Download from {url} failed: {exc}",
        ) from exc

    return dst


def load_dataset(ds: str, verbose: bool = True, data_dir: str | Path | None = None) -> NpzFile:
    dst = ensure_dataset(ds, data_dir=data_dir)
    real_data = np.load(dst, allow_pickle=True)

    if verbose:
        dataset_summary(real_data, ds, data_dir=data_dir, data_path=dst)

    return real_data


def dataset_summary(
    real_data: NpzFile,
    ds: str,
    data_dir: str | Path | None = None,
    data_path: str | Path | None = None,
) -> None:
    ds = normalize_dataset_id(ds)
    y = real_data["ts"]
    t = real_data["t"]
    resolved_path = (
        Path(data_path).expanduser().resolve(strict=False)
        if data_path is not None
        else ensure_dataset(ds, data_dir=data_dir)
    )

    print(f"\n=== Dataset {ds} ===")
    print(f"path:     {resolved_path}")
    print(f"dataset name: {real_data['dataset']}")
    print(f"ts shape: {y.shape}")
    print(f"ts max:   {y.max()}")
    print(f"t units:  {real_data['t_units']}")
    print(f"units:    {real_data['data_units']}")
    print(f"t start:  {t[0]}")
    print(f"t end:    {t[-1]}")
    print(f"t count:  {len(t)}")
    print(f"fs:       {real_data['sample_rate']}")


def main() -> None:
    data_dir = get_data_dir()
    print(f"Using dataset folder: {data_dir}")
    for ds in DATASETS:
        ensure_dataset(ds, data_dir=data_dir)


if __name__ == "__main__":
    main()
