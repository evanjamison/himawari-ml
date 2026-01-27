from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    while not (p / "pyproject.toml").exists() and not (p / ".git").exists():
        if p == p.parent:
            break
        p = p.parent
    return p


REPO_ROOT = _repo_root()


def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 78)
    print(" ".join(cmd))
    print("=" * 78)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def _abs_under_repo(p: Path) -> Path:
    """Return absolute path; if relative, interpret relative to repo root."""
    p = Path(p)
    return (REPO_ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def _looks_like_glob(s: str) -> bool:
    return any(ch in s for ch in ["*", "?", "["])


def _iter_images_in_dir(d: Path, recursive: bool, exts: set[str]) -> list[Path]:
    if recursive:
        it = d.rglob("*")
    else:
        it = d.glob("*")

    files = []
    for p in it:
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def _glob_images(pattern: str, exts: set[str]) -> list[Path]:
    """
    Glob pattern relative to repo root unless absolute.
    Supports ** patterns.
    """
    # If absolute-ish, try direct glob from its parent.
    pat = pattern
    if Path(pattern).is_absolute():
        base = Path(pattern).anchor
        # pathlib doesn't accept glob across anchors cleanly; use the absolute string via REPO_ROOT.glob is wrong.
        # Instead: treat as normal string and use Path().glob with a best-effort approach by splitting.
        # Simple approach: use the parent of the pattern up to first wildcard.
        # If this is too fancy, just fall back to REPO_ROOT.glob on a relative pattern.
        # We'll do a robust and simple fallback:
        root = Path(pattern)
        # find first wildcard index
        idx = min([pattern.find(ch) for ch in ["*", "?", "["] if ch in pattern] or [-1])
        if idx == -1:
            candidates = [root]
        else:
            prefix = pattern[:idx]
            prefix_dir = Path(prefix).parent
            rel_pat = pattern[len(str(prefix_dir)) + 1 :]
            candidates = [p for p in prefix_dir.glob(rel_pat)]
    else:
        candidates = [p for p in REPO_ROOT.glob(pat)]

    files = []
    for p in candidates:
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted({p.resolve() for p in files})


def _stage_files(files: list[Path], outdir: Path, stage_name: str = "_raw_staged") -> Path:
    """
    Copy files into a flat staging folder under outdir.
    This avoids changing downstream scripts that expect --raw-dir to be a single folder.
    """
    staged = outdir / stage_name
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(files):
        # Stable, collision-resistant name
        rel = str(src).encode("utf-8", errors="ignore")
        h = hashlib.sha1(rel).hexdigest()[:10]
        dst = staged / f"frame_{i:06d}_{h}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    return staged


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the sample CV pipeline end-to-end on frames (directory, recursive, or glob)."
    )

    ap.add_argument(
        "--raw-dir",
        default="data/sample/raw/sequence",
        help=(
            "Directory OR glob pattern of frames. Examples:\n"
            "  --raw-dir data/sample/raw/sequenceA\n"
            "  --raw-dir data/sample/raw --recursive\n"
            "  --raw-dir 'data/**/raw/**/*.png'\n"
            "(If relative, interpreted from repo root.)"
        ),
    )
    ap.add_argument("--recursive", action="store_true", help="If --raw-dir is a directory, search subfolders too.")
    ap.add_argument(
        "--exts",
        default=".png,.jpg,.jpeg,.webp",
        help="Comma-separated allowed extensions.",
    )
    ap.add_argument("--outdir", default="out", help="Output directory (absolute or relative to repo root).")
    ap.add_argument("--save-overlays", action="store_true", help="Save overlay/contact-sheet images.")
    args = ap.parse_args()

    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    outdir = _abs_under_repo(Path(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    raw_arg = args.raw_dir.strip()

    # Resolve inputs:
    # 1) If looks like a glob -> expand
    # 2) Else treat as a directory (absolute or relative to repo)
    files: list[Path] = []
    raw_dir_for_run: Path | None = None
    used_staging = False

    if _looks_like_glob(raw_arg):
        files = _glob_images(raw_arg, exts)
        if not files:
            raise SystemExit(f"No image files matched glob: {raw_arg}")
        raw_dir_for_run = _stage_files(files, outdir)
        used_staging = True
    else:
        raw_dir = _abs_under_repo(Path(raw_arg))
        if not raw_dir.exists():
            raise SystemExit(f"Raw dir not found: {raw_dir}")

        # If recursive requested, or if directory has no direct images but has nested ones -> stage
        direct = _iter_images_in_dir(raw_dir, recursive=False, exts=exts)
        if args.recursive:
            files = _iter_images_in_dir(raw_dir, recursive=True, exts=exts)
            if not files:
                raise SystemExit(f"No image files found under (recursive): {raw_dir}")
            raw_dir_for_run = _stage_files(files, outdir)
            used_staging = True
        else:
            if direct:
                raw_dir_for_run = raw_dir
            else:
                # helpful fallback: auto-recursive if direct folder is empty of images
                files = _iter_images_in_dir(raw_dir, recursive=True, exts=exts)
                if not files:
                    raise SystemExit(f"No image files found under: {raw_dir}")
                raw_dir_for_run = _stage_files(files, outdir)
                used_staging = True

    assert raw_dir_for_run is not None

    # Pretty print without crashing if path is already relative/odd
    try:
        raw_disp = raw_dir_for_run.relative_to(REPO_ROOT).as_posix()
    except Exception:
        raw_disp = str(raw_dir_for_run)

    if used_staging:
        print(f"Input resolved to {len(files)} frames (staged): {raw_disp}")
    else:
        print(f"Raw frames used: {raw_disp}")
    print(f"Output dir: {outdir}")

    py = str(Path(sys.executable))

    # --- STEP 1: heuristic cloud mask baseline ---
    cmd1 = [
        py,
        str(REPO_ROOT / "scripts" / "experiments" / "segmentation" / "cloud_mask_baseline.py"),
        "--raw-dir",
        str(raw_dir_for_run),
        "--outdir",
        str(outdir),
    ]
    if args.save_overlays:
        cmd1.append("--save-overlays")
    _run(cmd1)

    # --- STEP 2: cloud objects from masks ---
    cmd2 = [
        py,
        str(REPO_ROOT / "scripts" / "experiments" / "segmentation" / "cloud_objects.py"),
        "--metrics-csv",
        str(outdir / "cloud_metrics.csv"),
        "--outdir",
        str(outdir),
    ]
    if args.save_overlays:
        cmd2.append("--save-overlays")
    _run(cmd2)

    # --- STEP 3: track storms ---
    cmd3 = [
        py,
        str(REPO_ROOT / "scripts" / "experiments" / "tracking" / "track_storms.py"),
        "--frames-csv",
        str(outdir / "cloud_objects_frames.csv"),
        "--outdir",
        str(outdir),
    ]
    if args.save_overlays:
        cmd3.append("--save-contact-sheet")
    _run(cmd3)

    print("\nDONE. Key outputs:")
    for p in [
        outdir / "cloud_metrics.csv",
        outdir / "cloud_objects.csv",
        outdir / "cloud_objects_frames.csv",
        outdir / "storm_track.csv",
        outdir / "viz" / "cloud_fraction_timeseries.png",
        outdir / "viz" / "storm_track_overlay.png",
    ]:
        status = "OK" if p.exists() else "MISSING"
        # print relative if possible
        try:
            disp = p.relative_to(REPO_ROOT).as_posix()
        except Exception:
            disp = str(p)
        print(f"{status:7} {disp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
