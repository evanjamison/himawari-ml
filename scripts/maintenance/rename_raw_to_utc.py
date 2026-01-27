from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


HHMMSS_RE = re.compile(r"^(himawari)_(\d{6})(?:\.\d+)?\.png$", re.IGNORECASE)
FULL_TS_RE = re.compile(r"^himawari_\d{8}T\d{6}Z\.png$", re.IGNORECASE)


@dataclass
class RenamePlan:
    src: Path
    dst: Path
    reason: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rename himawari_HHMMSS.png files into himawari_YYYYMMDDTHHMMSSZ.png without re-downloading."
    )
    p.add_argument("--raw-dir", default="data/raw", help="Directory containing raw PNGs (default: data/raw)")
    p.add_argument(
        "--date-mode",
        choices=["mtime-local", "mtime-utc", "fixed-utc-date"],
        default="mtime-local",
        help=(
            "How to choose the date part for HHMMSS-only filenames:\n"
            "- mtime-local: use file modified time in LOCAL time, then convert to UTC date\n"
            "- mtime-utc: use file modified time interpreted as UTC\n"
            "- fixed-utc-date: use --fixed-utc-date for ALL files\n"
        ),
    )
    p.add_argument(
        "--fixed-utc-date",
        default=None,
        help="If date-mode=fixed-utc-date, use this UTC date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--keep-originals",
        action="store_true",
        help="Copy instead of rename (keeps originals). Default is rename/move.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen, but do not rename/copy.",
    )
    p.add_argument(
        "--manifest",
        default="out/rename_manifest.csv",
        help="Where to write the old->new mapping CSV (default: out/rename_manifest.csv)",
    )
    return p.parse_args()


def ensure_unique(dst: Path) -> Path:
    """If dst exists, append _dupN before .png."""
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        cand = parent / f"{stem}_dup{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def utc_date_for_file(p: Path, mode: str, fixed_utc_date: str | None) -> str:
    """
    Return YYYYMMDD (UTC) used for naming.
    """
    if mode == "fixed-utc-date":
        if not fixed_utc_date:
            raise SystemExit("--fixed-utc-date is required when --date-mode=fixed-utc-date")
        d = datetime.strptime(fixed_utc_date, "%Y-%m-%d").date()
        return d.strftime("%Y%m%d")

    st = p.stat()
    if mode == "mtime-utc":
        dt_utc = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
        return dt_utc.strftime("%Y%m%d")

    # mtime-local
    # Interpret mtime in local time, then convert to UTC and take the UTC date.
    dt_local = datetime.fromtimestamp(st.st_mtime)  # naive local
    dt_local = dt_local.astimezone()  # attach local tz
    dt_utc = dt_local.astimezone(timezone.utc)
    return dt_utc.strftime("%Y%m%d")


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise SystemExit(f"raw dir not found: {raw_dir.resolve()}")

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    plans: list[RenamePlan] = []
    skipped = 0

    # Only look at pngs at top-level (not sequence/ etc unless you want it)
    files = sorted([p for p in raw_dir.glob("*.png") if p.is_file()])

    for p in files:
        name = p.name

        # Already in desired format
        if FULL_TS_RE.match(name):
            skipped += 1
            continue

        m = HHMMSS_RE.match(name)
        if not m:
            # not a HHMMSS file; skip quietly (or you can print it)
            skipped += 1
            continue

        hhmmss = m.group(2)  # e.g. 003000
        yyyymmdd = utc_date_for_file(p, args.date_mode, args.fixed_utc_date)
        new_name = f"himawari_{yyyymmdd}T{hhmmss}Z.png"
        dst = ensure_unique(raw_dir / new_name)

        plans.append(RenamePlan(src=p, dst=dst, reason=f"{args.date_mode} -> {yyyymmdd} + {hhmmss}"))

    print(f"[info] raw_dir: {raw_dir.resolve()}")
    print(f"[info] found {len(files)} PNGs, planned renames: {len(plans)}, skipped: {skipped}")
    if args.dry_run:
        print("[dry-run] no files will be changed.\n")

    # Write manifest + perform operations
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old_path", "new_path", "reason"])
        for plan in plans:
            w.writerow([str(plan.src), str(plan.dst), plan.reason])

    for plan in plans:
        if args.dry_run:
            print(f"DRY  {plan.src.name}  ->  {plan.dst.name}   ({plan.reason})")
            continue

        if args.keep_originals:
            plan.dst.write_bytes(plan.src.read_bytes())
            print(f"COPY {plan.src.name}  ->  {plan.dst.name}")
        else:
            plan.src.rename(plan.dst)
            print(f"MOVE {plan.src.name}  ->  {plan.dst.name}")

    print(f"\n[done] manifest written to: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
