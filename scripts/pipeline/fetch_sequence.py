from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import himawari_ml.ingest.fetch_latest as fl
from himawari_ml.utils.paths import sample_sequence_dir


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _round_down_minutes(ts: datetime, step_minutes: int) -> datetime:
    ts = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
    m = (ts.minute // step_minutes) * step_minutes
    return ts.replace(minute=m)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="DEV: Fetch a Himawari sequence into data/sample/raw/<out-subdir>/"
    )
    ap.add_argument("--n", "--n-frames", dest="n_frames", type=int, default=100,
                    help="Number of frames to fetch (successful saves).")
    ap.add_argument("--step-minutes", dest="step_minutes", type=int, default=10,
                    help="Minutes between frames.")
    ap.add_argument("--out-subdir", dest="subdir", default="sequenceA",
                    help="Subdirectory under data/sample/raw/")
    ap.add_argument("--start-minutes-ago", type=int, default=0,
                    help="Start this many minutes ago (default now).")
    ap.add_argument("--max-tries", type=int, default=600,
                    help="How far back to search (attempts).")
    ap.add_argument("--image-size", type=int, default=fl.DEFAULT_IMAGE_SIZE,
                    help="Final stitched image size (defaults to src ingest default).")
    ap.add_argument("--retries", type=int, default=3,
                    help="HTTP retries per frame.")
    ap.add_argument("--timeout-s", type=int, default=30,
                    help="HTTP timeout in seconds.")
    args = ap.parse_args()

    out_dir = sample_sequence_dir(args.subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = fl.FetchCfg(retries=args.retries, timeout_s=args.timeout_s)

    start = _utcnow() - timedelta(minutes=args.start_minutes_ago)
    start = _round_down_minutes(start, args.step_minutes)

    saved = 0
    attempts = 0

    # Try timestamps going back until we save n_frames or run out of candidates
    for i in range(args.max_tries):
        if saved >= args.n_frames:
            break

        ts = start - timedelta(minutes=args.step_minutes * i)
        attempts += 1

        # Since core now uses full UTC timestamp filenames, existence check is simple
        out_path = out_dir / fl.frame_filename(ts) if hasattr(fl, "frame_filename") else None
        if out_path is not None and out_path.exists():
            saved += 1
            continue

        ok = fl.fetch_frame(ts=ts, out_dir=out_dir, image_size=args.image_size, cfg=cfg)
        if ok:
            saved += 1

    if saved == 0:
        raise SystemExit("No valid frames fetched (all candidates failed/blank).")

    print(f"DEV sequence fetched: saved={saved} attempts={attempts} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
