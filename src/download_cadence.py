"""
Download the 2017-06-24 ABACAD cadence of HIP13375 from the BL Open Data Archive.

URL pattern:
    https://bldata.berkeley.edu/pipeline/{PROJECT}/holding/
        spliced_blc0001020304050607_guppi_{MJD}_{SOD}_{TARGET}_{SCAN}.gpuspec.0000.h5

Resumable, skips files already complete, HEAD-probes the SOD field (which can
be off by ±2 sec from MJD rounding). Run from project root:

    python src/download_cadence.py
    python src/download_cadence.py --offs-only          # skip the 3 ON scans
    python src/download_cadence.py --workers 3          # parallel (be polite)
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


BASE = "https://bldata.berkeley.edu/pipeline"
PROJECT_2017 = "AGBT17A_999_86"
MJD_2017 = 57928
MAX_WORKERS = 6  # hard cap — BL Open Data Archive is a shared resource; do not exceed

# Note on integrity: this script verifies downloads by byte size only.
# The BL Open Data Archive does not publish per-file checksums, so a
# truncated-but-correct-size file is the worst-case failure mode the
# resume + size-check logic can detect. If checksums become available
# upstream, add SHA-256 verification after download_with_resume().


def make_session(retries: int = 3, backoff: float = 2.0) -> requests.Session:
    """HTTP session with automatic retries on transient network failures.

    Retries on 5xx and connection errors; exponential backoff; respects
    Retry-After headers. A blip mid-stream raises a ChunkedEncodingError
    that requests does NOT retry — handled at the call site instead.
    """
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("HEAD", "GET"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


SESSION = make_session()

# ---------------------------------------------------------------------------
# 2017-06-24 ABACAD cadence around HIP 13375 — Breakthrough Listen project
# AGBT17A_999_86. All six panels resolve via the HEAD probe in resolve_url();
# any files already present at full size are skipped by download_with_resume().
# ---------------------------------------------------------------------------

# Cadence panels: (scan_num, target, sec_of_day_guess, role)
# A1/A2/A3 and D SODs are confirmed from canonical filenames; B and C SODs
# are arithmetic guesses from the published observing times and are caught
# by the ±3 second HEAD probe in resolve_url().
CADENCE = [
    (44, "HIP13375", 54431, "A1 ON"),    # confirmed
    (45, "HIP12678", 54789, "B  OFF"),   # guess (15:13:09 UTC)
    (46, "HIP13375", 55145, "A2 ON"),    # confirmed
    (47, "HIP12790", 55498, "C  OFF"),   # guess (15:24:58 UTC)
    (48, "HIP13375", 55851, "A3 ON"),    # confirmed
    (49, "HIP12919", 56197, "D  OFF"),   # confirmed
]

DEST_DIR = Path(__file__).parent.parent / "data" / "raw"
CHUNK_SIZE = 1 << 20  # 1 MiB
SOD_PROBE_RANGE = range(-3, 4)  # try guess±3 if HEAD fails on the guess


def build_filename(mjd: int, sod: int, target: str, scan: int) -> str:
    return (
        f"spliced_blc0001020304050607_guppi_"
        f"{mjd}_{sod}_{target}_{scan:04d}.gpuspec.0000.h5"
    )


def url_for(filename: str, project: str = PROJECT_2017) -> str:
    return f"{BASE}/{project}/holding/{filename}"


def resolve_url(mjd: int, target: str, scan: int, sod_guess: int) -> tuple[str, str] | None:
    """HEAD-probe SOD ± 3 seconds. Return (url, filename) on hit, else None."""
    for delta in sorted(SOD_PROBE_RANGE, key=abs):
        sod = sod_guess + delta
        fname = build_filename(mjd, sod, target, scan)
        url = url_for(fname)
        try:
            r = SESSION.head(url, allow_redirects=True, timeout=15)
        except requests.RequestException:
            continue
        if r.status_code == 200:
            return url, fname
    return None


def download_with_resume(url: str, dest: Path, position: int = 0, max_attempts: int = 4) -> None:
    """Download `url` → `dest` with resume + auto-retry on streaming hiccups.

    `position` reserves a tqdm bar slot. SESSION already handles connection-
    level retries (HEAD/GET that fail before bytes arrive). What SESSION can't
    handle is a ChunkedEncodingError mid-stream — for that we re-enter the
    download loop up to `max_attempts` times, each attempt resuming from the
    bytes already on disk.
    """
    for attempt in range(1, max_attempts + 1):
        head = SESSION.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        remote_size = int(head.headers.get("Content-Length", 0))
        accepts_ranges = head.headers.get("Accept-Ranges", "").lower() == "bytes"
        local_size = dest.stat().st_size if dest.exists() else 0

        if remote_size and local_size == remote_size:
            tqdm.write(f"  [skip] {dest.name[-45:]}  already complete ({local_size/1e9:.2f} GB)")
            return

        headers, mode = {}, "wb"
        if local_size and accepts_ranges and local_size < remote_size:
            headers["Range"] = f"bytes={local_size}-"
            mode = "ab"
            label = "resume" if attempt == 1 else f"retry {attempt}/{max_attempts}"
            tqdm.write(f"  [{label}] {dest.name[-45:]}  from {local_size/1e9:.2f} / {remote_size/1e9:.2f} GB")
        elif local_size and not accepts_ranges:
            tqdm.write(f"  [warn] {dest.name[-45:]}  server lacks ranges, restarting")
            local_size = 0

        try:
            with SESSION.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                bar = tqdm(
                    total=remote_size, initial=local_size,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc=dest.name[-45:], position=position, leave=True,
                )
                with dest.open(mode) as f:
                    for chunk in r.iter_content(CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
                bar.close()
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            if attempt < max_attempts:
                wait = 2 ** attempt
                tqdm.write(f"  [retry] {dest.name[-45:]}  network blip ({type(e).__name__}); retrying in {wait}s")
                time.sleep(wait)
                continue
            raise

        final_size = dest.stat().st_size
        if remote_size and final_size != remote_size:
            raise RuntimeError(f"size mismatch: got {final_size}, expected {remote_size}")
        return  # success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--offs-only", action="store_true",
                        help="Skip the 3 HIP13375 ON scans (assume they came via browser).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve URLs and print plan, don't download.")
    parser.add_argument("--workers", type=int, default=1, metavar="N",
                        help=f"Parallel download workers (default 1 = sequential, max {MAX_WORKERS}). "
                             "Be polite: 3 is plenty for a shared archive.")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.workers > MAX_WORKERS:
        print(f"⚠ --workers={args.workers} exceeds hard cap of {MAX_WORKERS}; clamping to {MAX_WORKERS}.")
        args.workers = MAX_WORKERS

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    panels = [p for p in CADENCE
              if not (args.offs_only and p[1] == "HIP13375")]

    print(f"Destination: {DEST_DIR}")
    print(f"Resolving {len(panels)} URLs...\n")

    plan: list[tuple[str, str, str]] = []  # (role, url, filename)
    for scan, target, sod_guess, role in panels:
        hit = resolve_url(MJD_2017, target, scan, sod_guess)
        if hit is None:
            print(f"  ✗ {role:6s} scan {scan:04d} {target} — no match within ±3s of SOD={sod_guess}")
            continue
        url, fname = hit
        actual_sod = int(fname.split("_")[4])
        delta = actual_sod - sod_guess
        marker = "✓" if delta == 0 else f"✓ (Δsod={delta:+d})"
        print(f"  {marker} {role:6s} {fname}")
        plan.append((role, url, fname))

    if not plan:
        print("\nNo files resolved. Check PROJECT_2017 and the search-result times.")
        sys.exit(1)

    if args.dry_run:
        print(f"\nDry run — would download {len(plan)} file(s).")
        return

    workers = min(args.workers, len(plan))
    mode = "sequential" if workers == 1 else f"parallel x{workers}"
    print(f"\n{'─'*60}\nDownloading {len(plan)} file(s) [{mode}]\n{'─'*60}")

    if workers == 1:
        for i, (role, url, fname) in enumerate(plan, 1):
            print(f"\n[{i}/{len(plan)}] {role}  {fname}")
            try:
                download_with_resume(url, DEST_DIR / fname)
            except Exception as e:
                print(f"  ✗ failed: {e}")
        return

    # Parallel: each task gets a unique tqdm position so bars don't overlap.
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_with_resume, url, DEST_DIR / fname, slot): (role, fname)
            for slot, (role, url, fname) in enumerate(plan)
        }
        for fut in as_completed(futures):
            role, fname = futures[fut]
            try:
                fut.result()
            except Exception as e:
                failures.append((fname, str(e)))
                tqdm.write(f"  ✗ {role}  {fname}: {e}")

    # Push cursor below all bars before final summary.
    print("\n" * len(plan))
    if failures:
        print(f"\n{len(failures)} file(s) failed:")
        for fname, err in failures:
            print(f"  - {fname}: {err}")
        sys.exit(1)
    print(f"\nAll {len(plan)} file(s) downloaded successfully.")


if __name__ == "__main__":
    main()
