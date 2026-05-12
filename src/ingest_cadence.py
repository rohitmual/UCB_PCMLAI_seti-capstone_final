"""
Ingest Campaign 2 (2017-06-24) ABACAD cadence into engineered features.

Streams 6 GBT files (~93 GB total) → cadence_features.h5 (~15.5 GB).
Each raw HDF5 is read in frequency-axis chunks of CHUNK_W native channels
(~1.6 GB peak RAM per chunk), downsampled and median-normalized per snippet,
and appended to the output archive. Designed to run on a 16 GB laptop without
OOM.

Cadence:
  A1 = HIP 13375 (15:07:11 UTC)
  B  = HIP 12678 (15:13:09 UTC)
  A2 = HIP 13375 (15:19:05 UTC)
  C  = HIP 12790 (15:24:58 UTC)
  A3 = HIP 13375 (15:30:51 UTC)
  D  = HIP 12919 (15:36:37 UTC)

Snippet schema: 512 native channels → downsample 8× → (16 time, 64 freq).
"""

import argparse
import time
import numpy as np
import hdf5plugin  # noqa: F401  — required for bitshuffle decompression
import h5py
from pathlib import Path
from tqdm import tqdm

# Repo root = parent of src/. Matches src/download_cadence.py and the
# notebooks, which read with Path("data/processed/cadence_features.h5").
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "cadence_features.h5"

CADENCE = [
    ("spliced_blc0001020304050607_guppi_57928_54431_HIP13375_0044.gpuspec.0000.h5", "A1", 0),
    ("spliced_blc0001020304050607_guppi_57928_54789_HIP12678_0045.gpuspec.0000.h5", "B",  1),
    ("spliced_blc0001020304050607_guppi_57928_55145_HIP13375_0046.gpuspec.0000.h5", "A2", 2),
    ("spliced_blc0001020304050607_guppi_57928_55498_HIP12790_0047.gpuspec.0000.h5", "C",  3),
    ("spliced_blc0001020304050607_guppi_57928_55851_HIP13375_0048.gpuspec.0000.h5", "A3", 4),
    ("spliced_blc0001020304050607_guppi_57928_56197_HIP12919_0049.gpuspec.0000.h5", "D",  5),
]

SNIPPET_NATIVE_W = 512        # native frequency channels per snippet
DOWNSAMPLE_F     = 8          # → 64 effective channels per snippet
BATCH_SNIPPETS   = 50_000     # snippets per output-append batch
CHUNK_SNIPPETS   = 50_000     # snippets per raw-read chunk (~1.6 GB peak per chunk for float32: 50K × 512 × 16 × 4 bytes)


def iter_file_chunks(path):
    """Yield ``(snips_1024d, freqs_mhz, fch1, foff, n_snip_chunk)`` per chunk.

    Reads the raw HDF5 in frequency-axis chunks of CHUNK_SNIPPETS × SNIPPET_NATIVE_W
    native channels instead of loading the entire ~15 GB array at once. Each
    chunk peaks at ~1.6 GB RAM (float32), so the ingest runs on a 16 GB laptop
    without OOM. Mathematically identical to the prior whole-file load.
    """
    with h5py.File(path, "r") as f:
        d = f["data"]
        fch1 = float(d.attrs["fch1"])
        foff = float(d.attrs["foff"])
        total_chans = d.shape[2]
        n_snip_total = total_chans // SNIPPET_NATIVE_W

        for chunk_start_snip in range(0, n_snip_total, CHUNK_SNIPPETS):
            chunk_end_snip = min(chunk_start_snip + CHUNK_SNIPPETS, n_snip_total)
            n_snip_chunk = chunk_end_snip - chunk_start_snip

            ch_lo = chunk_start_snip * SNIPPET_NATIVE_W
            ch_hi = chunk_end_snip * SNIPPET_NATIVE_W
            raw_chunk = d[:, 0, ch_lo:ch_hi].astype(np.float32)   # (16, n_snip_chunk * 512)

            snips = (
                raw_chunk.reshape(16, n_snip_chunk, SNIPPET_NATIVE_W)
                         .transpose(1, 0, 2)
                         .reshape(n_snip_chunk, 16, SNIPPET_NATIVE_W // DOWNSAMPLE_F, DOWNSAMPLE_F)
                         .mean(axis=3)
            )
            del raw_chunk

            med = np.median(snips, axis=(1, 2), keepdims=True)
            med[med == 0] = 1
            snips = (snips / med).astype(np.float32).reshape(n_snip_chunk, -1)

            centers = (np.arange(chunk_start_snip, chunk_end_snip) * SNIPPET_NATIVE_W) + (SNIPPET_NATIVE_W // 2)
            freqs = fch1 + centers * foff

            yield snips, freqs, fch1, foff, chunk_start_snip, n_snip_total


def append(dset, arr):
    n = dset.shape[0]
    new_n = n + len(arr)
    if arr.ndim == 1:
        dset.resize((new_n,))
    else:
        dset.resize((new_n,) + arr.shape[1:])
    dset[n:new_n] = arr


def main():
    parser = argparse.ArgumentParser(
        description="Ingest the 2017-06-24 ABACAD cadence into cadence_features.h5."
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Directory containing the 6 raw GBT HDF5 files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_PATH,
        help=f"Output HDF5 path (default: {DEFAULT_OUT_PATH})",
    )
    args = parser.parse_args()

    data_dir = args.raw_dir
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for fname, *_ in CADENCE:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(f"Missing cadence file: {data_dir / fname}")
    print(f"All 6 cadence files present in {data_dir}")
    print(f"Output: {out_path}")

    t_start = time.time()
    fch1_ref = foff_ref = None

    with h5py.File(out_path, "w") as out:
        snippets_dset = out.create_dataset("snippets", shape=(0, 1024), maxshape=(None, 1024), chunks=(10000, 1024), dtype="f4")
        pos   = out.create_dataset("cadence_pos", shape=(0,),    maxshape=(None,),    chunks=(10000,),    dtype="i1")
        fidx  = out.create_dataset("file_idx",    shape=(0,),    maxshape=(None,),    chunks=(10000,),    dtype="i1")
        sidx  = out.create_dataset("snippet_idx", shape=(0,),    maxshape=(None,),    chunks=(10000,),    dtype="i4")
        freq  = out.create_dataset("freq_mhz",    shape=(0,),    maxshape=(None,),    chunks=(10000,),    dtype="f8")

        files_bar = tqdm(CADENCE, desc="Cadence files", unit="file", position=0)
        for file_i, (fname, tag, cadence_pos_val) in enumerate(files_bar):
            files_bar.set_description(f"[{file_i+1}/6] {tag}")
            t_file = time.time()

            tqdm.write(f"\n[{file_i+1}/6] {tag} ← {fname}")
            tqdm.write(f"   streaming raw HDF5 in {CHUNK_SNIPPETS:,}-snippet chunks (~1.6 GB peak each)...")

            chunk_iter = iter_file_chunks(data_dir / fname)
            first_chunk = True
            n_snip_total = 0
            freq_min = freq_max = None
            chunk_bar = None

            for snips, freqs, fch1, foff, chunk_start_snip, n_snip_total_this_file in chunk_iter:
                if first_chunk:
                    if fch1_ref is None:
                        fch1_ref, foff_ref = fch1, foff
                    elif not (np.isclose(fch1, fch1_ref) and np.isclose(foff, foff_ref)):
                        raise ValueError(
                            f"Frequency alignment mismatch in {fname}: "
                            f"fch1={fch1} vs ref={fch1_ref}, foff={foff} vs ref={foff_ref}"
                        )
                    n_snip_total = n_snip_total_this_file
                    chunk_bar = tqdm(
                        total=n_snip_total,
                        desc=f"   {tag} chunks",
                        unit="snip",
                        position=1,
                        leave=False,
                    )
                    first_chunk = False

                n_chunk = len(snips)
                # Append in BATCH_SNIPPETS sub-batches so HDF5 chunk-writes stay efficient
                for b0 in range(0, n_chunk, BATCH_SNIPPETS):
                    b1 = min(b0 + BATCH_SNIPPETS, n_chunk)
                    abs_start = chunk_start_snip + b0
                    abs_end   = chunk_start_snip + b1
                    append(snippets_dset, snips[b0:b1].astype(np.float32))
                    append(pos,  np.full(b1 - b0, cadence_pos_val, dtype="i1"))
                    append(fidx, np.full(b1 - b0, file_i,          dtype="i1"))
                    append(sidx, np.arange(abs_start, abs_end,     dtype="i4"))
                    append(freq, freqs[b0:b1])

                fmin, fmax = float(freqs.min()), float(freqs.max())
                freq_min = fmin if freq_min is None else min(freq_min, fmin)
                freq_max = fmax if freq_max is None else max(freq_max, fmax)
                chunk_bar.update(n_chunk)

                del snips, freqs

            if chunk_bar is not None:
                chunk_bar.close()
            tqdm.write(f"   snippets={n_snip_total:,}, freq range {freq_min:.2f}–{freq_max:.2f} MHz")
            tqdm.write(f"   file total {time.time()-t_file:.1f}s")
        files_bar.close()

        out.attrs["cadence_tags"]     = [tag for _, tag, _ in CADENCE]
        out.attrs["cadence_files"]    = [fname for fname, _, _ in CADENCE]
        out.attrs["fch1"]             = fch1_ref
        out.attrs["foff"]             = foff_ref
        out.attrs["snippet_native_w"] = SNIPPET_NATIVE_W
        out.attrs["downsample_f"]     = DOWNSAMPLE_F

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min")
    with h5py.File(out_path, "r") as out:
        n, k = out["snippets"].shape
        unique, counts = np.unique(out["cadence_pos"][:], return_counts=True)
        print(f"  cadence_features.h5: {n} snippets × {k} channels (raw)")
        print(f"  cadence_pos counts: {dict(zip(unique.tolist(), counts.tolist()))}")


if __name__ == "__main__":
    main()
