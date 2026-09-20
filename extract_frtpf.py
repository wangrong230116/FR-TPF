"""Extract label-free full-record three-phase transient descriptors."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import find_peaks


def phase_features(signal: np.ndarray, phase: int) -> tuple[dict[str, float], np.ndarray]:
    x = signal.astype(np.float32)
    dx = np.abs(np.diff(signal.astype(np.int16))).astype(np.float32)
    peaks, props = find_peaks(dx, height=3, distance=51)
    height = props["peak_heights"]
    prefix = f"p{phase}_"
    out = {prefix + "mean": float(x.mean()), prefix + "std": float(x.std()),
           prefix + "dx_mean": float(dx.mean()), prefix + "dx_std": float(dx.std()),
           prefix + "dx_q99": float(np.quantile(dx, .99)),
           prefix + "dx_q999": float(np.quantile(dx, .999)),
           prefix + "peak_count": float(len(peaks)),
           prefix + "peak_max": float(height.max()) if len(height) else 0.0}
    median = float(np.median(height)) if len(height) else 0.0
    mad = float(np.median(np.abs(height - median))) if len(height) else 0.0
    for threshold in (5, 8, 12, 20, 30, 50):
        out[prefix + f"dx_ge_{threshold}"] = float(np.count_nonzero(dx >= threshold))
        out[prefix + f"peak_ge_{threshold}"] = float(np.count_nonzero(height >= threshold))
    out[prefix + "adaptive_3mad"] = float(np.count_nonzero(height >= median + 3 * max(mad, 1)))
    out[prefix + "adaptive_6mad"] = float(np.count_nonzero(height >= median + 6 * max(mad, 1)))
    quarter = np.minimum(peaks * 4 // len(dx), 3)
    for threshold in (8, 12, 20):
        counts = np.bincount(quarter[height >= threshold], minlength=4)
        for region in range(4):
            out[prefix + f"peak_ge_{threshold}_q{region}"] = float(counts[region])
    return out, np.unique(peaks[height >= 20] // 200)


def group_features(wave: np.ndarray) -> dict[str, float]:
    out, strong = {}, []
    for phase in range(3):
        values, bins = phase_features(wave[phase], phase)
        out.update(values)
        strong.append(bins)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        out[f"coincidence_{left}{right}"] = float(
            len(np.intersect1d(strong[left], strong[right], assume_unique=True)))
    for name in ("peak_count", "adaptive_3mad", "adaptive_6mad",
                 "dx_q99", "dx_q999"):
        values = np.array([out[f"p{phase}_{name}"] for phase in range(3)])
        out[f"three_phase_{name}_mean"] = float(values.mean())
        out[f"three_phase_{name}_max"] = float(values.max())
        out[f"three_phase_{name}_range"] = float(np.ptp(values))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark/manifest.csv"))
    parser.add_argument("--waveforms", type=Path,
                        default=Path("data/raw/vsb-power-line-fault-detection/train.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/frtpf_features.parquet"))
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    parquet = pq.ParquetFile(args.waveforms)
    records = []
    for index, row in enumerate(manifest.itertuples(index=False), 1):
        ids = [str(getattr(row, f"signal_id_phase{phase}")) for phase in range(3)]
        table = parquet.read(columns=ids)
        wave = np.stack([table.column(i).to_numpy(zero_copy_only=False)
                         for i in range(3)])
        records.append({"id_measurement": row.id_measurement, **group_features(wave)})
        if index % 100 == 0:
            print(f"processed {index}/{len(manifest)} groups", flush=True)
    result = pd.DataFrame(records)
    if not np.isfinite(result.drop(columns="id_measurement").to_numpy()).all():
        raise ValueError("Non-finite feature value detected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"wrote {result.shape[0]} groups x {result.shape[1] - 1} features")


if __name__ == "__main__":
    main()

