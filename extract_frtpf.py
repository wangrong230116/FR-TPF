"""Extract label-free full-record three-phase transient descriptors."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import find_peaks


def q(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if len(values) else 0.0


def phase_features(signal: np.ndarray, phase: int) -> tuple[dict[str, float], np.ndarray]:
    x = signal.astype(np.float32)
    dx = np.abs(np.diff(signal.astype(np.int16))).astype(np.float32)
    peaks, props = find_peaks(dx, height=3, distance=51)
    height = props["peak_heights"]
    prefix = f"phase{phase}_"
    out = {
        prefix + "mean": float(x.mean()),
        prefix + "std": float(x.std()),
        prefix + "min": float(x.min()),
        prefix + "max": float(x.max()),
        prefix + "dx_mean": float(dx.mean()),
        prefix + "dx_std": float(dx.std()),
        prefix + "dx_max": float(dx.max()),
        prefix + "dx_q99": q(dx, .99),
        prefix + "dx_q999": q(dx, .999),
        prefix + "peak_count": float(len(peaks)),
        prefix + "peak_mean": float(height.mean()) if len(height) else 0.0,
        prefix + "peak_std": float(height.std()) if len(height) else 0.0,
        prefix + "peak_q90": q(height, .9),
        prefix + "peak_q99": q(height, .99),
        prefix + "peak_max": float(height.max()) if len(height) else 0.0,
    }
    for threshold in (5, 8, 12, 20, 30, 50):
        out[prefix + f"dx_count_ge{threshold}"] = float(np.count_nonzero(dx >= threshold))
        out[prefix + f"peak_count_ge{threshold}"] = float(np.count_nonzero(height >= threshold))
    if len(height):
        median = float(np.median(height))
        mad = float(np.median(np.abs(height - median)))
        out[prefix + "peak_median"] = median
        out[prefix + "peak_mad"] = mad
        for scale in (3, 6):
            cutoff = median + scale * max(mad, 1)
            out[prefix + f"peak_count_adaptive{scale}"] = float(np.count_nonzero(height >= cutoff))
    else:
        out[prefix + "peak_median"] = 0.0
        out[prefix + "peak_mad"] = 0.0
        out[prefix + "peak_count_adaptive3"] = 0.0
        out[prefix + "peak_count_adaptive6"] = 0.0

    quarter = np.minimum(peaks * 4 // len(dx), 3)
    for threshold in (8, 12, 20):
        counts = np.bincount(quarter[height >= threshold], minlength=4)
        for region in range(4):
            out[prefix + f"peak_ge{threshold}_quarter{region}"] = float(counts[region])
        out[prefix + f"peak_ge{threshold}_even_quarters"] = float(counts[0] + counts[2])
        out[prefix + f"peak_ge{threshold}_odd_quarters"] = float(counts[1] + counts[3])
    for threshold in (12, 20):
        selected = height[height >= threshold]
        out[prefix + f"peak_ge{threshold}_mean_height"] = float(selected.mean()) if len(selected) else 0.0
        out[prefix + f"peak_ge{threshold}_std_height"] = float(selected.std()) if len(selected) else 0.0

    strong = peaks[height >= 12]
    chosen = strong[(strong >= 1) & (strong + 2 < len(x))]
    if len(chosen):
        sharpness = np.abs(x[chosen] - (x[chosen - 1] + x[chosen + 2]) / 2)
        out[prefix + "strong_sharpness_mean"] = float(sharpness.mean())
        out[prefix + "strong_sharpness_q90"] = q(sharpness, .9)
    else:
        out[prefix + "strong_sharpness_mean"] = 0.0
        out[prefix + "strong_sharpness_q90"] = 0.0
    return out, np.unique(peaks[height >= 20] // 200)


def group_features(wave: np.ndarray) -> dict[str, float]:
    out, strong = {}, []
    for phase in range(3):
        values, bins = phase_features(wave[phase], phase)
        out.update(values)
        strong.append(bins)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        out[f"phase_coincidence_{left}{right}"] = float(
            len(np.intersect1d(strong[left], strong[right], assume_unique=True)))
    for name in ("peak_count", "peak_count_ge8", "peak_count_ge12", "peak_count_ge20",
                 "peak_count_adaptive3", "peak_q99", "dx_q999",
                 "strong_sharpness_mean"):
        values = np.array([out[f"phase{phase}_{name}"] for phase in range(3)])
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
    if result.shape[1] - 1 != 192:
        raise ValueError(f"Expected 192 FR-TPF descriptors, found {result.shape[1] - 1}")
    if not np.isfinite(result.drop(columns="id_measurement").to_numpy()).all():
        raise ValueError("Non-finite feature value detected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"wrote {result.shape[0]} groups x {result.shape[1] - 1} features")


if __name__ == "__main__":
    main()
