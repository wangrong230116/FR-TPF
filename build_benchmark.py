"""Reconstruct the 2,904-group three-phase VSB benchmark without copying waveforms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("data/raw/vsb-power-line-fault-detection"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--expected-groups", type=int, default=2904)
    args = parser.parse_args()

    metadata_path = args.source / "metadata_train.csv"
    waveform_path = args.source / "train.parquet"
    if not metadata_path.is_file() or not waveform_path.is_file():
        raise FileNotFoundError("Follow DATA_INSTRUCTIONS.md and provide both source files")

    meta = pd.read_csv(metadata_path)
    required = {"signal_id", "id_measurement", "phase", "target"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(meta.columns))}")
    if meta.signal_id.duplicated().any():
        raise ValueError("signal_id must be unique")

    phase_count = meta.groupby("id_measurement").phase.nunique()
    if not phase_count.eq(3).all():
        raise ValueError("Every measurement must contain exactly three phases")
    grouped = meta.sort_values(["id_measurement", "phase"]).groupby("id_measurement")
    rows = []
    for measurement, block in grouped:
        if block.phase.tolist() != [0, 1, 2]:
            raise ValueError(f"Invalid phase order for measurement {measurement}")
        ids = block.signal_id.astype(int).tolist()
        targets = block.target.astype(int).tolist()
        rows.append({"id_measurement": int(measurement),
                     "signal_id_phase0": ids[0], "signal_id_phase1": ids[1],
                     "signal_id_phase2": ids[2], "target_phase0": targets[0],
                     "target_phase1": targets[1], "target_phase2": targets[2],
                     "target_any": int(max(targets))})
    manifest = pd.DataFrame(rows)
    if len(manifest) != args.expected_groups:
        raise ValueError(f"Expected {args.expected_groups} groups, found {len(manifest)}")

    parquet = pq.ParquetFile(waveform_path)
    if parquet.metadata.num_rows != 800_000:
        raise ValueError(f"Expected 800000 samples/signal, found {parquet.metadata.num_rows}")
    if set(parquet.schema.names) != set(meta.signal_id.astype(str)):
        raise ValueError("Parquet columns do not match metadata signal identifiers")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output / "manifest.csv", index=False)
    card = {"name": "FR-TPF reconstructed VSB benchmark",
            "source": "VSB Power Line Fault Detection (Kaggle)",
            "groups": int(len(manifest)), "signals": int(len(meta)),
            "samples_per_phase": int(parquet.metadata.num_rows),
            "positive_groups": int(manifest.target_any.sum()),
            "metadata_sha256": sha256(metadata_path),
            "raw_data_redistributed": False}
    (args.output / "dataset_card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()

