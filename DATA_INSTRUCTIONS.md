# Data preparation

The original waveform archive is not redistributed by this repository.

1. Open the official [VSB Power Line Fault Detection competition page](https://www.kaggle.com/c/vsb-power-line-fault-detection).
2. Sign in to Kaggle and accept the competition rules.
3. Download `train.parquet` and `metadata_train.csv`.
4. Place both files under `data/raw/vsb-power-line-fault-detection/`:

   ```text
   data/raw/vsb-power-line-fault-detection/
   |-- train.parquet
   `-- metadata_train.csv
   ```

5. Reconstruct the measurement-level benchmark:

   ```bash
   python build_benchmark.py
   ```

6. The command validates the 800,000-sample records, groups the three phases by
   `id_measurement`, and writes `data/benchmark/manifest.csv`. With the complete
   official training archive, the expected result is 2,904 three-phase groups.

Then run `python extract_frtpf.py` to create the label-free FR-TPF feature table.
The raw Kaggle waveform values remain in `data/`, which is excluded by `.gitignore`.

