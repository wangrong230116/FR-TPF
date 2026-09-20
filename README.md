# FR-TPF

Reference implementation for **Measurement-Oriented Full-Record Transient
Representation for Three-Phase High-Frequency Voltage Fault Screening**.

FR-TPF preserves full-record transient evidence in each phase, then adds
three-phase aggregation and pulse-coincidence descriptors for measurement-level
screening. This repository contains code and feature definitions only. It does
not redistribute the VSB competition waveforms.

## Repository layout

```text
FR-TPF/
|-- README.md
|-- requirements.txt
|-- build_benchmark.py
|-- extract_frtpf.py
|-- train_nested_cv.py
|-- evaluate.py
|-- figures/
|-- configs/
`-- DATA_INSTRUCTIONS.md
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python build_benchmark.py
python extract_frtpf.py
python train_nested_cv.py
python evaluate.py
```

Read [DATA_INSTRUCTIONS.md](DATA_INSTRUCTIONS.md) before running the pipeline.
All splits are formed at `id_measurement` level to prevent three-phase leakage.
Threshold selection is performed only on training-side out-of-fold predictions.

## Data availability

The source waveforms are available from the VSB Power Line Fault Detection
competition subject to the applicable Kaggle competition rules. Users must
obtain the original waveform data directly from the official competition page
after accepting its terms. Raw competition data are not redistributed here.

## Status

This initial public release provides the benchmark constructor, FR-TPF feature
extractor, repeated nested group-CV trainer, evaluation script, and configuration.
Additional figure-generation utilities will be added with the archival release.

