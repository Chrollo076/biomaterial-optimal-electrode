# Biomaterial Optimal Electrode

Machine-learning and optimization framework for identifying optimal biomaterial and electrode-interface designs for EEG, ECG, and EMG applications.

## Overview

This project develops a literature-grounded dataset and machine-learning workflow for evaluating biomaterial-based electrodes across electrical, mechanical, biological, adhesion, surface, stability, and electrophysiological performance.

The framework treats electrode design as a context-dependent multi-objective optimization problem rather than attempting to identify one universally best material.

## Applications

- EEG — Electroencephalography
- ECG — Electrocardiography
- EMG — Electromyography

## Project Structure

```text
Data/
├── Biomaterial_Electrode_Literature_Dataset_REVISED.xlsx
└── Biomaterial_Electrode_Research_Report_REVISED.pdf

biomaterial_optimal_electrode/
biomaterial_electrode_ml/

app.py
streamlit_app.py
requirements.txt
run_app.bat
README.md
