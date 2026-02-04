# Himawari-ML: Cloud Segmentation and Temporal Forecasting from Geostationary Satellite Imagery

## Overview

**Himawari-ML** is an end-to-end machine learning pipeline for **cloud segmentation and short-term temporal forecasting** using imagery from the **Himawari-8/9 geostationary weather satellites** operated by the Japan Meteorological Agency (JMA).

The project is structured as a **multi-phase system** that progresses from:
1. deterministic image processing,
2. learned spatial segmentation (U-Net),
3. to **temporal sequence modeling (ConvLSTM)** for forecasting cloud evolution.

All stages are fully automated using **GitHub Actions**, enabling daily ingestion, training, inference, evaluation, and artifact versioning.

---

## Scientific Motivation

Cloud systems are inherently **spatiotemporal phenomena**:
- they have structure (shape, texture, boundaries),
- and dynamics (motion, growth, decay).

Traditional threshold-based cloud masks are:
- brittle across lighting conditions,
- sensitive to sensor artifacts,
- unable to model temporal continuity.

This project investigates:
- how far **weakly-supervised segmentation** can go without human labels,
- whether learned spatial masks improve downstream temporal forecasting,
- and how **deep sequence models** compare to static frame-based approaches.

---

## Pipeline Structure

### Phase 1 — Data Ingestion
- Automated ingestion of Himawari PNG imagery
- Timestamp-based foldering (`YYYY-MM-DD`)
- Frame filtering to remove black / invalid images
- Reproducible daily rollups

---

### Phase 2 — Deterministic Cloud Mask Baseline

**Purpose:** establish a physically interpretable reference mask.

**Method:**
- Luminance + saturation thresholding
- Morphological opening / closing
- Disk mask to restrict the Earth region

**Outputs:**
- Binary cloud masks
- Per-frame cloud metrics:
  - cloud fraction
  - connected components
  - object area statistics

These masks serve as **teacher labels** for later models.

---

### Phase 3 — Learned Cloud Segmentation (U-Net)

Phase 3 explores two complementary U-Net training strategies.

#### Phase 3A — Metric-Driven U-Net (Weak Supervision)
- Inputs: raw satellite frames
- Targets: teacher-generated cloud masks
- Loss: Binary Cross Entropy (Dice monitored)
- Optional disk masking

**Goal:** replicate and smooth deterministic masks while improving spatial coherence.

#### Phase 3B — Pixel-Mask U-Net (Direct Segmentation)
- Inputs: raw satellite frames
- Targets: binary pixel masks
- Data augmentation + temporal train/validation split
- Produces probabilistic cloud maps

**Key observation:**  
Pixel-mask U-Net outputs are spatially cleaner and better suited for temporal modeling than metric-only supervision.

**Outputs:**
- Multi-threshold inference masks
- Probability maps
- Training curves and validation previews

---

## Phase 4 — Temporal Forecasting (ConvLSTM)

**Objective:** predict **future cloud masks** given a short history of past frames.

### Dataset Construction
- Sequences of length *T* (e.g. 6 frames)
- Targets: next-step cloud mask
- Inputs:
  - binary cloud masks (primary)
  - optional raw RGB channels (disabled by default)

### Model
- Convolutional LSTM (ConvLSTM)
- Scheduled sampling (teacher forcing → rollout)
- Trained entirely on CPU (GitHub Actions compatible)

### Outputs
- Training and validation loss curves
- Qualitative rollout previews
- Per-sequence forecast metrics

---

## Evaluation Strategy

Rather than focusing solely on single-frame accuracy, evaluation emphasizes **rollout behavior**:

- multi-step forecasts
- error accumulation over time
- visual coherence of cloud motion

Planned metrics include:
- per-step BCE / Dice
- cloud fraction drift
- structural consistency across rollouts

Artifacts include:
- rollout contact sheets
- summary CSVs
- per-epoch diagnostics

---

## Key Findings (So Far)

- Deterministic cloud masks are sufficient as **weak supervision**
- Pixel-level U-Net segmentation significantly improves spatial stability
- ConvLSTM models converge reliably but are computationally expensive on CPU
- Temporal modeling benefits more from **clean segmentation** than from raw RGB alone
<img width="906" height="508" alt="5e3d37abb53cdb1d08ee291db1a96237" src="https://github.com/user-attachments/assets/00821255-7604-418c-8362-c6c186f73011" />

---

## Reproducibility & Automation

- Fully automated daily pipeline via GitHub Actions
- Versioned artifacts:
  - models
  - metrics
  - visual diagnostics
- No manual labeling required
- CPU-only compatible

---

## Project Status

**Current stage:**
- ✅ Spatial segmentation complete
- ✅ Temporal modeling functional
- 🔄 Rollout evaluation & ablation studies in progress

**Next steps:**
- Compare raw-RGB vs mask-only ConvLSTM inputs
- Longer rollout horizons
- Motion-aware losses
- Event-scale analysis (e.g., typhoons, frontal systems)

---


## Why This Matters

This project demonstrates a **scalable and reproducible approach** to learning from satellite imagery when:
- labeled data is scarce,
- automation is critical,
- and temporal dynamics matter more than static accuracy.

It also serves as a testbed for:
- weak supervision,
- spatiotemporal deep learning,
- and MLOps-style scientific workflows.

