# Himawari ML (Satellite CV + PCA + MLOps)

Automated machine-learning pipeline that ingests Japanese geostationary satellite imagery (Himawari), preprocesses frames, trains models (segmentation / nowcasting / autoencoding), and publishes continuously updated results.

## What you get out of the box
- **Professional repo layout** (src package, docs, workflows)
- **Working ingestion stub** (fetches a sample image URL you provide)
- **Preprocessing utilities** (resize/normalize, simple pseudo-label baseline)
- **Training stubs** (PyTorch placeholders)
- **Representation learning + PCA script** (runs on embeddings)
- **GitHub Actions**: ingest hourly, train weekly, publish docs to Pages
- **Local dev**: `make setup` + `make ingest` + `make report`

> Note: The ingestion script is intentionally configurable so you can start with PNG sample imagery and later swap in a richer data source without refactoring your whole pipeline.

---

## Quickstart (local)

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
# .venv\Scripts\activate  # windows powershell
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment
Copy `.env.example` to `.env` and set `HIMAWARI_SAMPLE_URL` (a direct PNG/JPG URL).

```bash
cp .env.example .env
```

### 4) Run ingestion + preprocessing + report
```bash
python -m himawari_ml.ingest.fetch_latest
python -m himawari_ml.preprocess.build_dataset
python -m himawari_ml.evaluation.visualize_predictions
```

Artifacts will appear under `outputs/`.

---

## Repo structure

- `src/himawari_ml/ingest/` – data fetching + validation
- `src/himawari_ml/preprocess/` – image cleaning, pseudo-labeling, dataset builder
- `src/himawari_ml/models/` – model definitions (PyTorch)
- `src/himawari_ml/train/` – training loops + checkpoints
- `src/himawari_ml/representation/` – embeddings + PCA + clustering
- `src/himawari_ml/evaluation/` – metrics + visualization
- `outputs/` – figures/metrics/reports (gitignored except small outputs)
- `docs/` – GitHub Pages site (auto-built)

---

## Makefile commands (optional convenience)
```bash
make setup
make ingest
make dataset
make report
```

---

## License
MIT (see `LICENSE`).
