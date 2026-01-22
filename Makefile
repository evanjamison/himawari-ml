.PHONY: setup ingest dataset report pca

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

ingest:
	python -m himawari_ml.ingest.fetch_latest

dataset:
	python -m himawari_ml.preprocess.build_dataset

report:
	python -m himawari_ml.evaluation.visualize_predictions

pca:
	python -m himawari_ml.representation.pca_analysis
