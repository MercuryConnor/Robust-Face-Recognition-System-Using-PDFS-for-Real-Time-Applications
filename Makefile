 .PHONY: all download preprocess train evaluate gradcam tsne prune quantize benchmark clean

all: download preprocess train evaluate

download:
	bash scripts/download_datasets.sh

preprocess:
	python scripts/preprocess.py --data-root data/raw --out data/processed --size 112

train:
	python scripts/train.py --data-root data/processed --out models/checkpoints/best.pth --epochs 5 --batch-size 64

evaluate:
	python scripts/evaluate.py --model models/checkpoints/best.pth --pairs data/manifests/test.csv

gradcam:
	python scripts/gradcam.py --model models/checkpoints/best.pth --samples demo_images/ --out paper/figures/gradcam

tsne:
	python scripts/tsne_plot.py --model models/checkpoints/best.pth --manifest data/manifests/tsne_pairs.csv --out paper/figures/tsne.png

prune:
	python scripts/prune_and_search.py --model models/checkpoints/best.pth --budget 0.7 --out models/checkpoints/pruned.pth

quantize:
	python scripts/quantize_export.py --model models/checkpoints/pruned.pth --out models/exports/pdfs_trt.engine

benchmark:
	python scripts/benchmark_fps.py --model models/checkpoints/best.pth

clean:
	rm -rf models/checkpoints/* models/exports/*
