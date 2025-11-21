"""Evaluation utilities for PDFS project.

Features:
- Compute embeddings for a list of images (batched) and save to CSV.
- Evaluate verification pairs CSV with columns (img1,img2,label).
- Support images specified as absolute paths or relative to a base `--images-dir`.
"""
import argparse
import os
import csv
import math
import sys
from typing import List, Dict
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from models.inference import load_checkpoint, embed_batch


def read_pairs(pairs_csv):
    pairs = []
    imgs = set()
    with open(pairs_csv, 'r', newline='') as f:
        r = csv.reader(f)
        # allow optional header
        first = next(r)
        try:
            int(first[2])
            # first row is data
            row = first
            img1, img2, label = row[0], row[1], int(row[2])
            pairs.append((img1, img2, label)); imgs.add(img1); imgs.add(img2)
        except Exception:
            # header present, continue normally
            pass
        for row in r:
            if len(row) < 3:
                continue
            img1, img2, label = row[0], row[1], int(row[2])
            pairs.append((img1, img2, label))
            imgs.add(img1); imgs.add(img2)
    return pairs, sorted(list(imgs))


class ImageListDataset(Dataset):
    def __init__(self, image_paths: List[str], images_dir: str = None, transform=None):
        self.image_paths = image_paths
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        if self.images_dir and not os.path.isabs(p):
            p_full = os.path.join(self.images_dir, p)
        else:
            p_full = p
        img = Image.open(p_full).convert('RGB')
        if self.transform:
            t = self.transform(img)
        else:
            t = transforms.ToTensor()(img)
        return p, t


def compute_embeddings_for_list(img_list: List[str], backbone, device, out_csv: str = None, images_dir: str = None, batch_size: int = 32, preprocess=None):
    print(f"Computing embeddings for {len(img_list)} images (batch_size={batch_size})")
    if preprocess is None:
        preprocess = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    ds = ImageListDataset(img_list, images_dir, transform=preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=lambda batch: batch)

    embeddings: Dict[str, np.ndarray] = {}
    for batch in loader:
        paths, tensors = zip(*batch)
        batch_t = torch.stack(tensors, dim=0)
        embs = embed_batch(batch_t, backbone, device=device)
        for p, e in zip(paths, embs):
            embeddings[p] = e

    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            dim = next(iter(embeddings.values())).shape[0]
            w.writerow(['image'] + [f'e{i}' for i in range(dim)])
            for p, e in embeddings.items():
                w.writerow([p] + e.tolist())
    return embeddings


def eval_pairs(pairs, embeddings):
    y_true = []
    y_score = []
    missing = 0
    for a,b,label in pairs:
        if a not in embeddings or b not in embeddings:
            missing += 1
            continue
        ea = embeddings[a]
        eb = embeddings[b]
        sim = float(np.dot(ea, eb) / (np.linalg.norm(ea)*np.linalg.norm(eb) + 1e-8))
        y_true.append(label)
        y_score.append(sim)
    if len(y_true) == 0:
        raise RuntimeError('No valid pairs found (missing embeddings for all pairs)')
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = None
    # simple accuracy at best threshold
    best_acc = 0.0
    best_th = 0.0
    for th in np.linspace(-1,1,201):
        preds = [1 if s>=th else 0 for s in y_score]
        acc = sum([p==t for p,t in zip(preds,y_true)])/len(y_true)
        if acc > best_acc:
            best_acc = acc; best_th = th
    return {'auc': auc, 'best_acc': best_acc, 'best_th': best_th, 'missing_pairs': missing}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Checkpoint path')
    p.add_argument('--pairs', help='CSV with img1,img2,label for verification')
    p.add_argument('--images-list', help='Text file with one image path per line (or use --pairs)')
    p.add_argument('--images-dir', help='Base dir to resolve relative image paths')
    p.add_argument('--out-embeddings', help='CSV to write embeddings for images')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    backbone, classifier, class_to_idx, device = load_checkpoint(args.model, device=args.device)

    # decide which images to compute
    image_paths = []
    pairs = None
    if args.pairs:
        pairs, imgs = read_pairs(args.pairs)
        image_paths = imgs
    elif args.images_list:
        with open(args.images_list, 'r') as f:
            image_paths = [l.strip() for l in f.readlines() if l.strip()]
    else:
        print('Provide --pairs or --images-list'); sys.exit(1)

    embeddings = compute_embeddings_for_list(image_paths, backbone, device, out_csv=args.out_embeddings, images_dir=args.images_dir, batch_size=args.batch_size)

    if pairs:
        stats = eval_pairs(pairs, embeddings)
        print(f"Results: AUC={stats['auc']}, best_acc={stats['best_acc']}, best_th={stats['best_th']}, missing_pairs={stats.get('missing_pairs',0)}")


if __name__ == '__main__':
    main()
# Evaluate script
