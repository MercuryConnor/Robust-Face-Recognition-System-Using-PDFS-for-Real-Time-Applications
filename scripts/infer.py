"""Inference helper: embed single image(s) using a checkpoint.

This helper ensures the project root is on `sys.path` so imports like
`models.inference` work when scripts are run directly.

Usage examples:
  python scripts/infer.py --model models/checkpoints/smoke.pth --image data/processed/a/image_0.ppm
  python scripts/infer.py --model models/checkpoints/smoke.pth --images-list images.txt --out embeddings.csv
"""
import sys
from pathlib import Path
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

import argparse
import os
import csv
import numpy as np
from models.inference import load_checkpoint, embed_image


def write_embedding_csv(out_path, image_path, emb):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image'] + [f'e{i}' for i in range(len(emb))])
        w.writerow([image_path] + emb.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--image', help='single image path to embed')
    p.add_argument('--images-list', help='text file with one image path per line')
    p.add_argument('--out', help='output CSV path to write embeddings')
    p.add_argument('--device', default=None)
    p.add_argument('--no-normalize', action='store_true', help='Do not L2-normalize embedding')
    args = p.parse_args()

    backbone, classifier, class_to_idx, device = load_checkpoint(args.model, device=args.device)

    if not args.image and not args.images_list:
        print('Provide --image or --images-list'); return

    if args.image:
        emb = embed_image(args.image, backbone, device=device, l2_normalize=not args.no_normalize)
        print('Embedding length:', emb.shape[0])
        print('First 8 values:', np.round(emb[:8], 6).tolist())
        if args.out:
            write_embedding_csv(args.out, args.image, emb)
            print('Wrote embedding to', args.out)
    else:
        # images list
        with open(args.images_list, 'r') as f:
            imgs = [l.strip() for l in f.readlines() if l.strip()]
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w', newline='') as f:
                w = csv.writer(f)
                # infer dim from first image
                first_emb = embed_image(imgs[0], backbone, device=device, l2_normalize=not args.no_normalize)
                dim = first_emb.shape[0]
                w.writerow(['image'] + [f'e{i}' for i in range(dim)])
                w.writerow([imgs[0]] + first_emb.tolist())
                for img in imgs[1:]:
                    e = embed_image(img, backbone, device=device, l2_normalize=not args.no_normalize)
                    w.writerow([img] + e.tolist())
            print('Wrote embeddings to', args.out)
        else:
            for img in imgs:
                e = embed_image(img, backbone, device=device, l2_normalize=not args.no_normalize)
                print(img, '->', e[:8].tolist())


if __name__ == '__main__':
    main()
