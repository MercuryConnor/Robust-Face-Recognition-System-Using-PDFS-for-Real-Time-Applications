"""Simple FPS benchmark script.

Loads a TorchScript or checkpoint model and runs inference repeatedly on a sample
image (or random tensor) to report average FPS.
"""
import argparse
import time
import numpy as np
import torch
from PIL import Image
from models.inference import load_checkpoint, embed_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--image', help='Sample image to run (optional)')
    p.add_argument('--iters', type=int, default=500)
    args = p.parse_args()

    backbone, classifier, class_to_idx, device = load_checkpoint(args.model)

    # warmup
    if args.image:
        _ = embed_image(args.image, backbone, device=device)
    else:
        dummy = torch.randn(1,3,112,112).to(device)
        with torch.no_grad():
            _ = backbone(dummy)

    # benchmark
    t0 = time.time()
    for i in range(args.iters):
        if args.image:
            _ = embed_image(args.image, backbone, device=device)
        else:
            dummy = torch.randn(1,3,112,112).to(device)
            with torch.no_grad():
                _ = backbone(dummy)
    dt = time.time() - t0
    fps = args.iters / dt if dt>0 else float('inf')
    print(f"Ran {args.iters} iters in {dt:.2f}s -> {fps:.2f} FPS")


if __name__ == '__main__':
    main()
# Benchmark FPS script
