"""Simple pruning utility for the embedding dimension.

This script performs a very simple channel/embedding-dimension pruning by
selecting the top-K embedding dimensions (by L1 norm of the classifier
weights) and reconstructing a smaller backbone.fc and classifier layers.
This is not a full research-quality pruning pipeline but helps test the
pruning/export flow in the repo.
"""
import argparse
import torch
import torch.nn as nn
import os
from models.inference import load_checkpoint


def prune_embedding_dims(ckpt_path, keep_ratio, out_path):
    backbone, classifier, class_to_idx, device = load_checkpoint(ckpt_path)
    # load raw checkpoint to access states
    raw = torch.load(ckpt_path, map_location='cpu')
    bstate = raw['backbone_state']
    cstate = raw.get('classifier_state')

    emb_dim = bstate['fc.weight'].shape[0]
    keep = max(1, int(emb_dim * keep_ratio))
    print(f"Pruning embedding dim {emb_dim} -> {keep}")

    if cstate is None:
        # prune backbone.fc only
        w = bstate['fc.weight']
        b = bstate['fc.bias']
        # prune by largest absolute mean across rows
        scores = w.abs().mean(dim=1)
        _, idx = torch.topk(scores, keep)
        new_w = w[idx,:]
        new_b = b[idx]
        new_state = bstate.copy()
        new_state['fc.weight'] = new_w
        new_state['fc.bias'] = new_b
        raw['backbone_state'] = new_state
    else:
        # determine important dims from classifier weight magnitude
        cw = cstate['weight']  # shape (num_classes, emb_dim)
        scores = cw.abs().sum(dim=0)
        _, idx = torch.topk(scores, keep)
        idx = idx.sort()[0]

        # prune backbone fc
        bw = bstate['fc.weight']
        bb = bstate['fc.bias']
        new_bw = bw[idx,:]
        new_bb = bb[idx]
        new_bstate = bstate.copy()
        new_bstate['fc.weight'] = new_bw
        new_bstate['fc.bias'] = new_bb

        # prune classifier (keep columns corresponding to selected dims)
        new_cw = cw[:, idx]
        new_cb = cstate['bias']
        new_cstate = cstate.copy()
        new_cstate['weight'] = new_cw
        new_cstate['bias'] = new_cb

        raw['backbone_state'] = new_bstate
        raw['classifier_state'] = new_cstate

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(raw, out_path)
    print('Saved pruned checkpoint to', out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--budget', type=float, default=0.7, help='fraction of embedding dims to keep')
    args = p.parse_args()
    prune_embedding_dims(args.model, args.budget, args.out)


if __name__ == '__main__':
    main()
# Prune and search script
