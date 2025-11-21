import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


def build_backbone(embedding_size=512, pretrained=True):
    backbone = models.resnet18(pretrained=pretrained)
    in_feats = backbone.fc.in_features
    backbone.fc = nn.Linear(in_feats, embedding_size)
    return backbone


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(path, map_location=device)
    class_to_idx = ckpt.get('class_to_idx', None)
    # infer sizes
    # try to reconstruct backbone
    backbone = build_backbone(embedding_size=ckpt['backbone_state']['fc.weight'].shape[0], pretrained=False)
    backbone.load_state_dict(ckpt['backbone_state'])
    classifier = None
    if ckpt.get('classifier_state') is not None:
        out_dim = ckpt['classifier_state']['weight'].shape[0]
        in_dim = ckpt['classifier_state']['weight'].shape[1]
        classifier = nn.Linear(in_dim, out_dim)
        classifier.load_state_dict(ckpt['classifier_state'])
    backbone.eval().to(device)
    if classifier is not None:
        classifier.eval().to(device)
    return backbone, classifier, class_to_idx, device


_DEFAULT_PREPROCESS = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


def embed_image(path, backbone, device=None, preprocess=None, l2_normalize=True):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if preprocess is None:
        preprocess = _DEFAULT_PREPROCESS
    img = Image.open(path).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = backbone(x)
    emb = emb.cpu().numpy().reshape(-1)
    if l2_normalize:
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
    return emb


def embed_batch(batch_tensors, backbone, device=None, l2_normalize=True):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        emb = backbone(batch_tensors.to(device)).cpu().numpy()
    if l2_normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms==0]=1.0
        emb = emb / norms
    return emb
