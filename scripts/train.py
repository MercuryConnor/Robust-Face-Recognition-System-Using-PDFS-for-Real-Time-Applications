"""Simple training script for PDFS project.

This script provides a minimal, runnable training loop using torchvision
backbones and an ImageFolder dataset layout. It trains a backbone to produce
embeddings and a linear classifier on top (softmax). This is a pragmatic
implementation to get the project running end-to-end; the research losses
(ArcFace, Triplet, purity losses) can be added later.
"""

import argparse
import os
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models


def build_model(embedding_size=512, num_classes=None, pretrained=True):
	backbone = models.resnet18(pretrained=pretrained)
	in_feats = backbone.fc.in_features
	# make embedding layer
	backbone.fc = nn.Linear(in_feats, embedding_size)
	if num_classes is not None:
		# simple classifier on top of embedding
		classifier = nn.Linear(embedding_size, num_classes)
	else:
		classifier = None
	return backbone, classifier


def save_checkpoint(out_path, backbone, classifier, class_to_idx):
	ckpt = {
		'backbone_state': backbone.state_dict(),
		'classifier_state': classifier.state_dict() if classifier is not None else None,
		'class_to_idx': class_to_idx,
	}
	torch.save(ckpt, out_path)


def train(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	transform = transforms.Compose([
		transforms.Resize((args.size, args.size)),
		transforms.ToTensor(),
		transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
	])

	dataset = datasets.ImageFolder(args.data_root, transform=transform)
	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

	num_classes = len(dataset.classes)
	backbone, classifier = build_model(args.embedding_size, num_classes, pretrained=not args.no_pretrain)
	backbone = backbone.to(device)
	if classifier is not None:
		classifier = classifier.to(device)

	criterion = nn.CrossEntropyLoss()
	params = list(backbone.parameters()) + (list(classifier.parameters()) if classifier is not None else [])
	optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

	backbone.train()
	if classifier is not None:
		classifier.train()

	for epoch in range(args.epochs):
		t0 = time.time()
		running_loss = 0.0
		for i, (imgs, labels) in enumerate(dataloader):
			imgs = imgs.to(device); labels = labels.to(device)
			optimizer.zero_grad()
			embeddings = backbone(imgs)
			if classifier is not None:
				logits = classifier(embeddings)
				loss = criterion(logits, labels)
			else:
				# if no classifier provided just compute a dummy loss (not useful)
				loss = embeddings.norm(2)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			if (i+1) % 20 == 0:
				print(f"Epoch {epoch+1}/{args.epochs} step {i+1}/{len(dataloader)} loss={running_loss/20:.4f}")
				running_loss = 0.0
		print(f"Epoch {epoch+1} done in {time.time()-t0:.1f}s")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	save_checkpoint(args.out, backbone, classifier, dataset.class_to_idx)
	print(f"Saved checkpoint to {args.out}")


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument('--data-root', required=True, help='ImageFolder root of training images')
	p.add_argument('--out', default='models/checkpoints/latest.pth', help='Checkpoint output path')
	p.add_argument('--epochs', type=int, default=2)
	p.add_argument('--batch-size', type=int, default=32)
	p.add_argument('--lr', type=float, default=0.01)
	p.add_argument('--embedding-size', type=int, default=512)
	p.add_argument('--size', type=int, default=112)
	p.add_argument('--no-pretrain', action='store_true', help='Disable pretrained backbone')
	return p.parse_args()


if __name__ == '__main__':
	args = parse_args()
	train(args)

