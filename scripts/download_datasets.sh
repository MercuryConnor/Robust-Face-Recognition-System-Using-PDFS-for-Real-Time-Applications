#!/bin/bash
set -e

mkdir -p data/raw

# WIDER FACE (detection) - download URLs may require manual registration
echo "Download WIDER FACE manually and place in data/raw/widerface/"

# VGGFace2 (pretrained / pretraining) - these are huge, suggest using pre-trained weights if available
echo "Download VGGFace2 following instructions: https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/"

# MaskedFace-Net (available on GitHub/Zenodo)
mkdir -p data/raw/maskedface
wget -O data/raw/maskedface/maskedface-net.zip "https://github.com/..." || echo "Please download MaskedFace-Net manually if wget fails."

# RMFRD (Real Masked Face Recognition Dataset)
# Provide link or instruction to download
echo "Place RMFRD dataset under data/raw/rmfrd/"

# IJB-B / IJB-C - provide instructions for download (registration)
echo "Download IJB-B/IJB-C via IARPA Janus license; place in data/raw/ijb/"

echo "Datasets: Please follow the printed instructions to place datasets in data/raw/"
