"""Basic Grad-CAM utility for PyTorch models.

This script computes Grad-CAM for a given image and model's target layer
and writes an overlay image to disk. It uses the final ResNet layer by default.
"""
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from models.inference import load_checkpoint


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        score = out[0, class_idx]
        score.backward(retain_graph=True)
        grads = self.gradients[0]
        acts = self.activations[0]
        weights = grads.mean(dim=(1,2), keepdim=True)
        cam = (weights * acts).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam_on_image(img_path, cam, out_path, alpha=0.5):
    img = cv2.imread(img_path)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    cv2.imwrite(out_path, overlay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--image', help='single image to process')
    p.add_argument('--samples', help='directory of images to process')
    p.add_argument('--out', required=True, help='output file or output directory')
    args = p.parse_args()

    backbone, classifier, class_to_idx, device = load_checkpoint(args.model)
    # use layer4 of resnet
    target_layer = backbone.layer4

    # preprocess image to tensor
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    from PIL import Image

    gcam = GradCAM(backbone, target_layer)

    if args.image:
        img = Image.open(args.image).convert('RGB')
        x = preprocess(img).unsqueeze(0).to(device)
        cam = gcam(x)
        overlay_cam_on_image(args.image, cam, args.out)
        print(f"Wrote Grad-CAM overlay to {args.out}")
    elif args.samples:
        import os
        os.makedirs(args.out, exist_ok=True)
        for fname in sorted(os.listdir(args.samples)):
            if not fname.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            inp = os.path.join(args.samples, fname)
            img = Image.open(inp).convert('RGB')
            x = preprocess(img).unsqueeze(0).to(device)
            cam = gcam(x)
            outp = os.path.join(args.out, fname)
            overlay_cam_on_image(inp, cam, outp)
            print(f"Wrote {outp}")
    else:
        print('Provide --image or --samples')


if __name__ == '__main__':
    main()
# GradCAM script
