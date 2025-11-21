import os, argparse, cv2, csv
import numpy as np
from pathlib import Path
# from retinaface import RetinaFace  # if you install a retinaface package or use MTCNN

def align_and_crop(img_path, out_path, size=112):
	img = cv2.imread(str(img_path))
	# TODO: call your detection (RetinaFace or MTCNN). Here we use bounding box center
	# Placeholder: assume faces are centered; for production, integrate RetinaFace.
	h,w = img.shape[:2]
	cx, cy = w//2, h//2
	s = min(w,h)//1.5
	x1 = max(0,int(cx - s/2)); y1 = max(0,int(cy - s/2))
	x2 = min(w,int(cx + s/2)); y2 = min(h,int(cy + s/2))
	crop = img[y1:y2,x1:x2]
	crop = cv2.resize(crop,(size,size))
	cv2.imwrite(str(out_path), crop)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-root', required=True)
	parser.add_argument('--out', required=True)
	parser.add_argument('--size', type=int, default=112)
	args = parser.parse_args()

	os.makedirs(args.out, exist_ok=True)
	# Walk raw data and process - example for a simple layout
	for root, dirs, files in os.walk(args.data_root):
		for f in files:
			if f.lower().endswith(('jpg','png','jpeg')):
				rel = os.path.relpath(root, args.data_root)
				outdir = os.path.join(args.out, rel)
				os.makedirs(outdir, exist_ok=True)
				inpath = os.path.join(root, f)
				outpath = os.path.join(outdir, f)
				align_and_crop(inpath, outpath, size=args.size)

if __name__ == "__main__":
	main()
