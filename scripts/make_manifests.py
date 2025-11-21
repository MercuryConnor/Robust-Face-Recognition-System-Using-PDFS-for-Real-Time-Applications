import os, csv
from glob import glob
from sklearn.model_selection import train_test_split

def gather(root):
	classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
	records = []
	for idx, cls in enumerate(classes):
		imgs = glob(os.path.join(root, cls, '*'))
		for im in imgs:
			# occlusion detection heuristic: filename contains 'mask' or similar
			occluded = 1 if 'mask' in os.path.basename(im).lower() else 0
			records.append((im, idx, occluded))
	return records, classes

def write_csv(rows, outpath):
	with open(outpath, 'w', newline='') as f:
		w = csv.writer(f)
		w.writerow(['path','label','occluded'])
		w.writerows(rows)

if __name__ == "__main__":
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('--processed', default='data/processed')
	p.add_argument('--out', default='data/manifests')
	args = p.parse_args()

	recs, classes = gather(args.processed)
	train, test = train_test_split(recs, test_size=0.2, stratify=[r[1] for r in recs])
	val, test = train_test_split(test, test_size=0.5, stratify=[r[1] for r in test])

	os.makedirs(args.out, exist_ok=True)
	write_csv(train, os.path.join(args.out,'train.csv'))
	write_csv(val, os.path.join(args.out,'val.csv'))
	write_csv(test, os.path.join(args.out,'test.csv'))
