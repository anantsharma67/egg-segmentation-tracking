"""Leakage-aware 4:1 dev/eval split for the egg dataset.

The Roboflow export has no augmented copies, but many photos share a scene
(pink bowl, steel pot, keyboard, rug...). Splitting at random would put the
same background in both dev and eval, so we split by GROUP instead:
  1. union near-duplicate images (perceptual hash distance <= DUP_T)
  2. cluster remaining images by background colour (KMeans on HSV histogram)
  3. assign whole groups to dev/eval until eval is ~20% of images

Usage: python make_split.py --data data --out splits.json
"""
import argparse, glob, json, random, itertools
import numpy as np
import cv2
import imagehash
from PIL import Image
from sklearn.cluster import KMeans

DUP_T = 40          # phash (16x16) distance for "same photo"
N_SCENES = 24       # coarse scene clusters
SEED = 0


def hsv_hist(path):
    img = cv2.resize(cv2.imread(path), (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="splits.json")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.data}/*/*/*.jpg"))
    parent = {f: f for f in files}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    ph = {f: imagehash.phash(Image.open(f).convert("RGB"), hash_size=16) for f in files}
    for x, y in itertools.combinations(files, 2):
        if ph[x] - ph[y] <= DUP_T:
            union(x, y)

    feats = np.stack([hsv_hist(f) for f in files])
    scene = KMeans(N_SCENES, n_init=10, random_state=SEED).fit_predict(feats)
    for s in set(scene):
        members = [f for f, c in zip(files, scene) if c == s]
        for m in members[1:]:
            union(members[0], m)

    groups = {}
    for f in files:
        groups.setdefault(find(f), []).append(f)
    groups = list(groups.values())

    random.Random(SEED).shuffle(groups)
    target_eval = round(len(files) * 0.2)
    ev, dev = [], []
    for g in sorted(groups, key=len):          # fill eval with small groups first
        (ev if len(ev) + len(g) <= target_eval else dev).extend(g)

    json.dump({"dev": sorted(dev), "eval": sorted(ev)}, open(a.out, "w"), indent=1)
    print(f"{len(files)} images -> {len(groups)} groups | dev={len(dev)} eval={len(ev)}")


if __name__ == "__main__":
    main()
