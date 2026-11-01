"""Tile the Task 1 overlays into labelled contact sheets for fast manual review.

Usage: python contact_sheet.py --overlays out_task1/overlays --out out_task1/review
Open each sheet, note the filenames of bad masks, then re-prompt or drop them.
"""
import argparse, glob, os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--overlays", default="out_task1/overlays")
ap.add_argument("--out", default="out_task1/review")
ap.add_argument("--cols", type=int, default=5)
ap.add_argument("--rows", type=int, default=4)
ap.add_argument("--tile", type=int, default=256)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
files = sorted(glob.glob(f"{a.overlays}/*.jpg"))
per = a.cols * a.rows
for p in range(0, len(files), per):
    sheet = np.zeros((a.rows * a.tile, a.cols * a.tile, 3), np.uint8)
    for i, f in enumerate(files[p:p + per]):
        im = cv2.resize(cv2.imread(f), (a.tile, a.tile))
        name = os.path.basename(f).replace("not_damaged_", "").split("_jpg")[0]
        cv2.putText(im, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(im, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        r, c = divmod(i, a.cols)
        sheet[r * a.tile:(r + 1) * a.tile, c * a.tile:(c + 1) * a.tile] = im
    out = f"{a.out}/sheet_{p // per + 1:02d}.jpg"
    cv2.imwrite(out, sheet)
    print(out)
