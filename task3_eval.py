import os, csv, sys, numpy as np, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import classical as c

fix = lambda l: [p for p in l]
files = fix(c.SPLITS['eval'])
out = 'results'; os.makedirs(out, exist_ok=True)
rows, store = [], []
for f in files:
    if not os.path.exists(f'{c.GT}/{c.stem(f)}.png'): continue
    img, gt = c.load(f)
    pred = c.t10(img)
    g, p = gt > 0, pred > 0
    inter = (g & p).sum()
    iou = inter / max((g | p).sum(), 1)
    dice = 2 * inter / max(g.sum() + p.sum(), 1)
    prec = inter / max(p.sum(), 1); rec = inter / max(g.sum(), 1)
    # boundary F1 (3px tolerance)
    eg = cv2.Canny(g.astype(np.uint8) * 255, 50, 150) > 0
    ep = cv2.Canny(p.astype(np.uint8) * 255, 50, 150) > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bp = (cv2.dilate(ep.astype(np.uint8), k) > 0 & eg).sum() / max(eg.sum(), 1) if False else \
         ((cv2.dilate(ep.astype(np.uint8), k) > 0) & eg).sum() / max(eg.sum(), 1)
    bg_ = ((cv2.dilate(eg.astype(np.uint8), k) > 0) & ep).sum() / max(ep.sum(), 1)
    bf1 = 2 * bp * bg_ / max(bp + bg_, 1e-9)
    n_gt, n_pr = len(np.unique(gt)) - 1, c.count(pred)
    rows.append(dict(image=int(c.stem(f).split('_')[2]), iou=iou, dice=dice, precision=prec, recall=rec,
                     boundary_f1=bf1, gt_area=g.mean(), pred_area=p.mean(), n_gt=n_gt, n_pred=n_pr))
    store.append((img, gt, pred, iou, c.stem(f)))
with open(f'{out}/eval_metrics.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
A = lambda k: np.array([r[k] for r in rows])
print(f"EVAL n={len(rows)}  mIoU={A('iou').mean():.3f} median={np.median(A('iou')):.3f} Dice={A('dice').mean():.3f} "
      f"P={A('precision').mean():.3f} R={A('recall').mean():.3f} BF1={A('boundary_f1').mean():.3f} "
      f"count-acc={np.mean(A('n_gt')==A('n_pred')):.2f}")
iou = A('iou'); print('IoU>=0.8:', (iou >= .8).sum(), ' 0.5-0.8:', ((iou >= .5) & (iou < .8)).sum(), ' <0.5:', (iou < .5).sum())
# plots
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].hist(iou, bins=10, range=(0, 1), color='#4c72b0'); ax[0].set_title('Per-image IoU (eval)'); ax[0].set_xlabel('IoU')
ax[1].scatter(A('gt_area'), A('pred_area'), s=18); ax[1].plot([0, .5], [0, .5], 'k--', lw=1)
ax[1].set_xlabel('GT egg area (fraction)'); ax[1].set_ylabel('Predicted area'); ax[1].set_title('Over/under-segmentation')
plt.tight_layout(); plt.savefig(f'{out}/iou_hist_and_area.png', dpi=130); plt.close()
worst = sorted(store, key=lambda s: s[3])[:6]
fig, ax = plt.subplots(3, 6, figsize=(16, 8.4))
for j, (img, gt, pred, i, s) in enumerate(worst):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ov = rgb.copy(); ov[(gt > 0) & (pred == 0)] = [255, 0, 0]; ov[(pred > 0) & (gt == 0)] = [0, 90, 255]
    ov[(pred > 0) & (gt > 0)] = [0, 200, 0]
    for r, im, t in [(0, rgb, f'#{s.split("_")[2]}  IoU={i:.2f}'), (1, (gt > 0) * 255, 'SAM (GT)'), (2, ov, 'green=TP red=FN blue=FP')]:
        ax[r, j].imshow(im, cmap='gray'); ax[r, j].set_title(t, fontsize=9); ax[r, j].axis('off')
plt.tight_layout(); plt.savefig(f'{out}/worst6.png', dpi=110); plt.close()
print('worst:', [(s[4].split('_')[2], round(s[3], 2)) for s in worst])
