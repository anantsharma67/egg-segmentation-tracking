import json, os, glob, sys
import numpy as np, cv2

DATA = 'data'
GT = os.environ.get('GT', 'out_task1_fixed/masks')
SPLITS = json.load(open('splits.json'))


def stem(p): return os.path.splitext(os.path.basename(p))[0]


def load(p):
    img = cv2.imread(p)
    gt = cv2.imread(f'{GT}/{stem(p)}.png', cv2.IMREAD_GRAYSCALE)
    return img, gt


def clean(m, min_frac=0.01, k=7):
    m = m.astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_frac * m.size:
            comp = (lab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, 1, -1)   # fill holes
    return out


# ---- trial 1: grayscale Otsu (bright object) ----
def t1(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, m = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return clean(m)


# ---- trial 2: Lab distance from border colour ----
def t2(img, T=None):
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = np.concatenate([lab[:20].reshape(-1, 3), lab[-20:].reshape(-1, 3),
                        lab[:, :20].reshape(-1, 3), lab[:, -20:].reshape(-1, 3)])
    bg = np.median(b, 0)
    d = np.linalg.norm(lab - bg, axis=2)
    d8 = np.clip(d * 2, 0, 255).astype(np.uint8)
    if T is None:
        _, m = cv2.threshold(d8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        m = (d8 > T).astype(np.uint8)
    return clean(m)


# ---- trial 3: Lab k-means (k=3) on a/b/L + keep compact egg-like blobs ----
def egg_like(m, min_solid=0.85, max_area=0.5):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        a = comp.sum() / comp.size
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        solid = cv2.contourArea(c) / max(hull, 1)
        if solid >= min_solid and a <= max_area:
            out |= comp
    return out


def t3(img):
    return egg_like(t2(img))


# ---- trial 4: t3 + GrabCut refinement seeded from t3 ----
def t4(img):
    m0 = t3(img)
    if m0.sum() == 0:
        return m0
    gc = np.full(m0.shape, cv2.GC_BGD, np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dil = cv2.dilate(m0, ker)
    ero = cv2.erode(m0, ker)
    gc[dil > 0] = cv2.GC_PR_BGD
    gc[m0 > 0] = cv2.GC_PR_FGD
    gc[ero > 0] = cv2.GC_FGD
    bgm = np.zeros((1, 65)); fgm = np.zeros((1, 65))
    try:
        cv2.grabCut(img, gc, None, bgm, fgm, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m0
    m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    return egg_like(clean(m), min_solid=0.85)


def metrics(pred, gt):
    g = gt > 0
    p = pred > 0
    inter = (g & p).sum(); uni = (g | p).sum()
    iou = inter / uni if uni else 1.0
    dice = 2 * inter / (g.sum() + p.sum()) if (g.sum() + p.sum()) else 1.0
    return iou, dice


def count(m):
    return cv2.connectedComponents(m.astype(np.uint8))[0] - 1


def evaluate(fn, files):
    ious, dices, cnt_ok = [], [], []
    for f in files:
        if not os.path.exists(f'{GT}/{stem(f)}.png'):
            continue
        img, gt = load(f)
        pred = fn(img)
        i, d = metrics(pred, gt)
        ious.append(i); dices.append(d)
        cnt_ok.append(count(pred) == len(np.unique(gt)) - 1)
    return np.mean(ious), np.median(ious), np.mean(dices), np.mean(cnt_ok), len(ious)


# ---- trial 5: GrabCut from a central rectangle prior ----
def t5(img, margin=0.08, it=4):
    h, w = img.shape[:2]
    rect = (int(w * margin), int(h * margin), int(w * (1 - 2 * margin)), int(h * (1 - 2 * margin)))
    gc = np.zeros((h, w), np.uint8)
    bgm = np.zeros((1, 65)); fgm = np.zeros((1, 65))
    cv2.grabCut(img, gc, rect, bgm, fgm, it, cv2.GC_INIT_WITH_RECT)
    m = ((gc == 1) | (gc == 3)).astype(np.uint8)
    return egg_like(clean(m), min_solid=0.8, max_area=0.7)


# ---- trial 6: many candidate masks, keep the most egg-like components ----
def candidates(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    chans = [lab[..., 0], lab[..., 1], lab[..., 2], hsv[..., 1], hsv[..., 2]]
    out = []
    for c in chans:
        _, m = cv2.threshold(c, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out += [m, 1 - m]
    out.append(t2(img))
    return out


def ellipse_score(comp):
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return 0
    area = cv2.contourArea(c)
    (cx, cy), (MA, ma), _ = cv2.fitEllipse(c)
    ell = np.pi * MA * ma / 4
    fit = min(area, ell) / max(area, ell, 1)
    hull = cv2.contourArea(cv2.convexHull(c))
    solid = area / max(hull, 1)
    ar = min(MA, ma) / max(MA, ma, 1)
    h, w = comp.shape
    ys, xs = np.where(comp)
    border = (xs.min() <= 1) + (ys.min() <= 1) + (xs.max() >= w - 2) + (ys.max() >= h - 2)
    s = fit * solid * (1 if 0.45 < ar else 0.5) * (0.5 ** border)
    return s


def t6(img, thr=0.80, min_a=0.015, max_a=0.6):
    h, w = img.shape[:2]
    comps = []
    for cand in candidates(img):
        n, lab, st, _ = cv2.connectedComponentsWithStats(clean(cand, 0.01, 7))
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA] / (h * w)
            if min_a <= a <= max_a:
                comp = (lab == i).astype(np.uint8)
                s = ellipse_score(comp)
                if s >= thr:
                    comps.append((s * (a ** 0.3), comp))
    comps.sort(key=lambda x: -x[0])
    out = np.zeros((h, w), np.uint8)
    for s, comp in comps:
        ov = (out & comp).sum() / comp.sum()
        if ov < 0.2:
            out |= comp
    return out


if __name__ == '__main__':
    for name, fn in [('t5 grabcut-rect', t5), ('t6 candidates+ellipse', t6)]:
        r = evaluate(fn, dev)
        print(f'{name:24s} dev n={r[4]} mIoU={r[0]:.3f} medIoU={r[1]:.3f} Dice={r[2]:.3f} count-acc={r[3]:.2f}')


# ---- trial 8: t6 candidates as seeds, GrabCut refinement at half resolution ----
def t8(img, thr=0.6, scale=0.5):
    m0 = t6(img, thr=thr)
    if m0.sum() == 0:
        return m0
    h, w = img.shape[:2]
    sm = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = cv2.resize(m0, (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(m, k(21)) > 0] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(m, k(15)) > 0] = cv2.GC_FGD
    bgm = np.zeros((1, 65)); fgm = np.zeros((1, 65))
    try:
        cv2.grabCut(sm, gc, None, bgm, fgm, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m0
    r = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    r = cv2.resize(r, (w, h), interpolation=cv2.INTER_NEAREST)
    return clean(r, 0.01, 7)


# ---- trial 9: t8, with a low-res central-rectangle GrabCut fallback when no candidate passes ----
def grabcut_rect_lowres(img, margin=0.1, scale=0.4, it=4):
    h, w = img.shape[:2]
    sm = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = sm.shape[:2]
    rect = (int(sw * margin), int(sh * margin), int(sw * (1 - 2 * margin)), int(sh * (1 - 2 * margin)))
    gc = np.zeros((sh, sw), np.uint8)
    bgm = np.zeros((1, 65)); fgm = np.zeros((1, 65))
    cv2.grabCut(sm, gc, rect, bgm, fgm, it, cv2.GC_INIT_WITH_RECT)
    r = ((gc == 1) | (gc == 3)).astype(np.uint8)
    r = cv2.resize(r, (w, h), interpolation=cv2.INTER_NEAREST)
    return egg_like(clean(r, 0.01, 7), min_solid=0.75, max_area=0.7)


def t9(img):
    m = t8(img)
    if m.sum() == 0:
        m = grabcut_rect_lowres(img)
    return m


# ---- trial 10: t9, plus a last-resort "largest GrabCut blob" when every filter rejects ----
def largest_blob(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n <= 1:
        return m
    i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    return (lab == i).astype(np.uint8)


def grabcut_raw(img, margin=0.1, scale=0.4, it=4):
    h, w = img.shape[:2]
    sm = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = sm.shape[:2]
    rect = (int(sw * margin), int(sh * margin), int(sw * (1 - 2 * margin)), int(sh * (1 - 2 * margin)))
    gc = np.zeros((sh, sw), np.uint8)
    bgm = np.zeros((1, 65)); fgm = np.zeros((1, 65))
    cv2.grabCut(sm, gc, rect, bgm, fgm, it, cv2.GC_INIT_WITH_RECT)
    r = ((gc == 1) | (gc == 3)).astype(np.uint8)
    return largest_blob(clean(cv2.resize(r, (w, h), interpolation=cv2.INTER_NEAREST), 0.01, 7))


def t10(img):
    m = t9(img)
    if m.sum() == 0:
        m = grabcut_raw(img)
    return m
