"""Task 4: segment + track individual eggs in a video.

  yt-dlp -f "bv*[height<=720]" -o egg_video.mp4 "<YOUTUBE_URL>"
  python track_video.py --video egg_video.mp4 --out tracked.mp4 [--stride 1] [--max-frames 0]

Per frame: classical segmentation (classical.t9) -> split touching eggs with a
distance-transform watershed -> match to existing tracks with the Hungarian
algorithm on (1 - IoU) with a centroid-distance gate -> draw persistent IDs.
"""
import argparse, csv
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment
import classical as c


def split_instances(mask, min_area=300):
    """Binary mask -> list of instance masks (watershed on distance transform)."""
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return []
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, peaks = cv2.threshold(dist, 0.55 * dist.max(), 1, cv2.THRESH_BINARY)
    n, markers = cv2.connectedComponents(peaks.astype(np.uint8))
    markers = markers + 1
    markers[mask == 0] = 0
    ws = cv2.watershed(cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
    out = []
    for i in range(2, n + 1):
        m = (ws == i).astype(np.uint8)
        if m.sum() >= min_area:
            out.append(m)
    return out


def segment_video_frame(img, v_min=165, s_max=75, roi_top=0.36, dark_table_v=115):
    """Bright, low-saturation blobs on a dark table (this video). Returns binary mask.
    Skips frames whose table region is light (title / text cards have no eggs)."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    y0 = int(roi_top * h)
    if np.median(hsv[y0:, :, 2]) > dark_table_v:
        return np.zeros((h, w), np.uint8)
    m = np.zeros((h, w), np.uint8)
    m[y0:] = ((hsv[y0:, :, 2] > v_min) & (hsv[y0:, :, 1] < s_max)).astype(np.uint8)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k(5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k(9))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 0.004 * h * w:
            continue
        # touching eggs merge into one blob: split each component on its own distance transform
        for inst in split_instances((lab == i).astype(np.uint8), min_area=int(0.004 * h * w)):
            a = inst.sum() / (h * w)
            x, y, bw, bh = cv2.boundingRect(inst)
            if a <= 0.10 and min(bw, bh) / max(bw, bh) >= 0.4 and c.ellipse_score(inst) >= 0.5:
                out |= inst
    # grow each detected core out to the shaded egg edge (hysteresis within a low threshold, max ~14 px)
    low = np.zeros((h, w), np.uint8)
    low[y0:] = ((hsv[y0:, :, 2] > 95) & (hsv[y0:, :, 1] < 95)).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(14):
        out = cv2.dilate(out, k3) & low | out
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, k(9))


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()])


class Tracker:
    def __init__(self, max_age=80, gate=140):
        self.tracks, self.next_id, self.max_age, self.gate = {}, 1, max_age, gate

    def update(self, masks):
        ids = list(self.tracks)
        assigned = {}
        if ids and masks:
            cost = np.ones((len(ids), len(masks)))
            for i, t in enumerate(ids):
                for j, m in enumerate(masks):
                    d = np.linalg.norm(self.tracks[t]['c'] - centroid(m))
                    if self.tracks[t]['age'] > 2:
                        # lost track (egg hidden by a hand, then carried away): re-identify by size, not position
                        ratio = abs(np.log(max(m.sum(), 1) / max(self.tracks[t]['m'].sum(), 1)))
                        cost[i, j] = 0.6 + 0.5 * ratio
                    elif d < self.gate:
                        cost[i, j] = 1 - max(iou(self.tracks[t]['m'], m), 0.3 * (1 - d / self.gate))
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 0.95:
                    assigned[j] = ids[i]
        for j, m in enumerate(masks):
            if j not in assigned:
                assigned[j] = self.next_id; self.next_id += 1
        for j, tid in assigned.items():
            self.tracks[tid] = dict(m=masks[j], c=centroid(masks[j]), age=0)
        for tid in list(self.tracks):
            if tid not in assigned.values():
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
        return {tid: masks[j] for j, tid in assigned.items()}


def colour(i):
    rng = np.random.RandomState(i * 7919 % 2**31)
    return tuple(int(x) for x in rng.randint(60, 255, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True); ap.add_argument('--out', default='tracked.mp4')
    ap.add_argument('--stride', type=int, default=1); ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--mode', choices=['video', 'general'], default='video', help='video = bright-blob segmenter tuned for this clip; general = classical.t9')
    ap.add_argument('--width', type=int, default=640, help='resize frames to this width for speed')
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W0, H0 = int(cap.get(3)), int(cap.get(4))
    scale = a.width / W0 if W0 > a.width else 1.0
    W, H = int(W0 * scale), int(H0 * scale)
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'mp4v'), fps / a.stride, (W, H))
    tr, rows, f = Tracker(), [], -1
    while True:
        ok, frame = cap.read()
        if not ok: break
        f += 1
        if f % a.stride: continue
        if a.max_frames and f // a.stride >= a.max_frames: break
        img = cv2.resize(frame, (W, H)) if scale != 1 else frame
        masks = split_instances(segment_video_frame(img) if a.mode == 'video' else c.t9(img))
        tracked = tr.update(masks)
        over = img.copy()
        for tid, m in tracked.items():
            col = colour(tid)
            over[m > 0] = (0.5 * over[m > 0] + 0.5 * np.array(col)).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cnts, -1, col, 2)
            cx, cy = centroid(m).astype(int)
            cv2.putText(over, f'egg {tid}', (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(over, f'egg {tid}', (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            rows.append([f, tid, cx, cy, int(m.sum())])
        wr.write(over)
    wr.release()
    csv.writer(open(a.out.rsplit('.', 1)[0] + '_tracks.csv', 'w')).writerows([['frame', 'id', 'cx', 'cy', 'area']] + rows)
    print(f'frames processed: {f + 1} | tracks created: {tr.next_id - 1} | wrote {a.out}')


if __name__ == '__main__':
    main()
