"""Clean up the Task 1 (SAM) masks after manual review.

1. Drop instances that are clearly not eggs: area > 40% of the image, or
   area > 30% with a low detector score (water surface, plates, containers).
2. Exclude images where the egg is not reliably segmentable (listed in EXCLUDE).
3. Add eggs the detector missed, by giving SAM a manual box (ADD_BOXES).

Everything is logged to fixes.json so the report can say exactly what was changed.

Usage (needs SAM weights only when ADD_BOXES is non-empty):
  python fix_masks.py --data data --ckpt sam_vit_b_01ec64.pth \
         --src out_task1 --out out_task1_fixed
"""
import argparse, glob, json, os
import numpy as np
import cv2

# image number (the N in not_damaged_N) -> reason
EXCLUDE = {
    116: "egg tray, no clearly visible single egg",
    42: "egg tray, eggs partly hidden by a rod",
}
# image number -> list of [x0, y0, x1, y1] boxes (pixels, 640x640) for missed eggs.
# These are ESTIMATES read off the review sheets: check the new overlays and nudge.
ADD_BOXES = {
    105: [[26, 373, 248, 488]],   # white egg bottom-left
    134: [[0, 340, 287, 496]],    # white egg bottom-left
    124: [[352, 248, 483, 379]],  # pale egg floating in water
    145: [[274, 274, 431, 405]],  # pale egg floating in water
}


def num(stem):
    return int(stem.split("_")[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="sam_vit_b_01ec64.pth")
    ap.add_argument("--src", default="out_task1")
    ap.add_argument("--out", default="out_task1_fixed")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/masks", exist_ok=True)
    os.makedirs(f"{a.out}/overlays", exist_ok=True)

    meta = json.load(open(f"{a.src}/meta.json"))["images"]
    paths = {os.path.splitext(os.path.basename(f))[0]: f
             for f in glob.glob(f"{a.data}/*/*/*.jpg")}

    predictor = None
    if ADD_BOXES:
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = SamPredictor(sam_model_registry["vit_b"](checkpoint=a.ckpt).to(dev))

    log = {"dropped_instances": [], "excluded": [], "added": []}
    rng = np.random.RandomState(1)
    for stem, path in sorted(paths.items(), key=lambda kv: num(kv[0])):
        n = num(stem)
        if n in EXCLUDE:
            log["excluded"].append({"image": n, "reason": EXCLUDE[n]})
            continue
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        label = cv2.imread(f"{a.src}/masks/{stem}.png", cv2.IMREAD_GRAYSCALE)
        masks = []
        for e in meta.get(stem, []):
            frac = e["area_px"] / (h * w)
            if frac > 0.40 or (frac > 0.30 and e["score"] < 0.45):
                log["dropped_instances"].append(
                    {"image": n, "id": e["id"], "area_frac": round(frac, 2),
                     "score": round(e["score"], 2)})
                continue
            masks.append(label == e["id"])
        for box in ADD_BOXES.get(n, []):
            predictor.set_image(img)
            m, _, _ = predictor.predict(box=np.array(box), multimask_output=False)
            masks.append(m[0])
            log["added"].append({"image": n, "box": box})
        if not masks:
            log["excluded"].append({"image": n, "reason": "no valid egg mask after cleanup"})
            continue
        out = np.zeros((h, w), np.uint8)
        over = img.copy()
        for i, m in enumerate(masks, start=1):
            out[m & (out == 0)] = i
            c = rng.randint(80, 255, 3)
            over[m] = (0.5 * over[m] + 0.5 * c).astype(np.uint8)
        cv2.imwrite(f"{a.out}/masks/{stem}.png", out)
        cv2.imwrite(f"{a.out}/overlays/{stem}.jpg", cv2.cvtColor(over, cv2.COLOR_RGB2BGR))

    json.dump(log, open(f"{a.out}/fixes.json", "w"), indent=1)
    kept = len(glob.glob(f"{a.out}/masks/*.png"))
    print(f"kept {kept} images | dropped {len(log['dropped_instances'])} instances | "
          f"added {len(log['added'])} | excluded {len(log['excluded'])}")
    print("check overlays for:", sorted({x['image'] for x in log['added']}))


if __name__ == "__main__":
    main()
