"""Task 1: pseudo ground-truth egg masks with a pretrained model.

Pipeline: Grounding DINO ("an egg.") -> boxes -> SAM (box prompts) -> masks.
The dataset has no annotations, so the text-prompted detector supplies the boxes.

Setup (Colab or local, GPU optional):
  pip install torch torchvision transformers opencv-python pillow numpy
  pip install git+https://github.com/facebookresearch/segment-anything.git
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Usage:
  python task1_sam.py --data data --ckpt sam_vit_b_01ec64.pth --out out_task1

Outputs (per image <stem>):
  out_task1/masks/<stem>.png     label map, 0 = background, 1..n = egg instance ids
  out_task1/overlays/<stem>.jpg  image with masks and boxes drawn, for review
  out_task1/meta.json            boxes, scores, mask areas; images with no detection
"""
import argparse, glob, json, os
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor

DINO_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "an egg."
BOX_T, TEXT_T = 0.30, 0.25
MAX_AREA_FRAC = 0.90     # drop boxes that cover (almost) the whole image
NMS_IOU = 0.5


def detect_boxes(img_pil, processor, model, device):
    inputs = processor(images=img_pil, text=PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    kw = dict(input_ids=inputs.input_ids, text_threshold=TEXT_T,
              target_sizes=[img_pil.size[::-1]])
    try:   # argument name differs between transformers versions
        res = processor.post_process_grounded_object_detection(out, threshold=BOX_T, **kw)[0]
    except TypeError:
        res = processor.post_process_grounded_object_detection(out, box_threshold=BOX_T, **kw)[0]
    boxes, scores = res["boxes"].cpu(), res["scores"].cpu()
    if len(boxes) == 0:
        return boxes, scores
    W, H = img_pil.size
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = area < MAX_AREA_FRAC * W * H
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes):
        k = nms(boxes, scores, NMS_IOU)
        boxes, scores = boxes[k], scores[k]
    return boxes, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="vit_b")
    ap.add_argument("--out", default="out_task1")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for d in ("masks", "overlays"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)

    processor = AutoProcessor.from_pretrained(DINO_ID)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(device).eval()
    sam = sam_model_registry[a.model](checkpoint=a.ckpt).to(device)
    predictor = SamPredictor(sam)

    files = sorted(glob.glob(f"{a.data}/*/*/*.jpg"))
    meta, no_det = {}, []
    rng = np.random.RandomState(0)
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        img_pil = Image.open(f).convert("RGB")
        img = np.array(img_pil)
        boxes, scores = detect_boxes(img_pil, processor, dino, device)
        if len(boxes) == 0:
            no_det.append(stem)
            Image.fromarray(img).save(f"{a.out}/overlays/{stem}.jpg")
            cv2.imwrite(f"{a.out}/masks/{stem}.png", np.zeros(img.shape[:2], np.uint8))
            continue

        predictor.set_image(img)
        label = np.zeros(img.shape[:2], np.uint8)
        over = img.copy()
        info = []
        for i, (b, s) in enumerate(zip(boxes.numpy(), scores.numpy()), start=1):
            m, _, _ = predictor.predict(box=b, multimask_output=False)
            m = m[0]
            label[m & (label == 0)] = i
            colour = rng.randint(80, 255, 3)
            over[m] = (0.5 * over[m] + 0.5 * colour).astype(np.uint8)
            x0, y0, x1, y1 = b.astype(int)
            cv2.rectangle(over, (x0, y0), (x1, y1), colour.tolist(), 2)
            info.append({"id": i, "box": b.round(1).tolist(), "score": float(s),
                         "area_px": int(m.sum())})
        cv2.imwrite(f"{a.out}/masks/{stem}.png", label)
        Image.fromarray(over).save(f"{a.out}/overlays/{stem}.jpg")
        meta[stem] = info
        print(stem, len(info), "egg(s)")

    json.dump({"images": meta, "no_detection": no_det}, open(f"{a.out}/meta.json", "w"), indent=1)
    print(f"done: {len(meta)} with eggs, {len(no_det)} with no detection -> {no_det}")


if __name__ == "__main__":
    main()
