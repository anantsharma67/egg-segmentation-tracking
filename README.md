# Egg segmentation and tracking (Nosh CV Intern task)

Pipeline: SAM pseudo ground truth (Task 1) -> classical image-processing segmentation (Task 2) -> evaluation and failure analysis (Task 3) -> video tracking (Task 4). Full write-up: `Egg_Segmentation_Report.pdf`.

## Setup
```bash
pip install -r requirements.txt
# put the Roboflow export (train/valid/test folders) in ./data
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Run
```bash
python make_split.py --data data --out splits.json                   # leakage-aware 4:1 dev/eval split
python task1_sam.py --data data --ckpt sam_vit_b_01ec64.pth --out out_task1   # GPU recommended
python contact_sheet.py --overlays out_task1/overlays --out out_task1/review   # review masks by eye
python fix_masks.py --data data --ckpt sam_vit_b_01ec64.pth --src out_task1 --out out_task1_fixed
python run_trial.py t10                 # dev-set score of any trial (t1..t10 in classical.py)
python task3_eval.py                    # Task 3 metrics and plots on the eval split -> results/
yt-dlp -f "bv*[height<=720]" -o egg_video.mp4 "<YOUTUBE_URL>"
python track_video.py --video egg_video.mp4 --out tracked.mp4 --stride 2   # --mode video (default: bright-blob segmenter for this clip); --mode general uses classical.t9
```
`classical.py` reads ground truth from `out_task1_fixed/masks` (override with env var `GT`).

## Files
| File | Purpose |
|---|---|
| make_split.py | Groups near-duplicate and same-scene images, then splits 4:1 |
| task1_sam.py | Grounding DINO boxes -> SAM masks |
| contact_sheet.py, fix_masks.py | Manual-review helper; mask cleanup (logs to fixes.json) |
| classical.py | All classical trials t1..t10 (final = `t10`) |
| run_trial.py, task3_eval.py | Dev scoring; eval metrics, plots, failure gallery |
| track_video.py | Segmentation + watershed split + Hungarian tracker |
| results/ | Eval CSV and figures used in the report |
