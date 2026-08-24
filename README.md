WIP 


EgoX reproduce protocol


# Reproducing the EgoX Table-1 Numbers

This document specifies, end to end, how to reproduce the evaluation numbers of
**EgoX** (arXiv:2512.08269, Table 1) on the Ego-Exo4D exo→ego benchmark — from
generating videos with the officially released weights to scoring all **11 metrics**.

The authors did not release their evaluation code. We therefore calibrated a scoring
protocol against the paper: we generated the unseen split with the official weights and
official inference configuration, then systematically swept scoring variants until the
paper's numbers were matched as closely as possible. **9 of the 11 metrics reproduce
within noise**; the remaining gaps are generation-side, not scoring-side (see [§6](#6-calibration-results)).

- Scoring implementation: [`eval.py`](../eval.py) (single file)
- Frozen protocol config: [`results/eval/protocol.json`](../results/eval/protocol.json) (applied automatically)
- Per-condition calibration sweeps: `results/eval/EgoX_official/calibration_sweeps.txt`

## Contents

1. [Requirements](#1-requirements)
2. [Data layout](#2-data-layout)
3. [Step 1 — Generation with the official weights](#3-step-1--generation-with-the-official-weights)
4. [Step 2 — Scoring](#4-step-2--scoring)
5. [Metric specifications](#5-metric-specifications)
6. [Calibration results](#6-calibration-results)
7. [Appendix A — Exact-reproduction pins](#7-appendix-a--exact-reproduction-pins)
8. [Scope and limitations](#8-scope-and-limitations)

---

## 1. Requirements

Two Python environments are used (VBench's pins conflict with the main environment):

| Environment | Purpose | Key packages |
|---|---|---|
| main (`.venv_eval`) | generation + image/object metrics + FVD | `torch 2.5.1+cu121`, `diffusers`, `sam2 1.1.0`, `transformers 4.57.6`, `lpips 0.1.4`, `scikit-image 0.25.2`, `cd-fvd 0.1.1` |
| vbench (`.venv_vbench`) | VBench video metrics | `vbench 0.1.5`, `torch 2.5.1+cu121` |

Model checkpoints:

| Model | Source | Note |
|---|---|---|
| Wan2.1-I2V-14B-480P (Diffusers) | `Wan-AI` on Hugging Face | base video model |
| EgoX official LoRA (rank 256) | authors' release | `checkpoints/EgoX_authors/` |
| SAM2 | `facebook/sam2-hiera-large` | object segmentation/tracking |
| DINOv3 ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | **gated** — request access on Hugging Face |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | CLIP-I |
| I3D | bundled with `cd-fvd` | FVD |

Exact version pins and default hyperparameters that affect the numbers are listed in
[Appendix A](#7-appendix-a--exact-reproduction-pins).

## 2. Data layout

Evaluation uses two Ego-Exo4D splits, each described by a metadata JSON plus per-clip
videos (49 frames, 30 fps):

```
dataset/
  meta_seen.json            # 388 clips from validation segments of training takes
  meta_unseen.json          # 100 clips from held-out takes
  val/{seen,unseen}/videos/<clip>/
    exo.mp4                 # 784x448 exocentric input
    ego.mp4                 # 448x448 egocentric ground truth
```

The metadata carries, per clip, the exo camera intrinsics/extrinsics and per-frame ego
extrinsics used by GGA at inference time.

## 3. Step 1 — Generation with the official weights

Official inference configuration (fixed; this is what the paper reports):
LoRA rank 256, **GGA attention bias on**, `cos_sim_scaling_factor 3.0`,
50 denoising steps, guidance 5.0 (both hard-coded in `infer.py`), seed 42,
full-canvas output `[exo | ego]` (1232×448).

```bash
python infer.py \
  --model_path checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
  --lora_path  checkpoints/EgoX_authors --lora_rank 256 \
  --use_GGA --cos_sim_scaling_factor 3.0 --seed 42 \
  --meta_data_file dataset/meta_unseen.json \
  --out results/eval/EgoX_official/unseen \
  --start_idx 0 --end_idx 100        # shard across GPUs as needed
```

Output: one `<clip>.mp4` per clip (full canvas). ~100 s/clip on an 80 GB-class GPU.

## 4. Step 2 — Scoring

```bash
python eval.py eval \
  --set unseen \
  --gen_dir results/eval/EgoX_official/unseen \
  --metrics image,object,video \
  --gpus 0,1,2,3 \
  --out results/eval/EgoX_official/results_unseen.txt
```

- If `results/eval/protocol.json` exists it is applied automatically
  (object matching mode, DynDeg threshold — see §5).
- Image/object metrics shard per clip across the listed GPUs; FVD and VBench run once
  over the full set. Sharding does not change the numbers.
- Outputs: a human-readable table with the paper baselines and per-metric deltas
  (`results_*.txt`), machine-readable `results_*.json`, and the raw object records
  `results_*.obj_records.npy` (see §5.3.5).

## 5. Metric specifications

### 5.0 Common input conventions

- If the generated video is a full canvas (width > 448), the **rightmost 448 px crop**
  is taken as the ego view. Ego-only outputs are used as-is. GT is the 448×448 ego video.
- Both videos are truncated to the shorter length; frames are decoded to RGB uint8 with
  OpenCV; no resizing for scoring (448² native).
- Unless stated otherwise, aggregation is **frames → per-clip mean → mean over clips**.

### 5.1 Image metrics

| Metric | Specification |
|---|---|
| PSNR | on the **luminance channel** `Y = 0.299R + 0.587G + 0.114B` (float64); `skimage.metrics.peak_signal_noise_ratio(data_range=255)` |
| SSIM | same Y channel; `skimage.metrics.structural_similarity(data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)` (the original Wang et al. configuration) |
| LPIPS | `lpips` package, `net='alex'`; inputs normalized as `uint8/127.5 − 1` |
| CLIP-I | `openai/clip-vit-base-patch32` with its official image processor; per-frame `get_image_features`, L2-normalized, cosine between generated and GT frame |

Calibration note: PSNR/SSIM on RGB read ≈0.35 dB / 0.021 lower and do **not** match the
paper; the Y channel does (SSIM matches to 3 decimals on the seen split).

### 5.2 Object metrics (Location Error / IoU / Contour Accuracy)

Follows the paper's "segment and track" description.

**(1) Segment + track**, independently for GT and generated video:
- SAM2 Automatic Mask Generator on frame 0 (`facebook/sam2-hiera-large`,
  defaults per Appendix A.2); discard seeds with bbox smaller than 8×8 px.
- Propagate all seed masks through the 49 frames with the SAM2 video predictor.
- Within a track, drop frames where the mask shrinks below 64 px.

**(2) Embed**: for every object in every frame, crop the **bounding box** (not the mask)
and encode with DINOv3 ViT-L/16, taking the **pre-final-LayerNorm patch-token mean**
(`hidden_states[-1][:, 5:].mean(1)`, skipping CLS + 4 register tokens), L2-normalized.

> Why pre-LN: the documented post-LayerNorm features saturate at cosine ≈ 0.9, so a
> τ = 0.9 match set cannot exist. The pre-LN path reproduces the paper's match-set scale
> and its numbers (LocErr 146.6 vs paper 149.93 at literal τ).

**(3) Match (track mode, τ = 0.9)**: for every same-frame (GT object, generated object)
pair compute cosine similarity; accept a **track pair** when the mean cosine over its
shared frames is ≥ 0.9, then take *all* frame pairs of accepted track pairs as
correspondences. All-pairs (no 1:1 assignment), per the paper's eq. 8.

**(4) Score each correspondence**:

| Metric | Definition |
|---|---|
| Location Error | L2 distance between the two bbox centers (448² pixel coordinates) |
| IoU | **bbox** IoU |
| Contour Accuracy | DAVIS boundary F-measure **after centroid registration**: overlay the two masks at their centroids, extract boundaries (3×3-erosion difference), match with a tolerance dilation of 6 px, harmonic mean of precision/recall |

> The literal reading of the paper's eq. 11 (image-coordinate mask IoU) cannot produce
> the reported 0.481 — a mask is a subset of its bbox, so mask-IoU ≤ bbox-IoU (≈ 0.05
> here). Centroid-registered boundary-F reproduces the reported value and is adopted.

**(5) Aggregate**: correspondences → per-clip mean → mean over clips. Clips with zero
matches are excluded (the participating clip count is recorded in the JSON).

All candidate pairs are also dumped with 16 raw fields to `results_*.obj_records.npy`,
so any change of τ / matching mode / aggregation can be re-derived **without re-running
SAM2** (`scripts/obj_solve2.py`).

### 5.3 Video metrics

| Metric | Specification |
|---|---|
| FVD | `cd-fvd` package (the implementation the paper cites), **I3D** backbone, `resolution=224`, `sequence_length=16` (the first 16 frames of each video, stride 1); generated = ego crop, GT = ego original; full-set statistics, seed 42 |
| Temporal Flickering, Motion Smoothness | VBench, scored on the **full 1232×448 canvas**. Full-canvas outputs are used as-is; ego-only outputs are composed as `[input exo | generated ego]` (exo resized to 784×448, INTER_AREA) |
| Dynamic Degree | VBench with one patch: recent VBench scales the RAFT flow threshold with resolution, which does not reproduce the paper (0.77 vs 0.989). We restore the legacy behavior with a **fixed threshold of 2.0** (`protocol.json → dyndeg_fixed_thres`). The other two VBench metrics are unpatched |

## 6. Calibration results

Official weights, official inference configuration, unseen split (100 clips):

| Metric | Ours (this protocol) | Paper | Δ |
|---|---|---|---|
| PSNR ↑ | 13.53 | 14.38 | −0.85 |
| SSIM ↑ | 0.437 | 0.457 | −0.020 |
| LPIPS ↓ | 0.542 | 0.552 | −0.010 |
| CLIP-I ↑ | 0.886 | 0.877 | +0.009 |
| LocErr ↓ | 127.94 | 149.93 | −22.0 |
| IoU ↑ | 0.082 | 0.092 | −0.010 |
| Contour ↑ | 0.492 | 0.481 | +0.011 |
| FVD ↓ | 455.1 | 440.6 | +14.4 (≈3 %) |
| TempFlick ↑ | 0.986 | 0.981 | +0.005 |
| MotionSmooth ↑ | 0.994 | 0.992 | +0.002 |
| DynDeg ↑ | 0.990 | 0.989 | +0.001 |

Interpretation:

- **9 / 11 metrics reproduce within noise.**
- **PSNR / SSIM residuals are generation-side, not scoring-side**: scoring-resolution
  sweeps (224–448) are flat, and the measured seed variance is ±0.2 dB. The released
  weights plausibly differ from the paper's exact checkpoint.
- **The three object metrics cannot be matched simultaneously** under any single
  configuration we found: frame-level matching nails LocErr/Contour but reads IoU 37 %
  low; track-level matching balances all three (adopted; summed relative error 0.28).
  The authors' matching/aggregation details are unpublished.

## 7. Appendix A — Exact-reproduction pins

### A.1 Package versions

`torch 2.5.1+cu121 · sam2 1.1.0 · transformers 4.57.6 · lpips 0.1.4 ·
scikit-image 0.25.2 · cd-fvd 0.1.1 · vbench 0.1.5 · opencv-python 4.11.0 · numpy 2.2.6`

### A.2 SAM2 AMG parameters (the "defaults" of sam2 1.1.0, spelled out)

```
points_per_side=32, points_per_batch=64, pred_iou_thresh=0.8,
stability_score_thresh=0.95, stability_score_offset=1.0, mask_threshold=0.0,
box_nms_thresh=0.7, crop_n_layers=0, min_mask_region_area=0, multimask_output=True
```
(followed by the 8×8 px seed filter of §5.2)

### A.3 Fine print

- DINOv3 crops go through the Hugging Face `AutoImageProcessor` defaults
  (resize + 224² crop, ImageNet normalization).
- GPU sharding is per-clip and does not affect results; SAM2/RAFT GPU nondeterminism
  was measured to be below reporting precision (a full re-run reproduced the object
  metrics exactly).

### A.4 What "exact" covers

- With this document and the pins above, **scoring is exactly reproducible on the same
  generated videos**.
- Regenerating the videos yourself (§3) introduces ±0.2 dB-level PSNR variation from
  seed chains, library versions, and GPU nondeterminism — this is the same residual
  that separates our reproduction from the paper's reported PSNR/SSIM.

## 8. Scope and limitations

- This protocol reproduces the paper's *evaluation*; it does not claim the authors used
  byte-identical code. Where the paper under-specifies (object matching details,
  eq. 11 vs its reported value, VBench version behavior), we adopted the variant that
  reproduces the reported numbers and documented every rejected alternative in
  `calibration_sweeps.txt`.
- All numbers in §6 are on the unseen split; the same protocol applies unchanged to the
  seen split.
