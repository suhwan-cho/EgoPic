WIP 


EgoX reproduce protocol
```
Notes:
  - generated videos: results/eval/EgoX_official/unseen (rightmost 448px cropped as the ego view)
  - PSNR/SSIM on the LUMINANCE (Y) channel (calibrated: authors' weights reproduce paper SSIM
  -   to 3 decimals on seen; RGB reads ~-0.02 SSIM / -0.35dB lower). LPIPS=AlexNet,
  -   CLIP-I=openai/clip-vit-base-patch32 (paper does not pin variants).
  - object MAIN row: PER-CLIP aggregation (mean within clip, then over clips) at tau=0.9;
  -     [per-clip] tau=0.95: clips=86 pairs=9045  LocErr=63.08  IoU=0.221  Contour=0.615  [Ego-Exo4D registered-mask-IoU=0.579]
  -     [per-clip] tau=0.90: clips=100 pairs=115740  LocErr=151.60  IoU=0.058  Contour=0.463  [Ego-Exo4D registered-mask-IoU=0.365]
  -     [per-clip] tau=0.85: clips=100 pairs=650320  LocErr=186.80  IoU=0.019  Contour=0.380  [Ego-Exo4D registered-mask-IoU=0.291]
  - raw object records dumped -> results_unseen.obj_records.npy (670624 pairs x 16: simC,simM,simL,loc,iou,contShape,contImg,regBF,assignCos,regMaskIoU,simLpost,simLclsPost,locCmid,gtTrack,genTrack,clipIdx)
  - object criteria, MAIN row: eq.8 ALL-pairs cosine >= tau (no 1:1), DINOv3 ViT-L feat;
  -   eq.9 LocErr=bbox-center L2; eq.10 bbox IoU; Contour = boundary-F after centroid registration
  -   (reproduces the paper; eq.11-literal image-coord mask IoU shown as [eq.11 maskIoU=..] diagnostic).
  -     [paper F.3] tau=0.95: n=9045  LocErr=65.76  IoU=0.245  Contour(boundary-F)=0.607  [eq.11 maskIoU=0.194]
  -     [paper F.3] tau=0.90: n=115740  LocErr=144.20  IoU=0.053  Contour(boundary-F)=0.471  [eq.11 maskIoU=0.039]
  -     [paper F.3] tau=0.85: n=650320  LocErr=179.85  IoU=0.015  Contour(boundary-F)=0.387  [eq.11 maskIoU=0.011]
  -   NOTE: MAIN Contour uses boundary-F (centroid-registered), which reproduces the paper
  -   (0.470 vs 0.481). The paper's STATED eq.11 (image-coord mask IoU) cannot yield 0.481: a
  -   mask is a subset of its bbox, so mask-IoU <= bbox-IoU (~0.05 here) << 0.481. The literal
  -   eq.11 value is shown as [eq.11 maskIoU=..]; the paper must have used an aligned method.
  -   [non-paper diagnostic] 1:1 Hungarian assignment (post-LN ViT-B patch-mean), Contour=boundary-F:
  -     [assigned-1to1] tau=0.95: n=9  LocErr=38.26  IoU=0.275  Contour(regBF)=0.349
  -     [assigned-1to1] tau=0.90: n=463  LocErr=28.03  IoU=0.364  Contour(regBF)=0.756
  -     [assigned-1to1] tau=0.85: n=2076  LocErr=34.99  IoU=0.333  Contour(regBF)=0.765
  -   STRICT diagnostic (documented ViT-B post-LN features; tau=0.9 keeps only genuine
  -   correspondences -> 'better' numbers by construction). Sweep over tau and pooling:
  -     [mean] tau=0.9: n=479  LocErr=28.44  IoU=0.367  Contour=0.844  Contour(img-coords)=0.282
  -     [mean] tau=0.8: n=6394  LocErr=55.52  IoU=0.265  Contour=0.814  Contour(img-coords)=0.215
  -     [mean] tau=0.7: n=23257  LocErr=85.01  IoU=0.157  Contour=0.759  Contour(img-coords)=0.125
  -     [mean] tau=0.6: n=75852  LocErr=123.86  IoU=0.073  Contour=0.698  Contour(img-coords)=0.054
  -     [mean] tau=0.5: n=263401  LocErr=157.99  IoU=0.030  Contour=0.641  Contour(img-coords)=0.021
  -     [ cls] tau=0.9: n=51  LocErr=23.48  IoU=0.061  Contour=0.884  Contour(img-coords)=0.042
  -     [ cls] tau=0.8: n=1094  LocErr=43.35  IoU=0.242  Contour=0.841  Contour(img-coords)=0.195
  -     [ cls] tau=0.7: n=6480  LocErr=70.23  IoU=0.187  Contour=0.808  Contour(img-coords)=0.155
  -     [ cls] tau=0.6: n=22313  LocErr=95.39  IoU=0.122  Contour=0.774  Contour(img-coords)=0.097
  -     [ cls] tau=0.5: n=62457  LocErr=121.40  IoU=0.073  Contour=0.728  Contour(img-coords)=0.055
  - FVD = cd-fvd package (paper's citation), I3D, one 16-frame clip/video, res 224, ego crop
  -   (calibrated on authors' weights: 451.3 vs paper 440.64 unseen, ~2%)
  - VBench = temporal_flickering/motion_smoothness/dynamic_degree on the FULL 1232x448 canvas
  -   (calibrated: full canvas reproduces paper TFlick 0.986~0.981 & MSmooth 0.994~0.992; the ego
  -   crop under-reads both. Ego-only outputs are composed as [input exo | generated ego].
  -   DynDeg remains below paper (0.76-0.82 vs 0.989) under every protocol tried — residual gap
  -   is generation-side motion amount, not the metric.)
  - 100/100 clips evaluated
```
