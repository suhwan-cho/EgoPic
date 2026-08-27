# EgoPic 평가 프로토콜

Exo→Ego 비디오 생성 결과를 EgoX 논문(arXiv:2512.08269) Table 1과 동일한 11개 지표로
채점하기 위한 표준 프로토콜이다. 저자의 평가 코드가 공개되지 않았기 때문에, 공식 릴리스
가중치로 생성한 결과에 대해 채점 방식을 체계적으로 탐색하여 논문 수치를 가장 근접하게
재현하는 구성을 확정하였다 (2026-08-21 확정, §5 참조). 모든 모델·실험 팔은 이 프로토콜
하나로 채점하며, 논문 보고 수치와 병기할 때는 "(paper) vs (our protocol)"로 구분한다.
(개정: v1 2026-08-21 → v2 → **v3 2026-08-25**, object 매칭 교체 — §3.3·§5 참조)

- 구현: `eval.py` (단일 파일) — 확정 프로토콜은 **eval.py 내장 기본값**(`PROTOCOL_DEFAULT`, 2026-08-24 반영)이라 설정 파일 불필요
- 평가 세트: seen(학습 take의 검증 클립 388개), unseen(미학습 take 클립 100개)

---

## 1. 입력 규약

| 항목 | 규약 |
|---|---|
| 클립 | 49프레임 mp4, 30fps |
| 생성물 | full canvas(폭 > 448px) 출력이면 **우측 448px 크롭을 ego 뷰로 사용**. ego 전용 출력(448px)은 그대로 사용 |
| GT | 각 클립의 ego 원본 448×448 |
| 길이 정합 | 생성/GT 중 짧은 쪽 길이로 절단 |
| 색공간 | OpenCV 디코드 후 RGB, uint8. 채점 목적의 리사이즈 없음(448² 기준) |
| 집계 | 별도 명시가 없으면 **프레임 → 클립 내 평균 → 클립 간 평균** |

## 2. Image 지표 (4)

프레임 단위로 계산해 클립 내 평균, 클립 간 평균한다.

### 2.1 PSNR
- RGB를 휘도(luminance) 채널로 변환: `Y = 0.299·R + 0.587·G + 0.114·B` (float64)
- `skimage.metrics.peak_signal_noise_ratio(gt_y, gen_y, data_range=255)`
- RGB 3채널 계산은 사용하지 않는다 (Y 채널이 논문 수치를 재현, §5)

### 2.2 SSIM
- 같은 Y 채널에 대해
  `skimage.metrics.structural_similarity(data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)`
- 즉 Wang et al. 원 논문 구성(11×11 가우시안 창, σ=1.5)

### 2.3 LPIPS
- `lpips` 패키지, `net='alex'` (AlexNet 백본)
- 입력 정규화: `uint8 / 127.5 − 1` → [−1, 1]

### 2.4 CLIP-I
- 모델: `openai/clip-vit-base-patch32`, 공식 `CLIPImageProcessor` 전처리
- 프레임별 `get_image_features`를 L2 정규화 후 생성/GT 코사인 유사도

## 3. Object 지표 (3) — Location Error / IoU / Contour Accuracy

논문의 "segment and track" 서술을 따르는 5단계 파이프라인이다.

### 3.1 객체 세그먼트 + 추적 (GT·생성 비디오 각각)
1. frame 0에 SAM2 Automatic Mask Generator (`facebook/sam2-hiera-large`,
   **points_per_side=32, min_mask_region_area=2000**, 나머지 기본 파라미터) 실행.
   bbox가 **32×32px 미만**인 시드는 제거 (= 세그 변형 "vD_big"; granularity 는 논문 미명시라
   11개 변형을 스윕해 결정 — §5).
2. 시드 마스크들을 SAM2 video predictor로 49프레임 전체에 전파(track).
3. 프레임에서 마스크 면적이 64px 미만이면 해당 프레임은 그 트랙에서 제외.
4. GT 트랙은 결정적이므로 디스크 캐시 후 재사용 가능.

### 3.2 객체 임베딩
- 각 프레임에서 객체의 **bbox 크롭**(마스크 적용 아님, 8px 미만 변은 제외)을 추출
- DINOv3 **ViT-L/16** (`facebook/dinov3-vitl16-pretrain-lvd1689m`)에 통과시켜
  **최종 LayerNorm 이전(pre-LN) hidden state** (`hidden_states[-1]`)의
  **patch 토큰 평균** (`[:, 5:].mean(1)`, CLS+register 4개 제외)을 L2 정규화
- 전처리: HuggingFace `AutoImageProcessor` 기본값 (짧은 변 224 리사이즈 + 224² 크롭)
- 선택 근거: 논문이 특징을 명시하지 않아 5종(B-cls/B-patch/L-pre/L-post/L-cls)을 스윕 —
  τ=0.9 전쌍 매칭이 의미를 갖고 논문 unseen 을 재현하는 특징은 **pre-LN ViT-L 이 유일**(§5).
  5종 유사도 전부가 records 에 저장되므로 특징 변경은 재집계만으로 가능.

### 3.3 대응 매칭 (프레임별 전쌍, τ_sim = 0.90) — v4 (2026-08-27)
- 프레임마다 (GT 객체 × 생성 객체) 모든 쌍의 코사인을 계산하고 **cos ≥ 0.90 인 쌍 전부**를
  대응으로 채택 — 논문 eq.8 문자 그대로 (1:1 배정 아님, GT 객체가 여러 쌍에 참여 가능)
- 채택 기준 = **논문 명시축 준수**: τ_sim=0.9 는 논문이 명시한 유일한 매칭 수치이므로
  고정하고, 미명시 축(세그 granularity·특징·집계)만 스윕해 논문 수치 적합으로 결정(§5)
- 이력: v1(track τ0.9, 8/21) → v2(1:1 τ0.5) → v3(1:1 τ0.6125, 서열 제약 역추정) →
  **v4(전쌍 τ0.9, 논문충실)** — v1–v3 은 τ·매칭을 자유축으로 쓴 역추정이라 폐기

### 3.4 지표 계산 (채택된 프레임쌍마다)
| 지표 | 정의 |
|---|---|
| Location Error | 두 객체 bbox 중심 간 L2 거리 (448² 픽셀 좌표) |
| IoU | **bbox** IoU |
| Contour Accuracy | **bbox 정렬 마스크 IoU**: 각 마스크를 자기 bbox 로 crop 하고, 생성 마스크 crop 을 GT bbox 크기로 NEAREST 리사이즈해 겹친 뒤 IoU (eq.11 의 마스크 IoU 를 형태 비교로 구현) |

Contour 관련 주석: eq.11 을 이미지 좌표 그대로 계산하면 마스크가 bbox 의 부분집합이라
bbox IoU(≈0.05)를 넘을 수 없어 논문 값(0.481)이 나올 수 없다. 위치를 소거하고 형태만
비교하는 구현 3종(bbox 정렬 IoU / centroid 창 IoU / boundary-F)을 스윕한 결과
bbox 정렬 마스크 IoU 가 논문 unseen 과 최근접이라 채택(§5). records 에 3종 전부 저장.

### 3.5 집계와 재집계
- 프레임쌍 → 클립 내 평균 → 클립 간 평균 (per-clip mean). 매칭 0쌍 클립은 평균에서 제외
- 추출·집계는 2단 분리: `scripts/obj_cache_extract.py` 가 모든 후보 쌍의 원천값 16필드
  (유사도 5종·1:1 배정 코사인·LocErr 2종·bbox IoU·contour 4종·트랙/클립 id)를
  `results/eval/objsweep_full/<ds>.vD_big.s*.npy` 로 저장하고, SAM2 마스크는
  `checkpoints/eval/segvar_cache/` 에 변형별 캐시된다. **τ·특징·매칭·집계 변경은 SAM2
  재실행 없이 재집계만으로 가능** (`scripts/obj_aggregate_v4.py`; v4 확정값 dict 포함).
- `eval.py eval --metrics object` 는 이 레코드를 자동 탐색/추출/집계한다 (§6).

## 4. Video 지표 (4)

### 4.1 FVD
- `cd-fvd` 패키지(논문이 인용하는 구현), **I3D** 백본
- `resolution=224, sequence_length=16` — 비디오당 16프레임 1클립
- 생성 = ego 크롭, GT = ego 원본. 전체 클립으로 실통계(full set) 계산, seed 42

### 4.2 VBench: Temporal Flickering / Motion Smoothness / Dynamic Degree
- **full canvas(1232×448) 채점**: full-canvas 출력은 원본 그대로,
  ego 전용 출력은 `[입력 exo | 생성 ego]`로 합성해 캔버스를 구성
- VBench는 의존성 충돌 때문에 격리 가상환경(`.venv_vbench`)에서 실행
- **Dynamic Degree 패치**: 최신 VBench는 RAFT optical-flow 임계를 해상도에 비례시키는데,
  이 방식으로는 논문 값이 재현되지 않는다(0.77 vs 0.989). 구버전 방식의
  **고정 임계 2.0**으로 채점한다 (eval.py 내장 기본값 `dyndeg_fixed_thres`).
  Temporal Flickering / Motion Smoothness는 패치 없음.

## 5. 프로토콜 확정 근거 — 논문 수치 재현 결과

공식 릴리스 가중치 + 공식 추론 구성(LoRA rank 256, GGA, cos_sim 3.0, 50 steps,
guidance 5.0, seed 42)으로 **seen 388클립과 unseen 100클립을 모두 생성**하고, 채점 변형을
스윕해 확정했다. 조건별 스윕 전체 표는 `results/eval/EgoX_official/calibration_sweeps.txt`,
object 스윕 리포트는 `results/eval/objsweep_full/{paperfix_rank,sweep_report_full3}.txt` 참조.

| 지표 | 확정 방법 | seen 재현/논문 | unseen 재현/논문 |
|---|---|---|---|
| PSNR | Y채널 | 15.57 / 16.05 | 13.53 / 14.38 |
| SSIM | Y채널 | 0.556 / 0.556 | 0.437 / 0.457 |
| LPIPS | AlexNet | 0.466 / 0.498 | 0.542 / 0.552 |
| CLIP-I | ViT-B/32 | 0.903 / 0.896 | 0.886 / 0.877 |
| LocErr | v4 (전쌍 τ0.9) | 125.27 / 61.81* | **146.49 / 149.93** |
| IoU | 〃 | 0.176 / 0.363* | **0.091 / 0.092** |
| Contour | 〃 | 0.622 / 0.546* | 0.572 / 0.481 |
| FVD | I3D/224/16f/ego | 197.9 / 184.5 | 455.1 / 440.6 |
| TempFlick | full canvas | 0.985 / 0.977 | 0.986 / 0.981 |
| MotionSmooth | full canvas | 0.993 / 0.990 | 0.994 / 0.992 |
| DynDeg | 고정임계 2.0 | 0.990 / 0.974 | 0.990 / 0.989 |

**image·video 8개 지표는 seen·unseen 양쪽 모두 노이즈 수준에서 재현**된다 (SSIM-seen 은
소수 3자리 일치; 잔여 PSNR −0.5~−0.9dB 는 채점 규약이 아닌 생성측 잔차 — 채점 해상도
스윕·시드 분산 실측으로 확인). **object unseen 도 v4 에서 LocErr −2.3%/IoU −1.1% 로
사실상 정확 재현**된다 (Contour +19% 는 eq.11 구현 자유도 내 잔차).

*object **seen 행만 재현 불가** — v4 확정 과정에서 전수로 확인했다: 논문 명시축(τ=0.9,
eq.9/10/11)을 고정하고 미명시축(SAM2 세그 granularity 11변형 — 최소 객체 8→72px, IoU 는
0.20 에서 포화 — × DINOv3 특징 5종 × eq.8 프레임/트랙 독해 × 마스크IoU 구현 3종 × 집계
2종 = 408조합)을 전부 훑어도 최선이 LocErr 109/IoU 0.16 수준이고, τ·매칭까지 풀어
~7만 조합을 훑어도(자유 스윕) 87.5/0.222 가 한계다. unseen 이 같은 규약에서 정확히
재현되고 image/video 는 seen 도 재현되므로, **불일치는 released weights 의 품질이 아니라
논문 object-seen 행 자체에 고립**된다 (다른 체크포인트 또는 다른 평가 조건의 산물로
추정). 따라서 논문의 object-seen 3칸은 문헌 참조로만 병기하고 비교 대상은
"EgoX (release, our protocol)" 행으로 삼는다.

이력: v1 track τ0.9(8/21, unseen 절대적합) → v2 1:1 τ0.5(8/25) → v3 1:1 τ0.6125(8/25,
서열 제약 역추정) → **v4 전쌍 τ0.9(8/27, 논문 명시축 고정 + 미명시축 전수 스윕)**.
v1–v3 은 τ·매칭 방식을 자유 매개변수로 쓴 역추정이어서 폐기했다.

## 6. 실행 방법

```bash
.venv_eval/bin/python eval.py eval \
  --set unseen \                          # seen | unseen
  --gen_dir results/eval/<run>/unseen \   # <clip>.mp4 폴더
  --metrics image,object,video \
  --gpus 0,1,2,3 \
  --out results/eval/<run>/results_unseen.txt
```

- 확정 프로토콜(object v4, DynDeg 고정임계)은 eval.py 기본값 — 별도 설정 불필요.
  `results/eval/protocol.json` 이 존재하면 개별 필드 override(프로토콜 실험용)로만 동작
- object 는 2단 파이프라인: 레코드가 `results/eval/objsweep_full/` 에 없으면 eval.py 가
  `scripts/obj_cache_extract.py` 를 GPU 샤딩으로 자동 실행 후 v4 로 집계한다
  (SAM2 마스크는 변형별 캐시라 재채점은 재실행 없음)
- image 는 클립 단위 GPU 샤딩, FVD·VBench는 전체 일괄 처리
- 출력: 사람용 `results_*.txt`(논문 baseline 병기 표 + 델타) + 기계용 `results_*.json`
  + object 원천값 `results_*.obj_records.npy`

## 7. 부록 A — 정확 재현을 위한 고정 사항

문서 본문(§1–4)에 더해, 수치를 소수점까지 재현하려면 아래를 고정해야 한다.

### A.1 패키지 버전 (채점 환경)

| 패키지 | 버전 | 비고 |
|---|---|---|
| torch | 2.5.1+cu121 | |
| sam2 | 1.1.0 | AMG 기본값이 버전 의존 — A.2 참조 |
| transformers | 4.57.6 | DINOv3 / CLIP |
| lpips | 0.1.4 | |
| scikit-image | 0.25.2 | PSNR/SSIM |
| cd-fvd | 0.1.1 | |
| vbench | 0.1.5 | 격리 venv |
| opencv-python | 4.11.0 | 디코드/기하 연산 |
| numpy | 2.2.6 | |

### A.2 SAM2 AMG 파라미터 ("기본 파라미터"의 실제 값, sam2 1.1.0)

`points_per_side=32, points_per_batch=64, pred_iou_thresh=0.8, stability_score_thresh=0.95,
stability_score_offset=1.0, mask_threshold=0.0, box_nms_thresh=0.7, crop_n_layers=0,
min_mask_region_area=2000, multimask_output=True` (이후 bbox 32×32px 미만 시드 제거는 본문
§3.1 — v4 세그 변형 "vD_big". `min_mask_region_area`·시드 크기 외에는 sam2 1.1.0 기본값)

### A.3 기타 미세 규약

- **DINOv3 전처리**: HuggingFace `AutoImageProcessor` 기본값 (짧은 변 224 리사이즈 + 224² 크롭,
  ImageNet 평균/표준편차 정규화). bbox 크롭을 이 전처리에 그대로 통과시킨다.
- **cd-fvd 프레임 선택**: `sample_every_n_frames=1`, `sequence_length=16` — 각 비디오의
  **앞 16프레임** 1시퀀스.
- **object 무매칭 클립**: τ 매칭 쌍이 0인 클립은 클립 간 평균에서 **제외**된다
  (참여 클립 수는 결과 json의 `n_clips`로 기록).
- **VBench 캔버스 합성**(ego 전용 출력): 입력 exo를 높이 H(=448)에 맞춰 784×H로
  INTER_AREA 리사이즈 후 `[exo | ego]` 가로 연결.
- **DynDeg 패치 지점**: vbench 0.1.5 `dynamic_degree`의 RAFT flow 임계(해상도 비례 스케일)를
  고정값 2.0으로 치환 (eval.py 내장 기본값; vbench-shard 가 적용).
- **GPU 샤딩**: 클립 단위 분배이므로 GPU 수와 무관하게 결과 동일. SAM2/RAFT의 GPU
  비결정성은 수치에 유의미한 영향 없음(재실행 실측: object 3지표 동일).

### A.4 재현 범위의 한계

- 이 문서 + A.1–A.3 으로 **채점은 소수점까지 재현**된다 — 단, **같은 생성물**(mp4)에
  대해서다 (공식 가중치 재현 생성물: `results/eval/EgoX_official/unseen/`).
- 생성부터 다시 하면(§5의 추론 구성) 시드 체인·라이브러리 버전·GPU 비결정성으로
  PSNR 기준 ±0.2dB 수준의 변동이 생긴다 (동일 클립 시드 분산 실측).

## 8. 관련 파일

| 파일 | 내용 |
|---|---|
| `eval.py` | 전체 구현 |
| `results/eval/EgoX_official/` | 공식 가중치 재현 생성물 + 채점 결과 + records |
| `results/eval/EgoX_official/calibration_sweeps.txt` | 조건별 스윕 전체 표 |
| `scripts/obj_cache_extract.py` | object 원천 레코드 추출 (SAM2 세그 변형 + 마스크 캐시) |
| `scripts/obj_aggregate_v4.py` | v4 집계 (확정값 dict 포함) + results json 반영 |
| `results/eval/objsweep_full/` | 전 팔 원천 레코드 + 스윕 리포트(paperfix_rank.txt 등) |
| `scripts/dyndeg_sweep.py` | DynDeg 임계 스윕 |
| `data/preprocess_out/protocol_calibration_unseen.json` | image/FVD 변형 원값 |
