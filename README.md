<div align="center">

# 🐄 스마트축사 데이터 활용대회
## 발정 소 자동 탐지 — Instance Segmentation

<br>

🥈 **우수상 (2위)** &nbsp;|&nbsp; 과학기술정보통신부 · NIA · SK주식회사 &nbsp;|&nbsp; 2022.01

<br>

[![Award](https://img.shields.io/badge/🥈_Award-우수상_2위-silver?style=for-the-badge)](.)
[![Score](https://img.shields.io/badge/Weighted_mAP-0.97434-brightgreen?style=for-the-badge&logo=checkmarx&logoColor=white)](.)
[![Paper](https://img.shields.io/badge/Paper-Swin_Transformer_ICCV_2021-blue?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2103.14030)
[![Framework](https://img.shields.io/badge/Framework-mmdetection-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/open-mmlab/mmdetection)
[![KSC](https://img.shields.io/badge/Follow--up-KSC_2023_학술발표-8B5CF6?style=for-the-badge&logo=academia&logoColor=white)](.)

</div>

---

## 📋 대회 개요

축사 CCTV 영상에서 **발정 상태인 소를 자동 판별**하여 적기 번식 관리를 가능하게 하는 AI 모델 개발 대회.

| 항목 | 내용 |
|------|------|
| 🏛️ 주최 | 과학기술정보통신부 · NIA · SK주식회사 |
| 📅 기간 | 2022년 1월 |
| 🎯 과제 | Instance Segmentation — 소 발정 여부 분류 (`normal` / `heat`) |
| 📊 평가지표 | Weighted mAP |
| 🥈 최종 순위 | **2위 (우수상)** |
| ✅ 최종 점수 | **Weighted mAP 0.97434** |

---

## 🎯 문제 정의

축사 CCTV 영상에서 소의 발정 상태를 육안으로 모니터링하는 방식은 **비효율적이고 누락이 발생하기 쉽습니다.**
AI 기반 Instance Segmentation으로 이를 자동화하여 농가의 번식 관리 효율을 높이는 것이 목표입니다.

```
🟢 Class 1 (normal) : 일반 상태의 소  →  초록 마스크
🟣 Class 2 (heat)   : 발정 상태의 소  →  보라 마스크
```

> **추론 출력**: confidence score + bounding box + segmentation mask 동시 출력

---

## 🖼️ 추론 결과 예시

![inference result](assets/inference_result.jpg)

> 초록 마스크(`normal`) / 보라 마스크(`heat`)로 발정 소를 인스턴스 단위로 구분.

---

## 🔬 접근 방법

### 1️⃣ 데이터 전처리

대회 제공 어노테이션을 **COCO 포맷**으로 변환:

- Segmentation polygon → bbox (`xmin, ymin, width, height`) 자동 계산
- Category ID 매핑 (`normal`: 1, `heat`: 2)
- Mask area 계산 후 annotation에 반영

```
CODE/
├── data/
│   └── sk/
│       ├── train_images/
│       ├── test_images/
│       └── annotations/
├── preparing_datasets.ipynb   # 데이터 전처리
└── to_submission.ipynb        # 예측 결과 → 제출 포맷 변환
```

---

### 2️⃣ 이미지 전처리 실험

다양한 전처리 조합을 실험했으나 유의미한 성능 향상이 없어 **원본 RGB를 그대로 사용**했습니다.

| 전처리 방법 | 결과 |
|------------|:----:|
| Grayscale 변환 | ❌ |
| Saturation 채널 분리 | ❌ |
| 원본 RGB | ✅ **최종 채택** |

> 💡 축사 CCTV 특성상 조명·환경이 일정하여 색상 전처리의 효과가 제한적이었습니다.
> 이후 전처리보다 **모델 선택**에 집중하는 전략으로 전환했습니다.

---

### 3️⃣ 모델 선정

Papers with Code 기준 Object Detection / Instance Segmentation SOTA를 조사하여
YOLOv5, Mask R-CNN과의 비교 실험 끝에 **Swin Transformer**를 최종 채택했습니다.

| | |
|---|---|
| 📄 **Paper** | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) |
| 🏆 **수상** | ICCV 2021 Best Paper Award |
| 🏢 **저자** | Ze Liu et al., Microsoft Research Asia |
| 💻 **Code** | [SwinTransformer/Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) |
| 🔧 **Framework** | [mmdetection](https://github.com/open-mmlab/mmdetection) |

---

## 🧠 Swin Transformer

![Swin Transformer Architecture](https://raw.githubusercontent.com/microsoft/Swin-Transformer/main/figures/teaser.png)

> *출처: Microsoft Research Asia, ICCV 2021*

Swin Transformer는 ViT의 한계였던 **고해상도 이미지 처리 비용 문제**를 해결한 계층형 비전 트랜스포머입니다.

### 💡 핵심 아이디어: Shifted Window Self-Attention

기존 ViT는 전체 이미지를 단일 시퀀스로 처리해 연산량이 이미지 크기의 **제곱에 비례**했습니다.
Swin Transformer는 **로컬 윈도우** 내에서만 self-attention을 수행하고,
레이어마다 윈도우를 Shift시켜 인접 윈도우 간 정보 교환을 가능하게 합니다.

| 구분 | ViT | Swin Transformer |
|:----:|:---:|:---------------:|
| Attention 범위 | 전체 이미지 | 로컬 윈도우 |
| 연산 복잡도 | O(n²) | O(n) |
| 다중 스케일 피처 | ✗ | ✅ |
| 계층적 구조 | ✗ | ✅ |
| 객체 탐지 적합성 | 낮음 | **높음** |

### 🏗️ 계층형 구조 (Hierarchical Feature Maps)

```
입력 (H × W × 3)
  → Stage 1: H/4  × W/4  × 48C   (Swin Block × 2)
  → Stage 2: H/8  × W/8  × 96C   (Swin Block × 2)
  → Stage 3: H/16 × W/16 × 192C  (Swin Block × 6)
  → Stage 4: H/32 × W/32 × 384C  (Swin Block × 2)
```

> 이 다중 스케일 피처맵이 Instance Segmentation에 특히 강점을 가집니다.

**벤치마크 (ICCV 2021 논문 기준)**

| 데이터셋 | 지표 | Swin-L |
|:-------:|:----:|:------:|
| ImageNet-1K | Top-1 Acc | 87.3% |
| COCO | Box AP | 58.7 |
| COCO | Mask AP | 51.1 |

---

## ⚙️ 개발 환경

```
OS      : Ubuntu 18.04
GPU     : RTX 3090
CUDA    : 11.0
PyTorch : 1.7.0+cu110
mmcv    : 1.4.0
```

### 📦 설치

```bash
python -m venv venv && source venv/bin/activate

pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

cd [코드디렉토리] && pip install -v -e .
cd [코드디렉토리]/apex && pip install -v --disable-pip-version-check --no-cache-dir ./
```

---

## 🚀 실행

```bash
# 추론 (제공 weights 사용)
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/swin/skcomB.py weights.pth \
    --show-dir outputs/images \
    --format-only \
    --eval-options "jsonfile_prefix=outputs/test"

# 학습
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/swin/skcomB.py --no-validate

# 학습 후 추론
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/swin/skcomB.py work_dirs/skcomB/latest.pth \
    --show-dir outputs/images \
    --format-only \
    --eval-options "jsonfile_prefix=outputs/test"
```

---

## 📊 결과

<div align="center">

YOLOv5, Mask R-CNN 등 후보 모델과의 비교 실험을 거쳐
**Swin Transformer 기반 Instance Segmentation 예측 모델**을 구성했습니다.

| 지표 | 점수 |
|:----:|:----:|
| Weighted mAP | **0.97434** |
| 최종 순위 | **🥈 2위 (우수상)** |

<br>

본 연구는 **KSC 2023 학술발표**로 이어졌습니다.

> *"Swin Transformer 모델을 적용한 한우 발정 탐지"*, KSC 2023, pp.609–611

</div>

---

## 📎 비고

> ⚠️ 코드는 대회 규정 및 데이터 보안상 **비공개**입니다.
