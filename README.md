# 🐄 스마트축사 데이터 활용대회 — 발정 소 탐지

> **우수상 (2위)** | 과학기술정보통신부 · NIA · SK주식회사 · 2022.01

[![Competition](https://img.shields.io/badge/Competition-스마트축사_데이터_활용대회-blue)](https://www.nia.or.kr)
[![Award](https://img.shields.io/badge/Award-우수상_2위-silver)](.)
[![Score](https://img.shields.io/badge/Weighted_mAP-0.97434-brightgreen)](.)
[![Framework](https://img.shields.io/badge/Framework-mmdetection-orange)](https://github.com/open-mmlab/mmdetection)

---

## 대회 개요

축사 CCTV 영상에서 **발정 상태인 소를 자동 판별**하여 적기 번식 관리로 생산성을 높이는 AI 모델 개발.

| 항목 | 내용 |
|------|------|
| 주최 | 과학기술정보통신부 · NIA · SK주식회사 |
| 기간 | 2022.01 |
| 과제 | Instance Segmentation — 소 발정 여부 분류 (`normal` / `heat`) |
| 평가 지표 | Weighted mAP |
| 최종 순위 | **2위 (우수상)** |
| 최종 점수 | **Weighted mAP 0.97434** |

---

## 문제 정의

축사 CCTV 영상에서 소의 발정 상태를 사람이 육안으로 모니터링하는 방식은 비효율적이고 누락이 잦습니다.  
이를 AI 기반 인스턴스 세그멘테이션으로 자동화하여 농가의 번식 관리 효율을 높이는 것이 목표입니다.

- **Class 1 (`normal`)**: 일반 상태의 소
- **Class 2 (`heat`)**: 발정 상태의 소

---

## 접근 방법

### 1. 데이터 전처리

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

### 2. 모델 선정

`papers-with-code` 기준 object detection / instance segmentation SOTA를 조사하여  
**Swin Transformer Object Detection** 채택.

| 모델 | 비고 |
|------|------|
| YOLOv5 | 초기 실험 |
| Mask R-CNN | 비교 실험 |
| **Swin Transformer** | **최종 채택** — Weighted mAP **0.97434** |

- Paper: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)
- Code Base: [SwinTransformer/Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)
- Framework: [mmdetection](https://github.com/open-mmlab/mmdetection)

### 3. 이미지 전처리 실험

다양한 전처리 조합을 실험하여 최적 입력 표현 탐색:

- 흑백(Grayscale) 변환
- 채도(Saturation) 채널 분리
- 원본 RGB 비교

### 4. 후처리 — 제출 포맷 변환

모델 출력(RLE mask) → polygon segmentation 변환 후 대회 제출 포맷으로 저장.

---

## 환경

```
OS      : Ubuntu 18.04
GPU     : RTX 3090
CUDA    : 11.0
PyTorch : 1.7.0+cu110
mmcv    : 1.4.0
```

### 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# PyTorch 설치
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# mmcv 설치
pip install mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

# mmdetection 설치
cd [코드디렉토리]
pip install -v -e .

# apex 설치
cd [코드디렉토리]/apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```

---

## 추론 및 학습

```bash
# 추론 (제공 weights 사용)
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/swin/skcomB.py weights.pth \
    --show-dir outputs/images \
    --format-only \
    --eval-options "jsonfile_prefix=outputs/test"

# 학습 후 추론
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/swin/skcomB.py --no-validate
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/swin/skcomB.py work_dirs/skcomB/latest.pth \
    --show-dir outputs/images \
    --format-only \
    --eval-options "jsonfile_prefix=outputs/test"
```

---

## 결과

| 지표 | 점수 |
|------|------|
| Weighted mAP | **0.97434** |
| 최종 순위 | **2위** (우수상) |

이 연구는 **KSC 2023 학술발표**로 이어졌습니다.

---

## 비고

- 코드는 대회 규정 및 데이터 보안상 비공개입니다.
- 방법론 및 결과는 `졸림_코드설명.docx` 기준으로 작성되었습니다.
