# ArchLens

AWS 아키텍처 다이어그램 분석을 위한 실험 프로젝트입니다. YOLO 모델을 사용하여 AWS 서비스 아이콘을 자동으로 분류하고 탐지합니다.

## 🎯 주요 기능

- **YOLO Classification**: AWS 아이콘 이미지를 클래스로 분류
- **YOLO Detection**: AWS 아키텍처 다이어그램에서 아이콘 위치 탐지
- **실험 결과 시각화**: 학습 곡선, Confusion Matrix, 예측 결과 시각화

## 🏗️ 아키텍처

```bash
ArchLens/
├── experiments/            # 실험 디렉터리
│   ├── classification/    # YOLO Classification 실험
│   └── detection/         # YOLO Detection 실험
├── backend/               # 백엔드 패키지
├── data/                  # 데이터
├── docs/                  # 문서
└── scripts/               # 스크립트 파일
```


## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd hit_archlens

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는 venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. AWS 아이콘 다운로드

```bash
# AWS 공식 아키텍처 아이콘 다운로드
wget https://d1.awsstatic.com/webteam/architecture-icons/q1-2024/Asset-Package_01242024.7c4f8b8b.zip -O Asset-Package.zip

# 또는 AWS 공식 사이트에서 수동 다운로드:
# https://aws.amazon.com/ko/architecture/icons/
```


## 📁 출력 구조

## 🎯 YOLO Classification 학습

AWS 아이콘 분류를 위한 YOLO 모델 학습 및 사용 방법입니다.

### 빠른 시작

```bash
# 1. 환경 설정
conda activate archlens
./scripts/setup_yolo_env.sh

# 2. 데이터셋 준비 (아직 안 했다면)
python scripts/prepare_yolo_dataset.py --mode fine

# 3. 모델 학습
python scripts/train_yolo_cls.py --mode fine --epochs 100 --imgsz 256

# 4. 모델 평가
python scripts/eval_yolo_cls.py \
    --model runs/classify/fine_cls_*/weights/best.pt \
    --mode fine \
    --split test

# 5. 이미지 예측
# 단일 이미지
python scripts/predict_yolo_cls.py \
    --model runs/classify/fine_cls_yolov8n-cls/weights/best.pt \
    --source "dataset/icons/images/fine/amazon s3/Arch_Amazon-S3_64.png" \
    --mode fine \
    --top-k 5

# 디렉터리 전체
python scripts/predict_yolo_cls.py \
    --model runs/classify/fine_cls_yolov8n-cls/weights/best.pt \
    --source "dataset/icons/images/fine/amazon s3" \
    --mode fine \
    --save-json \
    --save-txt
```

### 상세 가이드

## 📊 실험 결과

### 최고 성능 실험: YOLO Classification (yolo-cls16)

최고 성능을 달성한 YOLO Classification 실험 결과입니다.

**성능 지표:**
- Top-1 Accuracy: 높은 정확도 달성
- Top-5 Accuracy: 우수한 성능

**시각화 결과:**

| 항목 | 이미지 |
|------|--------|
| 학습 곡선 | [Results](experiments/classification/runs/classify/yolo-cls16/results.png) |
| Confusion Matrix | [Confusion Matrix](experiments/classification/runs/classify/yolo-cls16/confusion_matrix.png) |
| 정규화된 Confusion Matrix | [Normalized CM](experiments/classification/runs/classify/yolo-cls16/confusion_matrix_normalized.png) |
| 검증 예측 결과 (Batch 0) | [Val Batch 0](experiments/classification/runs/classify/yolo-cls16/val_batch0_pred.jpg) |
| 검증 예측 결과 (Batch 1) | [Val Batch 1](experiments/classification/runs/classify/yolo-cls16/val_batch1_pred.jpg) |

더 자세한 실험 결과는 [experiments/README.md](experiments/README.md)를 참고하세요.

## 📚 문서

프로젝트의 모든 문서는 `docs/` 디렉터리에 체계적으로 정리되어 있습니다.

- **사용 가이드** (`docs/01_guides/`): 
  - [YOLO 학습 가이드](docs/01_guides/01_yolo_training_guide.md)
  - [추론 가이드](docs/01_guides/02_inference_guide.md)
- **참고 자료** (`docs/02_reference/`): 프로젝트 개요, 모듈 비교, 기술 용어집 등

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🔗 관련 링크

- [AWS 공식 아이콘](https://aws.amazon.com/ko/architecture/icons/)
- [OpenAI API 문서](https://platform.openai.com/docs/)
- [CLIP 모델](https://github.com/openai/CLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.

---

```
conda create -n archlens python=3.11 -y
conda activate archlens
which python ; which python3 ; which pip ; which pip3
conda install ipykernel -y
python -m ipykernel install --user --name archlens --display-name "(archlens)"
jupyter kernelspec list | grep archlens
pip install pandas jupyterlab ipython
pip install -r requirements.txt

```