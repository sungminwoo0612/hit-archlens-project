# YOLO Classification 학습 가이드

**카테고리**: 사용 가이드  
**작성일**: 2025-11-28  
**관련 문서**: [02_inference_guide.md](02_inference_guide.md)

AWS 아이콘 분류를 위한 YOLO 모델 학습 및 사용 가이드입니다.

## 📋 목차

1. [환경 설정](#환경-설정)
2. [데이터셋 준비](#데이터셋-준비)
3. [모델 학습](#모델-학습)
4. [모델 평가](#모델-평가)
5. [이미지 예측](#이미지-예측)
6. [고급 사용법](#고급-사용법)

## 🔧 환경 설정

### 1. Conda 환경 활성화

```bash
conda activate archlens
```

### 2. 필수 패키지 설치

```bash
# 자동 설치 스크립트 실행
./scripts/setup_yolo_env.sh
```

또는 수동 설치:

```bash
pip install ultralytics>=8.0.0
pip install opencv-python pillow pandas numpy matplotlib seaborn tqdm pyyaml
```

### 3. 설치 확인

```bash
python -c "from ultralytics import YOLO; print('OK')"
```

## 📦 데이터셋 준비

### Fine-level (64 클래스) 데이터셋 생성

```bash
# 방법 1: Python 스크립트 사용 (권장)
conda activate archlens
python scripts/prepare_yolo_dataset.py --mode fine

# 방법 2: Bash 스크립트 사용
conda activate archlens
./aws_icon_yolo_cls_prepare_and_train.sh fine ./dataset/icons
```

### Coarse-level (19 클래스) 데이터셋 생성

```bash
# 방법 1: Python 스크립트 사용 (권장)
conda activate archlens
python scripts/prepare_yolo_dataset.py --mode coarse

# 방법 2: Bash 스크립트 사용
conda activate archlens
./aws_icon_yolo_cls_prepare_and_train.sh coarse ./dataset/icons
```

데이터셋 구조:
```
dataset/icons/yolo_cls_fine/
├── train/
│   ├── 0/    # amazon api gateway
│   ├── 1/    # amazon athena
│   └── ...
├── val/
│   ├── 0/
│   ├── 1/
│   └── ...
└── test/
    ├── 0/
    ├── 1/
    └── ...
```

## 🚀 모델 학습

### 기본 학습 (Fine-level, 64 클래스)

```bash
python scripts/train_yolo_cls.py \
    --mode fine \
    --epochs 100 \
    --imgsz 256 \
    --batch 16
```

### Coarse-level (19 클래스) 학습

```bash
python scripts/train_yolo_cls.py \
    --mode coarse \
    --epochs 50 \
    --imgsz 128 \
    --batch 32
```

### 고급 옵션

```bash
python scripts/train_yolo_cls.py \
    --mode fine \
    --model yolov8s-cls.pt \      # 더 큰 모델 사용
    --epochs 200 \
    --imgsz 512 \                  # 더 큰 이미지 크기
    --batch 8 \
    --lr0 0.001 \                  # 학습률 조정
    --patience 30 \                # Early stopping
    --device 0 \                   # GPU 지정
    --name my_experiment           # 실험 이름 지정
```

### 학습 재개 (Resume)

```bash
python scripts/train_yolo_cls.py \
    --mode fine \
    --resume runs/classify/fine_cls_v2/weights/last.pt \
    --epochs 200
```

### 주요 파라미터

- `--mode`: `fine` (64 클래스) 또는 `coarse` (19 클래스)
- `--model`: 사전 학습 모델 (`yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt`, `yolov8l-cls.pt`, `yolov8x-cls.pt`)
- `--epochs`: 학습 에포크 수 (기본: 100)
- `--imgsz`: 입력 이미지 크기 (기본: 256)
- `--batch`: 배치 크기 (기본: 16)
- `--device`: 디바이스 (`0`=GPU, `cpu`=CPU, `None`=자동)
- `--lr0`: 초기 학습률 (기본: 0.01)
- `--patience`: Early stopping patience (기본: 50)

## 📊 모델 평가

### Validation 세트 평가

```bash
python scripts/eval_yolo_cls.py \
    --model runs/classify/fine_cls_v2/weights/best.pt \
    --mode fine \
    --split val
```

### Test 세트 평가

```bash
python scripts/eval_yolo_cls.py \
    --model runs/classify/fine_cls_v2/weights/best.pt \
    --mode fine \
    --split test \
    --save-json
```

### 평가 결과

평가 결과는 다음 메트릭을 포함합니다:
- **Top-1 Accuracy**: 가장 높은 확률의 클래스가 정답인 비율
- **Top-5 Accuracy**: 상위 5개 예측 중 정답이 포함된 비율
- **Per-class metrics**: 클래스별 정확도, 정밀도, 재현율

## 🔮 이미지 예측

### 단일 이미지 예측

```bash
python scripts/predict_yolo_cls.py \
    --model runs/classify/fine_cls_v2/weights/best.pt \
    --source dataset/icons/images/amazon-s3/amazon-s3.png \
    --mode fine \
    --top-k 5
```

### 디렉터리 내 모든 이미지 예측

```bash
python scripts/predict_yolo_cls.py \
    --model runs/classify/fine_cls_v2/weights/best.pt \
    --source dataset/icons/images \
    --mode fine \
    --save-txt \
    --save-json
```

### 예측 결과 저장

- `--save-txt`: 텍스트 형식으로 결과 저장
- `--save-json`: JSON 형식으로 결과 저장
- `--top-k`: 상위 K개 예측 출력 (기본: 5)

## 🎯 고급 사용법

### 1. 커스텀 학습 설정

학습 스크립트를 직접 수정하여 다음을 조정할 수 있습니다:
- 데이터 증강 (augmentation) 파라미터
- 옵티마이저 설정
- 학습률 스케줄러
- Loss 함수 가중치

### 2. 모델 비교

여러 모델을 학습하고 비교:

```bash
# 작은 모델
python scripts/train_yolo_cls.py --mode fine --model yolov8n-cls.pt --name fine_nano

# 중간 모델
python scripts/train_yolo_cls.py --mode fine --model yolov8s-cls.pt --name fine_small

# 큰 모델
python scripts/train_yolo_cls.py --mode fine --model yolov8m-cls.pt --name fine_medium
```

### 3. 하이퍼파라미터 튜닝

학습률, 배치 크기, 이미지 크기 등을 조정하여 성능 최적화:

```bash
# 높은 해상도로 학습
python scripts/train_yolo_cls.py \
    --mode fine \
    --imgsz 512 \
    --batch 4 \
    --epochs 150

# 낮은 학습률로 fine-tuning
python scripts/train_yolo_cls.py \
    --mode fine \
    --lr0 0.001 \
    --epochs 200 \
    --patience 50
```

### 4. 클래스 불균형 처리

데이터셋에 클래스 불균형이 있는 경우:
- 클래스 가중치 조정
- 데이터 증강 강화
- Focal Loss 사용 (코드 수정 필요)

## 📁 출력 파일 구조

학습 후 생성되는 파일:

```
runs/classify/fine_cls_v2/
├── weights/
│   ├── best.pt          # 최고 성능 모델
│   └── last.pt          # 마지막 체크포인트
├── args.yaml            # 학습 설정
├── results.csv          # 학습 메트릭
├── confusion_matrix.png # 혼동 행렬
├── results.png          # 학습 곡선
└── ...
```

## 🐛 문제 해결

### CUDA Out of Memory

배치 크기나 이미지 크기를 줄이세요:

```bash
python scripts/train_yolo_cls.py --mode fine --batch 8 --imgsz 128
```

### 데이터셋을 찾을 수 없음

데이터셋이 준비되었는지 확인:

```bash
ls -la dataset/icons/yolo_cls_fine/train/
```

### 모델 다운로드 실패

수동으로 모델을 다운로드하거나 프록시 설정을 확인하세요.

## 📚 참고 자료

- [Ultralytics YOLO 문서](https://docs.ultralytics.com/)
- [YOLOv8 Classification 가이드](https://docs.ultralytics.com/tasks/classify/)
- 프로젝트 노트북:
  - `01_taxonomy_definition.ipynb`: 분류 체계 정의
  - `02_icon_mapping_from_assets.ipynb`: 아이콘 매핑
  - `03_icon_dataset_build_and_stats.ipynb`: 데이터셋 통계

## 💡 팁

1. **작은 모델로 시작**: `yolov8n-cls.pt`로 빠르게 프로토타입을 테스트한 후 필요시 더 큰 모델 사용
2. **이미지 크기 조정**: 아이콘은 작은 이미지이므로 `imgsz=128` 또는 `imgsz=256`이 적절할 수 있음
3. **Early Stopping 활용**: `--patience` 파라미터로 과적합 방지
4. **Validation 모니터링**: 학습 중 validation accuracy를 모니터링하여 최적의 모델 선택
5. **데이터 증강**: 작은 데이터셋의 경우 데이터 증강이 중요함

