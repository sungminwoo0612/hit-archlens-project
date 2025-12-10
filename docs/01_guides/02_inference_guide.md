# YOLO Classification 추론 가이드

**카테고리**: 사용 가이드  
**작성일**: 2025-11-28  
**관련 문서**: [01_yolo_training_guide.md](01_yolo_training_guide.md)

학습된 YOLO 분류 모델을 사용하여 AWS 아이콘을 예측하는 방법입니다.

## 🚀 빠른 시작

### 1. 단일 이미지 추론

```bash
conda activate archlens

# 단일 이미지 예측
python scripts/predict_yolo_cls.py \
    --model runs/classify/fine_cls_yolov8n-cls/weights/best.pt \
    --source "dataset/icons/images/fine/amazon s3/Arch_Amazon-S3_64.png" \
    --mode fine \
    --top-k 5
```

### 2. 디렉터리 전체 추론

```bash
# 특정 서비스의 모든 이미지 예측
python scripts/predict_yolo_cls.py \
    --model runs/classify/fine_cls_yolov8n-cls/weights/best.pt \
    --source "dataset/icons/images/fine/amazon s3" \
    --mode fine \
    --top-k 5 \
    --save-json \
    --save-txt
```

### 3. 전체 데이터셋 추론

```bash
# 모든 fine 이미지 예측
python scripts/predict_yolo_cls.py \
    --model runs/classify/fine_cls_yolov8n-cls/weights/best.pt \
    --source dataset/icons/images/fine \
    --mode fine \
    --save-json \
    --save-txt
```

## 📋 주요 옵션

### 필수 옵션

- `--model`: 학습된 모델 경로 (예: `runs/classify/fine_cls_yolov8n-cls/weights/best.pt`)
- `--source`: 입력 이미지 또는 디렉터리 경로
- `--mode`: `fine` (64 클래스) 또는 `coarse` (19 클래스)

### 선택 옵션

- `--top-k`: 상위 K개 예측 출력 (기본: 5)
- `--imgsz`: 입력 이미지 크기 (기본: 256, 학습 시 사용한 크기와 동일하게)
- `--conf`: 신뢰도 임계값 (기본: 0.25)
- `--save-json`: JSON 형식으로 결과 저장
- `--save-txt`: 텍스트 형식으로 결과 저장
- `--device`: 디바이스 지정 (`0`=GPU, `cpu`=CPU, `None`=자동)

## 📁 출력 결과

### 콘솔 출력

```
Arch_Amazon-S3_64.png:
  amazon s3: 0.9234
  amazon s3 glacier: 0.0456
  aws storage gateway: 0.0123
  ...
```

### JSON 결과 (`--save-json`)

```json
[
  {
    "image_path": "dataset/icons/images/fine/amazon s3/Arch_Amazon-S3_64.png",
    "predictions": [
      {
        "class_id": 27,
        "class_name": "amazon s3",
        "confidence": 0.9234
      },
      {
        "class_id": 28,
        "class_name": "amazon s3 glacier",
        "confidence": 0.0456
      }
    ]
  }
]
```

### 텍스트 결과 (`--save-txt`)

```
dataset/icons/images/fine/amazon s3/Arch_Amazon-S3_64.png
  amazon s3: 0.9234
  amazon s3 glacier: 0.0456
  aws storage gateway: 0.0123
```

결과는 `runs/classify/fine_cls_yolov8n-cls/predict/` 디렉터리에 저장됩니다.

## 🎯 사용 예제

### 예제 1: 특정 서비스 아이콘 테스트

```bash
# EC2 아이콘 예측
python scripts/predict_yolo_cls.py \
    --model runs/classify/fine_cls_yolov8n-cls/weights/best.pt \
    --source "dataset/icons/images/fine/amazon ec2" \
    --mode fine \
    --top-k 3 \
    --save-json
```

### 예제 2: Test 세트 전체 평가

```bash
# Test 세트의 모든 이미지 예측
python scripts/predict_yolo_cls.py \
    --model runs/classify/fine_cls_yolov8n-cls/weights/best.pt \
    --source dataset/icons/yolo_cls_fine/test \
    --mode fine \
    --save-json \
    --save-txt
```

### 예제 3: 커스텀 이미지 예측

```bash
# 외부 이미지 파일 예측
python scripts/predict_yolo_cls.py \
    --model runs/classify/fine_cls_yolov8n-cls/weights/best.pt \
    --source /path/to/your/icon.png \
    --mode fine \
    --top-k 5
```

## 🔍 결과 분석

### 정확도 확인

```bash
# JSON 결과를 사용하여 정확도 계산
python -c "
import json
from pathlib import Path

results = json.load(open('runs/classify/fine_cls_yolov8n-cls/predict/predictions.json'))
correct = 0
total = 0

for item in results:
    img_path = Path(item['image_path'])
    # 실제 클래스는 디렉터리 이름에서 추출 가능
    true_class = img_path.parent.name
    pred_class = item['predictions'][0]['class_name']
    
    if true_class == pred_class:
        correct += 1
    total += 1

print(f'정확도: {correct/total:.2%} ({correct}/{total})')
"
```

### 혼동 행렬 생성

```python
# confusion_matrix.py
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

results = json.load(open('runs/classify/fine_cls_yolov8n-cls/predict/predictions.json'))

y_true = []
y_pred = []

for item in results:
    img_path = Path(item['image_path'])
    true_class = img_path.parent.name
    pred_class = item['predictions'][0]['class_name']
    
    y_true.append(true_class)
    y_pred.append(pred_class)

# 혼동 행렬 생성
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.savefig('confusion_matrix.png')
```

## 💡 팁

1. **이미지 크기**: 학습 시 사용한 `imgsz`와 동일한 크기로 추론하는 것이 좋습니다.

2. **Top-K 값**: 불확실한 예측의 경우 `--top-k 10`으로 상위 10개를 확인해보세요.

3. **배치 처리**: 디렉터리 전체를 한 번에 처리하면 더 빠릅니다.

4. **결과 저장**: `--save-json`과 `--save-txt`를 함께 사용하면 다양한 형식으로 결과를 분석할 수 있습니다.

5. **GPU 사용**: GPU가 있다면 자동으로 사용되며, `--device 0`으로 명시적으로 지정할 수 있습니다.

## 🐛 문제 해결

### 모델을 찾을 수 없음

```bash
# 사용 가능한 모델 확인
find runs/classify -name "best.pt"
```

### 이미지를 찾을 수 없음

```bash
# 이미지 경로 확인 (공백이 있는 경우 따옴표 사용)
ls "dataset/icons/images/fine/amazon s3"
```

### 메모리 부족

```bash
# 배치 크기 줄이기 (스크립트 내부에서 자동 처리)
# 또는 이미지를 작은 그룹으로 나누어 처리
```

## 📚 관련 문서

- [01_yolo_training_guide.md](01_yolo_training_guide.md): 학습 가이드
- [프로젝트 README](../../README.md): 프로젝트 개요

