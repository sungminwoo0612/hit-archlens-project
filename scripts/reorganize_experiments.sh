#!/bin/bash
# 실험 디렉터리 구조 재구성 스크립트
# Classification과 Detection 실험을 experiments/ 디렉터리로 분리

set -e

PROJECT_ROOT="/home/wsm/workspace/hit-archlens-project"
cd "${PROJECT_ROOT}"

echo "🔄 실험 디렉터리 구조 재구성 시작..."
echo ""

# 1. experiments 디렉터리 구조 생성
echo "📁 디렉터리 구조 생성 중..."
mkdir -p experiments/classification/{notebooks,scripts,data,runs,weights}
mkdir -p experiments/detection/{notebooks,scripts,data,runs,experiments,weights}

# 2. Classification 노트북 이동
echo "📓 Classification 노트북 이동 중..."
mv 00_*.ipynb 01_*.ipynb 02_*.ipynb 03_*.ipynb \
   04_*.ipynb 05_*.ipynb 06_*.ipynb 07_*.ipynb \
   08_*.ipynb 09_*.ipynb 10_*.ipynb 11_*.ipynb \
   12_*.ipynb 13_*.ipynb \
   experiments/classification/notebooks/ 2>/dev/null || true

# 3. Detection 노트북 이동
echo "📓 Detection 노트북 이동 중..."
mv 14_*.ipynb experiments/detection/notebooks/ 2>/dev/null || true

# 4. Classification 스크립트 이동
echo "📜 Classification 스크립트 이동 중..."
mv scripts/train_yolo_cls.py experiments/classification/scripts/ 2>/dev/null || true
mv scripts/eval_yolo_cls.py experiments/classification/scripts/ 2>/dev/null || true
mv scripts/predict_yolo_cls.py experiments/classification/scripts/ 2>/dev/null || true

# 5. Detection 스크립트 및 설정 이동
echo "📜 Detection 스크립트 이동 중..."
mv obj_detection/train.py experiments/detection/scripts/ 2>/dev/null || true
mv obj_detection/dataset.yaml experiments/detection/ 2>/dev/null || true

# 6. Classification 데이터 이동
echo "📦 Classification 데이터 이동 중..."
if [ -d "dataset/icons" ]; then
    mv dataset/icons experiments/classification/data/dataset/
fi

# 7. Classification 결과 이동
echo "📊 Classification 결과 이동 중..."
if [ -d "runs/classify" ]; then
    mv runs/classify experiments/classification/runs/
fi

# 8. Detection 데이터 및 결과 이동
echo "📦 Detection 데이터 및 결과 이동 중..."
if [ -d "obj_detection/aws_diagram_data" ]; then
    mv obj_detection/aws_diagram_data experiments/detection/data/
fi
if [ -d "obj_detection/runs" ]; then
    mv obj_detection/runs experiments/detection/
fi
if [ -d "obj_detection/experiments" ]; then
    mv obj_detection/experiments experiments/detection/
fi

# 9. 모델 가중치 분리
echo "⚖️  모델 가중치 분리 중..."
# Classification 가중치
if [ -f "weights/yolov8n-cls.pt" ]; then
    mv weights/yolov8n-cls.pt experiments/classification/weights/
fi
if [ -f "weights/yolo11n-cls.pt" ]; then
    mv weights/yolo11n-cls.pt experiments/classification/weights/
fi

# Detection 가중치
if [ -f "weights/yolov8n.pt" ]; then
    mv weights/yolov8n.pt experiments/detection/weights/
fi
if [ -f "weights/yolov8s.pt" ]; then
    mv weights/yolov8s.pt experiments/detection/weights/
fi
if [ -f "weights/yolov8m.pt" ]; then
    mv weights/yolov8m.pt experiments/detection/weights/
fi
if [ -f "weights/yolo11n.pt" ]; then
    mv weights/yolo11n.pt experiments/detection/weights/
fi

# 루트의 가중치 파일도 이동
if [ -f "yolov8n.pt" ]; then
    mv yolov8n.pt experiments/detection/weights/
fi
if [ -f "yolov8s.pt" ]; then
    mv yolov8s.pt experiments/detection/weights/
fi
if [ -f "yolo11n.pt" ]; then
    mv yolo11n.pt experiments/detection/weights/
fi

# 10. 기타 파일 정리
echo "🧹 기타 파일 정리 중..."
if [ -f "coarse20_icons_result.csv" ]; then
    mv coarse20_icons_result.csv experiments/classification/data/
fi
if [ -f "stage1_classes.txt" ]; then
    mv stage1_classes.txt experiments/classification/data/
fi
if [ -f "stage2_classes.txt" ]; then
    mv stage2_classes.txt experiments/classification/data/
fi

# 11. 빈 디렉터리 정리
echo "🗑️  빈 디렉터리 정리 중..."
rmdir obj_detection 2>/dev/null || true
rmdir obj_classification 2>/dev/null || true
rmdir dataset 2>/dev/null || true
rmdir runs 2>/dev/null || true

echo ""
echo "✅ 실험 디렉터리 구조 재구성 완료!"
echo ""
echo "📋 새로운 구조:"
echo "  experiments/classification/  - 아이콘 분류 실험"
echo "  experiments/detection/      - 다이어그램 객체 탐지 실험"

