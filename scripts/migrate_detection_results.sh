#!/bin/bash
# Detection 실험 결과 마이그레이션 스크립트
# hit-aws-object-detection-project에서 hit-archlens-project로 결과 파일 이동

set -e

SOURCE_PROJECT="/home/wsm/workspace/hit-aws-object-detection-project"
TARGET_PROJECT="/home/wsm/workspace/hit-archlens-project"

echo "🔄 Detection 실험 결과 마이그레이션 시작..."
echo ""

# Detection 결과 복사
DETECTION_EXPERIMENTS=(
    "aws_diagram_yolov8m_v1"
    "aws_icon_detector2"
    "aws_icon_detector3"
    "aws_icon_detector4"
    "aws_icon_detector5"
    "aws_icon_detector6"
    "aws_icon_detector7"
)

TARGET_DIR="${TARGET_PROJECT}/obj_detection/runs"
mkdir -p "${TARGET_DIR}"

for exp in "${DETECTION_EXPERIMENTS[@]}"; do
    SOURCE_PATH="${SOURCE_PROJECT}/runs/detect/${exp}"
    TARGET_PATH="${TARGET_DIR}/${exp}"
    
    if [ -d "${SOURCE_PATH}" ]; then
        if [ -d "${TARGET_PATH}" ]; then
            echo "⚠️  이미 존재: ${exp} (건너뜀)"
        else
            echo "📦 복사 중: ${exp}"
            cp -r "${SOURCE_PATH}" "${TARGET_PATH}"
            echo "   ✅ 완료: ${TARGET_PATH}"
        fi
    else
        echo "⚠️  경로 없음: ${SOURCE_PATH}"
    fi
done

echo ""
echo "🔄 Classification 결과 확인 중..."

# Classification 결과도 확인
CLASSIFY_SOURCE="${SOURCE_PROJECT}/runs/classify"
CLASSIFY_TARGET="${TARGET_PROJECT}/runs/classify"

if [ -d "${CLASSIFY_SOURCE}/predict3" ]; then
    echo "📦 Classification predict3 결과 발견"
    mkdir -p "${CLASSIFY_TARGET}"
    if [ ! -d "${CLASSIFY_TARGET}/predict3_migrated" ]; then
        cp -r "${CLASSIFY_SOURCE}/predict3" "${CLASSIFY_TARGET}/predict3_migrated"
        echo "   ✅ 복사 완료: ${CLASSIFY_TARGET}/predict3_migrated"
    else
        echo "   ⚠️  이미 존재: predict3_migrated"
    fi
fi

echo ""
echo "✅ 마이그레이션 완료!"

