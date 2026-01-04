#!/bin/bash
# 전체 프로세스: 히스토리 정리 → 원격 푸시

set -euo pipefail

PROJECT_ROOT="/home/wsm/workspace/hit-archlens-project"
cd "$PROJECT_ROOT"

echo "🚀 Git 히스토리 정리 및 원격 푸시 프로세스"
echo "=========================================="
echo ""
echo "이 스크립트는 다음 작업을 수행합니다:"
echo "1. Git 히스토리에서 대용량 파일 제거"
echo "2. 정리된 히스토리를 원격 저장소에 force push"
echo ""
echo "⚠️  주의: 원격 저장소 히스토리가 덮어씌워집니다!"
echo "⚠️  다른 사람이 작업 중이라면 반드시 협의하세요!"
echo ""
read -p "계속하시겠습니까? (yes/no): " confirm

if [[ "$confirm" != "yes" ]]; then
    echo "작업이 취소되었습니다."
    exit 1
fi

# 1단계: 히스토리 정리
echo ""
echo "=========================================="
echo "1단계: Git 히스토리 정리"
echo "=========================================="
./clean_history_final.sh

if [[ $? -ne 0 ]]; then
    echo "❌ 히스토리 정리 실패"
    exit 1
fi

# 2단계: 원격 푸시
echo ""
echo "=========================================="
echo "2단계: 원격 저장소에 푸시"
echo "=========================================="

# 현재 크기 확인
CURRENT_SIZE=$(du -sh .git | awk '{print $1}')
echo "📊 정리된 .git 크기: $CURRENT_SIZE"

# 원격 저장소 확인
REMOTE_URL=$(git remote get-url origin)
echo "🔗 원격 저장소: $REMOTE_URL"

echo ""
read -p "원격 저장소에 force push 하시겠습니까? (yes/no): " push_confirm

if [[ "$push_confirm" != "yes" ]]; then
    echo "푸시가 취소되었습니다."
    exit 0
fi

# Force push
echo ""
echo "🚀 원격 저장소에 푸시 중..."
git push origin main --force

echo ""
echo "✅ 완료!"
echo ""
echo "📊 결과:"
echo "  - 로컬 .git 크기: $CURRENT_SIZE"
echo "  - 원격 저장소도 정리되었습니다."
echo ""
echo "💡 다른 클론을 사용하는 경우:"
echo "  git fetch origin"
echo "  git reset --hard origin/main"

