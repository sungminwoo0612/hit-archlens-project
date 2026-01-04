#!/bin/bash
# Git 히스토리에서 대용량 파일 제거

set -euo pipefail

PROJECT_ROOT="/home/wsm/workspace/hit-archlens-project"
cd "$PROJECT_ROOT"

echo "⚠️  주의: 이 스크립트는 Git 히스토리를 영구적으로 변경합니다!"
echo "⚠️  실행 전에 반드시 백업을 생성하세요!"
echo ""
read -p "계속하시겠습니까? (yes/no): " confirm

if [[ "$confirm" != "yes" ]]; then
    echo "작업이 취소되었습니다."
    exit 1
fi

# 백업 생성
echo ""
echo "💾 백업 생성 중..."
BACKUP_DIR="../hit-archlens-project-backup-$(date +%Y%m%d-%H%M%S)"
cp -r "$PROJECT_ROOT" "$BACKUP_DIR"
echo "✅ 백업 완료: $BACKUP_DIR"

# 제거할 경로 목록
PATHS_TO_REMOVE=(
    "data/Asset-Package"
    "data/outputs"
    "experiments/classification/runs"
    "experiments/detection/runs"
    "experiments/classification/data"
    "experiments/detection/data"
    "cache"
    "*.pdf"
    "*.zip"
    "*.pt"
    "*.pkl"
)

echo ""
echo "🗑️  Git 히스토리에서 대용량 파일 제거 중..."

# git filter-branch 사용 (git filter-repo가 없을 경우)
for path in "${PATHS_TO_REMOVE[@]}"; do
    echo "  제거 중: $path"
    git filter-branch --force --index-filter \
        "git rm -rf --cached --ignore-unmatch '$path'" \
        --prune-empty --tag-name-filter cat -- --all 2>/dev/null || true
done

# 또는 git filter-repo 사용 (더 빠르고 안전함, 설치 필요)
# git filter-repo 설치: pip install git-filter-repo
# git filter-repo --path data/Asset-Package --invert-paths
# git filter-repo --path data/outputs --invert-paths
# git filter-repo --path experiments/classification/runs --invert-paths
# git filter-repo --path experiments/detection/runs --invert-paths
# git filter-repo --path cache --invert-paths
# git filter-repo --path-glob '*.pdf' --invert-paths
# git filter-repo --path-glob '*.zip' --invert-paths
# git filter-repo --path-glob '*.pt' --invert-paths
# git filter-repo --path-glob '*.pkl' --invert-paths

echo ""
echo "🧹 Git 히스토리 정리 중..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "✅ 완료!"
echo "📊 새로운 리포지토리 크기:"
du -sh .git