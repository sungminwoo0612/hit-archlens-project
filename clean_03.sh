#!/bin/bash
# 원격 저장소 히스토리 정리 및 force push

set -euo pipefail

PROJECT_ROOT="/home/wsm/workspace/hit-archlens-project"
cd "$PROJECT_ROOT"

echo "⚠️  주의: 이 작업은 원격 저장소 히스토리를 덮어씁니다!"
echo "⚠️  다른 사람이 작업 중이라면 반드시 협의하세요!"
echo ""
read -p "계속하시겠습니까? (yes/no): " confirm

if [[ "$confirm" != "yes" ]]; then
    echo "작업이 취소되었습니다."
    exit 1
fi

# 1. 현재 로컬 히스토리 크기 확인
echo ""
echo "📊 현재 로컬 .git 크기:"
du -sh .git

# 2. 원격 추적 브랜치 제거 (안전을 위해)
echo ""
echo "🔗 원격 추적 정보 확인..."
git remote -v

# 3. 원격 저장소와의 연결 확인
REMOTE_URL=$(git remote get-url origin)
echo "원격 저장소: $REMOTE_URL"

# 4. 로컬 히스토리가 정리되었는지 확인
echo ""
echo "🔍 로컬 히스토리에서 큰 파일 확인 (상위 10개):"
LARGE_FILES=$(git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | \
  sort --numeric-sort --key=2 --reverse | \
  head -10 | \
  awk '{printf "%10s %s\n", $1, $2}')

echo "$LARGE_FILES"

# 1MB 이상의 파일이 있는지 확인
LARGE_COUNT=$(git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ && $3 > 1000000 {count++} END {print count+0}')

if [[ "$LARGE_COUNT" -gt 10 ]]; then
    echo ""
    echo "⚠️  경고: 1MB 이상의 큰 파일이 $LARGE_COUNT개 이상 발견되었습니다."
    echo "먼저 clean_history_final.sh를 실행하여 로컬 히스토리를 정리하세요."
    exit 1
fi

echo ""
echo "✅ 로컬 히스토리가 정리된 것으로 확인되었습니다."

# 5. 원격 저장소에 force push
echo ""
echo "🚀 원격 저장소에 정리된 히스토리 푸시 중..."
echo "⚠️  이 작업은 원격 저장소의 히스토리를 완전히 덮어씁니다!"

read -p "정말로 force push 하시겠습니까? (yes/no): " force_confirm

if [[ "$force_confirm" != "yes" ]]; then
    echo "작업이 취소되었습니다."
    exit 1
fi

# force push 실행
git push origin main --force

echo ""
echo "✅ 완료!"
echo "📊 이제 원격 저장소도 정리되었습니다."
echo ""
echo "다른 클론을 사용하는 경우:"
echo "  git fetch origin"
echo "  git reset --hard origin/main"