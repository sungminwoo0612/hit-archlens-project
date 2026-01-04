#!/bin/bash
# 원격 변경사항 확인 후 히스토리 정리

set -euo pipefail

PROJECT_ROOT="/home/wsm/workspace/hit-archlens-project"
cd "$PROJECT_ROOT"

# 1. 원격의 변경사항만 확인 (다운로드 없이)
echo "🔍 원격 저장소의 최근 커밋 확인..."
git ls-remote origin main

# 2. 원격의 변경사항이 중요한지 확인
echo ""
echo "원격에 있는 커밋 메시지 확인:"
git fetch origin main --dry-run 2>&1 | head -20

# 3. 원격 변경사항이 중요하지 않다면 force push
# 중요하다면 먼저 merge 후 히스토리 정리

echo ""
echo "원격 변경사항을 확인한 후 다음 중 선택하세요:"
echo "1. 원격 변경사항이 중요하지 않음 → force push"
echo "2. 원격 변경사항이 중요함 → 먼저 merge 후 히스토리 정리"