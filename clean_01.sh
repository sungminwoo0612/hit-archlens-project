#!/bin/bash
# Git 히스토리에서 대용량 파일 찾기 및 제거

set -euo pipefail

PROJECT_ROOT="/home/wsm/workspace/hit-archlens-project"
cd "$PROJECT_ROOT"

echo "🔍 Git 히스토리에서 큰 파일 찾기..."
echo ""

# 1. 큰 파일 찾기 (상위 20개)
echo "📊 히스토리에서 가장 큰 파일 Top 20:"
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | \
  sort --numeric-sort --key=2 --reverse | \
  head -20 | \
  awk '{printf "%10s %s\n", $1, $2}'

echo ""
echo "📦 큰 파일들의 총 크기 계산 중..."
LARGE_FILES=$(git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | \
  sort --numeric-sort --key=2 --reverse | \
  head -20 | \
  awk '{sum += $1} END {print sum}')

echo "상위 20개 파일 총 크기: $(numfmt --to=iec-i --suffix=B $LARGE_FILES 2>/dev/null || echo "${LARGE_FILES} bytes")"