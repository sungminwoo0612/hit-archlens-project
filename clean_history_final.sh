#!/bin/bash
# Git 히스토리에서 대용량 파일 완전 제거 (git-filter-repo 사용)

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

# git-filter-repo 또는 git filter-branch 사용
echo ""
echo "🔍 Git 히스토리 정리 도구 확인..."

USE_FILTER_REPO=false
if command -v git-filter-repo &> /dev/null; then
    USE_FILTER_REPO=true
    echo "✅ git-filter-repo 사용 가능"
else
    echo "⚠️  git-filter-repo가 없습니다. git filter-branch를 사용합니다."
    echo "   (git filter-branch는 더 느리지만 Git에 기본 포함되어 있습니다)"
    
    # git-filter-repo 설치 시도 (선택사항)
    read -p "git-filter-repo를 설치하시겠습니까? (더 빠름) (yes/no): " install_confirm
    if [[ "$install_confirm" == "yes" ]]; then
        # pipx로 설치 시도
        if command -v pipx &> /dev/null; then
            echo "  pipx를 사용하여 설치 중..."
            pipx install git-filter-repo
            USE_FILTER_REPO=true
        # pip --break-system-packages로 설치 시도
        elif command -v pip &> /dev/null; then
            echo "  pip --break-system-packages를 사용하여 설치 중..."
            pip install --break-system-packages git-filter-repo
            if command -v git-filter-repo &> /dev/null; then
                USE_FILTER_REPO=true
            fi
        fi
    fi
fi

# 현재 크기 확인
echo ""
echo "📊 현재 .git 크기:"
du -sh .git

# 히스토리에서 제거할 경로들
echo ""
echo "🗑️  Git 히스토리에서 대용량 파일 제거 중..."

# git-filter-repo로 히스토리 정리
# 주의: 소스 코드, 노트북, 설정 파일 등은 유지하고 생성된 대용량 파일만 제거
echo "  제거 대상:"
echo "    - 레거시 디렉터리 (aws_cv_clip, aws_llm_autolabel, aws_data_collectors, archive)"
echo "    - 대용량 데이터 (data/Asset-Package, data/outputs, data/images, data/labels, data/yolo_diagrams)"
echo "    - 실험 결과 (experiments/*/runs, experiments/*/data)"
echo "    - 캐시 및 임시 파일 (cache, out)"
echo "    - 대용량 바이너리 (PDF, ZIP, 모델 가중치, 캐시 파일)"
echo ""
echo "  유지 대상:"
echo "    - 소스 코드 (*.py, *.sh)"
echo "    - 노트북 파일 (*.ipynb)"
echo "    - 설정 파일 (*.yaml, *.yml, *.json, *.toml)"
echo "    - 문서 파일 (*.md, *.txt)"
echo ""

if [[ "$USE_FILTER_REPO" == "true" ]]; then
    # git-filter-repo 사용 (빠름)
    git filter-repo \
        --path aws_cv_clip --invert-paths \
        --path aws_llm_autolabel --invert-paths \
        --path aws_data_collectors --invert-paths \
        --path archive --invert-paths \
        --path data/Asset-Package --invert-paths \
        --path data/outputs --invert-paths \
        --path data/images --invert-paths \
        --path data/labels --invert-paths \
        --path data/yolo_diagrams --invert-paths \
        --path experiments/classification/runs --invert-paths \
        --path experiments/detection/runs --invert-paths \
        --path experiments/classification/data --invert-paths \
        --path experiments/detection/data --invert-paths \
        --path cache --invert-paths \
        --path out --invert-paths \
        --path-glob '*.pdf' --invert-paths \
        --path-glob '*.zip' --invert-paths \
        --path-glob '*.pt' --invert-paths \
        --path-glob '*.pkl' --invert-paths \
        --force
else
    # git filter-branch 사용 (느리지만 기본 포함)
    echo "  git filter-branch를 사용하여 히스토리 정리 중..."
    echo "  (이 작업은 시간이 오래 걸릴 수 있습니다)"
    
    PATHS_TO_REMOVE=(
        "aws_cv_clip"
        "aws_llm_autolabel"
        "aws_data_collectors"
        "archive"
        "data/Asset-Package"
        "data/outputs"
        "data/images"
        "data/labels"
        "data/yolo_diagrams"
        "experiments/classification/runs"
        "experiments/detection/runs"
        "experiments/classification/data"
        "experiments/detection/data"
        "cache"
        "out"
    )
    
    for path in "${PATHS_TO_REMOVE[@]}"; do
        echo "    제거 중: $path"
        git filter-branch --force --index-filter \
            "git rm -rf --cached --ignore-unmatch '$path'" \
            --prune-empty --tag-name-filter cat -- --all 2>/dev/null || true
    done
    
    # 파일 패턴 제거
    echo "    제거 중: *.pdf, *.zip, *.pt, *.pkl"
    git filter-branch --force --index-filter \
        "git rm -rf --cached --ignore-unmatch '*.pdf' '*.zip' '*.pt' '*.pkl' 2>/dev/null || true" \
        --prune-empty --tag-name-filter cat -- --all 2>/dev/null || true
fi

# 히스토리 정리
echo ""
echo "🧹 Git 히스토리 정리 중..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "✅ 완료!"
echo "📊 새로운 리포지토리 크기:"
du -sh .git

echo ""
echo "🔍 남은 큰 파일 확인 (상위 10개):"
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | \
  sort --numeric-sort --key=2 --reverse | \
  head -10 | \
  awk '{printf "%10s %s\n", $1, $2}'

