#!/usr/bin/env bash
# 디렉터리 구조 재구성 스크립트
# 주의: 실행 전에 반드시 백업을 생성하세요!

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 프로젝트 루트 디렉터리
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}=== Hit ArchLens 디렉터리 구조 재구성 ===${NC}\n"

# 백업 확인
echo -e "${YELLOW}⚠️  주의: 이 스크립트는 디렉터리 구조를 변경합니다.${NC}"
echo -e "${YELLOW}⚠️  실행 전에 Git 커밋 또는 백업을 생성하시기 바랍니다.${NC}"
read -p "계속하시겠습니까? (yes/no): " confirm

if [[ "$confirm" != "yes" ]]; then
    echo -e "${RED}작업이 취소되었습니다.${NC}"
    exit 1
fi

# Phase 1: 새 디렉터리 생성 및 파일 이동
echo -e "\n${BLUE}[Phase 1] 새 디렉터리 생성 및 파일 이동${NC}"

# 1.1 scripts/ 디렉터리 생성 및 스크립트 파일 이동
echo -e "${GREEN}1.1 scripts/ 디렉터리 생성 및 스크립트 파일 이동${NC}"
mkdir -p scripts

if [[ -f "connect_lmstudio_chat.py" ]]; then
    mv connect_lmstudio_chat.py scripts/
    echo "  ✓ connect_lmstudio_chat.py → scripts/"
fi

if [[ -f "connect_lmstudio_embed.py" ]]; then
    mv connect_lmstudio_embed.py scripts/
    echo "  ✓ connect_lmstudio_embed.py → scripts/"
fi

if [[ -f "connect_local_llm.py" ]]; then
    mv connect_local_llm.py scripts/
    echo "  ✓ connect_local_llm.py → scripts/"
fi

if [[ -f "auto_aws_cv_clip.sh" ]]; then
    mv auto_aws_cv_clip.sh scripts/
    echo "  ✓ auto_aws_cv_clip.sh → scripts/"
fi

if [[ -f "auto_label_aws.sh" ]]; then
    mv auto_label_aws.sh scripts/
    echo "  ✓ auto_label_aws.sh → scripts/"
fi

# 1.2 docs/ 디렉터리 생성 및 문서 이동
echo -e "\n${GREEN}1.2 docs/ 디렉터리 생성 및 문서 이동${NC}"
if [[ -d "notes" ]]; then
    mkdir -p docs
    mv notes/* docs/ 2>/dev/null || true
    rmdir notes 2>/dev/null || true
    echo "  ✓ notes/ → docs/"
fi

# 1.3 examples/ 디렉터리 생성 및 예제 파일 이동
echo -e "\n${GREEN}1.3 examples/ 디렉터리 생성 및 예제 파일 이동${NC}"
mkdir -p examples

if [[ -d "example_collected_icons" ]]; then
    mv example_collected_icons examples/collected_icons
    echo "  ✓ example_collected_icons/ → examples/collected_icons/"
fi

if [[ -d "example_github_icons" ]]; then
    mv example_github_icons examples/github_icons
    echo "  ✓ example_github_icons/ → examples/github_icons/"
fi

# 1.4 out/performance/ 디렉터리 생성 및 성능 테스트 결과 이동
echo -e "\n${GREEN}1.4 out/performance/ 디렉터리 생성 및 성능 테스트 결과 이동${NC}"
if [[ -d "performance_test_results" ]]; then
    mkdir -p out/performance
    mv performance_test_results/* out/performance/ 2>/dev/null || true
    rmdir performance_test_results 2>/dev/null || true
    echo "  ✓ performance_test_results/ → out/performance/"
fi

# Phase 2: 레거시 디렉터리 백업
echo -e "\n${BLUE}[Phase 2] 레거시 디렉터리 백업${NC}"

# 백업 디렉터리 생성
mkdir -p archive/legacy

# 레거시 디렉터리 백업
if [[ -d "aws_cv_clip" ]]; then
    echo -e "${GREEN}aws_cv_clip/ 백업 중...${NC}"
    cp -r aws_cv_clip archive/legacy/ 2>/dev/null || true
    echo "  ✓ aws_cv_clip/ → archive/legacy/aws_cv_clip/"
fi

if [[ -d "aws_llm_autolabel" ]]; then
    echo -e "${GREEN}aws_llm_autolabel/ 백업 중...${NC}"
    cp -r aws_llm_autolabel archive/legacy/ 2>/dev/null || true
    echo "  ✓ aws_llm_autolabel/ → archive/legacy/aws_llm_autolabel/"
fi

if [[ -d "aws_data_collectors" ]]; then
    echo -e "${GREEN}aws_data_collectors/ 백업 중...${NC}"
    cp -r aws_data_collectors archive/legacy/ 2>/dev/null || true
    echo "  ✓ aws_data_collectors/ → archive/legacy/aws_data_collectors/"
fi

# Phase 3: 레거시 디렉터리 의존성 확인
echo -e "\n${BLUE}[Phase 3] 레거시 디렉터리 의존성 확인${NC}"

check_dependency() {
    local module=$1
    local count=$(grep -r "from ${module}\|import ${module}" . \
        --exclude-dir=.git \
        --exclude-dir=.venv \
        --exclude-dir=archive \
        --exclude-dir=node_modules \
        2>/dev/null | wc -l || echo "0")
    
    if [[ "$count" -gt 0 ]]; then
        echo -e "${YELLOW}  ⚠️  ${module} 참조 발견: ${count}개${NC}"
        return 1
    else
        echo -e "${GREEN}  ✓ ${module} 참조 없음${NC}"
        return 0
    fi
}

echo -e "${GREEN}의존성 검사 중...${NC}"
check_dependency "aws_cv_clip"
check_dependency "aws_llm"
check_dependency "aws_data_collectors"

# Phase 4: 레거시 디렉터리 제거 확인
echo -e "\n${BLUE}[Phase 4] 레거시 디렉터리 제거${NC}"
echo -e "${YELLOW}⚠️  레거시 디렉터리를 제거하시겠습니까?${NC}"
echo -e "${YELLOW}⚠️  백업은 archive/legacy/에 저장되었습니다.${NC}"
read -p "레거시 디렉터리를 제거하시겠습니까? (yes/no): " remove_confirm

if [[ "$remove_confirm" == "yes" ]]; then
    if [[ -d "aws_cv_clip" ]]; then
        rm -rf aws_cv_clip
        echo -e "${GREEN}  ✓ aws_cv_clip/ 제거됨${NC}"
    fi
    
    if [[ -d "aws_llm_autolabel" ]]; then
        rm -rf aws_llm_autolabel
        echo -e "${GREEN}  ✓ aws_llm_autolabel/ 제거됨${NC}"
    fi
    
    if [[ -d "aws_data_collectors" ]]; then
        rm -rf aws_data_collectors
        echo -e "${GREEN}  ✓ aws_data_collectors/ 제거됨${NC}"
    fi
else
    echo -e "${YELLOW}  ⏭️  레거시 디렉터리 제거를 건너뜁니다.${NC}"
fi

# Phase 5: 요약
echo -e "\n${BLUE}[Phase 5] 작업 완료${NC}"
echo -e "${GREEN}✓ 디렉터리 구조 재구성이 완료되었습니다!${NC}\n"

echo -e "${BLUE}변경 사항 요약:${NC}"
echo "  • scripts/ 디렉터리 생성 및 스크립트 파일 이동"
echo "  • docs/ 디렉터리 생성 및 문서 이동"
echo "  • examples/ 디렉터리 생성 및 예제 파일 이동"
echo "  • out/performance/ 디렉터리 생성 및 성능 테스트 결과 이동"
echo "  • 레거시 디렉터리 백업: archive/legacy/"

echo -e "\n${YELLOW}다음 단계:${NC}"
echo "  1. configs/default.yaml의 경로를 확인하고 필요시 수정하세요"
echo "  2. dags/aws_data_pipeline.py의 레거시 참조를 수정하세요"
echo "  3. README.md 및 기타 문서의 경로를 업데이트하세요"
echo "  4. Git 커밋을 생성하세요: git add . && git commit -m 'Reorganize directory structure'"

echo -e "\n${GREEN}완료!${NC}"

