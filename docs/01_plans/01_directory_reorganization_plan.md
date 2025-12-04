# 디렉터리 구조 재구성 계획

## 📋 현재 상태 분석

### 최상단 디렉터리 목록

| 디렉터리 | 역할 | 상태 | 통합 필요성 |
|---------|------|------|------------|
| `core/` | 핵심 프레임워크 | ✅ 유지 | - |
| `tools/` | CLI 도구 | ✅ 유지 | - |
| `configs/` | 설정 파일 | ✅ 유지 | - |
| `aws_cv_clip/` | 레거시 CV 구현 | ⚠️ 통합 필요 | `core/providers/aws/cv/`로 통합됨 |
| `aws_llm_autolabel/` | 레거시 LLM 구현 | ⚠️ 통합 필요 | `core/providers/aws/llm/`로 통합됨 |
| `aws_data_collectors/` | 레거시 데이터 수집 | ⚠️ 통합 필요 | `core/data_collectors/`로 통합됨 |
| `out/` | 출력 결과물 | ✅ 유지 | - |
| `images/` | 테스트 이미지 | ✅ 유지 | - |
| `data/` | 데이터 파일 | ✅ 유지 | - |
| `cache/` | 캐시 파일 | ✅ 유지 | - |
| `notes/` | 문서 | ⚠️ 이동 필요 | `docs/`로 이동 |
| `dags/` | Airflow DAG | ⚠️ 검토 필요 | 레거시 참조 수정 필요 |
| `example_collected_icons/` | 예제 아이콘 | ⚠️ 통합 필요 | `examples/`로 통합 |
| `example_github_icons/` | 예제 아이콘 | ⚠️ 통합 필요 | `examples/`로 통합 |
| `performance_test_results/` | 성능 테스트 결과 | ⚠️ 이동 필요 | `out/performance/`로 이동 |

### 루트 레벨 파일

| 파일 | 역할 | 상태 | 통합 필요성 |
|------|------|------|------------|
| `connect_lmstudio_chat.py` | LM Studio 연결 스크립트 | ⚠️ 이동 필요 | `scripts/`로 이동 |
| `connect_lmstudio_embed.py` | LM Studio 임베딩 스크립트 | ⚠️ 이동 필요 | `scripts/`로 이동 |
| `connect_local_llm.py` | 로컬 LLM 연결 스크립트 | ⚠️ 이동 필요 | `scripts/`로 이동 |
| `auto_aws_cv_clip.sh` | CV CLIP 자동화 스크립트 | ⚠️ 이동 필요 | `scripts/`로 이동 |
| `auto_label_aws.sh` | AWS 라벨링 자동화 스크립트 | ⚠️ 이동 필요 | `scripts/`로 이동 |
| `Asset-Package.zip` | AWS 아이콘 패키지 | ✅ 유지 | - |

---

## 🎯 통합 목표

1. **레거시 코드 정리**: 이미 `core/`로 통합된 레거시 디렉터리 제거
2. **구조 명확화**: 관련 파일들을 논리적으로 그룹화
3. **유지보수성 향상**: 새로운 개발자가 프로젝트 구조를 쉽게 이해할 수 있도록
4. **일관성 확보**: 표준 프로젝트 구조 패턴 준수

---

## 📐 제안하는 새로운 구조

```
hit-archlens-project/
├── core/                          # 핵심 프레임워크 (유지)
│   ├── auto_labeler/
│   ├── data_collectors/
│   ├── providers/
│   ├── taxonomy/
│   ├── utils/
│   └── models.py
│
├── tools/                         # CLI 도구 (유지)
│   ├── cli.py
│   └── config_validator.py
│
├── scripts/                       # 🆕 스크립트 파일들
│   ├── connect_lmstudio_chat.py
│   ├── connect_lmstudio_embed.py
│   ├── connect_local_llm.py
│   ├── auto_aws_cv_clip.sh
│   └── auto_label_aws.sh
│
├── configs/                       # 설정 파일 (유지)
│   └── default.yaml
│
├── docs/                          # 🆕 문서 (notes/에서 이동)
│   ├── ideas.txt
│   ├── module-comparison.md
│   ├── project-overview.md
│   ├── technical-glossary.md
│   └── what-is-airflow.md
│
├── examples/                      # 🆕 예제 파일들
│   ├── collected_icons/
│   │   └── example_icons.json
│   └── github_icons/
│       └── github_high_quality_icons.json
│
├── data/                          # 데이터 파일 (유지)
│   └── aws/
│
├── images/                        # 테스트 이미지 (유지)
│
├── out/                           # 출력 결과물 (유지)
│   ├── aws/
│   ├── experiments/
│   ├── performance/               # 🆕 성능 테스트 결과
│   │   └── performance_test_*.json
│   └── unified/
│
├── cache/                         # 캐시 파일 (유지)
│
├── dags/                          # Airflow DAG (유지, 수정 필요)
│   ├── aws_data_pipeline.py       # ⚠️ 레거시 참조 수정 필요
│   └── rss_ingest.py
│
├── .venv/                         # 가상환경 (유지)
├── .git/                          # Git (유지)
│
├── Asset-Package.zip              # AWS 아이콘 패키지 (유지)
├── requirements.txt               # 의존성 (유지)
├── README.md                      # 프로젝트 문서 (유지)
├── PROJECT_ANALYSIS.md            # 분석 문서 (유지)
└── .gitignore                     # Git 무시 파일 (유지)
```

---

## 🔄 통합 작업 단계

### Phase 1: 새 디렉터리 생성 및 파일 이동

#### 1.1 스크립트 파일 이동
```bash
# scripts/ 디렉터리 생성
mkdir -p scripts

# 스크립트 파일 이동
mv connect_lmstudio_chat.py scripts/
mv connect_lmstudio_embed.py scripts/
mv connect_local_llm.py scripts/
mv auto_aws_cv_clip.sh scripts/
mv auto_label_aws.sh scripts/
```

#### 1.2 문서 통합
```bash
# docs/ 디렉터리 생성
mkdir -p docs

# notes/ 내용 이동
mv notes/* docs/
rmdir notes
```

#### 1.3 예제 파일 통합
```bash
# examples/ 디렉터리 생성
mkdir -p examples

# 예제 디렉터리 이동
mv example_collected_icons examples/collected_icons
mv example_github_icons examples/github_icons
```

#### 1.4 성능 테스트 결과 이동
```bash
# out/performance/ 디렉터리 생성
mkdir -p out/performance

# 성능 테스트 결과 이동
mv performance_test_results/* out/performance/
rmdir performance_test_results
```

### Phase 2: 레거시 디렉터리 정리

#### 2.1 레거시 디렉터리 백업 및 제거

**주의**: 레거시 디렉터리에는 아직 사용 중인 코드나 데이터가 있을 수 있으므로, 먼저 백업을 생성하고 의존성을 확인해야 합니다.

```bash
# 백업 디렉터리 생성
mkdir -p archive/legacy

# 레거시 디렉터리 백업
cp -r aws_cv_clip archive/legacy/
cp -r aws_llm_autolabel archive/legacy/
cp -r aws_data_collectors archive/legacy/

# 의존성 확인 후 제거 (주의!)
# grep -r "from aws_cv_clip\|import aws_cv_clip" . --exclude-dir=.git
# grep -r "from aws_llm\|import aws_llm" . --exclude-dir=.git
# grep -r "from aws_data_collectors\|import aws_data_collectors" . --exclude-dir=.git

# 의존성이 없음을 확인한 후 제거
rm -rf aws_cv_clip
rm -rf aws_llm_autolabel
rm -rf aws_data_collectors
```

#### 2.2 레거시 디렉터리의 유용한 리소스 보존

레거시 디렉터리에서 유용한 리소스를 보존해야 합니다:

**aws_cv_clip/**:
- `icons/` → `data/aws/icons/` (이미 `out/aws/icons/`에 있으면 중복 확인)
- `images/` → `images/` (중복 확인)
- `aws_resources_models.csv` → `data/aws/` (중복 확인)

**aws_llm_autolabel/**:
- `images/` → `images/` (중복 확인)
- `aws_resources_models.csv` → `data/aws/` (중복 확인)

**aws_data_collectors/**:
- `data/` → `data/aws/` (중복 확인)
- `Asset-Package.zip` → 루트 (이미 있으면 중복 확인)

### Phase 3: 의존성 수정

#### 3.1 DAG 파일 수정

`dags/aws_data_pipeline.py`는 레거시 모듈을 참조하고 있으므로 수정이 필요합니다:

```python
# 기존 (레거시)
from aws_icons_parser.aws_icons_zip_to_mapping import generate_mapping
from aws_products_scraper.fetch_products import fetch_products
from aws_service_boto3.export_service_codes import export_service_codes
from aws_service_boto3.infer_from_models import infer_from_models

# 수정 후 (core 프레임워크 사용)
from core.data_collectors.aws_collector import AWSDataCollector
```

#### 3.2 스크립트 파일 경로 수정

이동된 스크립트 파일들의 상대 경로를 수정해야 할 수 있습니다.

### Phase 4: 설정 파일 업데이트

#### 4.1 configs/default.yaml 경로 수정

디렉터리 구조 변경에 따라 설정 파일의 경로를 업데이트해야 합니다:

```yaml
# 기존
data:
  icons_dir: "aws_cv_clip/icons"
  taxonomy_csv: "aws_cv_clip/aws_resources_models.csv"

# 수정 후
data:
  icons_dir: "out/aws/icons"  # 또는 "data/aws/icons"
  taxonomy_csv: "data/aws/aws_resources_models.csv"
```

#### 4.2 .gitignore 업데이트

새로운 디렉터리 구조에 맞게 `.gitignore`를 업데이트:

```gitignore
# Archive
archive/

# Scripts output (필요시)
scripts/*.log
scripts/*.tmp
```

---

## ⚠️ 주의사항

### 1. 의존성 확인
- 레거시 디렉터리를 제거하기 전에 모든 의존성을 확인해야 합니다
- `grep` 명령어로 프로젝트 전체에서 레거시 모듈 참조를 검색하세요

### 2. 백업 필수
- 레거시 디렉터리를 제거하기 전에 반드시 백업을 생성하세요
- `archive/legacy/` 디렉터리에 백업을 저장하는 것을 권장합니다

### 3. 점진적 마이그레이션
- 한 번에 모든 것을 변경하지 말고 단계적으로 진행하세요
- 각 단계마다 테스트를 수행하여 문제를 조기에 발견하세요

### 4. 문서 업데이트
- README.md 및 기타 문서의 경로를 업데이트해야 합니다
- 사용자 가이드의 예제 코드도 수정이 필요할 수 있습니다

---

## 📝 체크리스트

### Phase 1: 새 디렉터리 생성 및 파일 이동
- [ ] `scripts/` 디렉터리 생성 및 스크립트 파일 이동
- [ ] `docs/` 디렉터리 생성 및 문서 이동
- [ ] `examples/` 디렉터리 생성 및 예제 파일 이동
- [ ] `out/performance/` 디렉터리 생성 및 성능 테스트 결과 이동

### Phase 2: 레거시 디렉터리 정리
- [ ] 레거시 디렉터리 의존성 확인
- [ ] 레거시 디렉터리 백업 생성
- [ ] 유용한 리소스 보존 (아이콘, 데이터 등)
- [ ] 레거시 디렉터리 제거

### Phase 3: 의존성 수정
- [ ] DAG 파일 수정
- [ ] 스크립트 파일 경로 수정
- [ ] 기타 코드 내 경로 수정

### Phase 4: 설정 및 문서 업데이트
- [ ] `configs/default.yaml` 경로 수정
- [ ] `.gitignore` 업데이트
- [ ] `README.md` 업데이트
- [ ] 기타 문서 업데이트

### Phase 5: 테스트 및 검증
- [ ] CLI 도구 테스트
- [ ] 데이터 수집 테스트
- [ ] 이미지 분석 테스트
- [ ] 전체 워크플로우 테스트

---

## 🚀 실행 스크립트

통합 작업을 자동화하는 스크립트를 제공합니다:

```bash
# 통합 스크립트 실행 (주의: 백업 후 실행)
./scripts/reorganize_directories.sh
```

---

## 📊 예상 효과

### Before (현재)
- 최상단 디렉터리: 15개
- 레거시 코드: 3개 디렉터리
- 문서 분산: `notes/` 디렉터리
- 스크립트 분산: 루트 레벨

### After (통합 후)
- 최상단 디렉터리: 12개
- 레거시 코드: `archive/legacy/`로 이동
- 문서 통합: `docs/` 디렉터리
- 스크립트 통합: `scripts/` 디렉터리

### 개선 사항
- ✅ 구조 명확화: 관련 파일들이 논리적으로 그룹화
- ✅ 유지보수성 향상: 새로운 개발자가 이해하기 쉬움
- ✅ 일관성 확보: 표준 프로젝트 구조 패턴 준수
- ✅ 레거시 코드 정리: 사용하지 않는 코드 제거

---

## 🔗 참고 자료

- [Python 프로젝트 구조 모범 사례](https://docs.python-guide.org/writing/structure/)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Hit ArchLens 프로젝트 분석](./PROJECT_ANALYSIS.md)

---

**작성일**: 2025-01-XX  
**작성자**: AI Assistant  
**상태**: 계획 단계

