# 추가 디렉터리 구조 정리 계획

## 분석 결과

### 1. dags/ 디렉터리 삭제 가능 여부

**현재 상태**:
- `dags/aws_data_pipeline.py`: 레거시 모듈 참조가 모두 주석 처리되어 있고 실제로 작동하지 않음
- `dags/rss_ingest.py`: RSS 수집용 DAG로, 이 프로젝트의 핵심 기능과 직접 관련 없음
- 프로젝트 내 다른 파일에서 참조되지 않음

**결론**: ✅ **삭제 가능** (archive/로 이동 권장)

### 2. images/와 out/을 data/에 통합 가능 여부

**현재 상태**:
- `images/`: 테스트 이미지 파일들 (6개 PNG 파일)
- `out/`: 모든 출력 결과물 (실험 결과, 시각화, 성능 테스트 등)
- `data/`: 이미 AWS 데이터가 있음 (icons, products, services, taxonomy)

**hit-aws-object-detection-project 참고**:
- `data/` 안에 `datasets/`, `runs/`, `raw/`, `configs/` 등이 있음
- 출력 결과물도 `data/` 안에 포함 가능

**결론**: ✅ **통합 가능**
- `images/` → `data/images/` (테스트 이미지)
- `out/` → `data/outputs/` (모든 출력 결과물)

### 3. backend/ 안에 configs/ 포함 가능 여부

**현재 상태**:
- `configs/default.yaml`: 프로젝트 전체 설정 파일
- `backend/tools/cli.py`에서 `configs/default.yaml` 참조
- `backend/core/performance_test.py`에서 `core/configs/ultra_performance_config.yaml` 참조

**hit-aws-object-detection-project 참고**:
- 설정 파일이 최상단에 있음 (별도 configs/ 디렉터리 없음)
- backend/core/config.py 형태로 설정 관리

**결론**: ✅ **backend/ 안으로 이동 가능**
- `configs/default.yaml` → `backend/configs/default.yaml`
- `backend/core/configs/` → `backend/configs/` (이미 있음)

---

## 제안하는 최종 구조

```
hit-archlens-project/
├── backend/                    # 백엔드 패키지
│   ├── core/                  # 핵심 프레임워크
│   ├── tools/                 # CLI 도구
│   └── configs/               # 🆕 설정 파일 (configs/에서 이동)
│       ├── default.yaml
│       └── ultra_performance_config.yaml
├── data/                      # 🆕 모든 데이터 통합
│   ├── aws/                   # AWS 데이터 (기존)
│   ├── images/                # 🆕 테스트 이미지 (images/에서 이동)
│   └── outputs/               # 🆕 출력 결과물 (out/에서 이동)
│       ├── aws/
│       ├── experiments/
│       ├── performance/
│       └── unified/
├── archive/                   # 레거시 백업
│   └── legacy/
│       ├── aws_cv_clip/
│       ├── aws_llm_autolabel/
│       ├── aws_data_collectors/
│       └── dags/              # 🆕 DAG 파일 백업
├── cache/                     # 캐시 파일
├── docs/                      # 문서
├── examples/                  # 예제 파일
├── scripts/                   # 스크립트 파일
├── pyproject.toml
├── requirements.txt
└── README.md
```

**최상단 디렉터리 수: 7개** (현재 10개 → 목표 달성!)

---

## 작업 단계

### Phase 1: dags/ 디렉터리 처리

1. **dags/ → archive/legacy/dags/ 이동**
   ```bash
   mv dags archive/legacy/
   ```

### Phase 2: images/와 out/을 data/로 통합

1. **images/ → data/images/ 이동**
   ```bash
   mv images data/
   ```

2. **out/ → data/outputs/ 이동**
   ```bash
   mv out data/outputs
   ```

### Phase 3: configs/를 backend/로 이동

1. **configs/default.yaml → backend/configs/default.yaml 이동**
   ```bash
   mkdir -p backend/configs
   mv configs/default.yaml backend/configs/
   rmdir configs
   ```

2. **backend/core/configs/ → backend/configs/ 통합**
   ```bash
   mv backend/core/configs/* backend/configs/
   rmdir backend/core/configs
   ```

### Phase 4: 경로 참조 수정

1. **configs/default.yaml 참조 수정**
   - `backend/tools/cli.py`: `configs/default.yaml` → `backend/configs/default.yaml` 또는 상대 경로
   - 모든 CLI 옵션의 기본값 수정

2. **images/ 경로 수정**
   - `configs/default.yaml`: `images_dir: "images"` → `images_dir: "data/images"`

3. **out/ 경로 수정**
   - `configs/default.yaml`: `output_dir: "out"` → `output_dir: "data/outputs"`
   - 모든 출력 경로 참조 수정

4. **performance_test.py 경로 수정**
   - `backend/core/performance_test.py`: `core/configs/ultra_performance_config.yaml` → `backend/configs/ultra_performance_config.yaml`

---

## 주요 변경 사항

### 이동되는 디렉터리/파일
- `dags/` → `archive/legacy/dags/`
- `images/` → `data/images/`
- `out/` → `data/outputs/`
- `configs/default.yaml` → `backend/configs/default.yaml`
- `backend/core/configs/` → `backend/configs/` (통합)

### 수정이 필요한 파일들
1. `backend/tools/cli.py` - config 경로 수정
2. `backend/core/configs/default.yaml` - images_dir, output_dir 경로 수정
3. `backend/core/performance_test.py` - config 경로 수정
4. `backend/core/data_collectors/setup_output_structure.py` - 출력 경로 수정
5. `README.md` - 구조 업데이트

---

## 주의사항

1. **경로 참조**: 모든 하드코딩된 경로를 상대 경로 또는 설정 기반으로 변경
2. **기존 결과물**: `out/`에 있는 기존 결과물은 `data/outputs/`로 이동되므로 사용자에게 알림 필요
3. **설정 파일**: `configs/default.yaml`의 기본 경로들이 모두 업데이트되어야 함

