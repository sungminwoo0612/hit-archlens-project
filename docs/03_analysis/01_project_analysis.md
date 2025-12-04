# Hit ArchLens 프로젝트 분석 보고서

**카테고리**: 분석 문서  
**작성일**: 2025-11-28  
**관련 문서**: 
- [프로젝트 개요](../04_reference/01_project_overview.md)
- [모듈 비교](../04_reference/02_module_comparison.md)

## 📋 프로젝트 개요

**Hit ArchLens**는 멀티 클라우드 아키텍처 다이어그램을 자동으로 분석하여 서비스 아이콘을 인식하고 분류하는 통합 프레임워크입니다. Computer Vision과 Large Language Model을 결합하여 클라우드 서비스 아이콘을 자동으로 인식하고 바운딩 박스를 생성합니다.

### 핵심 목표
- **자동화**: 수동 라벨링 작업을 AI로 대체
- **멀티 클라우드**: AWS, GCP, Azure, Naver Cloud 지원 (현재 AWS 구현 완료)
- **정확성**: CV + LLM 하이브리드 접근으로 높은 정확도 달성
- **확장성**: 모듈화된 아키텍처로 새로운 클라우드 제공자 추가 용이

---

## 🏗️ 시스템 아키텍처

### 전체 구조

```
hit-archlens-project/
├── core/                          # 핵심 프레임워크 (클라우드 중립)
│   ├── auto_labeler/              # 오토라벨링 추상 클래스
│   │   ├── base_auto_labeler.py   # 베이스 클래스
│   │   ├── cv_auto_labeler.py     # CV 기반 구현
│   │   ├── llm_auto_labeler.py    # LLM 기반 구현
│   │   └── hybrid_auto_labeler.py # 하이브리드 구현
│   ├── data_collectors/           # 데이터 수집 프레임워크
│   │   ├── base_collector.py      # 베이스 수집기
│   │   └── aws_collector.py       # AWS 수집기 구현
│   ├── taxonomy/                  # 서비스 분류 시스템
│   │   ├── base_taxonomy.py       # 베이스 택소노미
│   │   └── aws_taxonomy.py        # AWS 택소노미 구현
│   ├── providers/                 # 클라우드별 구현체
│   │   └── aws/                   # AWS 전용 모듈
│   │       ├── cv/                # CV 기반 오토라벨러
│   │       ├── llm/               # LLM 기반 오토라벨러
│   │       └── hybrid/            # 하이브리드 오토라벨러
│   ├── models.py                  # 통합 데이터 모델
│   └── utils/                     # 유틸리티 함수
├── tools/                         # CLI 도구
│   └── cli.py                     # 통합 명령행 인터페이스
├── configs/                       # 설정 파일
│   └── default.yaml               # 기본 설정
├── aws_cv_clip/                   # AWS CV CLIP 구현 (레거시)
├── aws_llm_autolabel/             # AWS LLM 구현 (레거시)
├── aws_data_collectors/           # AWS 데이터 수집 (레거시)
└── out/                           # 모든 결과물 저장소
```

### 아키텍처 패턴

1. **추상 베이스 클래스 (ABC)**: 클라우드 중립 인터페이스 제공
2. **전략 패턴**: CV, LLM, Hybrid 알고리즘 교체 가능
3. **팩토리 패턴**: 클라우드별 구현체 생성
4. **의존성 주입**: 설정을 통한 유연한 구성

---

## 🔧 핵심 모듈 분석

### 1. 데이터 모델 (`core/models.py`)

프로젝트 전반에서 사용하는 통합 데이터 모델:

- **`BoundingBox`**: 바운딩 박스 좌표 및 유틸리티 (IoU 계산 등)
- **`DetectionResult`**: 감지 결과 (바운딩 박스, 라벨, 신뢰도, 서비스 코드)
- **`AnalysisResult`**: 이미지 분석 결과 (감지 목록, 처리 시간, 메타데이터)
- **`BatchAnalysisResult`**: 배치 분석 결과 (통계, 성공률 등)
- **`AWSServiceInfo`**: AWS 서비스 정보
- **`AWSServiceIcon`**: AWS 서비스 아이콘 정보

**특징**:
- 타입 안정성을 위한 dataclass 사용
- JSON 직렬화/역직렬화 지원
- 통계 계산 프로퍼티 제공

### 2. 오토라벨러 프레임워크 (`core/auto_labeler/`)

#### BaseAutoLabeler (추상 클래스)
- `analyze_image()`: 단일 이미지 분석
- `analyze_batch()`: 배치 분석 (병렬 처리 지원)
- `analyze_directory()`: 디렉터리 전체 분석
- `_normalize_detections()`: 택소노미 기반 정규화

#### 구현체
- **CVAutoLabeler**: Computer Vision 기반
- **LLMAutoLabeler**: Large Language Model 기반
- **HybridAutoLabeler**: CV + LLM 융합

### 3. AWS CV Auto Labeler (`core/providers/aws/cv/`)

**기술 스택**:
- **CLIP (OpenCLIP)**: 이미지-텍스트 유사도 검색
- **OpenCV**: 이미지 처리 (Canny Edge, MSER)
- **FAISS**: 벡터 유사도 검색
- **ORB**: 특징점 매칭
- **EasyOCR**: 텍스트 인식 (선택적)

**처리 파이프라인**:
1. **객체 제안 생성**:
   - Canny Edge Detection
   - MSER (Maximally Stable Extremal Regions)
   - Sliding Window Scan
2. **CLIP 유사도 검색**: 각 제안 영역과 AWS 아이콘 비교
3. **ORB 정제**: 특징점 매칭으로 신뢰도 향상
4. **OCR 힌트**: 텍스트 정보 활용
5. **NMS (Non-Maximum Suppression)**: 중복 검출 제거
6. **택소노미 정규화**: 서비스명 표준화

### 4. AWS LLM Auto Labeler (`core/providers/aws/llm/`)

**기술 스택**:
- **OpenAI GPT-4 Vision**: 멀티모달 이미지 분석
- **DeepSeek Vision**: 대안 LLM 제공자
- **로컬 LLM**: LM Studio 지원

**처리 방식**:
- **전체 이미지 분석**: 컨텍스트 기반 인식
- **패치별 분석**: 세밀한 영역 분석
- **프롬프트 엔지니어링**: 구조화된 프롬프트로 정확도 향상

### 5. AWS Hybrid Auto Labeler (`core/providers/aws/hybrid/`)

**융합 전략**:
- **가중 합산**: CV 60% + LLM 40%
- **Ensemble**: 다수결 기반
- **Confidence 기반**: 신뢰도 높은 결과 우선
- **IoU 기반**: 바운딩 박스 겹침 고려

### 6. 데이터 수집기 (`core/data_collectors/`)

**수집 대상**:
1. **아이콘**: AWS Asset Package ZIP에서 추출
2. **서비스 정보**: AWS 공식 문서/API에서 수집
3. **제품 정보**: AWS 제품 페이지 스크래핑

**기능**:
- 실시간 진행 상황 모니터링
- 병렬 처리 지원
- 데이터 검증 및 정규화

### 7. 택소노미 시스템 (`core/taxonomy/`)

**기능**:
- 서비스명 정규화 (별칭 → 표준명)
- Fuzzy Matching (rapidfuzz 사용)
- 그룹 매핑 (카테고리 분류)
- 블랙리스트 관리

---

## 🛠️ CLI 도구 (`tools/cli.py`)

### 주요 명령어

1. **`analyze`**: 이미지 분석
   ```bash
   python tools/cli.py analyze \
     --input images/test.png \
     --method hybrid \
     --output out/experiments \
     --confidence-thresholds 0.0,0.2,0.4,0.6,0.8
   ```

2. **`collect-data`**: 데이터 수집
   ```bash
   python tools/cli.py collect-data \
     --data-type all \
     --monitor \
     --verbose
   ```

3. **`visualize`**: 결과 시각화
   ```bash
   python tools/cli.py visualize \
     --input out/experiments \
     --output out/visualizations
   ```

4. **`status`**: 시스템 상태 확인

5. **`compare-methods`**: 방법론 비교

### 특징
- Click 기반 CLI 프레임워크
- 다중 신뢰도 임계값 지원
- 배치 처리 자동화
- 상세 통계 및 시각화

---

## 📊 데이터 흐름

### 분석 파이프라인

```
입력 이미지
    ↓
[전처리]
    ├─ 리사이즈 (max_size: 1600)
    ├─ RGB 변환
    └─ 노이즈 제거
    ↓
[객체 제안 생성]
    ├─ Canny Edge Detection
    ├─ MSER Detection
    └─ Sliding Window
    ↓
[특징 추출]
    ├─ CLIP 임베딩
    ├─ ORB 특징점
    └─ OCR 텍스트
    ↓
[유사도 검색]
    ├─ FAISS 벡터 검색
    ├─ ORB 매칭
    └─ 점수 융합
    ↓
[NMS & 필터링]
    ├─ IoU 기반 중복 제거
    └─ 신뢰도 임계값 필터링
    ↓
[택소노미 정규화]
    ├─ 서비스명 표준화
    └─ 그룹 매핑
    ↓
출력 (DetectionResult 리스트)
```

### 데이터 수집 파이프라인

```
AWS Asset Package ZIP
    ↓
[아이콘 추출]
    ├─ ZIP 압축 해제
    ├─ PNG/SVG 파일 추출
    └─ 메타데이터 파싱
    ↓
[서비스 정보 수집]
    ├─ AWS 문서 스크래핑
    ├─ Boto3 API 호출
    └─ JSON/CSV 변환
    ↓
[제품 정보 수집]
    ├─ AWS 제품 페이지 스크래핑
    └─ API 호출
    ↓
[통합 택소노미 생성]
    ├─ 데이터 병합
    ├─ 정규화
    └─ CSV/JSON 출력
    ↓
out/aws/ 디렉터리 저장
```

---

## ⚙️ 설정 시스템 (`configs/default.yaml`)

### 주요 설정 항목

1. **데이터 설정**:
   - 아이콘 디렉터리 경로
   - 택소노미 CSV 경로
   - 출력 디렉터리

2. **CV 설정**:
   - CLIP 모델명 (ViT-B-32)
   - 사전 훈련 가중치
   - 디바이스 (auto/cuda/cpu)

3. **LLM 설정**:
   - 제공자 (openai/deepseek/local)
   - API 키 (환경변수)
   - Vision 모델명

4. **감지 설정**:
   - 최대 이미지 크기
   - 최소/최대 영역
   - Canny/MSER 파라미터

5. **검색 설정**:
   - Top-K 후보 수
   - 수락 임계값
   - 점수 가중치 (CLIP/ORB/OCR)

6. **하이브리드 설정**:
   - CV/LLM 가중치
   - 융합 방법
   - IoU 임계값

---

## 📈 성능 특성

### 정확도 (예상)
- **CV 기반**: 85-90%
- **LLM 기반**: 90-95%
- **하이브리드**: 92-97%

### 처리 속도
- **단일 이미지**: 2-5초 (CV), 1-3초 (LLM)
- **배치 처리**: 병렬 처리로 100장/분
- **캐싱**: CLIP 임베딩 캐싱으로 재처리 속도 향상

### 리소스 사용
- **메모리**: CLIP 모델 로드 시 ~2GB
- **GPU**: CUDA 사용 시 속도 향상
- **디스크**: 캐시 디렉터리 사용

---

## 🔍 코드 품질 분석

### 강점

1. **모듈화**: 명확한 계층 구조와 책임 분리
2. **타입 안정성**: Python 타입 힌팅 적극 활용
3. **확장성**: 추상 클래스 기반으로 새 클라우드 추가 용이
4. **에러 처리**: try-except와 결과 객체로 안정성 확보
5. **문서화**: docstring과 README 상세

### 개선 가능 영역

1. **의존성 관리**: `requirements.txt`가 시스템 패키지 포함 (정리 필요)
2. **테스트 커버리지**: 테스트 파일은 있으나 실행 여부 불명
3. **레거시 코드**: `aws_cv_clip/`, `aws_llm_autolabel/` 디렉터리 정리 필요
4. **설정 검증**: 설정 파일 유효성 검증 강화 필요
5. **로깅**: print 대신 logging 모듈 사용 권장

---

## 🚀 사용 시나리오

### 1. AWS 아키텍처 다이어그램 분석

```bash
# 1. 데이터 수집
python tools/cli.py collect-data --data-type all

# 2. 이미지 분석
python tools/cli.py analyze \
  --input images/aws_diagram.png \
  --method hybrid \
  --output out/experiments

# 3. 결과 시각화
python tools/cli.py visualize \
  --input out/experiments \
  --output out/visualizations
```

### 2. 배치 처리

```bash
# 디렉터리 내 모든 이미지 분석
python tools/cli.py analyze \
  --input images/ \
  --method cv \
  --output out/batch_results
```

### 3. 방법론 비교

```bash
# CV, LLM, Hybrid 결과 비교
python tools/cli.py compare-methods \
  --input images/test.png \
  --output out/comparison
```

---

## 📝 주요 파일 요약

| 파일/디렉터리 | 역할 | 중요도 |
|--------------|------|--------|
| `core/models.py` | 통합 데이터 모델 | ⭐⭐⭐⭐⭐ |
| `core/auto_labeler/base_auto_labeler.py` | 오토라벨러 프레임워크 | ⭐⭐⭐⭐⭐ |
| `core/providers/aws/cv/aws_cv_auto_labeler.py` | AWS CV 구현 | ⭐⭐⭐⭐⭐ |
| `core/providers/aws/llm/aws_llm_auto_labeler.py` | AWS LLM 구현 | ⭐⭐⭐⭐ |
| `core/providers/aws/hybrid/aws_hybrid_auto_labeler.py` | 하이브리드 구현 | ⭐⭐⭐⭐ |
| `core/taxonomy/aws_taxonomy.py` | 택소노미 시스템 | ⭐⭐⭐⭐ |
| `core/data_collectors/aws_collector.py` | 데이터 수집 | ⭐⭐⭐ |
| `tools/cli.py` | CLI 인터페이스 | ⭐⭐⭐⭐ |
| `configs/default.yaml` | 기본 설정 | ⭐⭐⭐⭐ |

---

## 🎯 향후 개선 제안

### 단기 (1-2개월)
1. **레거시 코드 정리**: `aws_cv_clip/`, `aws_llm_autolabel/` 통합
2. **테스트 강화**: 단위 테스트 및 통합 테스트 추가
3. **로깅 시스템**: print → logging 모듈 전환
4. **에러 처리**: 더 구체적인 에러 메시지 및 복구 전략

### 중기 (3-6개월)
1. **멀티 클라우드 확장**: GCP, Azure 지원 추가
2. **웹 인터페이스**: Gradio/Streamlit 기반 UI
3. **API 서버**: FastAPI 기반 REST API
4. **성능 최적화**: 모델 양자화, 배치 처리 최적화

### 장기 (6개월+)
1. **실시간 처리**: 웹소켓 기반 실시간 분석
2. **모델 학습**: 도메인 특화 모델 파인튜닝
3. **상용화**: SaaS 서비스로 전환
4. **커뮤니티**: 오픈소스 커뮤니티 구축

---

## 📚 기술 스택 요약

### 핵심 라이브러리
- **Computer Vision**: OpenCV, PIL, OpenCLIP
- **Machine Learning**: PyTorch, FAISS
- **Language Models**: OpenAI API, DeepSeek API
- **데이터 처리**: Pandas, NumPy
- **CLI**: Click
- **설정**: PyYAML
- **유틸리티**: tqdm, rapidfuzz

### 인프라
- **언어**: Python 3.10+
- **패키지 관리**: pip (requirements.txt)
- **버전 관리**: Git
- **문서화**: Markdown

---

## ✅ 결론

**Hit ArchLens**는 잘 설계된 멀티 클라우드 아키텍처 분석 프레임워크입니다. 모듈화된 구조, 타입 안정성, 확장 가능한 아키텍처가 강점입니다. 현재 AWS 지원이 완료되었으며, 다른 클라우드 제공자 추가가 용이한 구조입니다.

**주요 성과**:
- ✅ CV + LLM 하이브리드 접근법 구현
- ✅ 클라우드 중립 프레임워크 설계
- ✅ 통합 CLI 도구 제공
- ✅ 자동 데이터 수집 시스템

**개선 필요**:
- ⚠️ 레거시 코드 정리
- ⚠️ 테스트 커버리지 향상
- ⚠️ 로깅 시스템 개선
- ⚠️ 문서화 보완

전반적으로 프로덕션 환경에 배포 가능한 수준의 코드 품질을 보유하고 있으며, 지속적인 개선을 통해 더욱 강력한 도구로 발전할 수 있을 것입니다.

