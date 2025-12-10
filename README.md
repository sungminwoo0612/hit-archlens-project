# Hit ArchLens

멀티 클라우드 아키텍처 다이어그램 자동 분석을 위한 통합 프레임워크입니다. Computer Vision과 Large Language Model을 결합하여 클라우드 서비스 아이콘을 자동으로 인식하고 분류합니다.

## 🎯 주요 기능

- **Computer Vision 기반 분석**: CLIP 모델을 사용한 이미지 유사도 검색
- **LLM 기반 분석**: GPT-4 Vision을 활용한 텍스트 기반 분석
- **하이브리드 분석**: CV와 LLM 결과를 융합한 고정확도 분석
- **AWS 데이터 수집**: 아이콘, 서비스 정보, 제품 정보 자동 수집
- **실시간 모니터링**: 데이터 수집 및 분석 과정 실시간 추적
- **성능 시각화**: 분석 결과 및 통계 데이터 시각화

## 🏗️ 아키텍처

```bash
Hit ArchLens/
├── backend/                 # 백엔드 패키지
│   ├── core/               # 핵심 프레임워크
│   │   ├── auto_labeler/   # 오토라벨링 추상 클래스
│   │   ├── data_collectors/# 데이터 수집 프레임워크
│   │   ├── models.py       # 통합 데이터 모델
│   │   ├── taxonomy/       # 서비스 분류 시스템
│   │   ├── providers/      # 클라우드별 구현체
│   │   │   └── aws/        # AWS 전용 구현체
│   │   │       ├── cv/     # CV 기반 오토라벨러
│   │   │       ├── llm/    # LLM 기반 오토라벨러
│   │   │       └── hybrid/ # 하이브리드 오토라벨러
│   │   └── utils/          # 유틸리티 함수
│   ├── tools/              # CLI 도구
│   └── configs/            # 설정 파일
├── data/                   # 모든 데이터 통합
│   ├── aws/                # AWS 데이터
│   ├── images/             # 테스트 이미지
│   └── outputs/            # 출력 결과물
├── archive/                # 레거시 백업
├── cache/                  # 캐시 파일
├── docs/                   # 문서
├── examples/               # 예제 파일
└── scripts/                # 스크립트 파일
```

## ⚙️ 설정

### 기본 설정 파일: `backend/configs/default.yaml`

```yaml
# 데이터 설정
data:
  icons_dir: "data/outputs/aws/icons"
  taxonomy_csv: "data/aws/aws_resources_models.csv"
  images_dir: "data/images"
  output_dir: "data/outputs"

# CV 설정
cv:
  clip_name: "ViT-B-32"
  clip_pretrained: "laion2b_s34b_b79k"
  device: "auto"

# LLM 설정
llm:
  provider: "openai"
  api_key: "${OPENAI_API_KEY}"
  vision_model: "gpt-4-vision-preview"

# 분석 설정
detection:
  max_size: 1600
  min_area: 900
  max_area: 90000

# 성능 설정
performance:
  parallel_processing: true
  max_workers: 4
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd hit_archlens

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는 venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. AWS 아이콘 다운로드

```bash
# AWS 공식 아키텍처 아이콘 다운로드
wget https://d1.awsstatic.com/webteam/architecture-icons/q1-2024/Asset-Package_01242024.7c4f8b8b.zip -O Asset-Package.zip

# 또는 AWS 공식 사이트에서 수동 다운로드:
# https://aws.amazon.com/ko/architecture/icons/
```

### 3. 데이터 수집

```bash
# 모든 AWS 데이터 수집 (아이콘, 서비스, 제품 정보)
python cli.py collect-data --data-type all --monitor --verbose

# 또는 pyproject.toml 설치 후
archlens collect-data --data-type all --monitor --verbose

# 특정 데이터만 수집
python cli.py collect-data --data-type icons --verbose
python cli.py collect-data --data-type services --verbose
python cli.py collect-data --data-type products --verbose
```

### 4. 오토라벨링 분석

```bash
# CV 기반 분석 (API 키 불필요)
python cli.py analyze --input data/images/test_diagram.png --method cv --output data/outputs/experiments/cv_results --verbose

# LLM 기반 분석 (OpenAI API 키 필요)
export OPENAI_API_KEY="your-api-key-here"
python cli.py analyze --input data/images/test_diagram.png --method llm --output data/outputs/experiments/llm_results --verbose

# 하이브리드 분석 (CV + LLM 결합)
python cli.py analyze --input data/images/test_diagram.png --method hybrid --output data/outputs/experiments/hybrid_results --verbose
```

### 5. 배치 분석

```bash
# 여러 이미지 동시 분석
python cli.py analyze --input data/images/ --method hybrid --output data/outputs/experiments/batch_results --verbose
```

### 6. 결과 시각화

```bash
# 분석 결과 시각화
python cli.py visualize --input data/outputs/experiments/hybrid_results --output data/outputs/visualizations --verbose
```

## 📊 순차적 사용 가이드

### Phase 1: 초기 설정 및 데이터 수집

```bash
# 1. 환경 설정
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. AWS 아이콘 다운로드
wget https://d1.awsstatic.com/webteam/architecture-icons/q1-2024/Asset-Package_01242024.7c4f8b8b.zip -O Asset-Package.zip

# 3. 데이터 수집 (실시간 모니터링 포함)
python cli.py collect-data --data-type all --monitor --verbose
```

**예상 시간**: 5-10분 (네트워크 속도에 따라 다름)

### Phase 2: CV 기반 분석 테스트

```bash
# 1. 테스트 이미지 준비
mkdir -p data/images
# AWS 아키텍처 다이어그램을 data/images/ 디렉터리에 복사

# 2. CV 기반 분석 실행
python cli.py analyze --input data/images/test_diagram.png --method cv --output data/outputs/experiments/cv_results --verbose

# 3. 결과 확인
ls -la data/outputs/experiments/cv_results/
cat data/outputs/experiments/cv_results/analysis_result_000.json
```

**예상 시간**: 2-5분 (첫 실행 시 모델 다운로드 포함)

### Phase 3: LLM 기반 분석 (선택사항)

```bash
# 1. OpenAI API 키 설정
export OPENAI_API_KEY="your-api-key-here"

# 2. LLM 기반 분석 실행
python cli.py analyze --input data/images/test_diagram.png --method llm --output data/outputs/experiments/llm_results --verbose

# 3. 결과 확인
ls -la data/outputs/experiments/llm_results/
cat data/outputs/experiments/llm_results/analysis_result_000.json
```

**예상 시간**: 1-3분 (API 응답 시간에 따라 다름)

### Phase 4: 하이브리드 분석

```bash
# 1. 하이브리드 분석 실행 (CV + LLM 결합)
python cli.py analyze --input data/images/test_diagram.png --method hybrid --output data/outputs/experiments/hybrid_results --verbose

# 2. 결과 비교
ls -la data/outputs/experiments/
```

**예상 시간**: 3-8분 (CV + LLM 처리 시간)

### Phase 5: 성능 평가 및 시각화

```bash
# 1. 분석 결과 시각화
python cli.py visualize --input data/outputs/experiments/hybrid_results --output data/outputs/visualizations --verbose

# 2. 성능 통계 확인
python cli.py status --method hybrid --verbose

# 3. 결과 파일 확인
tree data/outputs/ -L 3
```

## 📁 출력 구조

## 🎯 YOLO Classification 학습

AWS 아이콘 분류를 위한 YOLO 모델 학습 및 사용 방법입니다.

### 빠른 시작

```bash
# 1. 환경 설정
conda activate archlens
./scripts/setup_yolo_env.sh

# 2. 데이터셋 준비 (아직 안 했다면)
python scripts/prepare_yolo_dataset.py --mode fine

# 3. 모델 학습
python scripts/train_yolo_cls.py --mode fine --epochs 100 --imgsz 256

# 4. 모델 평가
python scripts/eval_yolo_cls.py \
    --model runs/classify/fine_cls_*/weights/best.pt \
    --mode fine \
    --split test

# 5. 이미지 예측
# 단일 이미지
python scripts/predict_yolo_cls.py \
    --model runs/classify/fine_cls_yolov8n-cls/weights/best.pt \
    --source "dataset/icons/images/fine/amazon s3/Arch_Amazon-S3_64.png" \
    --mode fine \
    --top-k 5

# 디렉터리 전체
python scripts/predict_yolo_cls.py \
    --model runs/classify/fine_cls_yolov8n-cls/weights/best.pt \
    --source "dataset/icons/images/fine/amazon s3" \
    --mode fine \
    --save-json \
    --save-txt
```

### 상세 가이드

## 📊 실험 결과

### 최고 성능 실험: YOLO Classification (yolo-cls16)

최고 성능을 달성한 YOLO Classification 실험 결과입니다.

**성능 지표:**
- Top-1 Accuracy: 높은 정확도 달성
- Top-5 Accuracy: 우수한 성능

**시각화 결과:**

| 항목 | 이미지 |
|------|--------|
| 학습 곡선 | [Results](experiments/classification/runs/classify/yolo-cls16/results.png) |
| Confusion Matrix | [Confusion Matrix](experiments/classification/runs/classify/yolo-cls16/confusion_matrix.png) |
| 정규화된 Confusion Matrix | [Normalized CM](experiments/classification/runs/classify/yolo-cls16/confusion_matrix_normalized.png) |
| 검증 예측 결과 (Batch 0) | [Val Batch 0](experiments/classification/runs/classify/yolo-cls16/val_batch0_pred.jpg) |
| 검증 예측 결과 (Batch 1) | [Val Batch 1](experiments/classification/runs/classify/yolo-cls16/val_batch1_pred.jpg) |

더 자세한 실험 결과는 [experiments/README.md](experiments/README.md)를 참고하세요.

## 📚 문서

프로젝트의 모든 문서는 `docs/` 디렉터리에 체계적으로 정리되어 있습니다.

- **사용 가이드** (`docs/01_guides/`): 
  - [YOLO 학습 가이드](docs/01_guides/01_yolo_training_guide.md)
  - [추론 가이드](docs/01_guides/02_inference_guide.md)
- **참고 자료** (`docs/02_reference/`): 프로젝트 개요, 모듈 비교, 기술 용어집 등

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🔗 관련 링크

- [AWS 공식 아이콘](https://aws.amazon.com/ko/architecture/icons/)
- [OpenAI API 문서](https://platform.openai.com/docs/)
- [CLIP 모델](https://github.com/openai/CLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.

---

```
conda create -n archlens python=3.11 -y
conda activate archlens
which python ; which python3 ; which pip ; which pip3
conda install ipykernel -y
python -m ipykernel install --user --name archlens --display-name "(archlens)"
jupyter kernelspec list | grep archlens
pip install pandas jupyterlab ipython
pip install -r requirements.txt

```