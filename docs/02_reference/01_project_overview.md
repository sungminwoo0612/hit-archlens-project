# ArchLens 프로젝트 개요

**카테고리**: 참고 자료  
**작성일**: 2025-11-28  
**관련 문서**: 
- [모듈 비교](02_module_comparison.md)

## 🎯 **프로젝트 목적**

**ArchLens**는 클라우드 아키텍처 다이어그램을 자동으로 분석하여 서비스 아이콘을 인식하고 바운딩 박스를 생성하는 실험입니다.

### **핵심 가치**
- **자동화**: AWS 아이콘 자동 분류 및 탐지
- **정확성**: YOLO 모델 기반 고정확도 인식
- **확장성**: 다양한 AWS 서비스 아이콘 지원

---

## 🏗️ **시스템 아키텍처**

```
ArchLens
├── experiments/            # 실험 디렉터리
│   ├── classification/    # YOLO Classification 실험
│   └── detection/         # YOLO Detection 실험
├── backend/               # 백엔드 패키지
├── data/                  # 데이터
└── docs/                  # 문서
```

---

## 📋 **실험 개요**

### **1. YOLO Classification 실험**

AWS 아이콘 이미지를 클래스로 분류하는 모델 학습 실험입니다.

#### **데이터셋**
- **Train**: 447개 이미지
- **Validation**: 96개 이미지
- **Test**: 96개 이미지
- **클래스 수**: 64개 (fine-grained)

#### **성능**
- **Top-1 Accuracy**: 20.83%
- **Top-5 Accuracy**: 75.00%

### **2. YOLO Detection 실험**

AWS 아키텍처 다이어그램에서 아이콘의 위치를 바운딩 박스로 탐지하는 실험입니다.

#### **데이터셋**
- **Train**: ~800개 이미지
- **Validation**: ~200개 이미지
- **클래스 수**: 119개

#### **성능**
- **mAP@0.5**: 최고 성능 실험에서 달성

---

## 🔧 **핵심 기술 개념**

### **Computer Vision (CV)**
- **Object Detection**: 이미지에서 객체 위치 찾기
- **Feature Extraction**: 이미지의 특징점 추출
- **Image Similarity**: 이미지 간 유사도 계산
- **Image Preprocessing**: 노이즈 제거, 정규화

### **Large Language Models (LLM)**
- **Multimodal AI**: 이미지와 텍스트 동시 처리
- **Vision-Language Models**: CLIP, GPT-4V
- **Prompt Engineering**: 효과적인 프롬프트 설계
- **Context Understanding**: 컨텍스트 기반 이해

### **Machine Learning**
- **Transfer Learning**: 사전 훈련된 모델 활용
- **Fine-tuning**: 특정 도메인에 맞춤 조정
- **Ensemble Methods**: 여러 모델 결과 통합
- **Confidence Scoring**: 신뢰도 점수 계산

### **Data Processing**
- **ETL (Extract, Transform, Load)**: 데이터 파이프라인
- **Data Validation**: 데이터 품질 검증
- **Data Normalization**: 데이터 표준화
- **Data Augmentation**: 데이터 증강

---

## 🎯 **사용 사례**

### **1. AWS 아이콘 분류**
```
입력: AWS 아이콘 이미지
출력: 서비스 클래스 예측 (Top-5)
```

### **2. 다이어그램 분석**
```
입력: AWS 아키텍처 다이어그램
출력: 아이콘 위치 및 클래스 정보
```

---

## 🚀 **향후 로드맵**

### **Phase 1: 모델 성능 개선** 🔄
- [ ] Classification 정확도 향상
- [ ] Detection mAP 향상
- [ ] 데이터셋 확장

### **Phase 2: 기능 확장** 📋
- [ ] 더 많은 AWS 서비스 지원
- [ ] 실시간 추론 최적화
- [ ] 배치 처리 개선

---

## 📊 **성능 지표**

### **Classification 성능**
- **Top-1 Accuracy**: 20.83%
- **Top-5 Accuracy**: 75.00%

### **Detection 성능**
- **mAP@0.5**: 최고 성능 실험에서 달성

### **지원 형식**
- **입력**: PNG, JPG
- **출력**: JSON, YAML

---

## 🤝 **기여 가이드**

### **개발 환경**
```bash
# 가상환경 설정
uv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행
python tools/cli.py --dev
```

### **코딩 컨벤션**
- **언어**: Python 3.10+
- **타입 힌트**: 필수
- **문서화**: docstring 필수
- **테스트**: pytest 사용

### **브랜치 전략**
- `main`: 안정 버전
- `develop`: 개발 버전
- `feature/*`: 기능 개발
- `hotfix/*`: 긴급 수정
