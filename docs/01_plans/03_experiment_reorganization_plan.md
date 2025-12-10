# 실험 디렉터리 구조 재구성 계획 및 실행 결과

**작성일**: 2025-01-XX  
**상태**: ✅ 완료

## 📋 목적

두 가지 실험(Classification, Detection)이 혼재되어 있고 부수적인 코드와 파일이 많은 문제를 해결하기 위해 프로젝트 구조를 정리했습니다.

## 🔍 발견된 문제

### 1. Detection 실험 결과 파일 누락
- `hit-aws-object-detection-project/runs/detect/`에 이전 detection 실험 결과들이 잘못 저장되어 있었음
- 누락된 실험 결과:
  - `aws_diagram_yolov8m_v1/` (완전한 결과 + weights)
  - `aws_icon_detector2/` ~ `aws_icon_detector7/` (완전한 결과들)

### 2. Classification 결과 파일 잘못 저장
- `13_yolo_inference.ipynb`에서 잘못된 경로로 결과가 저장됨
- `hit-aws-object-detection-project/runs/classify/predict3/`에 저장됨

### 3. 프로젝트 구조 혼재
- 노트북 파일들이 루트에 15개 혼재
- `obj_classification/` 폴더는 비어있음
- `runs/`는 classification만, `obj_detection/runs/`는 detection만
- `weights/`에 모델 파일 혼재
- `data/`에 여러 종류 데이터 혼재

## ✅ 실행된 작업

### Phase 1: Detection 결과 파일 마이그레이션
- `hit-aws-object-detection-project/runs/detect/`에서 누락된 detection 실험 결과 복사
- `hit-aws-object-detection-project/runs/classify/predict3/`에서 classification 결과 복사

### Phase 2: 실험 디렉터리 구조 재구성
- `experiments/classification/` 디렉터리 생성
- `experiments/detection/` 디렉터리 생성
- 각 실험별로 notebooks, scripts, data, runs, weights 분리

### Phase 3: 파일 이동
- Classification 노트북 (00~13) → `experiments/classification/notebooks/`
- Detection 노트북 (14) → `experiments/detection/notebooks/`
- Classification 스크립트 → `experiments/classification/scripts/`
- Detection 스크립트 → `experiments/detection/scripts/`
- 데이터 및 결과 파일들 이동
- 모델 가중치 분리

### Phase 4: 경로 참조 수정
- 모든 스크립트의 경로 참조 업데이트
- `dataset.yaml` 경로 수정

## 📐 최종 구조

```
hit-archlens-project/
├── experiments/                      # 🆕 실험별 분리
│   ├── classification/               # 아이콘 분류 실험
│   │   ├── notebooks/                # 노트북 파일들 (00~13)
│   │   ├── scripts/                  # 학습/평가/추론 스크립트
│   │   ├── data/                     # 실험 데이터
│   │   │   └── dataset/icons/        # 아이콘 데이터셋
│   │   ├── runs/                     # 실험 결과
│   │   │   └── classify/             # YOLO classification 결과
│   │   └── weights/                  # 모델 가중치
│   │       ├── yolov8n-cls.pt
│   │       └── yolo11n-cls.pt
│   │
│   └── detection/                    # 다이어그램 객체 탐지 실험
│       ├── notebooks/                # 노트북 파일 (14)
│       ├── scripts/                  # 학습 스크립트
│       │   └── train.py
│       ├── data/                     # 실험 데이터
│       │   └── aws_diagram_data/     # 다이어그램 데이터
│       ├── runs/                     # 실험 결과
│       │   ├── aws_icon_detector_trial_0~49/  # Optuna 실험
│       │   ├── aws_diagram_yolov8m_v1/        # 마이그레이션된 결과
│       │   └── aws_icon_detector2~7/          # 마이그레이션된 결과
│       ├── experiments/              # Optuna 실험 메타데이터
│       ├── dataset.yaml              # YOLO 데이터셋 설정
│       └── weights/                  # 모델 가중치
│           ├── yolov8n.pt
│           ├── yolov8s.pt
│           ├── yolov8m.pt
│           └── yolo11n.pt
│
├── backend/                          # 백엔드 프레임워크 (유지)
├── data/                             # 공통 데이터 (유지)
├── scripts/                          # 공통 스크립트 (유지)
├── docs/                             # 문서 (유지)
└── ...
```

## 📝 주요 변경 사항

| 항목 | 이전 위치 | 새 위치 |
|------|----------|---------|
| Classification 노트북 | 루트 (00~13) | `experiments/classification/notebooks/` |
| Detection 노트북 | 루트 (14) | `experiments/detection/notebooks/` |
| Classification 스크립트 | `scripts/` | `experiments/classification/scripts/` |
| Detection 스크립트 | `obj_detection/train.py` | `experiments/detection/scripts/train.py` |
| Classification 데이터 | `dataset/icons/` | `experiments/classification/data/dataset/icons/` |
| Detection 데이터 | `obj_detection/aws_diagram_data/` | `experiments/detection/data/aws_diagram_data/` |
| Classification 결과 | `runs/classify/` | `experiments/classification/runs/classify/` |
| Detection 결과 | `obj_detection/runs/` | `experiments/detection/runs/` |
| Detection 설정 | `obj_detection/dataset.yaml` | `experiments/detection/dataset.yaml` |

## 🔧 수정된 파일

### 스크립트 파일
1. `experiments/classification/scripts/train_yolo_cls.py`
   - `--data-dir` 기본값 수정
   
2. `experiments/classification/scripts/eval_yolo_cls.py`
   - `--data-dir` 기본값 수정
   
3. `experiments/classification/scripts/predict_yolo_cls.py`
   - `--data-dir` 기본값 수정
   
4. `experiments/detection/scripts/train.py`
   - `EXPERIMENTS_DIR`, `RUNS_DIR` 경로 수정
   - `DATASET_YAML_PATH` 경로 수정
   - `prepare_dataset()` 함수의 이미지/라벨 경로 수정
   - 테스트 이미지 경로 수정

### 설정 파일
5. `experiments/detection/dataset.yaml`
   - `train`, `val` 경로 수정

## 🎯 장점

1. **실험 분리**: Classification과 Detection이 명확히 구분됨
2. **구조 일관성**: 각 실험마다 동일한 구조 (notebooks, scripts, data, runs, weights)
3. **확장성**: 새 실험 추가 시 `experiments/new_experiment/`로 추가 가능
4. **유지보수**: 실험별로 독립적으로 관리 가능
5. **가독성**: 루트 디렉터리 정리

## 📊 마이그레이션 결과

### Detection 실험 결과
- ✅ `aws_diagram_yolov8m_v1/` 복사 완료
- ✅ `aws_icon_detector2~7/` 복사 완료
- ✅ `aws_icon_detector_trial_0~49/` 기존 유지

### Classification 실험 결과
- ✅ `predict3_migrated/` 복사 완료

## ⚠️ 주의사항

1. **노트북 파일**: 노트북 내부의 경로 참조는 수동으로 확인 및 수정 필요
2. **상대 경로**: 가능하면 상대 경로 사용 권장
3. **테스트**: 각 스크립트 실행 테스트 필요

## 🚀 다음 단계

1. 노트북 파일들의 경로 참조 확인 및 수정
2. README.md 업데이트
3. 각 스크립트 실행 테스트

---

**작성자**: AI Assistant  
**검토 필요**: 노트북 파일 경로 참조 확인

