# 작업 단위별 커밋 메시지

GitHub push를 위한 논리적인 커밋 단위로 나눈 메시지입니다.

---

## 커밋 1: Detection 실험 결과 파일 마이그레이션

```bash
git add experiments/detection/runs/aws_diagram_yolov8m_v1/
git add experiments/detection/runs/aws_icon_detector2/
git add experiments/detection/runs/aws_icon_detector3/
git add experiments/detection/runs/aws_icon_detector4/
git add experiments/detection/runs/aws_icon_detector5/
git add experiments/detection/runs/aws_icon_detector6/
git add experiments/detection/runs/aws_icon_detector7/
git add experiments/classification/runs/classify/predict3_migrated/
git add scripts/migrate_detection_results.sh
git commit -m "feat: Detection 실험 결과 파일 마이그레이션

hit-aws-object-detection-project에서 누락된 detection 실험 결과를 
현재 프로젝트로 마이그레이션했습니다.

- aws_diagram_yolov8m_v1 실험 결과 추가
- aws_icon_detector2~7 실험 결과 추가
- Classification predict3 결과 마이그레이션
- 마이그레이션 스크립트 추가"
```

---

## 커밋 2: 실험 디렉터리 구조 재구성

```bash
git add experiments/
git add -u 00_*.ipynb 01_*.ipynb 02_*.ipynb 03_*.ipynb 04_*.ipynb \
          05_*.ipynb 06_*.ipynb 07_*.ipynb 08_*.ipynb 09_*.ipynb \
          10_*.ipynb 11_*.ipynb 12_*.ipynb 13_*.ipynb 14_*.ipynb
git add -u dataset/ obj_detection/ runs/ weights/
git add scripts/reorganize_experiments.sh
git commit -m "refactor: 실험 디렉터리 구조 재구성

Classification과 Detection 실험을 experiments/ 디렉터리로 분리하여
명확한 구조로 재구성했습니다.

- experiments/classification/ 디렉터리 생성 (notebooks, scripts, data, runs, weights)
- experiments/detection/ 디렉터리 생성 (notebooks, scripts, data, runs, experiments, weights)
- 루트 디렉터리 정리 (노트북 및 임시 디렉터리 제거)
- 재구성 스크립트 추가"
```

---

## 커밋 3: 스크립트 경로 참조 수정

```bash
git add experiments/classification/scripts/
git add experiments/detection/scripts/
git add experiments/detection/dataset.yaml
git commit -m "fix: 실험 스크립트 경로 참조 수정

새로운 디렉터리 구조에 맞게 모든 스크립트의 경로 참조를 업데이트했습니다.

- classification 스크립트: data-dir 기본값 수정
- detection 스크립트: EXPERIMENTS_DIR, RUNS_DIR, DATASET_YAML_PATH 경로 수정
- dataset.yaml: train/val 경로 수정"
```

---

## 커밋 4: experiments/README.md 추가

```bash
git add experiments/README.md
git commit -m "docs: experiments/README.md 추가

실험 결과를 시각적으로 보여주는 README를 추가했습니다.

- Classification 및 Detection 실험 개요
- 최고 성능 실험 결과 및 성능 지표
- 학습 곡선, Confusion Matrix, PR 곡선 등 시각화
- 성능 비교 테이블 및 사용 방법"
```

---

## 커밋 5: .gitignore 업데이트 및 레거시 정리

```bash
git add .gitignore
git add -u archive/legacy/dags/
git add archive/legacy/aws_cv_clip/
git commit -m "chore: .gitignore 업데이트 및 레거시 디렉터리 정리

.gitignore 업데이트:
- 실험 결과 가중치 파일 제외
- 캐시 및 대용량 데이터 파일 제외
- 가상환경 제외 강화

레거시 디렉터리 정리:
- 사용하지 않는 Airflow DAG 삭제
- aws_cv_clip 디렉터리 archive/legacy로 이동"
```

---

## 커밋 6: 문서 추가

```bash
git add docs/01_plans/03_experiment_reorganization_plan.md
git add docs/01_plans/04_pre_commit_checklist.md
git add docs/01_plans/05_commit_messages.md
git commit -m "docs: 실험 재구성 계획 및 체크리스트 문서 추가

- 실험 디렉터리 재구성 계획 및 실행 결과
- GitHub push 전 최종 체크리스트
- 작업 단위별 커밋 메시지 가이드"
```

---

## 전체 커밋 순서 요약

1. **Detection 실험 결과 파일 마이그레이션** - 누락된 결과 파일 복구
2. **실험 디렉터리 구조 재구성** - 파일 이동 및 구조 정리
3. **스크립트 경로 참조 수정** - 새 구조에 맞게 경로 업데이트
4. **experiments/README.md 추가** - 실험 결과 문서화
5. **.gitignore 업데이트 및 레거시 정리** - 불필요한 파일 제외 및 정리
6. **문서 추가** - 계획 및 체크리스트 문서

---

## 한 번에 실행하는 스크립트 (참고용)

```bash
#!/bin/bash
# 주의: 실제 실행 전에 각 단계를 확인하세요

# 커밋 1
git add experiments/detection/runs/aws_diagram_yolov8m_v1/ \
        experiments/detection/runs/aws_icon_detector[2-7]/ \
        experiments/classification/runs/classify/predict3_migrated/ \
        scripts/migrate_detection_results.sh
git commit -m "feat: Detection 실험 결과 파일 마이그레이션

hit-aws-object-detection-project에서 누락된 detection 실험 결과를 
현재 프로젝트로 마이그레이션했습니다."

# 커밋 2
git add experiments/ scripts/reorganize_experiments.sh
git add -u 00_*.ipynb 01_*.ipynb 02_*.ipynb 03_*.ipynb 04_*.ipynb \
          05_*.ipynb 06_*.ipynb 07_*.ipynb 08_*.ipynb 09_*.ipynb \
          10_*.ipynb 11_*.ipynb 12_*.ipynb 13_*.ipynb 14_*.ipynb \
          dataset/ obj_detection/ runs/ weights/
git commit -m "refactor: 실험 디렉터리 구조 재구성

Classification과 Detection 실험을 experiments/ 디렉터리로 분리하여
명확한 구조로 재구성했습니다."

# 커밋 3
git add experiments/classification/scripts/ \
        experiments/detection/scripts/ \
        experiments/detection/dataset.yaml
git commit -m "fix: 실험 스크립트 경로 참조 수정

새로운 디렉터리 구조에 맞게 모든 스크립트의 경로 참조를 업데이트했습니다."

# 커밋 4
git add experiments/README.md
git commit -m "docs: experiments/README.md 추가

실험 결과를 시각적으로 보여주는 README를 추가했습니다."

# 커밋 5
git add .gitignore archive/
git commit -m "chore: .gitignore 업데이트 및 레거시 디렉터리 정리"

# 커밋 6
git add docs/01_plans/
git commit -m "docs: 실험 재구성 계획 및 체크리스트 문서 추가"
```

---

**작성일**: 2025-01-XX

