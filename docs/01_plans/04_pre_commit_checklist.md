# GitHub Push 전 최종 체크리스트

**작성일**: 2025-01-XX  
**상태**: ✅ 완료

## ✅ 완료된 작업

### 1. 실험 디렉터리 구조 재구성
- [x] `experiments/classification/` 디렉터리 생성 및 파일 이동
- [x] `experiments/detection/` 디렉터리 생성 및 파일 이동
- [x] 모든 노트북 파일 이동 완료
- [x] 스크립트 파일 경로 수정 완료

### 2. Detection 결과 파일 마이그레이션
- [x] `hit-aws-object-detection-project`에서 누락된 detection 결과 복사
- [x] Classification 결과 파일 복사

### 3. 문서화
- [x] `experiments/README.md` 생성 (이미지 포함)
- [x] 실험 결과 요약 및 시각화 포함

### 4. .gitignore 업데이트
- [x] 실험 결과 가중치 파일 제외 설정
- [x] 캐시 파일 제외 설정
- [x] 대용량 데이터 파일 제외 설정

### 5. 레거시 디렉터리 정리
- [x] `aws_cv_clip/` 디렉터리 확인 및 처리
- [x] 빈 디렉터리 정리

## 📋 최종 확인 사항

### 파일 구조
- [x] 모든 노트북이 `experiments/` 디렉터리로 이동됨
- [x] 스크립트 파일 경로 참조 수정 완료
- [x] 실험 결과 파일 정리 완료

### .gitignore
- [x] Python 캐시 파일 제외
- [x] 실험 결과 가중치 파일 제외 (runs/**/weights/*.pt)
- [x] 기본 weights 폴더 가중치는 포함
- [x] 대용량 데이터 파일 제외

### 문서
- [x] `experiments/README.md` 생성
- [x] 실험 결과 이미지 및 설명 포함
- [x] 성능 지표 요약 포함

## ⚠️ 주의사항

### 1. args.yaml 파일의 경로
- `experiments/detection/runs/*/args.yaml` 파일들에 이전 경로가 남아있음
- 이는 실험 결과 메타데이터이므로 수정 불필요 (과거 기록)

### 2. 대용량 파일
- 실험 결과 가중치 파일(`*.pt`)은 `.gitignore`에 의해 제외됨
- 필요시 LFS 사용 고려

### 3. 노트북 파일
- 노트북 내부의 경로 참조는 수동으로 확인 및 수정 필요
- 실행 시 경로 오류가 발생할 수 있음

## 🚀 Push 전 최종 확인

```bash
# 1. 변경사항 확인
git status

# 2. .gitignore 확인
git check-ignore -v experiments/detection/runs/*/weights/*.pt

# 3. 대용량 파일 확인
find . -type f -size +50M | grep -v ".git"

# 4. 실험 README 확인
cat experiments/README.md
```

## 📝 커밋 메시지 제안

```
feat: 실험 디렉터리 구조 재구성 및 결과 정리

- Classification과 Detection 실험을 experiments/ 디렉터리로 분리
- Detection 실험 결과 파일 마이그레이션 (hit-aws-object-detection-project)
- experiments/README.md 생성 (이미지 및 성능 지표 포함)
- .gitignore 업데이트 (실험 결과 가중치 제외)
- 레거시 디렉터리 정리 (aws_cv_clip → archive/legacy)
```

---

**검토 완료**: 2025-01-XX

