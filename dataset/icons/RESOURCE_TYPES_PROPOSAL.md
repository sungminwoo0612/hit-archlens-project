# AWS 리소스 타입 처리 방안 제안

## 문제 상황
Role, Policy, User, Admin, Template, Image 등은 서비스가 아니라 리소스 타입입니다. 
이러한 리소스들을 현재의 fine-grained classification 체계에서 어떻게 처리할지 결정이 필요합니다.

## 현재 구조
- **Coarse**: 서비스 카테고리 (19개) - Compute, Storage, Database 등
- **Fine**: 서비스 이름 (81개) - amazon ec2, amazon s3 등

## 처리 방안 옵션

### 옵션 1: IAM 서비스 추가 + 리소스 타입을 fine으로 분류 (추천) ⭐

**장점:**
- IAM은 AWS의 핵심 서비스이므로 별도로 분류하는 것이 자연스러움
- Role, Policy, User 등은 자주 사용되는 리소스이므로 fine 레벨에서 구분 가능
- 기존 구조와 일관성 유지

**구현:**
```
Security & Identity:
  - aws iam (서비스 자체)
  - aws iam role
  - aws iam policy
  - aws iam user
  - aws iam group
  - aws iam identity provider
  - aws iam access analyzer

DevOps & Developer Tools:
  - aws cloudformation (기존)
  - aws cloudformation template (리소스 타입)
  - aws cloudformation stack (리소스 타입)

Compute:
  - amazon ec2 (기존)
  - amazon ec2 ami (기존 - 이미 추가됨)
  - amazon ec2 instance (리소스 타입)
```

**추가할 항목:**
- aws iam (ID: 81)
- aws iam role (ID: 82)
- aws iam policy (ID: 83)
- aws iam user (ID: 84)
- aws iam group (ID: 85)
- aws cloudformation template (ID: 86)
- aws cloudformation stack (ID: 87)

### 옵션 2: 기존 서비스에 리소스 타입 포함

**장점:**
- 구조가 단순함
- 서비스와 리소스의 관계가 명확함

**단점:**
- IAM이 없으면 IAM 리소스들을 어디에 둘지 애매함
- 리소스 타입이 많아지면 관리가 복잡해질 수 있음

**구현:**
```
Security & Identity:
  - aws iam role
  - aws iam policy
  - aws iam user
  (IAM 서비스 자체는 없음)

DevOps & Developer Tools:
  - aws cloudformation (서비스)
  - aws cloudformation template (리소스)
```

### 옵션 3: 별도의 "Resources & Components" 카테고리 추가

**장점:**
- 리소스 타입을 명확히 구분 가능
- 확장성이 좋음

**단점:**
- 새로운 coarse 카테고리를 추가해야 함
- 기존 구조 변경이 큼
- 서비스와 리소스의 경계가 모호해질 수 있음

**구현:**
```
새로운 Coarse 카테고리: "Resources & Components" (ID: 19)

Resources & Components:
  - iam role
  - iam policy
  - iam user
  - cloudformation template
  - ec2 instance
  - s3 bucket
  등등...
```

## 추천 방안: 옵션 1

**이유:**
1. IAM은 AWS의 핵심 서비스이므로 반드시 포함되어야 함
2. Role, Policy, User 등은 다이어그램에서 자주 사용되므로 fine 레벨에서 구분하는 것이 유용
3. 기존 구조를 크게 변경하지 않고 확장 가능
4. 서비스와 리소스의 관계가 명확함

## 구현 예시

### 추가할 서비스/리소스 목록

**Security & Identity:**
- aws iam (서비스)
- aws iam role
- aws iam policy  
- aws iam user
- aws iam group
- aws iam identity provider

**DevOps & Developer Tools:**
- aws cloudformation template (리소스)
- aws cloudformation stack (리소스)

**Compute:**
- amazon ec2 instance (리소스 - 선택적)

## 다음 단계

1. 사용자와 협의하여 최종 방안 결정
2. 선택한 방안에 따라 파일 업데이트
3. labels_fine.csv에 실제 이미지 경로 추가
4. generate_aws_icon_schemas.py 실행하여 일관성 확인

