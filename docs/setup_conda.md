# Conda 환경 설정 (Optional)

이 문서는 conda를 사용하여 프로젝트를 설정하는 방법을 설명합니다. **정본 환경 관리 도구는 uv입니다.** conda는 선택적(Optional)입니다.

## Conda 환경 생성

```bash
conda create -n archlens python=3.11 -y
conda activate archlens
```

## 의존성 설치

### 방법 1: requirements.txt 사용 (자동 생성됨)

```bash
# requirements.txt는 uv export로 생성됩니다
uv export --no-hashes -o requirements.txt
pip install -r requirements.txt
```

### 방법 2: pyproject.toml 직접 사용

```bash
pip install -e .
```

## Jupyter Kernel 설정 (선택)

```bash
conda install ipykernel -y
python -m ipykernel install --user --name archlens --display-name "(archlens)"
jupyter kernelspec list | grep archlens
```

## YOLO 실험 환경 설정

```bash
conda activate archlens
./scripts/setup_yolo_env.sh
```

## 주의사항

- **정본 환경 관리 도구는 uv입니다.** CI/CD와 공식 문서는 uv 기준으로 작성됩니다.
- conda 환경에서도 동일한 `pyproject.toml`을 사용하지만, 일부 패키지 버전 충돌이 발생할 수 있습니다.
- 문제가 발생하면 uv 환경을 사용하는 것을 권장합니다.

