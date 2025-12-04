#!/usr/bin/env python3
"""
쉘 스크립트와 Python 스크립트를 Jupyter 노트북으로 변환하는 유틸리티

사용법:
    python scripts/convert_to_notebooks.py
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


def create_markdown_cell(source: List[str]) -> Dict[str, Any]:
    """마크다운 셀 생성"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }


def create_code_cell(source: List[str], outputs: List[Any] = None) -> Dict[str, Any]:
    """코드 셀 생성"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": source,
        "outputs": outputs or []
    }


def bash_to_notebook(bash_file: Path, notebook_file: Path, title: str, description: str = ""):
    """Bash 스크립트를 노트북으로 변환"""
    with open(bash_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cells = []
    
    # 헤더 마크다운 셀
    header = [f"# {title}\n"]
    if description:
        header.append(f"\n{description}\n")
    header.append(f"\n**원본 스크립트**: `{bash_file.name}`\n")
    cells.append(create_markdown_cell(header))
    
    lines = content.splitlines(keepends=True)
    
    # 주석 블록과 코드 블록 분리
    current_block = []
    in_python = False
    python_code = []
    python_delimiter = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Python heredoc 시작 감지
        if re.match(r'python\s+-\s+<<\s*[\'"]?PY', stripped):
            if current_block:
                cells.append(create_code_cell(current_block))
                current_block = []
            in_python = True
            python_delimiter = re.search(r'[\'"]?(\w+)[\'"]?', stripped).group(1) if re.search(r'[\'"]?(\w+)[\'"]?', stripped) else 'PY'
            i += 1
            # 다음 줄이 delimiter인지 확인
            if i < len(lines) and lines[i].strip() == f"'{python_delimiter}'":
                i += 1
            continue
        
        # Python heredoc 종료 감지
        if in_python and stripped == f"'{python_delimiter}'" or stripped == python_delimiter:
            if python_code:
                cells.append(create_code_cell(python_code))
                python_code = []
            in_python = False
            python_delimiter = None
            i += 1
            continue
        
        if in_python:
            python_code.append(line)
            i += 1
            continue
        
        # 주석 블록 시작 (구분선이나 섹션 헤더)
        if stripped.startswith('#') and ('=' in stripped or '#' * 5 in stripped or stripped.startswith('# ')):
            if current_block:
                cells.append(create_code_cell(current_block))
                current_block = []
            # 섹션 헤더를 마크다운으로
            comment_text = line.lstrip('#').strip()
            if comment_text:
                # echo 문은 그대로 코드로
                if 'echo' in comment_text.lower():
                    current_block.append(line)
                else:
                    cells.append(create_markdown_cell([f"## {comment_text}\n"]))
            i += 1
            continue
        
        # 일반 주석 (설명용)
        if stripped.startswith('#'):
            # echo나 실행 명령어가 포함된 주석은 코드로
            if any(keyword in stripped.lower() for keyword in ['사용법:', 'usage:', 'example:', '예시:']):
                if current_block:
                    cells.append(create_code_cell(current_block))
                    current_block = []
                comment_text = line.lstrip('#').strip()
                if comment_text:
                    cells.append(create_markdown_cell([f"**{comment_text}**\n"]))
            else:
                # 일반 주석은 코드 블록에 포함
                current_block.append(line)
            i += 1
            continue
        
        # 일반 bash 명령어
        if stripped:
            current_block.append(line)
        i += 1
    
    if python_code:
        cells.append(create_code_cell(python_code))
    if current_block:
        cells.append(create_code_cell(current_block))
    
    # 노트북 생성
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(notebook_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 변환 완료: {bash_file} → {notebook_file}")


def py_to_notebook(py_file: Path, notebook_file: Path, title: str, description: str = ""):
    """Python 스크립트를 노트북으로 변환"""
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cells = []
    
    # 헤더 마크다운 셀
    header = [f"# {title}\n"]
    if description:
        header.append(f"\n{description}\n")
    header.append(f"\n**원본 스크립트**: `{py_file.name}`\n")
    cells.append(create_markdown_cell(header))
    
    # docstring 추출
    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if docstring_match:
        doc = docstring_match.group(1).strip()
        cells.append(create_markdown_cell([f"{doc}\n"]))
    
    # 코드를 셀로 분할
    lines = content.splitlines(keepends=True)
    current_cell = []
    in_docstring = False
    docstring_started = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # 모듈 레벨 docstring 건너뛰기 (이미 추출함)
        if i == 0 and stripped.startswith('#!/usr/bin/env'):
            continue
        if docstring_match and i < len(lines):
            # docstring 범위 확인
            if not docstring_started and '"""' in line:
                docstring_started = True
                if line.count('"""') == 2:  # 한 줄에 시작과 끝
                    continue
                continue
            if docstring_started:
                if '"""' in line:
                    docstring_started = False
                continue
        
        # 함수/클래스 정의 시작 시 새 셀
        if (stripped.startswith('def ') or 
            stripped.startswith('class ') or
            (stripped.startswith('@') and i > 0 and not lines[i-1].strip().startswith('@'))):
            if current_cell:
                cells.append(create_code_cell(current_cell))
                current_cell = []
        
        # import 문이 많으면 별도 셀로
        if stripped.startswith('import ') or stripped.startswith('from '):
            if current_cell and not any(l.strip().startswith(('import ', 'from ')) for l in current_cell):
                # 이전 셀에 import가 없으면 새 셀 시작
                if current_cell:
                    cells.append(create_code_cell(current_cell))
                    current_cell = []
        
        current_cell.append(line)
    
    if current_cell:
        cells.append(create_code_cell(current_cell))
    
    # 노트북 생성
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(notebook_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 변환 완료: {py_file} → {notebook_file}")


def main():
    """메인 함수"""
    base_dir = Path(__file__).parent.parent
    
    # 변환할 파일 목록 (순서대로)
    conversions = [
        # 환경 설정
        ("scripts/setup_yolo_env.sh", "00_environment_setup.ipynb", 
         "환경 설정", "YOLO Classification 학습에 필요한 패키지 설치"),
        
        # 데이터 수집
        ("backend/core/data_collectors/aws_collector.py", "02_aws_data_collection.ipynb",
         "AWS 데이터 수집", "AWS 아이콘, 서비스, 제품 정보 수집"),
        
        # Dataset 관련
        ("aws_icon_dataset_rebuild.sh", "08_dataset_rebuild.ipynb",
         "Dataset 재구성", "Taxonomy 및 train/val/test split 재생성"),
        ("aws_icon_class_mapping_and_yaml.sh", "09_class_mapping_and_yaml.ipynb",
         "Class 매핑 및 YAML 생성", "Coarse/Fine 클래스 ID 매핑 및 YOLO names.yaml 생성"),
        ("aws_icon_yolo_cls_prepare_and_train.sh", "10_yolo_dataset_preparation.ipynb",
         "YOLO 데이터셋 준비", "YOLO Classification용 데이터셋 생성"),
        
        # YOLO 학습/평가/추론
        ("scripts/train_yolo_cls.py", "11_yolo_training.ipynb",
         "YOLO Classification 학습", "YOLO 모델 학습"),
        ("scripts/eval_yolo_cls.py", "12_yolo_evaluation.ipynb",
         "YOLO 모델 평가", "학습된 모델 평가"),
        ("scripts/predict_yolo_cls.py", "13_yolo_inference.ipynb",
         "YOLO 추론", "이미지 분류 추론"),
        
        # Auto Labeling
        ("scripts/auto_aws_cv_clip.sh", "14_auto_labeling_cv.ipynb",
         "CV 기반 자동 라벨링", "CLIP 모델을 사용한 Computer Vision 기반 라벨링"),
        ("scripts/auto_label_aws.sh", "15_auto_labeling_llm.ipynb",
         "LLM 기반 자동 라벨링", "GPT-4 Vision을 사용한 LLM 기반 라벨링"),
        
        # 기타
        ("scripts/train_yolov8_diagrams.sh", "17_yolo_diagrams_training.ipynb",
         "YOLO Diagrams 학습", "다이어그램 객체 탐지를 위한 YOLO 학습"),
        ("generate_aws_icon_schemas.py", "18_label_studio_schema_generation.ipynb",
         "Label Studio 스키마 생성", "Label Studio용 라벨 스키마 XML 생성"),
        ("check_and_clean_yolo_cls_fine.sh", "19_dataset_validation_and_cleanup.ipynb",
         "데이터셋 검증 및 정리", "YOLO 데이터셋 검증 및 캐시 정리"),
    ]
    
    for source_path, target_path, title, description in conversions:
        source_file = base_dir / source_path
        target_file = base_dir / target_path
        
        if not source_file.exists():
            print(f"⚠️  파일 없음: {source_file}")
            continue
        
        try:
            if source_file.suffix == '.sh':
                bash_to_notebook(source_file, target_file, title, description)
            elif source_file.suffix == '.py':
                py_to_notebook(source_file, target_file, title, description)
            else:
                print(f"⚠️  지원하지 않는 파일 형식: {source_file}")
        except Exception as e:
            print(f"❌ 변환 실패: {source_file} - {e}")


if __name__ == "__main__":
    main()

