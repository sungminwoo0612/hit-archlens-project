#!/usr/bin/env python3
"""
YOLO Classification 데이터셋 준비 스크립트

AWS 아이콘 분류용 YOLO 데이터셋을 생성합니다.
conda archlens 환경에서 실행합니다.

사용법:
    python scripts/prepare_yolo_dataset.py --mode fine
    python scripts/prepare_yolo_dataset.py --mode coarse
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="YOLO Classification 데이터셋 준비",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fine", "coarse"],
        default="fine",
        help="데이터셋 모드: fine (64 클래스) 또는 coarse (19 클래스)",
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "dataset" / "icons",
        help="데이터셋 루트 디렉터리",
    )
    
    return parser.parse_args()


def main() -> None:
    """메인 함수"""
    args = parse_args()
    
    DATA_DIR = args.data_dir
    MODE = args.mode
    
    IMG_DIR = DATA_DIR / "images"
    if not IMG_DIR.is_dir():
        raise FileNotFoundError(f"images dir not found: {IMG_DIR}")
    
    train_path = DATA_DIR / "train_fine.csv"
    val_path = DATA_DIR / "val_fine.csv"
    test_path = DATA_DIR / "test_fine.csv"
    
    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("train/val/test_fine.csv 중 하나 이상이 없습니다.")
    
    # ---------- 1. taxonomy 로드 및 클래스 ID 매핑 생성 ----------
    if MODE == "fine":
        tax_path = DATA_DIR / "taxonomy_fine.csv"
        label_col = "canonical_service_name"
        if not tax_path.exists():
            raise FileNotFoundError(tax_path)
        df_tax = pd.read_csv(tax_path)
        
        # canonical_service_name 기준 정렬 후 ID 부여
        class_names = sorted(df_tax["canonical_service_name"].unique())
        print(f"[INFO] fine 클래스 개수: {len(class_names)}")
    elif MODE == "coarse":
        tax_path = DATA_DIR / "taxonomy_coarse.csv"
        label_col = "coarse_class"
        if not tax_path.exists():
            raise FileNotFoundError(tax_path)
        df_tax = pd.read_csv(tax_path)
        
        class_names = sorted(df_tax["coarse_class"].unique())
        print(f"[INFO] coarse 클래스 개수: {len(class_names)}")
    else:
        raise ValueError(f"Unsupported MODE: {MODE}")
    
    name_to_id = {name: i for i, name in enumerate(class_names)}
    id_to_name = {i: name for name, i in name_to_id.items()}
    
    print("[INFO] 클래스 매핑 예시 10개:")
    for i in range(min(10, len(id_to_name))):
        print(f"  {i}: {id_to_name[i]}")
    
    # ---------- 2. split CSV 로드 ----------
    def load_split(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if label_col not in df.columns:
            raise KeyError(f"{label_col} not in {path.name}")
        return df
    
    df_train = load_split(train_path)
    df_val = load_split(val_path)
    df_test = load_split(test_path)
    
    print(f"[INFO] train rows: {len(df_train)}")
    print(f"[INFO] val   rows: {len(df_val)}")
    print(f"[INFO] test  rows: {len(df_test)}")
    
    # ---------- 3. 출력 디렉터리 구조 준비 ----------
    OUT_ROOT = DATA_DIR / f"yolo_cls_{MODE}"
    if OUT_ROOT.exists():
        print(f"[WARN] {OUT_ROOT} 이미 존재합니다. 내용을 모두 지웁니다.")
        shutil.rmtree(OUT_ROOT)
    
    for split in ["train", "val", "test"]:
        split_dir = OUT_ROOT / split
        split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[OK] 출력 루트 디렉터리: {OUT_ROOT}")
    
    # ---------- 4. 이미지 심볼릭 링크 생성 ----------
    def build_split(df: pd.DataFrame, split_name: str):
        split_dir = OUT_ROOT / split_name
        n_missing = 0
        n_created = 0
        total = len(df)
        
        for idx, row in df.iterrows():
            label_name = row[label_col]
            class_id = name_to_id.get(label_name)
            if class_id is None:
                raise KeyError(f"'{label_name}' 에 해당하는 class_id가 없습니다. MODE={MODE}")
            
            src_rel = row["file_path"]
            src_path = IMG_DIR / src_rel
            if not src_path.is_file():
                n_missing += 1
                if n_missing <= 10:
                    print(f"[WARN] missing image: {src_path}")
                continue
            
            class_dir = split_dir / str(class_id)
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # 파일명 충돌 방지를 위해 split, 인덱스 prefix 사용
            dst_name = f"{split_name}_{idx}_{src_path.name}"
            dst_path = class_dir / dst_name
            
            try:
                # 심볼릭 링크 사용 (공간 절약). 필요시 shutil.copy2로 대체 가능.
                if not dst_path.exists():
                    os.symlink(src_path, dst_path)
                    n_created += 1
            except OSError:
                # 일부 환경에서 symlink가 안 될 수도 있으므로 fallback: copy
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
                    n_created += 1
        
        print(f"[OK] {split_name}: {n_created}개 링크 생성, {n_missing}개 미존재 (전체: {total})")
    
    build_split(df_train, "train")
    build_split(df_val, "val")
    build_split(df_test, "test")
    
    print("\n" + "=" * 80)
    print("[DONE] YOLO 분류용 데이터셋 생성 완료.")
    print(f"       경로: {OUT_ROOT}")
    print("=" * 80)
    
    # 생성된 이미지 수 확인
    for split in ["train", "val", "test"]:
        split_dir = OUT_ROOT / split
        total_images = sum(1 for _ in split_dir.rglob("*.png"))
        total_images += sum(1 for _ in split_dir.rglob("*.jpg"))
        print(f"  {split}: {total_images}개 이미지")


if __name__ == "__main__":
    main()

