#!/usr/bin/env python3
"""
YOLO Classification 평가 스크립트

학습된 YOLO 분류 모델을 평가합니다.

사용법:
    python scripts/eval_yolo_cls.py --model runs/classify/fine_cls_v2/weights/best.pt --mode fine
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics 패키지가 설치되지 않았습니다.")
    print("        conda activate archlens")
    print("        pip install ultralytics")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="YOLO Classification 모델 평가",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="평가할 모델 경로 (best.pt 또는 last.pt)",
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fine", "coarse"],
        default="fine",
        help="평가 모드: fine (64 클래스) 또는 coarse (19 클래스)",
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "dataset" / "icons",
        help="데이터셋 루트 디렉터리",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="val",
        help="평가할 split (val 또는 test)",
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=256,
        help="입력 이미지 크기",
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="배치 크기",
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="데이터 로더 워커 수",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="평가 디바이스 (None=auto, 0=cuda:0, cpu=cpu)",
    )
    
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="결과를 JSON 파일로 저장",
    )
    
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="결과 저장 디렉터리 (None이면 모델 디렉터리 사용)",
    )
    
    return parser.parse_args()


def validate_paths(model_path: Path, data_dir: Path, mode: str) -> tuple:
    """경로 검증"""
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    
    yolo_data_dir = data_dir / f"yolo_cls_{mode}"
    if not yolo_data_dir.exists():
        raise FileNotFoundError(f"데이터셋 디렉터리가 없습니다: {yolo_data_dir}")
    
    return model_path, yolo_data_dir


def main() -> None:
    """메인 함수"""
    args = parse_args()
    
    # 경로 검증
    model_path, data_path = validate_paths(args.model, args.data_dir, args.mode)
    
    # 저장 디렉터리 설정
    if args.save_dir is None:
        args.save_dir = model_path.parent.parent
    
    print("=" * 80)
    print("YOLO Classification 평가")
    print("=" * 80)
    print(f"모델          : {model_path}")
    print(f"데이터셋      : {data_path}")
    print(f"Split         : {args.split}")
    print(f"모드          : {args.mode}")
    print(f"이미지 크기   : {args.imgsz}")
    print(f"배치 크기     : {args.batch}")
    print(f"워커 수       : {args.workers}")
    print(f"디바이스      : {args.device or 'auto'}")
    print("=" * 80)
    
    # 모델 로드
    print(f"\n[1/2] 모델 로드: {model_path}")
    model = YOLO(str(model_path))
    
    # 평가 실행
    print(f"[2/2] {args.split} 세트 평가 중...")
    try:
        results = model.val(
            data=str(data_path),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            save_json=args.save_json,
            save_dir=str(args.save_dir),
            verbose=True,
        )
        
        print("\n" + "=" * 80)
        print("평가 완료!")
        print("=" * 80)
        
        # 주요 메트릭 출력
        if hasattr(results, "top1"):
            print(f"Top-1 Accuracy: {results.top1:.4f}")
        if hasattr(results, "top5"):
            print(f"Top-5 Accuracy: {results.top5:.4f}")
        if hasattr(results, "metrics"):
            print(f"\n전체 메트릭:")
            for key, value in results.metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        print("=" * 80)
        
        # JSON 저장
        if args.save_json:
            json_path = args.save_dir / f"eval_{args.split}_results.json"
            metrics_dict: Dict[str, Any] = {
                "model": str(model_path),
                "data": str(data_path),
                "split": args.split,
                "mode": args.mode,
            }
            
            if hasattr(results, "top1"):
                metrics_dict["top1_accuracy"] = float(results.top1)
            if hasattr(results, "top5"):
                metrics_dict["top5_accuracy"] = float(results.top5)
            if hasattr(results, "metrics"):
                metrics_dict["metrics"] = {
                    k: float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in results.metrics.items()
                }
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
            print(f"\n결과 저장: {json_path}")
        
    except Exception as e:
        print(f"\n[ERROR] 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

