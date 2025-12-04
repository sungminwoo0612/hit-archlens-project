#!/usr/bin/env python3
"""
YOLO Classification 학습 스크립트

AWS 아이콘 분류 모델을 YOLO로 학습합니다.
conda archlens 환경에서 실행합니다.

사용법:
    python scripts/train_yolo_cls.py --mode fine --epochs 100 --imgsz 256
    python scripts/train_yolo_cls.py --mode coarse --epochs 50 --imgsz 128
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

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
        description="YOLO Classification 모델 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fine", "coarse"],
        default="fine",
        help="학습 모드: fine (64 클래스) 또는 coarse (19 클래스)",
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "dataset" / "icons",
        help="데이터셋 루트 디렉터리",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-cls.pt",
        help="사전 학습된 모델 (yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt, yolov8l-cls.pt, yolov8x-cls.pt)",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="학습 에포크 수",
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
        help="학습 디바이스 (None=auto, 0=cuda:0, cpu=cpu)",
    )
    
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).parent.parent / "runs" / "classify",
        help="프로젝트 디렉터리",
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="실험 이름 (None이면 자동 생성)",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="재개할 체크포인트 경로",
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience",
    )
    
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="초기 학습률",
    )
    
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="최종 학습률 (lr0 * lrf)",
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="가중치 감쇠",
    )
    
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        help="Warmup 에포크 수",
    )
    
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Automatic Mixed Precision 사용",
    )
    
    parser.add_argument(
        "--cache",
        action="store_true",
        help="이미지를 메모리에 캐시",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="상세 출력",
    )
    
    return parser.parse_args()


def validate_data_dir(data_dir: Path, mode: str) -> Path:
    """데이터셋 디렉터리 검증"""
    if not data_dir.exists():
        raise FileNotFoundError(f"데이터셋 디렉터리가 없습니다: {data_dir}")
    
    yolo_data_dir = data_dir / f"yolo_cls_{mode}"
    if not yolo_data_dir.exists():
        raise FileNotFoundError(
            f"YOLO 데이터셋 디렉터리가 없습니다: {yolo_data_dir}\n"
            f"먼저 다음 명령을 실행하세요:\n"
            f"  ./aws_icon_yolo_cls_prepare_and_train.sh {mode} {data_dir}"
        )
    
    for split in ["train", "val"]:
        split_dir = yolo_data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split 디렉터리가 없습니다: {split_dir}")
    
    return yolo_data_dir


def main() -> None:
    """메인 함수"""
    args = parse_args()
    
    # 데이터셋 검증
    data_path = validate_data_dir(args.data_dir, args.mode)
    
    # 실험 이름 생성
    if args.name is None:
        args.name = f"{args.mode}_cls_{args.model.replace('.pt', '')}"
    
    print("=" * 80)
    print("YOLO Classification 학습 시작")
    print("=" * 80)
    print(f"모드          : {args.mode}")
    print(f"데이터셋      : {data_path}")
    print(f"모델          : {args.model}")
    print(f"에포크        : {args.epochs}")
    print(f"이미지 크기   : {args.imgsz}")
    print(f"배치 크기     : {args.batch}")
    print(f"워커 수       : {args.workers}")
    print(f"디바이스      : {args.device or 'auto'}")
    print(f"프로젝트      : {args.project}")
    print(f"실험 이름     : {args.name}")
    print(f"학습률        : {args.lr0} -> {args.lr0 * args.lrf}")
    print(f"Patience      : {args.patience}")
    print("=" * 80)
    
    # 모델 로드
    print(f"\n[1/3] 모델 로드: {args.model}")
    model = YOLO(args.model)
    
    # 학습 설정
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": str(args.project),
        "name": args.name,
        "patience": args.patience,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "amp": args.amp,
        "cache": args.cache,
        "verbose": args.verbose,
        "exist_ok": True,
    }
    
    if args.resume:
        train_kwargs["resume"] = args.resume
        print(f"[2/3] 체크포인트에서 재개: {args.resume}")
    else:
        print("[2/3] 새로 학습 시작")
    
    # 학습 실행
    print("[3/3] 학습 시작...")
    try:
        results = model.train(**train_kwargs)
        
        print("\n" + "=" * 80)
        print("학습 완료!")
        print("=" * 80)
        print(f"최종 모델 경로: {args.project / args.name / 'weights' / 'best.pt'}")
        print(f"결과 디렉터리: {args.project / args.name}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n[WARN] 사용자에 의해 학습이 중단되었습니다.")
        print(f"체크포인트는 저장되었을 수 있습니다: {args.project / args.name}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

