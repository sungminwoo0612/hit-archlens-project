#!/usr/bin/env python3
"""
YOLO Classification 추론 스크립트

학습된 YOLO 분류 모델로 이미지를 예측합니다.

사용법:
    python scripts/predict_yolo_cls.py --model runs/classify/fine_cls_v2/weights/best.pt --source dataset/icons/images/...
    python scripts/predict_yolo_cls.py --model runs/classify/fine_cls_v2/weights/best.pt --source dataset/icons/images --save-txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    from ultralytics import YOLO
    import cv2
    from PIL import Image
except ImportError:
    print("[ERROR] 필요한 패키지가 설치되지 않았습니다.")
    print("        conda activate archlens")
    print("        pip install ultralytics opencv-python pillow")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="YOLO Classification 모델 추론",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="추론할 모델 경로 (best.pt 또는 last.pt)",
    )
    
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="입력 이미지 또는 디렉터리 경로",
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fine", "coarse"],
        default="fine",
        help="모드: fine (64 클래스) 또는 coarse (19 클래스)",
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "dataset" / "icons",
        help="데이터셋 루트 디렉터리 (클래스 이름 매핑용)",
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=256,
        help="입력 이미지 크기",
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="신뢰도 임계값",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="추론 디바이스 (None=auto, 0=cuda:0, cpu=cpu)",
    )
    
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="결과 저장 디렉터리 (None이면 모델 디렉터리/predict 사용)",
    )
    
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="텍스트 결과 저장",
    )
    
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="JSON 결과 저장",
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="결과 이미지 표시",
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="상위 K개 예측 출력",
    )
    
    return parser.parse_args()


def load_class_names(data_dir: Path, mode: str) -> Dict[int, str]:
    """클래스 이름 매핑 로드"""
    mapping_path = data_dir / f"class_mapping_{mode}.json"
    
    if not mapping_path.exists():
        print(f"[WARN] 클래스 매핑 파일이 없습니다: {mapping_path}")
        return {}
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    id_to_fine = mapping.get("id_to_fine", {})
    return {int(k): v for k, v in id_to_fine.items()}


def collect_image_paths(source: Path) -> List[Path]:
    """이미지 파일 경로 수집"""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".svg"}
    image_paths: List[Path] = []
    
    if source.is_file():
        if source.suffix.lower() in image_extensions:
            image_paths.append(source)
        else:
            print(f"[WARN] 지원하지 않는 이미지 형식: {source}")
    elif source.is_dir():
        for ext in image_extensions:
            image_paths.extend(source.rglob(f"*{ext}"))
            image_paths.extend(source.rglob(f"*{ext.upper()}"))
    else:
        raise FileNotFoundError(f"경로가 존재하지 않습니다: {source}")
    
    return sorted(image_paths)


def main() -> None:
    """메인 함수"""
    args = parse_args()
    
    # 경로 검증
    if not args.model.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {args.model}")
    
    # 이미지 경로 수집
    image_paths = collect_image_paths(args.source)
    if not image_paths:
        print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {args.source}")
        sys.exit(1)
    
    # 클래스 이름 로드
    class_names = load_class_names(args.data_dir, args.mode)
    
    # 저장 디렉터리 설정
    if args.save_dir is None:
        args.save_dir = args.model.parent.parent / "predict"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("YOLO Classification 추론")
    print("=" * 80)
    print(f"모델          : {args.model}")
    print(f"입력          : {args.source}")
    print(f"이미지 개수   : {len(image_paths)}")
    print(f"모드          : {args.mode}")
    print(f"이미지 크기   : {args.imgsz}")
    print(f"신뢰도 임계값 : {args.conf}")
    print(f"Top-K         : {args.top_k}")
    print(f"저장 디렉터리 : {args.save_dir}")
    print("=" * 80)
    
    # 모델 로드
    print(f"\n[1/3] 모델 로드: {args.model}")
    model = YOLO(str(args.model))
    
    # 추론 실행
    print(f"[2/3] 추론 중... ({len(image_paths)}개 이미지)")
    
    results_list: List[Dict[str, Any]] = []
    
    try:
        for i, img_path in enumerate(image_paths, 1):
            if i % 10 == 0:
                print(f"  진행: {i}/{len(image_paths)}")
            
            # 추론
            results = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            
            if not results:
                continue
            
            result = results[0]
            
            # 결과 파싱
            if hasattr(result, "probs") and result.probs is not None:
                probs = result.probs.data.cpu().numpy()
                top_k_indices = probs.argsort()[-args.top_k:][::-1]
                top_k_probs = probs[top_k_indices]
                
                predictions = []
                for idx, prob in zip(top_k_indices, top_k_probs):
                    class_name = class_names.get(int(idx), f"class_{idx}")
                    predictions.append({
                        "class_id": int(idx),
                        "class_name": class_name,
                        "confidence": float(prob),
                    })
                
                result_dict = {
                    "image_path": str(img_path),
                    "predictions": predictions,
                }
                results_list.append(result_dict)
                
                # 콘솔 출력
                print(f"\n{img_path.name}:")
                for pred in predictions[:3]:  # 상위 3개만 출력
                    print(f"  {pred['class_name']}: {pred['confidence']:.4f}")
    
    except KeyboardInterrupt:
        print("\n[WARN] 사용자에 의해 추론이 중단되었습니다.")
    except Exception as e:
        print(f"\n[ERROR] 추론 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 결과 저장
    print(f"\n[3/3] 결과 저장...")
    
    if args.save_json:
        json_path = args.save_dir / "predictions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)
        print(f"  JSON 저장: {json_path}")
    
    if args.save_txt:
        txt_path = args.save_dir / "predictions.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for result in results_list:
                f.write(f"{result['image_path']}\n")
                for pred in result["predictions"]:
                    f.write(f"  {pred['class_name']}: {pred['confidence']:.4f}\n")
                f.write("\n")
        print(f"  텍스트 저장: {txt_path}")
    
    print("\n" + "=" * 80)
    print("추론 완료!")
    print("=" * 80)
    print(f"처리된 이미지: {len(results_list)}/{len(image_paths)}")
    print(f"결과 디렉터리: {args.save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

