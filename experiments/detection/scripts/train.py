import os
import shutil
import random
import json
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
import optuna
import pandas as pd

# --- 1. 학습 설정 ---

# 스크립트 파일의 디렉토리 경로 (어디서 실행하든 상대 경로가 올바르게 작동)
SCRIPT_DIR = Path(__file__).parent.absolute()

# 실험 결과 저장 디렉터리 설정
EXPERIMENTS_DIR = SCRIPT_DIR.parent / 'experiments'
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_STUDIES_DIR = EXPERIMENTS_DIR / 'optuna_studies'
OPTUNA_STUDIES_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR = EXPERIMENTS_DIR / 'summaries'
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

# YOLOv8 학습 결과 저장 디렉터리
RUNS_DIR = SCRIPT_DIR.parent / 'runs'

# YOLOv8 모델 크기 선택 (n: nano, s: small, m: medium, l: large)
# AWS 다이어그램 아이콘은 크기가 비교적 작고 단순하므로 'n' 또는 's'로 시작하는 것을 추천합니다.
MODEL_SIZE = 'yolov8s.pt'  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# dataset.yaml 파일 경로 (스크립트 디렉토리 기준)
DATASET_YAML_PATH = str(SCRIPT_DIR.parent / 'dataset.yaml')

# 모델 학습 파라미터
EPOCHS = 200    # 300장 데이터셋으로 좋은 결과를 얻기 위해 더 많은 에포크가 필요할 수 있습니다 (예: 100~300)
BATCH_SIZE = 12 # GPU 메모리 크기에 따라 조정
IMG_SIZE = 640  # 입력 이미지 크기 (표준값)
PATIENCE = 30   # validation mAP가 30 에포크 동안 개선되지 않으면 학습 중단

# 데이터셋 분할 비율
TRAIN_RATIO = 0.8  # 80% 학습, 20% 검증
VAL_RATIO = 0.2

# --- 2. 데이터셋 준비 (train/val 폴더 생성) ---
def prepare_dataset():
    """이미지와 라벨을 train/val로 분할합니다."""
    images_dir = SCRIPT_DIR.parent / 'data' / 'aws_diagram_data' / 'images'
    labels_dir = SCRIPT_DIR.parent / 'data' / 'aws_diagram_data' / 'labels'
    train_images_dir = images_dir / 'train'
    val_images_dir = images_dir / 'val'
    train_labels_dir = labels_dir / 'train'
    val_labels_dir = labels_dir / 'val'
    
    # train/val 폴더 생성
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 목록 가져오기 (라벨 파일과 매칭)
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    # 라벨 파일과 매칭되는 이미지만 사용
    matched_pairs = []
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            matched_pairs.append((img_file, label_file))
    
    print(f"📊 총 {len(matched_pairs)}개의 이미지-라벨 쌍을 찾았습니다.")
    
    # 이미 분할되어 있으면 스킵
    if train_images_dir.exists() and any(train_images_dir.iterdir()):
        if val_images_dir.exists() and any(val_images_dir.iterdir()):
            print("✅ train/val 폴더가 이미 존재하고 데이터가 있습니다. 분할을 건너뜁니다.")
            return
    
    # 랜덤 셔플
    random.seed(42)  # 재현성을 위한 시드 고정
    random.shuffle(matched_pairs)
    
    # train/val 분할
    n_train = int(len(matched_pairs) * TRAIN_RATIO)
    train_pairs = matched_pairs[:n_train]
    val_pairs = matched_pairs[n_train:]
    
    print(f"📦 분할: train={len(train_pairs)}, val={len(val_pairs)}")
    
    # train 파일 복사/이동
    for img_file, label_file in train_pairs:
        shutil.copy2(img_file, train_images_dir / img_file.name)
        shutil.copy2(label_file, train_labels_dir / label_file.name)
    
    # val 파일 복사/이동
    for img_file, label_file in val_pairs:
        shutil.copy2(img_file, val_images_dir / img_file.name)
        shutil.copy2(label_file, val_labels_dir / label_file.name)
    
    print("✅ 데이터셋 분할 완료!")

# --- 3. 모델 학습 및 최적화 ---

def objective(trial: optuna.Trial):
    """
    Optuna Trial을 사용하여 YOLOv8 학습을 실행하고 검증 mAP를 반환하는 목적 함수.
    """
    
    # 1. 하이퍼파라미터 탐색 공간 정의
    
    # EPOCHS와 BATCH_SIZE는 고정 파라미터로 설정하거나, 학습 시간/리소스에 따라 탐색 공간을 줄이는 것을 고려
    # trial.suggest_int('epochs', 50, 200, step=50) 
    # trial.suggest_categorical('batch', [8, 16, 32])
    
    # Learning Rate 관련 파라미터 (일반적으로 가장 중요한 파라미터 중 하나)
    lr0 = trial.suggest_float('lr0', 1e-4, 1e-2, log=True) # 초기 학습률
    lrf = trial.suggest_float('lrf', 1e-3, 0.1) # 최종 학습률 비율 (lr0 * lrf)
    
    # Weights decay (정규화)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.001)
    
    # Optimizer 선택 (SGD vs AdamW)
    optimizer = trial.suggest_categorical('optimizer', ['SGD', 'AdamW'])
    
    # Momentum (SGD Optimizer 사용 시에만 의미가 있음)
    # AdamW를 사용할 경우 momentum은 무시되지만, Optuna가 자동으로 선택하므로 둘 다 탐색
    momentum = trial.suggest_float('momentum', 0.9, 0.98)
    
    # Early stopping patience (과적합 방지)
    # 작은 데이터셋의 경우 patience를 작게 설정하는 것이 좋을 수 있습니다
    patience = trial.suggest_int('patience', 20, 50, step=5)
    
    # 2. YOLOv8 모델 로드 (매 Trial마다 새로 로드)
    # aws_icon_detector_trial_{n} 폴더에 결과가 저장되도록 name을 설정
    model = YOLO(MODEL_SIZE)  
    
    # 3. 학습 시작
    # 학습 결과를 저장할 이름에 Trial 번호를 포함시켜 충돌 방지
    experiment_name = f'aws_icon_detector_trial_{trial.number}'
    
    # 학습 파라미터 구성
    train_params = {
        'data': DATASET_YAML_PATH,  
        'epochs': EPOCHS,           # 고정 값 사용 (자원 및 시간 고려)
        'imgsz': IMG_SIZE,          
        'batch': BATCH_SIZE,        # 고정 값 사용
        'name': experiment_name,
        'project': str(RUNS_DIR),   # obj_detection/runs에 저장
        
        # Optuna에서 제안된 하이퍼파라미터 적용
        'lr0': lr0, 
        'lrf': lrf,
        'weight_decay': weight_decay,
        'optimizer': optimizer,
        'patience': patience,  # Early stopping
    }
    
    # Momentum은 SGD에만 적용 (AdamW는 내부적으로 다른 방식 사용)
    if optimizer == 'SGD':
        train_params['momentum'] = momentum
    
    # GPU 메모리 제한 등으로 인해 BATCH_SIZE를 고정하거나, 더 작은 탐색 공간을 설정할 수 있습니다.
    results = model.train(**train_params)
    
    # 4. 성능 지표 추출 및 반환
    # YOLOv8 학습 결과에서 best.pt에 해당하는 mAP50-95를 가져옵니다.
    # results 객체는 딕셔너리와 같은 구조로 metric을 가집니다.
    # mAP50-95 (mean Average Precision across IoU thresholds 0.5 to 0.95)
    
    # Ultralytics results 객체는 results.results_dict에 metrics를 저장합니다.
    map_score = results.results_dict['metrics/mAP50-95(B)'] 
    
    # Optuna에 현재 Trial의 mAP를 보고합니다.
    trial.report(map_score, EPOCHS)
    
    # 최적화 방향이 maximize이므로, mAP를 반환합니다.
    return map_score

# --- 4. 모델 추론 (새 이미지 테스트) ---
def predict_on_new_image(model, image_path):
    """학습된 모델을 사용하여 새로운 AWS 다이어그램 이미지에서 아이콘을 탐지합니다."""
    
    print(f"\n🔍 이미지 분석 시작: {image_path}")
    
    # 추론 수행
    results = model.predict(
        source=image_path, 
        conf=0.25, # 최소 신뢰도(Confidence) 임계값
        iou=0.7,   # IOU(Intersection Over Union) 임계값
        save=True, # 결과 이미지 저장 (runs/detect/predict 폴더)
        # show=True  # 실시간으로 결과 보기 (터미널 환경에서는 주석 처리)
    )
    
    # 탐지 결과 출력
    for r in results:
        # 박스 정보 (경계 상자)
        boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        # 클래스 ID
        class_ids = r.boxes.cls.cpu().numpy().astype(int) 
        # 신뢰도 점수
        confidences = r.boxes.conf.cpu().numpy()
        
        # 클래스 이름 매핑 (dataset.yaml에서 가져온 names)
        class_names = r.names
        
        print(f"--- 탐지된 아이콘 수: {len(boxes)} ---")
        for box, class_id, conf in zip(boxes, class_ids, confidences):
            icon_name = class_names[class_id]
            print(f"아이콘: {icon_name}, 신뢰도: {conf:.2f}, 박스: {box.round().astype(int)}")

    print(f"결과 이미지는 {RUNS_DIR}/detect/predict 폴더에 저장되었습니다.")


if __name__ == '__main__':
    
    # 0. 데이터셋 준비 (train/val 폴더 생성 및 분할)
    prepare_dataset()
    
    # 1. Optuna Study 생성 및 실행
    
    # Study 저장 경로 설정
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    study_name = f'aws_icon_detector_{timestamp}'
    study_db_path = OPTUNA_STUDIES_DIR / f'{study_name}.db'
    
    # Study 생성: maximize (최대화) 목표 (mAP를 높여야 하므로)
    # SQLite DB에 저장하여 나중에 재개 가능
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=f'sqlite:///{study_db_path}',
        load_if_exists=False  # 기존 study가 있으면 로드
    )
    
    # N번의 Trial 실행 (자원과 시간에 맞게 횟수 조정)
    N_TRIALS = 50 
    
    print(f"\n✨ Optuna Study 시작: 총 {N_TRIALS}회 Trial 진행")
    print(f"📁 Study 저장 위치: {study_db_path}")
    study.optimize(objective, n_trials=N_TRIALS)
    
    # 2. 최적의 하이퍼파라미터와 성능 출력
    print("\n--- Optuna 최적화 결과 ---")
    print(f"최적의 mAP50-95: {study.best_value:.4f}")
    print("최적의 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 3. 실험 결과 저장 (JSON 및 CSV)
    # JSON 요약 저장
    summary = {
        'study_name': study_name,
        'timestamp': timestamp,
        'n_trials': N_TRIALS,
        'best_value': float(study.best_value),
        'best_params': {k: float(v) if isinstance(v, (int, float)) else v 
                       for k, v in study.best_params.items()},
        'best_trial_number': study.best_trial.number,
    }
    
    summary_json_path = SUMMARIES_DIR / f'{study_name}_summary.json'
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n💾 실험 요약 저장: {summary_json_path}")
    
    # 모든 Trial 결과를 DataFrame으로 저장
    try:
        trials_df = study.trials_dataframe()
        summary_csv_path = SUMMARIES_DIR / f'{study_name}_trials.csv'
        trials_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        print(f"💾 Trial 결과 저장: {summary_csv_path}")
    except Exception as e:
        print(f"⚠️ Trial 결과 CSV 저장 실패: {e}")
        
    # 4. 최적의 모델 로드 및 추론 (선택 사항)
    # 최적의 Trial 번호를 기반으로 best.pt 파일 경로를 추론합니다.
    best_trial_num = study.best_trial.number
    best_run_name = f'aws_icon_detector_trial_{best_trial_num}'
    best_model_path = RUNS_DIR / 'detect' / best_run_name / 'weights' / 'best.pt'
    
    if best_model_path.exists():
        final_model = YOLO(str(best_model_path)) 
        print(f"\n--- 최적의 Trial {best_trial_num} 모델로 테스트 시작 ---")
        
        # 테스트 이미지 경로 설정
        test_image_filename = '0bdfa8fb-imgi_633_Face-blurring_serverless_architecture.png'
        test_image_path = SCRIPT_DIR.parent / 'data' / 'aws_diagram_data' / 'images' / test_image_filename
        
        # 추론 실행
        predict_on_new_image(final_model, str(test_image_path))
    else:
        print(f"\n⚠️ 최적 모델 파일({best_model_path})을 찾을 수 없습니다. 학습이 완료되지 않았을 수 있습니다.")