"""
Hit ArchLens CLI

통합 명령행 인터페이스 도구
"""

import click
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv # load_dotenv 임포트
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# .env 파일 로드
load_dotenv()

from ..core.providers.aws import (
    AWSCVAutoLabeler,
    AWSLLMAutoLabeler,
    AWSHybridAutoLabeler
)
from ..core.data_collectors import AWSDataCollector
from ..core.models import AnalysisResult, BatchAnalysisResult # BatchAnalysisResult도 임포트
from ..core.utils import visualizer # visualizer 모듈 임포트
import os


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Hit ArchLens CLI
    
    멀티 클라우드 아키텍처 다이어그램 자동 분석 도구
    """
    pass


def visualize_analysis_results(batch_results: BatchAnalysisResult, output_dir: Path, verbose: bool):
    """분석 결과를 시각화합니다 (그래프 및 바운딩 박스 이미지)."""
    visualizations_output_dir = output_dir / "visualizations"
    visualizations_output_dir.mkdir(parents=True, exist_ok=True)

    # 주요 통계 그래프
    visualizer.plot_detection_confidence(batch_results.results, visualizations_output_dir / "confidence_distribution.png", f"Detection Confidence ({batch_results.results[0].analysis_method.value})")
    visualizer.plot_service_distribution(batch_results.results, visualizations_output_dir / "service_distribution.png", title=f"Top Detected Services ({batch_results.results[0].analysis_method.value})")
    visualizer.plot_processing_time(batch_results, visualizations_output_dir / "processing_time.png", f"Processing Time ({batch_results.results[0].analysis_method.value})")
    visualizer.plot_detection_counts(batch_results.results, visualizations_output_dir / "detection_counts.png", f"Detection Counts ({batch_results.results[0].analysis_method.value})")
    visualizer.plot_normalization_success_rate(batch_results.results, visualizations_output_dir / "normalization_success_rate.png", f"Normalization Success Rate ({batch_results.results[0].analysis_method.value})")
    visualizer.plot_detection_status_distribution(batch_results.results, visualizations_output_dir / "detection_status_distribution.png", f"Detection Status ({batch_results.results[0].analysis_method.value})")

    # 각 이미지에 바운딩 박스 그리기
    for ar in batch_results.results:
        if ar.success:
            # 원본 이미지 파일명에 _detections 접미사 추가
            image_name = ar.image_path.stem
            image_extension = ar.image_path.suffix
            output_image_path = visualizations_output_dir / f"{image_name}_detections{image_extension}"
            
            visualizer.draw_detections_on_image(
                image_path=ar.image_path,
                detections=ar.detections,
                output_path=output_image_path
            )
        else:
            if verbose:
                click.echo(f"⚠️ 분석 실패한 이미지 ({ar.image_path.name})는 시각화하지 않습니다.")

    # 요약 보고서 생성
    visualizer.create_summary_report(batch_results, visualizations_output_dir)


def create_threshold_summary_report(
    all_threshold_results: Dict[float, BatchAnalysisResult], 
    output_base_dir: Path,
    method: str
) -> None:
    """
    모든 신뢰도 임계값에 대한 분석 통계를 요약하여 보고서로 저장합니다.
    """
    summary_data = []
    for threshold, result in all_threshold_results.items():
        summary_data.append({
            "threshold": threshold,
            "method": method,
            "total_images": result.total_images,
            "successful_images": result.success_count,
            "failed_images": result.error_count,
            "success_rate": result.success_rate,
            "total_detections": result.total_detections,
            "average_detections_per_image": result.average_detections_per_image,
            "average_processing_time": result.average_processing_time,
            "average_confidence": result.average_confidence,
            "average_normalization_success_rate": result.average_normalization_success_rate
        })
    
    if not summary_data:
        click.echo("⚠️ 신뢰도 임계값별 요약 보고서를 생성할 데이터가 없습니다.")
        return

    df_summary = pd.DataFrame(summary_data)
    
    summary_report_dir = output_base_dir / "evaluation" / "threshold_analysis"
    summary_report_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = summary_report_dir / f"{method}_threshold_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    click.echo(f"\n✅ 신뢰도 임계값별 요약 보고서 저장: {csv_path}")

    # 시각화 추가 (예시: 신뢰도별 성공률 변화)
    visualizer.setup_plot_style()
    plt.figure()
    sns.lineplot(data=df_summary, x="threshold", y="success_rate", marker='o')
    plt.title(f"{method.upper()} Success Rate by Confidence Threshold")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.savefig(summary_report_dir / f"{method}_success_rate_by_threshold.png")
    plt.close()
    click.echo(f"✅ 신뢰도 임계값별 성공률 그래프 저장: {summary_report_dir / f'{method}_success_rate_by_threshold.png'}")


@cli.command()
@click.option('--config', '-c', default='backend/configs/default.yaml', 
              help='설정 파일 경로')
@click.option('--method', '-m', 
              type=click.Choice(['cv', 'llm', 'hybrid']), 
              default='hybrid',
              help='분석 방법')
@click.option('--output', '-o', default='data/outputs/experiments', # 기본 출력 디렉토리
              help='출력 디렉터리')
@click.option('--format', '-f',
              type=click.Choice(['json', 'csv', 'yaml']),\
              default='json',
              help='출력 형식')
@click.option('--verbose', '-v', is_flag=True,
              help='상세 출력')
@click.option('--confidence-thresholds', '-t', default='0.0,0.2,0.4,0.6,0.8', # 추가된 옵션
              help='분석에 사용할 신뢰도 임계값 목록 (콤마로 구분, 예: 0.1,0.5,0.9)')
@click.argument('input_path')
def analyze(config, method, output, format, verbose, input_path, confidence_thresholds): # 인자에 confidence_thresholds 추가
    """
    이미지 분석
    
    AWS 아키텍처 다이어그램을 분석하여 서비스 아이콘을 인식합니다.
    
    INPUT_PATH: 분석할 이미지 파일 또는 디렉터리 경로
    """
    try:
        # 설정 로드
        config_data = load_config(config)
        
        # LLM API 키 환경 변수에서 로드
        if "llm" in config_data:
            if "api_key" in config_data["llm"] and config_data["llm"]["api_key"].startswith("${") and config_data["llm"]["api_key"].endswith("}"):
                env_var_name = config_data["llm"]["api_key"][2:-1]
                config_data["llm"]["api_key"] = os.getenv(env_var_name, config_data["llm"]["api_key"])
                if config_data["llm"]["api_key"] == env_var_name:
                    print(f"⚠️ 환경 변수 '{env_var_name}'가 설정되지 않았습니다. LLM 기능이 작동하지 않을 수 있습니다.")
        
        # 신뢰도 임계값 파싱
        threshold_list = [float(t.strip()) for t in confidence_thresholds.split(',')]
        threshold_list.sort() # 낮은 값부터 높은 값 순으로 정렬
        
        all_threshold_results: Dict[float, BatchAnalysisResult] = {}

        for threshold in threshold_list:
            click.echo(f"\n--- 📈 분석 시작: 신뢰도 임계값 = {threshold:.1f} ---")
            
            # config_data['runtime']이 없을 경우 초기화
            if 'runtime' not in config_data:
                config_data['runtime'] = {}
            config_data['runtime']['conf_threshold'] = threshold # 런타임 설정 업데이트

            # 오토라벨러 생성 (새로운 임계값으로)
            labeler = create_auto_labeler(method, config_data)
            
            # 입력 경로 처리
            current_input_path = Path(input_path)
            
            batch_results: BatchAnalysisResult
            if current_input_path.is_file():
                # 단일 파일 분석
                results = [labeler.analyze_image(current_input_path)] 
                batch_results = BatchAnalysisResult(
                    results=results,
                    total_images=1,
                    total_detections=len(results[0].detections) if results and results[0].success else 0,
                    average_processing_time=results[0].processing_time if results and results[0].success else 0.0,
                    success_count=1 if results and results[0].success else 0,
                    error_count=0 if results and results[0].success else 1,
                    errors=[results[0].errors[0]] if results and not results[0].success and results[0].errors else []
                )
            elif current_input_path.is_dir():
                # 디렉터리 분석
                image_files = find_image_files(current_input_path)
                if not image_files:
                    click.echo(f"❌ 이미지 파일을 찾을 수 없습니다: {current_input_path}")
                    return
                
                click.echo(f" {len(image_files)}개 이미지 파일 발견")
                batch_results = labeler.analyze_batch(image_files)
            else:
                click.echo(f"❌ 유효하지 않은 경로: {current_input_path}")
                return

            all_threshold_results[threshold] = batch_results

            # 결과 저장 경로 설정 (임계값 포함)
            threshold_output_dir = Path(output) / f"{method}_results_conf_{str(threshold).replace('.', '_')}"
            threshold_output_dir.mkdir(parents=True, exist_ok=True)

            save_results(batch_results.results, threshold_output_dir, format, verbose)
            
            # 바운딩 박스 시각화
            visualize_analysis_results(batch_results, threshold_output_dir, verbose)

            # 통계 출력
            print_analysis_statistics(batch_results, verbose)
        
        # 모든 임계값에 대한 최종 통계 보고서 생성
        create_threshold_summary_report(all_threshold_results, Path(output), method)

    except Exception as e:
        click.echo(f"❌ 분석 실패: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@cli.command()
@click.option('--data-type', default='all', help='수집할 데이터 타입 (icons, services, products, all)')
@click.option('--config', default='backend/configs/default.yaml', help='설정 파일 경로')
@click.option('--verbose', is_flag=True, help='상세 출력')
@click.option('--monitor', is_flag=True, help='실시간 진행 상황 모니터링')
def collect_data(data_type, config, verbose, monitor):
    """데이터 수집 실행"""
    try:
        # 출력 디렉터리 구조 설정
        from ..core.data_collectors.setup_output_structure import setup_output_structure
        setup_output_structure()
        
        # 설정 로드
        with open(config, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # AWS 설정과 collectors 설정을 모두 포함
        aws_config = {
            "region": config_data.get("aws", {}).get("region", "us-east-1"),
            "collectors": config_data.get("collectors", {}),
            "data_types": [data_type] if data_type != "all" else ["icons", "services", "products"],
            "output_dir": "data/outputs"  # data/outputs/ 디렉터리로 설정
        }
        
        if verbose:
            print(f"📋 설정 정보:")
            print(f"   - 리전: {aws_config['region']}")
            print(f"   - 수집 타입: {aws_config['data_types']}")
            print(f"   - 출력 디렉터리: {aws_config['output_dir']}")
            print(f"   - 아이콘 ZIP: {aws_config['collectors'].get('icons', {}).get('zip_path', 'Not set')}")
            print(f"   - 제품 API: {aws_config['collectors'].get('products', {}).get('api_url', 'Not set')}")
        
        # AWS 데이터 수집기 생성
        collector = AWSDataCollector(aws_config)
        
        if monitor:
            # 실시간 모니터링 모드
            print("🔍 실시간 모니터링 모드 시작")
            print("Ctrl+C로 중단할 수 있습니다")
            
            import threading
            import time
            
            def monitor_progress():
                while True:
                    progress = collector.get_progress()
                    if progress["current_task"]:
                        print(f"\r {progress['current_task']}: {progress['progress_percentage']:.1f}% "
                              f"({progress['detailed_status'].get('processed', 0)}/"
                              f"{progress['detailed_status'].get('total', 0)})", end="")
                    time.sleep(1)
            
            # 모니터링 스레드 시작
            monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
            monitor_thread.start()
        
        # 데이터 수집 실행
        result = collector.collect()
        
        if result.success:
            print(f"\n✅ 데이터 수집 완료!")
            print(f"   - 수집된 항목: {result.data_count}개")
            print(f"   - 처리 시간: {result.processing_time:.2f}초")
            print(f"   - 출력 파일: {len(result.output_paths)}개")
            print(f"   - 저장 위치: data/outputs/ 디렉터리")
        else:
            print(f"\n❌ 데이터 수집 실패!")
            print(f"   - 오류: {', '.join(result.errors)}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@cli.command()
@click.option('--config', '-c', default='backend/configs/default.yaml',
              help='설정 파일 경로')
@click.option('--method', '-m',
              type=click.Choice(['cv', 'llm', 'hybrid']),
              default='hybrid',
              help='분석 방법')
def status(config, method):
    """
    시스템 상태 확인
    
    오토라벨러와 데이터 수집기의 상태를 확인합니다.
    """
    try:
        # 설정 로드
        config_data = load_config(config)
        
        click.echo("🔍 시스템 상태 확인")
        click.echo("=" * 50)
        
        # 오토라벨러 상태
        click.echo("📊 오토라벨러 상태:")
        labeler = create_auto_labeler(method, config_data)
        labeler_stats = labeler.get_statistics()
        
        for key, value in labeler_stats.items():
            click.echo(f"   {key}: {value}")
        
        click.echo()
        
        # 데이터 수집기 상태
        click.echo("📊 데이터 수집기 상태:")
        collector = AWSDataCollector(config_data)
        collector_stats = collector.get_collection_status()
        
        for key, value in collector_stats.items():
            click.echo(f"   {key}: {value}")
        
        click.echo()
        click.echo("✅ 상태 확인 완료")
        
    except Exception as e:
        click.echo(f"❌ 상태 확인 실패: {e}")


@cli.command()
@click.option('--config', '-c', default='backend/configs/default.yaml',
              help='설정 파일 경로')
@click.option('--method', '-m',
              type=click.Choice(['cv', 'llm', 'hybrid']),
              default='hybrid',
              help='분석 방법')
@click.option('--output', '-o', default='output',
              help='출력 디렉터리')
@click.argument('input_path')
def compare_methods(config, method, output, input_path):
    """
    분석 방법 비교
    
    CV, LLM, 하이브리드 방법의 결과를 비교합니다.
    """
    try:
        # 설정 로드
        config_data = load_config(config)
        
        # 입력 경로 처리
        input_path = Path(input_path)
        if not input_path.is_file():
            click.echo(f"❌ 파일이 아닙니다: {input_path}")
            return
        
        click.echo("🔍 분석 방법 비교 시작")
        
        # 각 방법별 분석
        methods = ['cv', 'llm', 'hybrid']
        results = {}
        
        for method_name in methods:
            click.echo(f"   �� {method_name.upper()} 분석 중...")
            labeler = create_auto_labeler(method_name, config_data)
            result = labeler.analyze_image(input_path)
            results[method_name] = result
        
        # 결과 비교
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_data = {
            "input_image": str(input_path),
            "image_size": (results['cv'].width, results['cv'].height),
            "methods": {}
        }
        
        for method_name, result in results.items():
            comparison_data["methods"][method_name] = {
                "detection_count": len(result.detections),
                "processing_time": result.processing_time,
                "average_confidence": result.average_confidence,
                "detections": [det.to_dict() for det in result.detections]
            }
        
        # 결과 저장
        comparison_path = output_dir / "method_comparison.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        # 비교 결과 출력
        click.echo("\n📊 분석 방법 비교 결과:")
        click.echo("-" * 50)
        for method_name, data in comparison_data["methods"].items():
            click.echo(f"{method_name.upper():>8}: "
                      f"{data['detection_count']:>3}개 감지, "
                      f"{data['processing_time']:>6.2f}초, "
                      f"평균 신뢰도: {data['average_confidence']:.3f}")
        
        click.echo(f"\n📄 상세 결과 저장: {comparison_path}")
        
    except Exception as e:
        click.echo(f"❌ 방법 비교 실패: {e}")


@cli.command()
@click.argument('results_dir')
def visualize(output, results_dir):
    """
    분석 결과 시각화
    
    저장된 분석 결과 JSON 파일들을 기반으로 다양한 통계 그래프를 생성합니다.
    
    RESULTS_DIR: analyze 명령어로 생성된 JSON 결과 파일들이 있는 디렉터리
    """
    try:
        results_path = Path(results_dir)
        if not results_path.is_dir():
            click.echo(f"❌ 유효하지 않은 디렉터리: {results_dir}")
            return
        
        # JSON 결과 파일 로드
        analysis_results: List[AnalysisResult] = []
        for file_path in results_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    analysis_results.append(AnalysisResult.from_dict(data))
            except Exception as e:
                click.echo(f"⚠️ 결과 파일 로드 실패: {file_path} - {e}")
                continue
        
        if not analysis_results:
            click.echo("❌ 시각화할 분석 결과 JSON 파일을 찾을 수 없습니다.")
            return
        
        # 배치 분석 결과 객체 생성 (통계 계산용)
        total_detections = sum(r.detection_count for r in analysis_results)
        total_processing_time = sum(r.processing_time for r in analysis_results)
        
        batch_results = BatchAnalysisResult(
            results=analysis_results,
            total_images=len(analysis_results),
            total_detections=total_detections,
            average_processing_time=total_processing_time / len(analysis_results) if analysis_results else 0.0,
            success_count=len(analysis_results), # 여기서는 모든 로드된 파일이 성공으로 간주
            error_count=0
        )
        
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"📊 분석 결과 시각화 시작: {results_dir}")
        
        # 1. 신뢰도 분포
        visualizer.plot_confidence_distribution(
            analysis_results, output_dir / "confidence_distribution.png",
            title=f"Confidence Distribution ({batch_results.results[0].analysis_method.value.upper()})"
        )
        
        # 2. 처리 시간
        visualizer.plot_processing_time(
            analysis_results, output_dir / "processing_time.png"
        )
        
        # 3. 감지 수
        visualizer.plot_detection_counts(
            analysis_results, output_dir / "detection_counts.png"
        )

        # 4. 서비스 분포
        visualizer.plot_service_distribution(
            analysis_results, output_dir / "service_distribution.png"
        )

        # 5. 정규화 성공률
        visualizer.plot_normalization_success_rate(
            analysis_results, output_dir / "normalization_success_rate.png"
        )
        
        # 6. 요약 보고서
        visualizer.create_summary_report(batch_results, output_dir)
        
        click.echo(f"✅ 시각화 완료. 결과는 '{output_dir}' 디렉터리에 저장되었습니다.")
        
    except Exception as e:
        click.echo(f"❌ 시각화 실패: {e}")
        import traceback
        traceback.print_exc()


@cli.command()
@click.option('--data-dir', default='out', help='데이터 디렉터리 경로')
@click.option('--verbose', is_flag=True, help='상세 출력')
def generate_unified_taxonomy(data_dir, verbose):
    """통합 택소노미 생성"""
    try:
        from ..core.data_collectors.unified_taxonomy_generator import generate_unified_taxonomy
        
        print("🔍 통합 택소노미 생성 시작")
        
        success = generate_unified_taxonomy(data_dir)
        
        if success:
            print("✅ 통합 택소노미 생성 완료")
            print(f"   - 출력 위치: {data_dir}/unified/")
            print(f"   - 파일: aws_unified_taxonomy.csv, aws_unified_taxonomy.json")
        else:
            print("❌ 통합 택소노미 생성 실패")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@cli.command()
@click.option('--config', '-c', default='backend/configs/default.yaml',
              help='설정 파일 경로')
@click.option('--input', '-i', required=True, help='시각화할 분석 결과 디렉터리 또는 파일 경로')
@click.option('--output', '-o', default='data/outputs/visualizations', help='시각화 결과 저장 디렉터리')
@click.option('--verbose', '-v', is_flag=True, help='상세 출력')
def visualize(config, input, output, verbose):
    """
    분석 결과 시각화
    
    분석 결과를 그래프 및 차트로 시각화합니다.
    """
    try:
        config_data = load_config(config)
        
        input_path = Path(input)
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 분석 결과 시각화 시작: {input_path}")
        
        # BatchAnalysisResult 로드 (analyze 명령어에서 저장한 형식)
        if input_path.is_dir():
            # 디렉터리인 경우 모든 analysis_result_*.json 파일을 로드하여 BatchAnalysisResult로 통합
            analysis_results = []
            for json_file in input_path.glob("analysis_result_*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    analysis_results.append(AnalysisResult(**data))
            
            if not analysis_results:
                click.echo(f"❌ 시각화할 분석 결과를 찾을 수 없습니다: {input_path}")
                return
            
            batch_results = BatchAnalysisResult(
                results=analysis_results,
                total_images=len(analysis_results),
                total_detections=sum(len(ar.detections) for ar in analysis_results),
                average_processing_time=sum(ar.processing_time for ar in analysis_results) / len(analysis_results) if analysis_results else 0,
                success_count=sum(1 for ar in analysis_results if ar.success),
                error_count=sum(1 for ar in analysis_results if not ar.success)
            )
            
        elif input_path.is_file() and input_path.name.startswith("analysis_result_") and input_path.name.endswith(".json"):
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                analysis_result = AnalysisResult(**data)
                batch_results = BatchAnalysisResult(
                    results=[analysis_result],
                    total_images=1,
                    total_detections=len(analysis_result.detections),
                    average_processing_time=analysis_result.processing_time,
                    success_count=1 if analysis_result.success else 0,
                    error_count=0 if analysis_result.success else 1
                )
        else:
            click.echo(f"❌ 유효하지 않은 입력 경로 또는 파일 형식: {input_path}")
            return
        
        # 시각화 함수 호출
        visualizer.plot_detection_confidence(batch_results.results, output_path / "confidence_distribution.png")
        visualizer.plot_service_distribution(batch_results.results, output_path / "service_distribution.png")
        visualizer.plot_processing_time(batch_results, output_path / "processing_time.png")
        
        print(f"✅ 시각화 완료. 결과는 '{output_path}'에 저장되었습니다.")
        
    except Exception as e:
        click.echo(f"❌ 시각화 실패: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@cli.command()
@click.option('--config', '-c', default='backend/configs/default.yaml',
              help='설정 파일 경로')
@click.option('--method', '-m',
              type=click.Choice(['cv', 'llm', 'hybrid']),
              default='hybrid',
              help='분석 방법')
@click.option('--output', '-o', default='runs/demo',
              help='출력 디렉터리')
@click.option('--verbose', '-v', is_flag=True,
              help='상세 출력')
def demo(config, method, output, verbose):
    """
    데모 실행
    
    기본 데모 이미지로 빠른 테스트를 실행합니다.
    """
    demo_image = Path("assets/demo_diagram.jpg")
    
    if not demo_image.exists():
        click.echo(f"❌ 데모 이미지를 찾을 수 없습니다: {demo_image}")
        click.echo("💡 assets/demo_diagram.jpg 파일이 필요합니다.")
        return
    
    click.echo(f"🚀 데모 실행 중...")
    click.echo(f"   입력: {demo_image}")
    click.echo(f"   방법: {method}")
    click.echo(f"   출력: {output}\n")
    
    # analyze 함수를 직접 호출
    try:
        config_data = load_config(config)
        
        # LLM API 키 환경 변수에서 로드
        if "llm" in config_data:
            if "api_key" in config_data["llm"] and config_data["llm"]["api_key"].startswith("${") and config_data["llm"]["api_key"].endswith("}"):
                env_var_name = config_data["llm"]["api_key"][2:-1]
                config_data["llm"]["api_key"] = os.getenv(env_var_name, config_data["llm"]["api_key"])
        
        # 단일 임계값으로 분석 (데모용)
        threshold_list = [0.5]
        
        from ..core.providers.aws import AWSHybridAutoLabeler
        labeler = AWSHybridAutoLabeler(config_data)
        
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 분석
        result = labeler.analyze_image(demo_image, confidence_threshold=threshold_list[0])
        
        if result.success:
            click.echo(f"✅ 분석 완료!")
            click.echo(f"   감지된 서비스 수: {len(result.detections)}")
            if result.detections:
                click.echo(f"   첫 번째 감지: {result.detections[0].service_name} (신뢰도: {result.detections[0].confidence:.3f})")
            
            # 결과 저장
            result_path = output_dir / "demo_result.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            click.echo(f"   결과 저장: {result_path}")
        else:
            click.echo(f"⚠️ 분석 실패: {result.error_message}")
            
    except Exception as e:
        click.echo(f"❌ 데모 실행 실패: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


@cli.command()
@click.option('--config', '-c', default='backend/configs/default.yaml',
              help='설정 파일 경로')
@click.option('--method', '-m',
              type=click.Choice(['cv', 'llm', 'hybrid']),
              default='hybrid',
              help='분석 방법')
@click.option('--verbose', '-v', is_flag=True,
              help='상세 출력')
def status(config, method, verbose):
    """
    시스템 상태 및 통계 확인
    
    현재 시스템의 설정, 수집된 데이터 및 분석 통계를 확인합니다.
    """
    try:
        config_data = load_config(config)
        
        # 데이터 수집기 상태
        collector = AWSDataCollector(config_data.get("aws", {}))
        collection_status = collector.get_collection_status()
        
        click.echo("\n--- 📊 데이터 수집 상태 ---")
        for key, value in collection_status.items():
            if isinstance(value, float):
                click.echo(f"   {key}: {value:.2f}")
            else:
                click.echo(f"   {key}: {value}")
        
        # 오토라벨러 통계 (임시 구현)
        # 실제로는 각 라벨러에서 통계를 가져와야 함
        click.echo("\n--- 📈 오토라벨러 통계 (예정) ---")
        click.echo(f"   선택된 방법: {method}")
        click.echo("   * 이 부분은 향후 각 오토라벨러 모듈의 상세 통계로 채워질 예정입니다.")
        
    except Exception as e:
        click.echo(f"❌ 상태 확인 실패: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일을 로드합니다."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    return config_data


def create_auto_labeler(method: str, config: Dict[str, Any]):
    """선택된 방법에 따라 오토라벨러 인스턴스를 생성합니다."""
    if method == "cv":
        return AWSCVAutoLabeler(config)
    elif method == "llm":
        return AWSLLMAutoLabeler(config)
    elif method == "hybrid":
        return AWSHybridAutoLabeler(config)
    else:
        raise ValueError(f"지원하지 않는 분석 방법: {method}")


def find_image_files(directory: Path) -> List[Path]:
    """디렉터리에서 이미지 파일 목록을 찾습니다."""
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff']
    files = []
    for ext in image_extensions:
        files.extend(directory.rglob(ext))
    return files


def save_results(results: List[AnalysisResult], output_dir: Path, format: str, verbose: bool):
    """분석 결과를 파일로 저장합니다."""
    for i, result in enumerate(results):
        file_name = f"analysis_result_{i:03d}.json"
        output_path = output_dir / file_name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        if verbose:
            click.echo(f"📄 결과 저장: {output_path}")


def print_analysis_statistics(batch_results: BatchAnalysisResult, verbose: bool):
    """분석 통계를 출력합니다."""
    click.echo("\n 분석 통계:")
    click.echo(f"   이미지 수: {batch_results.total_images}")
    click.echo(f"   총 감지 수: {batch_results.total_detections}")
    click.echo(f"   평균 처리 시간: {batch_results.average_processing_time:.2f}초")
    
    # 평균 신뢰도 계산 (모든 AnalysisResult의 평균)
    if batch_results.results:
        all_confidences = [d.confidence for ar in batch_results.results for d in ar.detections]
        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            click.echo(f"   평균 신뢰도: {avg_confidence:.3f}")
        else:
            click.echo(f"   평균 신뢰도: N/A (감지된 객체 없음)")
    else:
        click.echo(f"   평균 신뢰도: N/A (분석 결과 없음)")
        
    if verbose:
        click.echo(f"   성공한 이미지: {batch_results.success_count}")
        click.echo(f"   실패한 이미지: {batch_results.error_count}")


def print_collection_statistics(stats):
    """수집 통계 출력"""
    click.echo(f"\n�� 수집 통계:")
    click.echo(f"   총 수집: {stats.total_collections}")
    click.echo(f"   성공: {stats.successful_collections}")
    click.echo(f"   실패: {stats.failed_collections}")
    click.echo(f"   성공률: {stats.success_rate:.1%}")
    click.echo(f"   총 데이터 수: {stats.total_data_count}")
    click.echo(f"   평균 처리 시간: {stats.average_processing_time:.2f}초")


if __name__ == '__main__':
    cli()
