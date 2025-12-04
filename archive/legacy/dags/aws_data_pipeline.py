"""Airflow DAG to run AWS data gathering tasks in sequence."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.decorators import task

# Ensure project root on path so our local packages are importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# NOTE: 레거시 모듈 참조 - backend.core.data_collectors로 마이그레이션 필요
# 현재는 레거시 모듈이 제거되어 주석 처리됨
# from backend.core.data_collectors.aws_collector import AWSDataCollector

# 레거시 모듈 참조 (아직 구현되지 않음)
# TODO: backend.core.data_collectors를 사용하도록 마이그레이션
# from aws_icons_parser.aws_icons_zip_to_mapping import generate_mapping
# from aws_products_scraper.fetch_products import fetch_products
# from aws_service_boto3.export_service_codes import export_service_codes
# from aws_service_boto3.infer_from_models import infer_from_models


default_args = {"retries": 1}

with DAG(
    dag_id="aws_data_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["aws", "batch"],
) as dag:

    @task()
    def parse_icons():
        # TODO: backend.core.data_collectors를 사용하도록 마이그레이션
        # collector = AWSDataCollector(config)
        # collector.collect_icons()
        pass

    @task()
    def scrape_products():
        # TODO: backend.core.data_collectors를 사용하도록 마이그레이션
        # collector = AWSDataCollector(config)
        # collector.collect_products()
        pass

    @task()
    def export_codes():
        # TODO: backend.core.data_collectors를 사용하도록 마이그레이션
        # collector = AWSDataCollector(config)
        # collector.collect_services()
        pass

    @task()
    def infer_resources():
        # TODO: backend.core.data_collectors를 사용하도록 마이그레이션
        pass

    # TODO: 마이그레이션 후 활성화
    # parse_icons() >> scrape_products() >> export_codes() >> infer_resources()
