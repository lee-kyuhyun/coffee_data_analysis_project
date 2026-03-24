#!/usr/bin/env python
# coding: utf-8

"""
[데이터 수집 및 적재 파이프라인 (Data Ingestion Pipeline)]
- 데이터 소스: CSV, Excel 등 다양한 로컬 파일 (기후, 환율, 원유, 인덱스, 커피 무역 등)
- 타겟 DB: DuckDB 로컬 파일 기반 데이터베이스
- 주요 작업: 데이터 형변환, 결측치 처리, 파생변수 생성 및 DB 테이블 일괄 적재
"""

# =====================================================================
# [1] 라이브러리 임포트 및 전역 환경 설정
# =====================================================================
import os
import re
import glob
import duckdb
import pandas as pd
import numpy as np
from collections import defaultdict

# 실무에서는 파일 경로를 하드코딩하지 않고 상단에 전역 변수로 관리하여 유지보수성을 높입니다.
DB_PATH = 'my_database.duckdb'
BASE_DIR = '/mnt/c/data/Coffee'

print("🚀 데이터베이스 연결 시작...")
con = duckdb.connect(DB_PATH)


# =====================================================================
# [2] 공통 데이터 전처리 및 적재 함수 정의
# =====================================================================

def db_save(df_name, table_name):
    """
    전처리된 DataFrame을 DuckDB에 적재하고 확인용 결과를 반환하는 공통 함수
    - if_exists='replace': 기존 테이블이 존재하면 덮어쓰기 (멱등성 보장)
    """
    df_name.to_sql(table_name, con=con, if_exists='replace', index=False)
    result_df = con.execute(f"SELECT * FROM {table_name};").fetchdf()
    return result_df


def coffee_import_preprocessing(sheet_name):
    """
    커피 수입/수출 Excel 데이터를 불러와 결측치를 처리하고 시계열 형태로 변환하는 함수
    """
    file_path = os.path.join(BASE_DIR, "coffee_import.xlsx")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df_copy = df.copy()

    # '년도' 컬럼의 공백을 NaN으로 대체 후 ffill(앞방향 채우기) 적용
    df_copy['년도'] = df_copy['년도'].replace(' ', np.nan).ffill()

    # '월' 컬럼이 '합'인 불필요한 총계 행 제거
    df_cleaned = df_copy[df_copy['월'] != '합'].copy()

    # 숫자 컬럼 내 쉼표(,) 제거 (에러 방지)
    numeric_cols = ['수출(중량)', '수출(금액)', '수입(중량)', '수입(금액)']
    for col in numeric_cols:
        df_cleaned[col] = df_cleaned[col].str.replace(',', '')

    # '년도'와 '월'을 결합하여 분석용 시계열 'Date' 컬럼(YYYY-MM) 생성
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['년도'] + '-' + df_cleaned['월'], format='%Y-%m')

    # 데이터 타입 캐스팅 (문자열 -> 실수형)
    df_cleaned = df_cleaned.astype({col: float for col in numeric_cols})

    return df_cleaned[['Date', '수출(중량)', '수출(금액)', '수입(중량)', '수입(금액)']]


def oil_prepro(file_path):
    """원유 가격 데이터 로드용 단순 래퍼 함수"""
    return pd.read_excel(file_path)


def er_prepro(df):
    """
    환율 및 인덱스 데이터 공통 전처리 함수
    - 불필요한 '거래량' 컬럼 제거 및 퍼센트(%) 문자열 실수형 변환
    """
    df_dropped = df.drop(columns=['거래량'], errors='ignore')
    if '변동 %' in df_dropped.columns:
        df_dropped.rename(columns={'변동 %': '변동'}, inplace=True)
    if '변동' in df_dropped.columns and df_dropped['변동'].dtype == 'object':
        df_dropped['변동'] = df_dropped['변동'].str.replace('%', '').astype(float)

    df_dropped['날짜'] = pd.to_datetime(df_dropped['날짜'])
    return df_dropped


def climate_prepro(df):
    """
    기후 데이터 결측치 제거 및 컬럼명 영/한 매핑 전처리 함수
    """
    df_dropped = df.dropna().copy()
    columns_names = ['Date', '일평균 기온', '일최고 기온', '일최저 기온', '일일 총강수량', '일평균 상대습도', '일일 총 일사량', '일일 총 잠재 증발산량']
    df_dropped.columns = columns_names
    df_dropped['일평균 상대습도'] = df_dropped['일평균 상대습도'].astype(float)
    return df_dropped


# =====================================================================
# [3] 메인 데이터 처리 및 적재 파이프라인 실행
# =====================================================================

print("\n▶ [Task 1] 커피 가격 (1973~) 데이터 저장")
# ---------------------------------------------------------------------
coffee_price_path = os.path.join(BASE_DIR, "coffee prices 1973~ usd per pound.csv")
df_coffee_price = pd.read_csv(coffee_price_path, sep='\t')
df_coffee_price['Date'] = pd.to_datetime(df_coffee_price['Date'])
df_coffee_price = df_coffee_price.rename(columns={'Value': 'Price'})
db_save(df_coffee_price, 'coffee_price')

print("\n▶ [Task 2] 커피 수출입 데이터 (Excel 4개 시트) 처리 및 View/Table 등록")
# ---------------------------------------------------------------------
sheet_name_list = ['Unroasted_Caf', 'Unroasted_DeCaf', 'Roasted_Caf', 'Roasted_DeCaf']
for name in sheet_name_list:
    table_name = f'coffee_price_{name}'
    df_processed = coffee_import_preprocessing(name)
    con.register(table_name, df_processed)  # 가상 뷰 등록
    con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM {table_name};")

print("\n▶ [Task 3] 국가별 커피 생산량 데이터 처리 (메모리 최적화 및 BIGINT 적용)")
# ---------------------------------------------------------------------
prod_path = os.path.join(BASE_DIR, "coffee_production.csv")
df_prod = pd.read_csv(prod_path, sep=',')
df_prod['Total_Value'] = df_prod['Value'] * 1000 * 60  # 60kg 백 단위 환산

# 데이터 용량 최적화를 위한 다운캐스팅 (Downcasting)
df_prod = df_prod.astype({
    'Country_Code': 'category', 'Country_Name': 'category',
    'Market_Year': 'int16', 'Calendar_Year': 'int16', 'Month': 'int16',
    'Attribute_Description': 'category', 'Value': 'int64', 'Total_Value': 'int64',
})

cols_to_keep = ['Country_Code', 'Country_Name', 'Market_Year', 'Calendar_Year', 'Month', 'Attribute_Description',
                'Value', 'Total_Value']
# 명시적으로 BIGINT 타입을 할당하여 DB 오버플로우 방지
df_prod[cols_to_keep].to_sql('coffee_production', con, if_exists='replace', index=False,
                             dtype={'Total_Value': 'BIGINT'})

print("\n▶ [Task 4] 원유 가격 (WTI, Brent, Dubai) 데이터 처리")
# ---------------------------------------------------------------------
oil_files = [
    ("추가 데이터/원유 데이터/WTI_oil_prices_20250925.xlsx", "WIT_oil_price"),
    ("추가 데이터/원유 데이터/Brent_oil_prices_20250925.xlsx", "Brent_oil_price"),
    ("추가 데이터/원유 데이터/Dubai_oil_prices_20250926.xlsx", "Dubai_oil_price")
]
for file_suffix, table_name in oil_files:
    file_path = os.path.join(BASE_DIR, file_suffix)
    df_oil = oil_prepro(file_path).dropna()
    db_save(df_oil, table_name)

print("\n▶ [Task 5] 국가별 환율 데이터 정규표현식 파싱 및 통합")
# ---------------------------------------------------------------------
exchange_path = os.path.join(BASE_DIR, "추가 데이터/환율 데이터/*.csv")
grouped_dfs = defaultdict(list)

for file in glob.glob(exchange_path):
    file_name = os.path.basename(file)
    currency_pair = file_name.split(' ')[0]
    match = re.search(r'\d{4}', file_name)  # 연도(4자리 숫자) 추출

    if match:
        prefix = currency_pair  # 연도를 제외한 통화명(예: USD_KRW)을 키값으로 사용
        grouped_dfs[prefix].append(pd.read_csv(file))

for prefix, df_list in grouped_dfs.items():
    merged_df = pd.concat(df_list, ignore_index=True)
    clean_df = er_prepro(merged_df)
    db_save(clean_df, prefix)

print("\n▶ [Task 6] 인덱스 데이터 (USD Index, BADI) 처리")
# ---------------------------------------------------------------------
usd_idx_path = os.path.join(BASE_DIR, "추가 데이터/인덱스 데이터/USDollarIndex_20250926.xlsx")
db_save(pd.read_excel(usd_idx_path).dropna(), 'USDdollarIndex')

badi_path = os.path.join(BASE_DIR, "추가 데이터/인덱스 데이터/*.csv")
df_badi_merged = pd.concat([pd.read_csv(f) for f in glob.glob(badi_path)], ignore_index=True)
df_badi_clean = er_prepro(df_badi_merged)

for col in ['종가', '시가', '고가', '저가']:
    df_badi_clean[col] = df_badi_clean[col].astype(str).str.replace(',', '').astype(float)
db_save(df_badi_clean, 'BADI')

print("\n▶ [Task 7] 주요 국가 기후 데이터 처리")
# ---------------------------------------------------------------------
climate_path = os.path.join(BASE_DIR, "추가 데이터/기후데이터/최종/xlsx/*.xlsx")
for file in glob.glob(climate_path):
    key_name = os.path.basename(file).split('_')[0]
    df_climate = pd.read_excel(file)
    df_climate_clean = climate_prepro(df_climate)
    db_save(df_climate_clean, f'{key_name}_climate')

print("\n▶ [Task 8] 커피 가격 (C 선물) 투자 데이터 병합 및 적재")
# ---------------------------------------------------------------------
investing_files = [
    "커피가격/미국 커피 C 선물 과거 데이터1.csv",
    "커피가격/미국 커피 C 선물 과거 데이터2.csv",
    "커피가격/미국 커피 C 선물 과거 데이터3.csv"
]
df_investing_list = [pd.read_csv(os.path.join(BASE_DIR, f)) for f in investing_files]
df_investing_merged = pd.concat(df_investing_list, ignore_index=True)

df_investing_clean = er_prepro(df_investing_merged)
df_investing_clean.sort_values(by='날짜', inplace=True)  # 시계열 오름차순 정렬 필수
db_save(df_investing_clean, 'coffee_price_investing')

print("\n✅ 모든 데이터 전처리 및 데이터베이스 적재가 성공적으로 완료되었습니다.")
con.close()