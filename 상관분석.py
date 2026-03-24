#!/usr/bin/env python
# coding: utf-8

# =====================================================================
# [1] 라이브러리 임포트 및 전역 환경 설정
# =====================================================================
import os
import platform
import warnings
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 경고 무시 및 멀티스레드 충돌 방지 설정
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"

# 한글 글꼴 및 마이너스 부호 깨짐 방지 전역 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
    
plt.rc('axes', unicode_minus=False)
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# [2] 데이터베이스 연결 및 다중 데이터 추출 (Data Extraction)
# =====================================================================
print("🚀 [Step 1] 데이터베이스 연결 및 거시 지표 데이터 추출 시작...")
con = duckdb.connect('my_database.duckdb')

# 1. 커피 가격 및 운임 지수
coffee_price_investing = con.execute("SELECT 날짜, 종가, 변동 FROM coffee_price_investing;").df()
BADI = con.execute("SELECT 날짜, 종가, 변동 FROM BADI;").df()

# 2. 유가 데이터
WIT_oil_price = con.execute("SELECT * FROM WIT_oil_price;").df()
Brent_oil_price = con.execute("SELECT * FROM Brent_oil_price;").df()
Dubai_oil_price = con.execute("SELECT * FROM Dubai_oil_price;").df()

# 3. 주요 커피 생산국 기후 데이터
Brazil_climate = con.execute("SELECT * FROM Brazil_climate;").df()
Colombia_climate = con.execute("SELECT * FROM Colombia_climate;").df()
Ethiopia_climate = con.execute("SELECT * FROM Ethiopia_climate;").df()
Indonesia_climate = con.execute("SELECT * FROM Indonesia_climate;").df()
Vietnam_climate = con.execute("SELECT * FROM Vietnam_climate;").df()

# 4. 환율 및 인덱스 데이터
USDdollarIndex = con.execute("SELECT * FROM USDdollarIndex;").df()
EUR_USD = con.execute("SELECT 날짜, 종가, 변동 FROM EUR_USD;").df()
USD_BRL = con.execute("SELECT 날짜, 종가, 변동 FROM USD_BRL;").df()
USD_COP = con.execute("SELECT 날짜, 종가, 변동 FROM USD_COP;").df()
USD_ETB = con.execute("SELECT 날짜, 종가, 변동 FROM USD_ETB;").df()
USD_IDR = con.execute("SELECT 날짜, 종가, 변동 FROM USD_IDR;").df()
USD_VND = con.execute("SELECT 날짜, 종가, 변동 FROM USD_VND;").df()

con.close()
print("✅ 데이터베이스 추출 및 연결 종료 완료.")


# =====================================================================
# [3] 데이터 전처리 자동화 및 병합 (Preprocessing & Merging)
# =====================================================================
print("\n🛠️ [Step 2] 데이터 전처리 및 인덱스 병합 진행 중...")

# 반복문 처리를 위해 추출한 데이터프레임들을 딕셔너리로 매핑합니다.
data_frames_map = {
    'coffee': coffee_price_investing,
    'BADI': BADI,
    'WTI_oil': WIT_oil_price,
    'Brent_oil': Brent_oil_price,
    'Dubai_oil': Dubai_oil_price,
    'Brazil_climate': Brazil_climate,
    'Colombia_climate': Colombia_climate,
    'Ethiopia_climate': Ethiopia_climate,
    'Indonesia_climate': Indonesia_climate,
    'Vietnam_climate': Vietnam_climate,
    'USD_Index': USDdollarIndex,
    'EUR_USD': EUR_USD,
    'USD_BRL': USD_BRL,
    'USD_COP': USD_COP,
    'USD_ETB': USD_ETB,
    'USD_IDR': USD_IDR,
    'USD_VND': USD_VND
}

processed_dfs = []

# 각 데이터프레임에 대해 공통 전처리를 수행합니다.
for name, df in data_frames_map.items():
    df_copy = df.copy()
    
    # 1. 날짜 컬럼명 통일
    if 'Date' in df_copy.columns:
        df_copy.rename(columns={'Date': '날짜'}, inplace=True)
    
    if '날짜' not in df_copy.columns:
        print(f"⚠️ '{name}' DataFrame에 '날짜' 컬럼이 없어 제외합니다.")
        continue
    
    # 2. 문자열 숫자(콤마 포함) 실수형 변환
    cols_to_clean = ['종가', '시가', '고가', '저가']
    for col in cols_to_clean:
        if col in df_copy.columns and df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str).str.replace(',', '')
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
    # 3. 시계열 병합을 위한 인덱스 설정
    try:
        df_copy.set_index('날짜', inplace=True)
    except KeyError:
        print(f"⚠️ '{name}' DataFrame에서 '날짜' 인덱스 설정 실패.")
        continue
    
    # 4. 변수 구분을 위해 기존 컬럼명 앞에 데이터셋 이름(Prefix) 추가
    df_copy.columns = [f"{name}_{col}" for col in df_copy.columns]
    
    processed_dfs.append(df_copy)

print("✅ 개별 데이터 전처리 완료 (날짜 인덱스 설정 및 컬럼명 변경).")

# 5. 시계열 빈도(Frequency) 조정 및 최종 병합
resampled_dfs = []
for df in processed_dfs:
    # 두바이유 데이터의 경우 결측을 방지하기 위해 일간(D) 리샘플링 후 이전 값(ffill)으로 채웁니다.
    if 'Dubai_oil_Price_USD' in df.columns:
        df = df.resample('D').ffill()
        print("💡 Dubai_oil_price 일간 리샘플링 완료 (ffill).")
    resampled_dfs.append(df)

# axis=1(열 기준), join='inner'(공통 날짜만 교집합)으로 모든 데이터를 하나로 병합합니다.
merged_df = pd.concat(resampled_dfs, axis=1, join='inner')

# 하나라도 결측치(NaN)가 있는 행은 분석의 정확도를 위해 제거합니다.
merged_df.dropna(inplace=True)

print("\n--- 최종 병합 데이터 정보 ---")
print(f"병합된 데이터 기간: {merged_df.index.min()} ~ {merged_df.index.max()}")
print(f"병합된 데이터 크기: {merged_df.shape}")
# print(merged_df.info()) # 필요 시 주석 해제하여 확인


# =====================================================================
# [4] 피어슨 상관관계 분석 (Correlation Analysis)
# =====================================================================
print("\n📊 [Step 3] 커피 종가와 타 변수 간의 상관관계 분석...")

if merged_df.empty:
    print("🚨 경고: 병합된 데이터프레임이 비어있습니다. 공통된 날짜가 없습니다.")
    print("해결 방안: 'join' 방식을 'inner'에서 'outer'로 변경해 보거나 원본 데이터 기간을 확인하세요.")
else:
    # 전체 변수 간의 피어슨 상관계수 행렬 계산
    corr_matrix = merged_df.corr()
    
    # 타겟 변수 설정 (커피 종가)
    target_variable = 'coffee_종가'
    
    if target_variable not in corr_matrix:
        print(f"🚨 오류: '{target_variable}'이 병합된 데이터에 없습니다. 컬럼명을 확인하세요.")
    else:
        # 타겟 변수와의 상관계수를 내림차순으로 정렬
        coffee_correlations = corr_matrix[target_variable].sort_values(ascending=False)
        
        print(f"\n📈 --- {target_variable}와 상관관계 높은 변수 (Top 20 / 양의 상관관계) ---")
        print(coffee_correlations.head(20))
        
        print(f"\n📉 --- {target_variable}와 상관관계 낮은 변수 (Bottom 20 / 음의 상관관계) ---")
        print(coffee_correlations.tail(20))


# =====================================================================
# [5] 분석 결과 시각화 (Visualizations)
# =====================================================================
print("\n🎨 [Step 4] 상관관계 시각화 출력...")

if not merged_df.empty:
    # 시각화 1: 전체 변수 간 상관관계 히트맵 (Heatmap)
    plt.figure(figsize=(25, 20))
    sns.heatmap(
        corr_matrix,
        annot=False,          # 변수가 너무 많으므로 숫자는 표시하지 않음
        cmap='coolwarm',      # 양수는 빨강, 음수는 파랑으로 표시
        vmin=-1,
        vmax=1
    )
    plt.title('전체 변수 간 상관관계 히트맵', fontsize=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap_all_variables.png') # 이미지 저장 코드 추가
    plt.show()

# 타겟 변수 추출에 성공했을 경우 막대그래프 출력
if not merged_df.empty and 'coffee_correlations' in locals():
    # 타겟 변수 자기 자신(상관계수 1.0)은 그래프에서 제외하고 오름차순 정렬
    coffee_corr_filtered = coffee_correlations.drop(target_variable).sort_values()
    
    # 시각화 2: 타겟 변수와 다른 변수들의 상관관계 (Horizontal Bar Chart)
    plt.figure(figsize=(12, 18))
    
    # 양수는 빨간색, 음수는 파란색으로 막대 색상 조건부 설정
    coffee_corr_filtered.plot(
        kind='barh',
        color=coffee_corr_filtered.apply(lambda x: 'red' if x > 0 else 'blue')
    )
    
    plt.title(f'{target_variable}과 다른 변수들의 상관관계', fontsize=16)
    plt.xlabel('피어슨 상관계수 (Pearson Correlation Coefficient)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('correlation_bar_chart_coffee_price.png') # 이미지 저장 코드 추가
    plt.show()