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

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt
from prophet import Prophet

# 경고 무시 및 멀티스레드 충돌 방지 설정
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"

# 한글 글꼴 및 시각화 전역 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
    
plt.rc('axes', unicode_minus=False)
plt.rcParams['axes.unicode_minus'] = False

# pandas 출력 설정
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)

# =====================================================================
# [2] 데이터베이스 연결 및 기초 데이터 프레임 구축
# =====================================================================
print("🚀 [Step 1] 데이터 로드 및 초기 병합 시작...")
con = duckdb.connect('my_database.duckdb')

df_rc = con.execute("SELECT * FROM coffee_price_Roasted_Caf;").df()
df_rd = con.execute("SELECT * FROM coffee_price_Roasted_DeCaf;").df()
df_uc = con.execute("SELECT * FROM coffee_price_Unroasted_Caf;").df()
df_ud = con.execute("SELECT * FROM coffee_price_Unroasted_DeCaf;").df()
con.close()

# 원본 보존 및 카테고리 열 추가
df_rc['구분_로스팅'], df_rc['구분_카페인'] = 'Roasted', 'Caf'
df_rd['구분_로스팅'], df_rd['구분_카페인'] = 'Roasted', 'DeCaf'
df_uc['구분_로스팅'], df_uc['구분_카페인'] = 'Unroasted', 'Caf'
df_ud['구분_로스팅'], df_ud['구분_카페인'] = 'Unroasted', 'DeCaf'

# 4개 데이터 통합
df_all = pd.concat([df_rc, df_rd, df_uc, df_ud], ignore_index=True)

# 기초 단가 계산
df_all['수입(중량)_corr'] = df_all['수입(중량)'].replace(0, np.nan)
df_all['수출(중량)_corr'] = df_all['수출(중량)'].replace(0, np.nan)
df_all['수입단가'] = df_all['수입(금액)'] / df_all['수입(중량)_corr']
df_all['수출단가'] = df_all['수출(금액)'] / df_all['수출(중량)_corr']

df_all['월'] = df_all['Date'].dt.month
df_all['연도'] = df_all['Date'].dt.year

# 환율 데이터 로드 및 전처리
exchange_path = "/mnt/c/data/Coffee/추가 데이터/exchange_rate.csv"
try:
    df_exchange = pd.read_csv(exchange_path)
    df_exchange_processed = df_exchange.copy()
    df_exchange_processed.rename(columns={'변환': 'Date', '원자료': 'ExchangeRate'}, inplace=True)
    df_exchange_processed['Date'] = pd.to_datetime(df_exchange_processed['Date'], format='%Y/%m')
    df_exchange_processed['ExchangeRate'] = df_exchange_processed['ExchangeRate'].str.replace(',', '').astype(float)
except FileNotFoundError:
    print(f"경고: {exchange_path} 파일을 찾을 수 없습니다.")

# =====================================================================
# [3] 데이터 집계 (월별, 연도별) 및 파생변수 생성
# =====================================================================
print("📊 [Step 2] 월별/연도별 시계열 집계...")

# --- 1. 월별 총계 (df_total) ---
df_total = df_all.groupby('Date').sum(numeric_only=True)
df_total['총_수입단가'] = df_total['수입(금액)'] / df_total['수입(중량)'].replace(0, np.nan)
df_total['총_수출단가'] = df_total['수출(금액)'] / df_total['수출(중량)'].replace(0, np.nan)
df_total['연도'] = df_total.index.year
df_total['월'] = df_total.index.month

# 12개월 이동평균 생성
df_total['수입중량_12MA'] = df_total['수입(중량)'].rolling(window=12).mean()
df_total['수입단가_12MA'] = df_total['총_수입단가'].rolling(window=12).mean()

# 변동률 및 변동성 생성
df_total['중량_변동률'] = df_total['수입(중량)'].pct_change() * 100
df_total['금액_변동률'] = df_total['수입(금액)'].pct_change() * 100
df_total['중량_변동성'] = df_total['중량_변동률'].rolling(window=12).std()
df_total['금액_변동성'] = df_total['금액_변동률'].rolling(window=12).std()

# 환율 병합 및 원화단가 등 계산
df_total_merged = pd.merge(df_total.reset_index(), df_exchange_processed, on='Date', how='left')
df_total_merged['총_원화단가'] = df_total_merged['총_수입단가'] * df_total_merged['ExchangeRate']
df_total_merged['원화단가_변동률'] = df_total_merged['총_원화단가'].pct_change() * 100
df_total_merged['변동성_지표_12M'] = df_total_merged['원화단가_변동률'].rolling(window=12).std()

# 볼린저 밴드 지표
df_total_merged['단가_12M_MA'] = df_total_merged['총_원화단가'].rolling(window=12).mean()
df_total_merged['단가_12M_STD'] = df_total_merged['총_원화단가'].rolling(window=12).std()
df_total_merged['볼린저_상단'] = df_total_merged['단가_12M_MA'] + (df_total_merged['단가_12M_STD'] * 2)
df_total_merged['볼린저_하단'] = df_total_merged['단가_12M_MA'] - (df_total_merged['단가_12M_STD'] * 2)

# --- 2. 연도별 총계 (df_yearly) ---
df_yearly = df_total.groupby('연도')[['수입(중량)', '수출(중량)', '수입(금액)', '수출(금액)']].sum()
df_yearly['국내소비_추정치'] = df_yearly['수입(중량)'] - df_yearly['수출(중량)']
df_yearly['재수출_비중(%)'] = (df_yearly['수출(중량)'] / df_yearly['수입(중량)']) * 100
df_yearly['재수출_비중_금액(%)'] = (df_yearly['수출(금액)'] / df_yearly['수입(금액)']) * 100
df_yearly['무역수지(USD)'] = df_yearly['수출(금액)'] - df_yearly['수입(금액)']

# =====================================================================
# [4] 기본 트렌드 및 기초 시각화 파트
# =====================================================================
print("📈 [Step 3] 기초 트렌드 시각화 생성 중...")

# 1. 대한민국 커피 총 수입 중량 및 단가 추세
plt.figure(figsize=(12, 6))
plt.plot(df_total.index, df_total['수입(중량)'], label='총 수입 중량 (kg)')
plt.title('대한민국 커피 총 수입 중량 추세 (1999-2025)'); plt.xlabel('연도'); plt.ylabel('수입 중량 (kg)'); plt.legend(); plt.grid(True)
plt.savefig('대한민국 커피 총 수입 중량 추세.png'); plt.close()

plt.figure(figsize=(12, 6))
plt.plot(df_total.index, df_total['총_수입단가'], label='평균 수입 단가 ($/kg)', color='orange')
plt.title('대한민국 커피 평균 수입 단가 추세 (1999-2025)'); plt.xlabel('연도'); plt.ylabel('평균 단가 ($/kg)'); plt.legend(); plt.grid(True)
plt.savefig('대한민국 커피 총 수입 단가 추세.png'); plt.close()

# 2. 카테고리별 수입 중량/단가 추세
plt.figure(figsize=(14, 7))
sns.lineplot(data=df_all, x='Date', y='수입(중량)', hue='구분_로스팅', style='구분_카페인')
plt.title('카테고리별 커피 수입 중량 추세'); plt.legend(title='구분'); plt.grid(True)
plt.savefig('카테고리별 커피 수입 중량 추세.png'); plt.close()

plt.figure(figsize=(14, 7))
sns.lineplot(data=df_all, x='Date', y='수입단가', hue='구분_로스팅', style='구분_카페인')
plt.title('카테고리별 커피 수입 단가 추세'); plt.legend(title='구분'); plt.grid(True, which='both')
plt.savefig('카테고리별 커피 수입 단가 추세.png'); plt.close()

# 3. 12개월 이동 평균 추세
plt.figure(figsize=(12, 6))
plt.plot(df_total.index, df_total['수입(중량)'], label='월별 수입 중량', alpha=0.5)
plt.plot(df_total.index, df_total['수입중량_12MA'], label='12개월 이동 평균', color='red', linewidth=2)
plt.title('커피 수입 중량 및 12개월 이동 평균 추세'); plt.legend(); plt.grid(True)
plt.savefig('커피 수입 중량 및 12개월 이동 평균 추세.png'); plt.close()

# 4. 월별 평균 수입 중량 (계절성 확인)
monthly_avg = df_all.groupby('월')['수입(중량)'].mean()
plt.figure(figsize=(10, 5))
monthly_avg.plot(kind='bar')
plt.title('월별 평균 커피 수입 중량 (계절성 확인)'); plt.xticks(rotation=0); plt.grid(axis='y')
plt.savefig('월별 평균 커피 수입 중량.png'); plt.close()

# 5. 연도별/월별 합산 비교 (중량 & 금액)
df_yearly_sum = df_total.groupby('연도')['수입(중량)'].sum()
plt.figure(figsize=(12, 6))
df_yearly_sum.plot(kind='bar', color='royalblue')
plt.title('연도별 총 커피 수입 중량 추세'); plt.xticks(rotation=45); plt.grid(axis='y'); plt.tight_layout()
plt.savefig('total_import_weight_by_year.png'); plt.close()

df_monthly_sum = df_all.groupby('월')['수입(중량)'].sum()
plt.figure(figsize=(10, 5))
df_monthly_sum.plot(kind='bar')
plt.title('월별 총 커피 수입 중량 (모든 연도 합산)'); plt.xticks(rotation=0); plt.grid(axis='y'); plt.tight_layout()
plt.savefig('total_import_weight_by_month.png'); plt.close()

df_yearly_amount_sum = df_total.groupby('연도')['수입(금액)'].sum()
plt.figure(figsize=(12, 6))
df_yearly_amount_sum.plot(kind='bar', color='forestgreen')
plt.title('연도별 총 커피 수입 금액(USD) 추세'); plt.xticks(rotation=45); plt.grid(axis='y'); plt.tight_layout()
plt.savefig('total_import_amount_by_year.png'); plt.close()

df_monthly_amount_sum = df_all.groupby('월')['수입(금액)'].sum()
plt.figure(figsize=(10, 5))
df_monthly_amount_sum.plot(kind='bar', color='goldenrod')
plt.title('월별 총 커피 수입 금액(USD) (모든 연도 합산)'); plt.xticks(rotation=0); plt.grid(axis='y'); plt.tight_layout()
plt.savefig('total_import_amount_by_month.png'); plt.close()

# =====================================================================
# [5] 거시 지표 및 무역 수지 시각화
# =====================================================================
print("💸 [Step 4] 무역수지 및 원가 변동성 시각화 중...")

# 1. 수입 vs 소비 추세
plt.figure(figsize=(12, 6))
plt.plot(df_yearly.index, df_yearly['수입(중량)'], label='총 수입량 (A)', color='blue', marker='o')
plt.plot(df_yearly.index, df_yearly['국내소비_추정치'], label='국내소비 추정치 (순수입량, A-B)', color='orange', marker='x', linestyle='--')
plt.title('연도별 커피 총 수입량 vs 국내소비 추정치'); plt.legend(); plt.grid(True)
plt.savefig('import_vs_consumption_trend.png'); plt.close()

# 2. 재수출 비중 추세
plt.figure(figsize=(12, 6))
plt.plot(df_yearly.index, df_yearly['재수출_비중(%)'], label='재수출 비중 (%)', color='green', marker='s')
plt.title('커피 총 수입량 대비 재수출 비중 추세'); plt.ylim(bottom=0); plt.legend(); plt.grid(True)
plt.savefig('re_export_ratio_trend.png'); plt.close()

# 3. 수입 vs 수출 중량/금액 비교
plt.figure(figsize=(12, 6))
plt.plot(df_yearly.index, df_yearly['수입(중량)'], label='총 수입 중량', color='blue', linewidth=3)
plt.plot(df_yearly.index, df_yearly['수출(중량)'], label='총 수출 중량', color='red', linestyle='--')
plt.title('연도별 커피 수입 vs 수출 [중량] 비교'); plt.legend(); plt.grid(True)
plt.savefig('import_vs_export_weight.png'); plt.close()

plt.figure(figsize=(12, 6))
plt.plot(df_yearly.index, df_yearly['수입(금액)'], label='총 수입 금액 (USD)', color='blue', linewidth=3)
plt.plot(df_yearly.index, df_yearly['수출(금액)'], label='총 수출 금액 (USD)', color='red', linestyle='--')
plt.title('연도별 커피 수입 vs 수출 [금액] 비교'); plt.legend(); plt.grid(True)
plt.savefig('import_vs_export_amount.png'); plt.close()

# 4. 금액 기준 재수출 비중
plt.figure(figsize=(12, 6))
plt.plot(df_yearly.index, df_yearly['재수출_비중_금액(%)'], label='재수출 비중 (금액 기준 %)', color='darkorange', marker='o')
plt.title('커피 총 수입액 대비 재수출액 비중 추세'); plt.ylim(bottom=0); plt.grid(True); plt.legend()
plt.savefig('re_export_ratio_amount.png'); plt.close()

# 5. 무역수지 추세
plt.figure(figsize=(12, 6))
colors = ['red' if x < 0 else 'blue' for x in df_yearly['무역수지(USD)']]
df_yearly['무역수지(USD)'].plot(kind='bar', color=colors)
plt.title('연도별 커피 무역수지 추세 (USD)'); plt.grid(axis='y', linestyle='--'); plt.tight_layout()
plt.savefig('trade_balance_trend.png'); plt.close()

# =====================================================================
# [6] 환율/단가 시계열 분석 및 볼린저 밴드
# =====================================================================
# 1. 원화단가 추세
plt.figure(figsize=(12, 6))
plt.plot(df_total_merged['Date'], df_total_merged['총_원화단가'], label='평균 수입 단가 (KRW/kg)', color='crimson')
plt.title('대한민국 커피 평균 수입 단가 추세 (원화 기준)'); plt.legend(); plt.grid(True)
plt.savefig('total_import_price_krw_trend.png'); plt.close()

# 2. USD vs KRW 추세 비교
fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(df_total_merged['Date'], df_total_merged['총_수입단가'], color='tab:blue', label='USD 단가 ($/kg)')
ax1.set_ylabel('평균 단가 ($/kg)', color='tab:blue'); ax1.tick_params(axis='y', labelcolor='tab:blue'); ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(df_total_merged['Date'], df_total_merged['총_원화단가'], color='tab:red', linestyle='--', label='KRW 단가 (원/kg)')
ax2.set_ylabel('평균 단가 (KRW/kg)', color='tab:red'); ax2.tick_params(axis='y', labelcolor='tab:red'); ax2.legend(loc='upper right')
plt.title('커피 수입 단가 추세 비교 (USD vs KRW)'); fig.tight_layout(); plt.grid(True)
plt.savefig('커피 수입 단가 추세 비교 (USD vs KRW).png'); plt.close()

# 3. 원화단가 변동성 추세
plt.figure(figsize=(12, 6))
plt.plot(df_total_merged['Date'], df_total_merged['변동성_지표_12M'], label='원화단가 변동성 (12개월 이동표준편차)', color='purple')
plt.title('커피 수입 원화단가 변동성 추세'); plt.legend(); plt.grid(True)
plt.savefig('커피 수입 원화단가 변동성 추세.png'); plt.close()

# 4. 볼린저 밴드 시각화
plt.figure(figsize=(14, 8))
plt.plot(df_total_merged['Date'], df_total_merged['총_원화단가'], label='총 원화단가 (KRW/kg)', color='blue', alpha=0.7)
plt.plot(df_total_merged['Date'], df_total_merged['단가_12M_MA'], label='중심선 (12M MA)', color='red', linestyle='--')
plt.plot(df_total_merged['Date'], df_total_merged['볼린저_상단'], label='상단 밴드', color='gray', linestyle=':')
plt.plot(df_total_merged['Date'], df_total_merged['볼린저_하단'], label='하단 밴드', color='gray', linestyle=':')
plt.fill_between(df_total_merged['Date'], df_total_merged['볼린저_하단'], df_total_merged['볼린저_상단'], color='gray', alpha=0.1)
plt.title('커피 수입 원화단가와 볼린저 밴드(12개월)'); plt.legend(loc='upper left'); plt.grid(True)
plt.savefig('커피 수입 원화단가와 볼린저 밴드(12개월).png'); plt.close()

# =====================================================================
# [7] 최근 5년 특화 분석 (이동평균, HP필터)
# =====================================================================
print("🔬 [Step 5] 최근 5년 집중 분석 및 이벤트 분석 중...")
start_date = '2020-10-01'
df_5y = df_total_merged[df_total_merged['Date'] >= start_date].copy()
df_5y.set_index('Date', inplace=True)
price_krw = df_5y['총_원화단가']

# 1. 3M, 12M 이평선
ma_3m, ma_12m = price_krw.rolling(window=3).mean(), price_krw.rolling(window=12).mean()
plt.figure(figsize=(12, 6))
plt.plot(price_krw, label='월별 원화단가', color='blue', alpha=0.5)
plt.plot(ma_3m, label='3개월 이동평균 (단기)', color='orange', linewidth=2, linestyle='--')
plt.plot(ma_12m, label='12개월 이동평균 (장기)', color='red', linewidth=2)
plt.title('최근 5년 커피 원화단가 이동평균 추세'); plt.legend(); plt.grid(True)
plt.savefig('price_trend_5y_ma.png'); plt.close()

# 2. 최근 5년 변동성
pct_change_5y = price_krw.pct_change() * 100
volatility_12m_5y = pct_change_5y.rolling(window=12).std()
plt.figure(figsize=(12, 6))
plt.plot(volatility_12m_5y, label='12개월 변동성 지수 (%)', color='purple')
plt.title('최근 5년 커피 원화단가 변동성 추세'); plt.legend(); plt.grid(True)
plt.savefig('price_volatility_5y.png'); plt.close()

# 3. YoY 변동률
yoy_change = price_krw.pct_change(periods=12) * 100
plt.figure(figsize=(12, 6))
plt.plot(yoy_change, label='YoY 변동률 (%)', color='green', marker='o')
plt.axhline(0, color='black', linestyle='--')
plt.title('최근 5년 커피 원화단가 전년 동월 대비(YoY) 변동률 (%)'); plt.legend(); plt.grid(True)
plt.savefig('price_yoy_change_5y.png'); plt.close()

# 4. HP 필터
price_krw_no_nan = price_krw.dropna()
krw_cycle, krw_trend = sm.tsa.filters.hpfilter(price_krw_no_nan, lamb=129600)
plt.figure(figsize=(12, 6))
plt.plot(krw_cycle, label='가격 순환변동 (Cycle)', color='darkcyan')
plt.axhline(0, color='black', linestyle='--')
plt.title('최근 5년 커피 원화단가 순환 변동 (HP필터)'); plt.legend(); plt.grid(True)
plt.savefig('price_cycle_hpfilter_5y.png'); plt.close()

# =====================================================================
# [8] 글로벌 이벤트 매핑 및 Ruptures 변화점 탐지
# =====================================================================
price_krw_full = df_total_merged.set_index('Date')['총_원화단가']

# 1. 이벤트 v2 (위기 초점)
events_v2 = {
    '1999-06-01': '커피 위기\n(공급 과잉)', '2008-09-15': '글로벌 금융위기\n(리먼)',
    '2011-01-01': '1차 원자재 붐\n(커피 녹병)', '2020-03-11': 'COVID-19\n팬데믹',
    '2021-07-01': '브라질 서리\n(공급 충격)', '2022-02-24': '러-우 전쟁\n(인플레)',
}
plt.figure(figsize=(15, 7))
plt.plot(price_krw_full.index, price_krw_full, label='월별 원화단가 (KRW/kg)', color='blue')
for date_str, label in events_v2.items():
    event_date = pd.to_datetime(date_str)
    if event_date >= price_krw_full.index.min() and event_date <= price_krw_full.index.max():
        plt.axvline(x=event_date, color='red', linestyle='--', linewidth=1.5, label=f'이벤트: {label}')
        plt.text(event_date, price_krw_full.min(), f'{label}', color='red', rotation=0, horizontalalignment='center', verticalalignment='bottom', fontsize=9)
plt.title('전체 기간 커피 가격과 주요 위기 이벤트 (1999-2025)')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
plt.grid(True); plt.savefig('price_with_events_full_v2.png'); plt.close()

# 2. 이벤트 v3 (통화정책 포함)
events_v3 = events_v2.copy()
events_v3['2022-03-17'] = '美 연준 금리인상\n(환율 충격 시작)'
plt.figure(figsize=(15, 7))
plt.plot(price_krw_full.index, price_krw_full, label='월별 원화단가 (KRW/kg)', color='blue')
for date_str, label in events_v3.items():
    event_date = pd.to_datetime(date_str)
    if event_date >= price_krw_full.index.min() and event_date <= price_krw_full.index.max():
        plt.axvline(x=event_date, color='red', linestyle='--', linewidth=1.5, label=f'이벤트: {label}')
        plt.text(event_date, price_krw_full.min(), f'{label}', color='red', rotation=0, horizontalalignment='center', verticalalignment='bottom', fontsize=9)
plt.title('전체 기간 커피 가격과 주요 위기 이벤트 (1999-2025)')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
plt.grid(True); plt.savefig('price_with_events_full_v3.png'); plt.close()

# 3. Ruptures (Dynp) 구조적 변화점 탐지
price_krw_full_no_nan = price_krw_full.dropna()
algo = rpt.Dynp(model="rbf").fit(price_krw_full_no_nan.values)
result = algo.predict(n_bkps=7)

plt.figure(figsize=(15, 7))
plt.plot(price_krw_full_no_nan.index, price_krw_full_no_nan, label='월별 원화단가 (KRW/kg)')
for i in result:
    if i < len(price_krw_full_no_nan):
        changepoint_date = price_krw_full_no_nan.index[i]
        plt.axvline(x=changepoint_date, color='purple', linestyle='--', linewidth=2, label=f'통계적 변화점 ({changepoint_date.strftime("%Y-%m")})')

plt.title('전체 기간 커피 원화단가 통계적 구조 변화점 탐지 (Ruptures/Dynp)')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc='upper left')
plt.grid(True); plt.savefig('price_changepoint_detection_full_dynp.png'); plt.close()

# =====================================================================
# [9] M/S, CAGR, 시계열 분해 및 Prophet 머신러닝 예측
# =====================================================================
print("🔮 [Step 6] CAGR 계산 및 Prophet 수요 예측 모델 구동 중...")

# 1. CAGR 분석 (2000~2024년)
df_yearly_cagr = df_total.groupby('연도')[['수입(중량)', '수출(중량)', '수입(금액)']].sum()
df_yearly_cagr['국내소비_추정치'] = df_yearly_cagr['수입(중량)'] - df_yearly_cagr['수출(중량)']
start_year, end_year = 2000, 2024
num_years = end_year - start_year
start_consum = df_yearly_cagr.loc[start_year, '국내소비_추정치']
end_consum = df_yearly_cagr.loc[end_year, '국내소비_추정치']
cagr_consum = (end_consum / start_consum) ** (1 / num_years) - 1

start_amount = df_yearly_cagr.loc[start_year, '수입(금액)']
end_amount = df_yearly_cagr.loc[end_year, '수입(금액)']
cagr_amount = (end_amount / start_amount) ** (1 / num_years) - 1

print(f"   [CAGR 결과] 실질 성장(중량): {cagr_consum:.2%} / 명목 성장(금액): {cagr_amount:.2%}")

# 2. 리스크 변동성 비교
plt.figure(figsize=(14, 7))
plt.plot(df_total.index, df_total['중량_변동성'], label='공급망 리스크 (중량 변동성)', color='blue', linewidth=2)
plt.plot(df_total.index, df_total['금액_변동성'], label='가격 리스크 (금액 변동성)', color='red', linewidth=2, linestyle='--')
plt.title('커피 수입 리스크 분석 (공급망 vs 가격)'); plt.legend(); plt.grid(True)
plt.savefig('import_risk_volatility_compare.png'); plt.close()

# 3. 시계열 분해 (계절성, 추세 파악)
ts_data = df_total['수입(중량)'].fillna(method='ffill')
decomposition = seasonal_decompose(ts_data, model='additive', period=12)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
decomposition.observed.plot(ax=ax1, grid=True, title='Observed (원본)')
decomposition.trend.plot(ax=ax2, grid=True, title='Trend (장기 추세)')
decomposition.seasonal.plot(ax=ax3, grid=True, title='Seasonal (계절성 패턴)')
decomposition.resid.plot(ax=ax4, grid=True, title='Residual (잔차/노이즈)')
plt.tight_layout(); plt.savefig('time_series_decomposition.png'); plt.close()

plt.figure(figsize=(10, 5))
decomposition.seasonal.tail(24).plot()
plt.title('월별 소비 패턴 (계절성)'); plt.grid(True); plt.savefig('monthly_pattern_seasonal.png'); plt.close()

# 4. Prophet 모델 구동
df_prophet = df_total.reset_index().rename(columns={'Date': 'ds', '수입(중량)': 'y'})
df_prophet = df_prophet[df_prophet['ds'] < '2025-01-01']

model = Prophet(seasonality_mode='multiplicative')
model.fit(df_prophet)
future = model.make_future_dataframe(periods=12, freq='MS')
forecast = model.predict(future)

fig1 = model.plot(forecast)
ax = fig1.gca()
last_history_date = df_prophet['ds'].max()
ax.axvline(x=last_history_date, color='red', linestyle='--', linewidth=2, label='예측 시작점')
plt.title('커피 수입 중량(수요) 예측 (1년)')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.savefig('demand_forecast_prophet_with_line.png'); plt.close()

fig2 = model.plot_components(forecast)
plt.savefig('demand_forecast_components_v2.png'); plt.close()

# =====================================================================
# [10] 세부 비중 분석 및 볼린저 밴드, 원가/리스크 분석
# =====================================================================
print("⚖️ [Step 7] 세부 비중 및 상관관계/계절성 분석 마무리 중...")

# 1. 로스팅/언로스팅 비중 시각화
df_roasted = df_all.groupby(['연도', '구분_로스팅'], as_index=False)[['수입(중량)', '수입(금액)']].sum()
df_yearly_sum_r = df_roasted.groupby('연도')[['수입(중량)', '수입(금액)']].sum().rename(columns={'수입(중량)': '총합(중량)', '수입(금액)': '총합(금액)'})
df_roasted_merged = pd.merge(df_roasted, df_yearly_sum_r, on='연도')
df_roasted_merged['비중(중량)'] = 100 * df_roasted_merged['수입(중량)'] / df_roasted_merged['총합(중량)']
df_roasted_merged['비중(금액)'] = 100 * df_roasted_merged['수입(금액)'] / df_roasted_merged['총합(금액)']

df_roasted_pivot_w = df_roasted_merged.pivot(index='연도', columns='구분_로스팅', values='비중(중량)')
plt.figure(figsize=(12, 6))
df_roasted_pivot_w.plot(kind='area', stacked=True, figsize=(12, 6))
plt.title('커피 수입 비중 변화 (물량/중량 기준)'); plt.legend(title='구분 (로스팅)', loc='upper left'); plt.grid(True)
plt.savefig('import_share_by_roasting_weight.png'); plt.close()

df_roasted_pivot_a = df_roasted_merged.pivot(index='연도', columns='구분_로스팅', values='비중(금액)')
plt.figure(figsize=(12, 6))
df_roasted_pivot_a.plot(kind='area', stacked=True, figsize=(12, 6), colormap='viridis')
plt.title('커피 수입 비중 변화 (비용/금액 기준)'); plt.legend(title='구분 (로스팅)', loc='upper left'); plt.grid(True)
plt.savefig('import_share_by_roasting_amount.png'); plt.close()

# 2. 카페인/디카페인 비중 시각화
df_caf = df_all.groupby(['연도', '구분_카페인'], as_index=False)[['수입(중량)', '수입(금액)']].sum()
df_yearly_sum_caf = df_caf.groupby('연도')[['수입(중량)', '수입(금액)']].sum().rename(columns={'수입(중량)': '총합(중량)', '수입(금액)': '총합(금액)'})
df_caf_merged = pd.merge(df_caf, df_yearly_sum_caf, on='연도')
df_caf_merged['비중(중량)'] = 100 * df_caf_merged['수입(중량)'] / df_caf_merged['총합(중량)']
df_caf_merged['비중(금액)'] = 100 * df_caf_merged['수입(금액)'] / df_caf_merged['총합(금액)']

df_caf_pivot_w = df_caf_merged.pivot(index='연도', columns='구분_카페인', values='비중(중량)')
plt.figure(figsize=(12, 6))
df_caf_pivot_w.plot(kind='area', stacked=True, figsize=(12, 6))
plt.title('커피 수입 비중 변화 (물량/중량 기준)'); plt.legend(title='구분 (카페인)', loc='upper left'); plt.grid(True)
plt.savefig('import_share_by_caffeine_weight.png'); plt.close()

df_caf_pivot_a = df_caf_merged.pivot(index='연도', columns='구분_카페인', values='비중(금액)')
plt.figure(figsize=(12, 6))
df_caf_pivot_a.plot(kind='area', stacked=True, figsize=(12, 6), colormap='viridis')
plt.title('커피 수입 비중 변화 (비용/금액 기준)'); plt.legend(title='구분 (카페인)', loc='upper left'); plt.grid(True)
plt.savefig('import_share_by_caffeine_amount.png'); plt.close()

# 3. 핵심 품목 (카페인 생두) 원화단가 볼린저 밴드
df_all_merged = pd.merge(df_all, df_exchange_processed, on='Date', how='left')
df_all_merged['원화단가'] = df_all_merged['수입단가'] * df_all_merged['ExchangeRate']
df_uc = df_all_merged[(df_all_merged['구분_로스팅'] == 'Unroasted') & (df_all_merged['구분_카페인'] == 'Caf')].copy()
df_uc.set_index('Date', inplace=True)
price_uc_krw = df_uc['원화단가'].dropna()

WINDOW_SIZE = 12
price_uc_ma = price_uc_krw.rolling(window=WINDOW_SIZE).mean()
price_uc_std = price_uc_krw.rolling(window=WINDOW_SIZE).std()
price_uc_upper = price_uc_ma + (price_uc_std * 2)
price_uc_lower = price_uc_ma - (price_uc_std * 2)

plt.figure(figsize=(14, 8))
plt.plot(price_uc_krw.index, price_uc_krw, label='카페인 생두 원화단가', color='blue', alpha=0.7)
plt.plot(price_uc_ma.index, price_uc_ma, label='중심선 (12M MA)', color='red', linestyle='--')
plt.plot(price_uc_upper.index, price_uc_upper, label='상단 밴드 (고가 영역)', color='gray', linestyle=':')
plt.plot(price_uc_lower.index, price_uc_lower, label='하단 밴드 (저가 영역)', color='gray', linestyle=':')
plt.fill_between(price_uc_ma.index, price_uc_lower, price_uc_upper, color='gray', alpha=0.1, label='정상 변동 범위 (±2 std)')
plt.title('핵심 품목(카페인 생두) 원화단가 및 볼린저 밴드'); plt.legend(loc='upper left'); plt.grid(True)
plt.savefig('price_negotiation_bollinger_bands.png'); plt.close()

# 4. 원가 구성요소 추세 및 변동성 시각화
df_risk_factors = df_total_merged[['총_원화단가', '총_수입단가', 'ExchangeRate']].dropna()
df_norm = (df_risk_factors / df_risk_factors.iloc[0]) * 100

plt.figure(figsize=(12, 6))
plt.plot(df_norm.index, df_norm['총_원화단가'], label='최종 원가 (KRW)', color='red', linewidth=2.5)
plt.plot(df_norm.index, df_norm['총_수입단가'], label='국제 시세 (USD)', color='blue', linestyle='--')
plt.plot(df_norm.index, df_norm['ExchangeRate'], label='환율 (FX)', color='green', linestyle=':')
plt.title('원가 구성요소별 누적 추세 비교 (시작점=100)'); plt.legend(); plt.grid(True)
plt.savefig('cost_component_trend_normalized.png'); plt.close()

df_pct_change_factors = df_risk_factors.pct_change() * 100
vol_krw = df_pct_change_factors['총_원화단가'].rolling(window=12).std()
vol_usd = df_pct_change_factors['총_수입단가'].rolling(window=12).std()
vol_fx = df_pct_change_factors['ExchangeRate'].rolling(window=12).std()

plt.figure(figsize=(12, 6))
plt.plot(vol_krw.index, vol_krw, label='최종 원가 (KRW) 변동성', color='red', linewidth=2.5)
plt.plot(vol_usd.index, vol_usd, label='국제 시세 (USD) 변동성', color='blue', linestyle='--')
plt.plot(vol_fx.index, vol_fx, label='환율 (FX) 변동성', color='green', linestyle=':')
plt.title('원가 구성요소별 변동성 크기 비교'); plt.legend(); plt.grid(True)
plt.savefig('cost_component_volatility_compare.png'); plt.close()

rolling_corr = df_pct_change_factors['총_수입단가'].rolling(window=12).corr(df_pct_change_factors['ExchangeRate'])
plt.figure(figsize=(12, 6))
plt.plot(rolling_corr.index, rolling_corr, label='롤링 상관관계 (국제 시세 vs 환율)', color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('국제 시세-환율 리스크 관계 분석 (Rolling Correlation)'); plt.ylim(-1, 1); plt.legend(); plt.grid(True)
plt.savefig('risk_correlation_rolling.png'); plt.close()

# 5. 계절성 분석 심화 및 주기/잔차 분석
five_years_ago = df_total.index.max() - pd.DateOffset(years=5)
df_5y_seasonal = df_total[df_total.index >= five_years_ago].copy()

seasonality_all = df_total.groupby('월')['수입(중량)'].mean()
plt.figure(figsize=(10, 5))
seasonality_all.plot(kind='bar')
plt.title('월별 평균 수입 중량 (전체 기간: 1999-2025)'); plt.xticks(rotation=0); plt.grid(axis='y')
plt.savefig('seasonality_overall.png'); plt.close()

seasonality_5y = df_5y_seasonal.groupby('월')['수입(중량)'].mean()
plt.figure(figsize=(10, 5))
seasonality_5y.plot(kind='bar', color='green')
plt.title('월별 평균 수입 중량 (최근 5년: 2020-2025)'); plt.xticks(rotation=0); plt.grid(axis='y')
plt.savefig('seasonality_recent_5y.png'); plt.close()

ts_data_weight = df_total['수입(중량)'].fillna(method='ffill')
decomposition_weight = seasonal_decompose(ts_data_weight, model='additive', period=12)

plt.figure(figsize=(12, 6))
decomposition_weight.trend.plot()
plt.title('수입 중량 장기 추세 (Long-term Trend)'); plt.grid(True)
plt.savefig('import_cycle_trend.png'); plt.close()

plt.figure(figsize=(12, 6))
decomposition_weight.resid.plot(marker='o', linestyle=' ')
plt.title('불규칙 변동/잔차 (Residuals / Shocks)'); plt.grid(True)
plt.savefig('import_cycle_residual.png'); plt.close()
