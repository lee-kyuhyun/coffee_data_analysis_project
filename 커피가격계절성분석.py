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

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

# 경고 무시 및 멀티스레드 충돌 방지 설정
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"

# 운영체제별 한글 폰트 및 마이너스 부호 깨짐 방지 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
    
plt.rc('axes', unicode_minus=False)
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# [2] 데이터베이스 연결 및 데이터 로드
# =====================================================================
print("🚀 [Step 1] 데이터베이스 연결 및 커피 가격 데이터 로드 시작...")

con = duckdb.connect('my_database.duckdb')

# 커피 가격 데이터 프레임 로드
coffee_price_investing = con.execute("SELECT 날짜, 종가, 변동 FROM coffee_price_investing;").df()

print("\n---- coffee_price_investing 데이터 정보 ----")
print(coffee_price_investing.info())
# print(coffee_price_investing.head()) # 필요시 주석 해제

con.close()
print("✅ 데이터 로드 및 DB 연결 종료 완료.")


# =====================================================================
# [3] 데이터 전처리 (Preprocessing)
# =====================================================================
print("\n🛠️ [Step 2] 시계열 분석을 위한 데이터 전처리 진행...")

# 시계열 분석의 핵심: '날짜' 컬럼을 인덱스로 설정
coffee_price_investing = coffee_price_investing.set_index('날짜')

# 계절성(월별 패턴) 확인을 위한 'month' 파생변수 생성
coffee_price_investing['month'] = coffee_price_investing.index.month

# 가격 데이터의 이상치(Outlier) 영향 완화 및 분산 안정화를 위한 로그 변환
# np.log1p()는 log(1 + x)를 계산하여 값이 0일 때 발생할 수 있는 무한대(-inf) 오류를 방지합니다.
coffee_price_investing['price_log'] = np.log1p(coffee_price_investing['종가'])


# =====================================================================
# [4] 시계열 탐색적 데이터 분석 (Time Series EDA)
# =====================================================================
print("\n📈 [Step 3] 장기 가격 추세 및 월별 계절성 탐색 시각화...")

# --- 시각화 1: 장기 커피 가격 변동 추세 (매년 1월 1일 수직선 표시) ---
fig, ax = plt.subplots(figsize=(14, 6))
coffee_price_investing['종가'].plot(ax=ax, title='커피 가격 변동 (1979-12-27 ~ 현재)', label='가격')

# 데이터의 시작~종료 범위 내에서 '매년 1월 1일(Year Start, YS)' 날짜 리스트 생성
yearly_dates = pd.date_range(
    start=coffee_price_investing.index.min(),
    end=coffee_price_investing.index.max(),
    freq='YS'
)

# 매년 시작일마다 연도 구분을 위한 수직 점선(axvline) 추가
for date in yearly_dates:
    ax.axvline(x=date, color='red', linestyle=':', linewidth=1, alpha=0.8)

ax.set_ylabel('가격')
ax.set_xlabel('날짜')
ax.legend()
ax.grid(True, axis='y') # 세로줄과 겹치지 않게 가로 그리드만 표시
plt.tight_layout()
plt.show()


# --- 시각화 2: 월별 커피 가격 분포 (원본 데이터) ---
plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='종가', data=coffee_price_investing)
plt.title('월별 커피 가격 분포 (원본)')
plt.xlabel('월 (Month)')
plt.ylabel('가격 (종가)')
plt.savefig('월별 커피 가격 분포.png') # 확장자(.png) 명시 권장
plt.show()


# --- 시각화 3: 월별 커피 가격 분포 (로그 변환 후) ---
plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='price_log', data=coffee_price_investing)
plt.title('월별 커피 가격 분포 (로그 변환 후)')
plt.xlabel('월 (Month)')
plt.ylabel('가격 (log(1 + price))')
plt.savefig('월별 커피 가격 분포(log변환후).png')
plt.show()


# =====================================================================
# [5] 시계열 분해 (Seasonal Decomposition)
# =====================================================================
print("\n🔍 [Step 4] 시계열 분해(Trend, Seasonal, Residual) 분석...")

# 시계열 데이터를 추세(Trend), 계절성(Seasonal), 잔차(Residual) 요소로 분해합니다.
# model='multiplicative': 곱셈 모델 (Y = T * S * R). 
# 가격 데이터처럼 추세 수준에 비례해 계절성 진폭이 커지는 경제/금융 지표에 주로 적합합니다.

# --- 시각화 4: 시계열 분해 (주기: 365일, 연간 패턴) ---
decomposition_365 = seasonal_decompose(
    coffee_price_investing['종가'], 
    model='multiplicative', 
    period=365
)
fig1 = decomposition_365.plot()
fig1.set_size_inches(12, 10)
plt.suptitle('시계열 분해 결과 (곱셈 모델, 주기=365일)', y=1.02)
plt.savefig('커피 가격 시계열 분해 결과.png')
plt.show()


# --- 시각화 5: 시계열 분해 (주기: 90일, 분기별/단기 패턴) ---
# 참고: 일별 데이터에서 90일 주기는 분기별(Quarterly) 변동성을 확인하는 데 의미가 있습니다.
decomposition_90 = seasonal_decompose(
    coffee_price_investing['종가'], 
    model='multiplicative', 
    period=90
)
fig2 = decomposition_90.plot()
fig2.set_size_inches(12, 10)
plt.suptitle('시계열 분해 결과 (곱셈 모델, 주기=90일)', y=1.02)
plt.show()


# =====================================================================
# [6] 정상성 및 자기상관 (ACF & Differencing) 분석
# =====================================================================
print("\n📉 [Step 5] 추세 제거 및 자기상관함수(ACF) 시각화...")


# --- 시각화 6: 원본 데이터의 ACF 플롯 ---
plt.figure(figsize=(12, 5))
plot_acf(coffee_price_investing['종가'], lags=40, title='커피 가격 ACF (자기상관, 차분 전)')
plt.xlabel('시차 (Lag)')
plt.ylabel('상관계수')
plt.show()


# --- 시각화 7: 1차 차분(Differencing)을 통한 정상성 확보 및 ACF 비교 ---
price_series = coffee_price_investing['종가']

# .diff()를 적용해 1차 차분 (Y_t - Y_{t-1}) 수행
# 추세(Trend)가 있는 비정상(Non-stationary) 시계열을 정상(Stationary) 시계열로 변환하는 핵심 과정입니다.
price_diff = price_series.diff().dropna() # 첫 행의 결측치(NaN) 제거 필수

# 그래프 2개를 위아래로 그릴 도화지 준비
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# (Before) 원본 데이터 ACF
plot_acf(price_series, lags=40, ax=axes[0], title='원본 데이터 ACF (추세 제거 전)')
axes[0].set_xlabel('시차 (Lag)')
axes[0].set_ylabel('상관계수')

# (After) 1차 차분 데이터 ACF
plot_acf(price_diff, lags=40, ax=axes[1], title='1차 차분 데이터 ACF (추세 제거 후)')
axes[1].set_xlabel('시차 (Lag)')
axes[1].set_ylabel('상관계수')

plt.tight_layout()
plt.savefig('ACF_comparison.png') # 파일명에 확장자 및 명확한 이름 추가
plt.show()
