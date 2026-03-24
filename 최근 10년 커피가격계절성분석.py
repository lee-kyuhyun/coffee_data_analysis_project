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
# [2] 데이터베이스 연결 및 최근 10년 데이터 로드
# =====================================================================
print("🚀 [Step 1] 데이터베이스 연결 및 최근 10년 데이터 로드 시작...")

con = duckdb.connect('my_database.duckdb')

# SQL 쿼리를 통해 2010년 1월 1일 이후의 최근 10년치 데이터를 필터링하여 가져옵니다.
# (금융위기 이후의 거시경제 사이클 트렌드를 파악하기 위함입니다.)
query = "SELECT 날짜, 종가, 변동 FROM coffee_price_investing WHERE 날짜 > '2010-01-01';"
coffee_price_investing = con.execute(query).df()

print("\n---- 최근 10년 coffee_price_investing 데이터 정보 ----")
print(coffee_price_investing.info())
# print(coffee_price_investing.head()) # 필요시 주석 해제

con.close()
print("✅ 데이터 로드 및 DB 연결 종료 완료.")


# =====================================================================
# [3] 데이터 전처리 (Preprocessing)
# =====================================================================
print("\n🛠️ [Step 2] 시계열 분석을 위한 데이터 전처리 진행...")

# 시계열 분석의 기본: '날짜' 컬럼을 시계열 인덱스로 지정
coffee_price_investing = coffee_price_investing.set_index('날짜')

# 계절성(월별 패턴) 확인을 위한 'month' 파생변수 추출
coffee_price_investing['month'] = coffee_price_investing.index.month


# =====================================================================
# [4] 시계열 탐색적 데이터 분석 (Time Series EDA)
# =====================================================================
print("\n📈 [Step 3] 최근 10년 장기 추세 및 월별 계절성 탐색 시각화...")

# --- 시각화 1: 최근 10년 커피 가격 변동 추세 (매년 1월 1일 기준선 추가) ---
fig, ax = plt.subplots(figsize=(14, 6))

# 원본 코드의 '1979-12-27~' 타이틀을 쿼리 필터링 조건에 맞게 '최근 10년(2010년~)'으로 수정하여 명확성을 높였습니다.
coffee_price_investing['종가'].plot(ax=ax, title='최근 10년 커피 가격 변동 (2010년~)', label='가격')

# 데이터의 시작~종료 범위 내에서 '매년 1월 1일'의 날짜 리스트를 생성합니다.
# freq='YS' (Year Start): 연도의 첫 번째 날을 추출하여 연간 주기성을 파악하기 좋게 만듭니다.
yearly_dates = pd.date_range(
    start=coffee_price_investing.index.min(),
    end=coffee_price_investing.index.max(),
    freq='YS'
)

# 찾은 날짜(매년 1월 1일)마다 반복하면서 세로선(axvline)을 긋습니다.
for date in yearly_dates:
    ax.axvline(
        x=date,           # 선을 그을 x축 위치 (날짜)
        color='red',      # 선 색상
        linestyle=':',    # 선 스타일 (점선)
        linewidth=1,      # 선 두께
        alpha=0.8         # 선 투명도
    )

ax.set_ylabel('가격')
ax.set_xlabel('날짜')
ax.legend()
ax.grid(True, axis='y') # Y축 가로선만 표시하여 세로 점선과 시각적으로 겹치지 않게 처리
plt.tight_layout()
plt.savefig('최근 10년 커피 가격 변동 시계열.png', dpi=300, bbox_inches='tight') # 포트폴리오 저장을 위해 추가
plt.show()


# --- 시각화 2: 최근 10년 월별 커피 가격 분포 (Boxplot) ---
plt.figure(figsize=(12, 6))

# Boxplot(상자 수염 그림)을 통해 월별 가격의 중앙값, 사분위수, 그리고 계절적 이상치를 한눈에 파악합니다.
sns.boxplot(x='month', y='종가', data=coffee_price_investing)

# 차트 제목을 데이터 범위에 맞게 업데이트했습니다.
plt.title('최근 10년 월별 커피 가격 분포')
plt.xlabel('월 (Month)')
plt.ylabel('가격 (종가)')

# 원본 코드에 없던 저장 기능을 추가하여, 시각화 자료를 로컬에 안전하게 보존합니다.
plt.savefig('최근 10년 월별 커피 가격 분포.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 --- 최근 10년 커피 가격 계절성 시각화가 완료되었습니다 ---")