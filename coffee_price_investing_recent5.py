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

# 운영체제별 한글 폰트 및 마이너스 부호 깨짐 방지 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
    
plt.rc('axes', unicode_minus=False)
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# [2] 데이터베이스 연결 및 최근 5년 데이터 로드
# =====================================================================
print("🚀 [Step 1] 데이터베이스 연결 및 최근 5년 데이터 로드 시작...")

con = duckdb.connect('my_database.duckdb')

# 2020년 이후 데이터만 필터링하여 변동성이 극대화된 최근 시장에 집중합니다.
query = "SELECT 날짜, 종가, 변동 FROM coffee_price_investing WHERE 날짜 > '2020-01-01';"
coffee_price_investing = con.execute(query).df()

print("\n---- 최근 5년 coffee_price 데이터 정보 ----")
print(coffee_price_investing.info())
# print(coffee_price_investing.head())

con.close()
print("✅ 데이터 로드 및 DB 연결 종료 완료.")

# 시계열 분석을 위해 '날짜' 컬럼을 인덱스로 설정
coffee_price_investing = coffee_price_investing.set_index('날짜')


# =====================================================================
# [3] 기초 가격 추세 및 이동평균선(MA / EMA) 분석
# =====================================================================
print("\n📈 [Step 2] 최근 5년 장기 가격 추세 및 이동평균선 분석 중...")

# --- 시각화 1: 단순 커피 가격 선그래프 ---
coffee_price_investing.plot(kind='line', y='종가', title='최근 5년 커피 가격', figsize=(10, 6))
plt.xlabel('Date (Day)')
plt.ylabel('가격')
plt.grid(True)
plt.savefig('최근_5년_커피_가격_기본추세.png', dpi=300, bbox_inches='tight') # 저장 기능 보완
plt.show()

# --- 파생변수 생성: 50일 & 200일 단순이동평균(SMA) ---
coffee_price_investing['MA_50'] = coffee_price_investing['종가'].rolling(window=50).mean()
coffee_price_investing['MA_200'] = coffee_price_investing['종가'].rolling(window=200).mean()

# --- 시각화 2: 단순이동평균선 (50일 & 200일) ---
plt.figure(figsize=(10, 6))
plt.plot(coffee_price_investing.index, coffee_price_investing['종가'], label='origin_price', alpha=0.3, color='gray')
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_50'], label='50day', color='orange', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_200'], label='200day', color='red', linewidth=2.5)

plt.title('최근 5년 이동평균 (50일 & 200일)')
plt.xlabel('Day')
plt.ylabel('가격')
plt.legend()
plt.grid(True)
plt.savefig('최근_5년_단순이동평균선.png', dpi=300, bbox_inches='tight') # 저장 기능 보완
plt.show()

# --- 파생변수 생성: 50일 & 200일 지수이동평균(EMA) ---
# 최근 가격에 가중치를 두어 추세 전환을 더 빠르게 포착합니다.
coffee_price_investing['EMA_50'] = coffee_price_investing['종가'].ewm(span=50, adjust=False).mean()
coffee_price_investing['EMA_200'] = coffee_price_investing['종가'].ewm(span=200, adjust=False).mean()

# --- 시각화 3: 지수이동평균선 (50일 & 200일) ---
plt.figure(figsize=(10, 6))
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_50'], label='50day', color='gray', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['EMA_50'], label='50day^2 (EMA)', color='orange', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['EMA_200'], label='200day^2 (EMA)', color='red', linewidth=2.5)

plt.title('최근 5년 지수이동평균 (50일 & 200일)')
plt.xlabel('Day')
plt.ylabel('price')
plt.legend()
plt.grid(True)
plt.savefig('최근_5년_지수이동평균선.png', dpi=300, bbox_inches='tight') # 저장 기능 보완
plt.show()


# =====================================================================
# [4] 1차 선형 회귀(Linear Regression)를 통한 최근 5년 추세선 도출
# =====================================================================
print("\n📏 [Step 3] np.polyfit 기반 선형 추세선 및 기울기 도출...")

# 결측치(NaN) 제거 후 유효 데이터로 x, y값 생성
valid_data = coffee_price_investing['종가'].dropna()
x_trend = np.arange(len(valid_data))
y_trend = valid_data.values

# 1차 방정식(직선)의 계수(기울기와 절편) 탐색
coefficients = np.polyfit(x_trend, y_trend, 1) 
polynomial = np.poly1d(coefficients)
trend_line = polynomial(x_trend)

# --- 시각화 4: 원본 가격 및 선형 추세선 ---
plt.figure(figsize=(10, 6))
plt.plot(coffee_price_investing.index, coffee_price_investing['종가'], label='원본 가격', alpha=0.3, color='gray')

# 추세선 매핑 (길이를 맞춘 인덱스 사용)
valid_index = coffee_price_investing.index[:len(y_trend)]
plt.plot(valid_index, trend_line, label='추세선 (Trend Line)', color='dodgerblue', linestyle='--', linewidth=3)

plt.title('최근 5년 추세 분석')
plt.xlabel('날짜')
plt.ylabel('가격')
plt.legend()
plt.grid(True)
plt.savefig('최근5년_커피_가격_추세.png', dpi=300, bbox_inches='tight') # 확장자 추가
plt.show()

# 추세선의 기울기 기반 인사이트 출력
slope = coefficients[0]
print(f"\n-> 추세선의 기울기: {slope:.10f}")
if slope > 0:
    print("-> 결론: 데이터는 최근 5년 '상승'하는 추세를 보입니다.")
elif slope < 0:
    print("-> 결론: 데이터는 최근 5년 '하락'하는 추세를 보입니다.")
else:
    print("-> 결론: 데이터는 최근 5년 뚜렷한 추세가 없습니다.")


# =====================================================================
# [5] 변동성 및 모멘텀 지표: MACD 및 볼린저 밴드
# =====================================================================
print("\n📊 [Step 4] 볼린저 밴드 및 MACD 지표 계산 및 시각화...")

# --- 1. MACD 계산 ---
ema_12 = coffee_price_investing['종가'].ewm(span=12, adjust=False).mean()
ema_26 = coffee_price_investing['종가'].ewm(span=26, adjust=False).mean()
coffee_price_investing['MACD'] = ema_12 - ema_26
coffee_price_investing['Signal'] = coffee_price_investing['MACD'].ewm(span=9, adjust=False).mean()
coffee_price_investing['Histogram'] = coffee_price_investing['MACD'] - coffee_price_investing['Signal']

# --- 2. 볼린저 밴드 계산 ---
coffee_price_investing['MA_20'] = coffee_price_investing['종가'].rolling(window=20).mean()
coffee_price_investing['StdDev'] = coffee_price_investing['종가'].rolling(window=20).std()
coffee_price_investing['UpperBand'] = coffee_price_investing['MA_20'] + (coffee_price_investing['StdDev']*2)
coffee_price_investing['LowerBand'] = coffee_price_investing['MA_20'] - (coffee_price_investing['StdDev']*2)

# --- 시각화 5: 볼린저 밴드 & MACD 서브플롯 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# 상단: 가격 + 볼린저 밴드
ax1.plot(coffee_price_investing.index, coffee_price_investing['종가'], color='gray', label='가격', alpha=0.7)
ax1.plot(coffee_price_investing.index, coffee_price_investing['MA_20'], color='k', linestyle='--', label='중간(20일 SMA)')
ax1.plot(coffee_price_investing.index, coffee_price_investing['UpperBand'], color='r', label='상단밴드')
ax1.plot(coffee_price_investing.index, coffee_price_investing['LowerBand'], color='b', label='하단밴드')
ax1.fill_between(coffee_price_investing.index, coffee_price_investing['UpperBand'], coffee_price_investing['LowerBand'], color='lightgray', alpha=0.4)
ax1.set_title('최근 5년 커피 가격과 볼린저 밴드', fontsize=16)
ax1.set_ylabel('가격')
ax1.legend(loc='upper left')
ax1.grid(True)

# 하단: MACD
ax2.plot(coffee_price_investing.index, coffee_price_investing['MACD'], color='blue', label='MACD')
ax2.plot(coffee_price_investing.index, coffee_price_investing['Signal'], color='red', linestyle='--', label='신호선')
ax2.bar(coffee_price_investing.index, coffee_price_investing['Histogram'], color='g', alpha=0.5, label='히스토그램')
ax2.axhline(0, color='k', linestyle='-') # MACD 0기준선
ax2.set_title('최근 5년 MACD (이동평균 수렴확산)', fontsize=16)
ax2.set_xlabel('날짜')
ax2.set_ylabel('MACD')
ax2.legend(loc='upper left')
ax2.grid(True)

plt.tight_layout()
plt.savefig('최근_5년_커피_가격과_볼린저_밴드와_MACD.png', dpi=300, bbox_inches='tight') # 확장자 추가
plt.show()

print("\n🎉 모든 분석 및 시각화가 완료되었습니다.")