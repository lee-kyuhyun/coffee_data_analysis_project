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

# 실무 배치(Batch) 환경을 위한 경고 무시 및 멀티스레드 충돌 방지 설정
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"

# 운영체제별 한글 폰트 및 마이너스 부호 깨짐 방지 전역 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
    
plt.rc('axes', unicode_minus=False)
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# [2] 데이터베이스 연결 및 최근 20년 데이터 로드
# =====================================================================
print("🚀 [Step 1] 데이터베이스 연결 및 최근 20년 커피 가격 데이터 로드 시작...")

con = duckdb.connect('my_database.duckdb')

# 2005년 이후 데이터만 필터링하여 최근 20년의 장기 빅사이클에 집중합니다.
query = "SELECT 날짜, 종가, 변동 FROM coffee_price_investing WHERE 날짜 > '2005-01-01';"
coffee_price_investing = con.execute(query).df()

print("\n---- 최근 20년 coffee_price_investing 데이터 정보 ----")
print(coffee_price_investing.info())
# print(coffee_price_investing.head())

con.close()
print("✅ 데이터 로드 및 DB 연결 종료 완료.")

# 시계열 분석을 위해 '날짜' 컬럼을 인덱스로 설정합니다.
coffee_price_investing.set_index('날짜', inplace=True)


# =====================================================================
# [3] 기초 가격 추세 및 이동평균선(MA / EMA) 분석
# =====================================================================
print("\n📈 [Step 2] 최근 20년 장기 가격 추세 및 이동평균선 시각화 중...")

# --- 시각화 1: 단순 커피 가격 선그래프 ---
coffee_price_investing.plot(kind='line', y='종가', title='최근 20년 커피 가격 변화 (2005년~)', figsize=(10, 6))
plt.xlabel('Date (Day)')
plt.ylabel('가격')
plt.grid(True)
plt.savefig('최근_20년_커피_가격_기본추세.png', dpi=300, bbox_inches='tight') # 이미지 자동 저장 추가
plt.show()


# --- 파생변수 생성: 50일 & 200일 단순이동평균(SMA) ---
# 가격의 단기 노이즈를 평활화하여 장기적인 지지선/저항선을 파악합니다.
coffee_price_investing['MA_50'] = coffee_price_investing['종가'].rolling(window=50).mean()
coffee_price_investing['MA_200'] = coffee_price_investing['종가'].rolling(window=200).mean()

# --- 시각화 2: 단순이동평균선 (50일 & 200일) ---
plt.figure(figsize=(10, 6))
plt.plot(coffee_price_investing.index, coffee_price_investing['종가'], label='Origin Price', alpha=0.3, color='gray')
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_50'], label='50-Day SMA', color='orange', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_200'], label='200-Day SMA', color='red', linewidth=2.5)

plt.title('최근 20년 이동평균 (50일 & 200일)')
plt.xlabel('Day')
plt.ylabel('가격')
plt.legend()
plt.grid(True)
plt.savefig('최근_20년_단순이동평균선.png', dpi=300, bbox_inches='tight') # 이미지 자동 저장 추가
plt.show()


# --- 파생변수 생성: 50일 & 200일 지수이동평균(EMA) ---
# 최신 가격 변화에 가중치를 두어 추세 전환(변곡점)을 더 신속하게 반영합니다.
coffee_price_investing['EMA_50'] = coffee_price_investing['종가'].ewm(span=50, adjust=False).mean()
coffee_price_investing['EMA_200'] = coffee_price_investing['종가'].ewm(span=200, adjust=False).mean()

# --- 시각화 3: 지수이동평균선 (50일 & 200일) ---
plt.figure(figsize=(10, 6))
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_50'], label='50-Day SMA (Reference)', color='gray', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['EMA_50'], label='50-Day EMA', color='orange', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['EMA_200'], label='200-Day EMA', color='red', linewidth=2.5)

plt.title('최근 20년 지수이동평균 (50일 & 200일)')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('최근_20년_지수이동평균선.png', dpi=300, bbox_inches='tight') # 이미지 자동 저장 추가
plt.show()


# =====================================================================
# [4] 1차 선형 회귀(Linear Regression)를 통한 최근 20년 추세선 분석
# =====================================================================
print("\n📏 [Step 3] np.polyfit 기반 선형 추세선 및 기울기 도출...")

# 결측치(NaN)로 인한 polyfit 계산 오류 방지를 위해 dropna() 적용
valid_data = coffee_price_investing['종가'].dropna()
x_trend = np.arange(len(valid_data))
y_trend = valid_data.values

# 1차 방정식(직선)의 계수(기울기, y절편) 도출
coefficients = np.polyfit(x_trend, y_trend, 1) 
polynomial = np.poly1d(coefficients)
trend_line = polynomial(x_trend)

# --- 시각화 4: 원본 가격 및 선형 회귀 추세선 ---
plt.figure(figsize=(10, 6))
plt.plot(coffee_price_investing.index, coffee_price_investing['종가'], label='원본 가격', alpha=0.3, color='gray')

# 추세선 매핑 (y_trend 배열의 길이와 인덱스를 일치시켜 에러 방지)
valid_index = coffee_price_investing.index[:len(y_trend)]
plt.plot(valid_index, trend_line, label='추세선 (Trend Line)', color='dodgerblue', linestyle='--', linewidth=3)

plt.title('최근 20년 추세 분석 (선형 회귀)')
plt.xlabel('날짜')
plt.ylabel('가격')
plt.legend()
plt.grid(True)
plt.savefig('최근_20년_커피_가격_추세.png', dpi=300, bbox_inches='tight') # 이미지 자동 저장 추가
plt.show()

# 장기 추세 기울기 결과 출력
slope = coefficients[0]
print(f"\n-> 추세선의 기울기: {slope:.10f}")
if slope > 0:
    print("-> 결론: 데이터는 최근 20년간 '상승'하는 추세를 보입니다.")
elif slope < 0:
    print("-> 결론: 데이터는 최근 20년간 '하락'하는 추세를 보입니다.")
else:
    print("-> 결론: 데이터는 최근 20년간 뚜렷한 추세가 없습니다.")


# =====================================================================
# [5] 변동성 및 모멘텀 지표: MACD 및 볼린저 밴드
# =====================================================================
print("\n📊 [Step 4] 볼린저 밴드 및 MACD 지표 계산 및 시각화...")

# --- 1. MACD (이동평균 수렴확산지수) 계산 ---
# 추세의 방향과 모멘텀(강도)을 파악하는 핵심 지표
ema_12 = coffee_price_investing['종가'].ewm(span=12, adjust=False).mean()
ema_26 = coffee_price_investing['종가'].ewm(span=26, adjust=False).mean()

coffee_price_investing['MACD'] = ema_12 - ema_26 # MACD 선
coffee_price_investing['Signal'] = coffee_price_investing['MACD'].ewm(span=9, adjust=False).mean() # Signal 선
coffee_price_investing['Histogram'] = coffee_price_investing['MACD'] - coffee_price_investing['Signal'] # 히스토그램

# --- 2. 볼린저 밴드 (Bollinger Bands) 계산 ---
# 가격의 상대적인 높낮이와 변동폭 확장을 측정하는 지표
coffee_price_investing['MA_20'] = coffee_price_investing['종가'].rolling(window=20).mean() # 중심선 (20일 SMA)
coffee_price_investing['StdDev'] = coffee_price_investing['종가'].rolling(window=20).std()   # 20일 이동 표준편차
coffee_price_investing['UpperBand'] = coffee_price_investing['MA_20'] + (coffee_price_investing['StdDev'] * 2) # 상단 저항선
coffee_price_investing['LowerBand'] = coffee_price_investing['MA_20'] - (coffee_price_investing['StdDev'] * 2) # 하단 지지선

# --- 시각화 5: 볼린저 밴드 & MACD 서브플롯 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# 상단 플롯: 가격 + 볼린저 밴드
ax1.plot(coffee_price_investing.index, coffee_price_investing['종가'], color='gray', label='가격', alpha=0.7)
ax1.plot(coffee_price_investing.index, coffee_price_investing['MA_20'], color='k', linestyle='--', label='중심선 (20일 SMA)')
ax1.plot(coffee_price_investing.index, coffee_price_investing['UpperBand'], color='r', label='상단 밴드')
ax1.plot(coffee_price_investing.index, coffee_price_investing['LowerBand'], color='b', label='하단 밴드')
ax1.fill_between(coffee_price_investing.index, coffee_price_investing['UpperBand'], coffee_price_investing['LowerBand'], color='lightgray', alpha=0.4)
ax1.set_title('최근 20년 커피 가격과 볼린저 밴드 (Volatility)', fontsize=16)
ax1.set_ylabel('가격')
ax1.legend(loc='upper left')
ax1.grid(True)

# 하단 플롯: MACD 지표
ax2.plot(coffee_price_investing.index, coffee_price_investing['MACD'], color='blue', label='MACD 선')
ax2.plot(coffee_price_investing.index, coffee_price_investing['Signal'], color='red', linestyle='--', label='Signal 선')
ax2.bar(coffee_price_investing.index, coffee_price_investing['Histogram'], color='g', alpha=0.5, label='히스토그램')
ax2.axhline(0, color='k', linestyle='-') # MACD 0 기준선
ax2.set_title('최근 20년 MACD (이동평균 수렴확산)', fontsize=16)
ax2.set_xlabel('날짜')
ax2.set_ylabel('MACD')
ax2.legend(loc='upper left')
ax2.grid(True)

plt.tight_layout()
plt.savefig('최근_20년_볼린저_밴드_및_MACD.png', dpi=300, bbox_inches='tight') # 이미지 자동 저장 추가
plt.show()

print("\n🎉 최근 20년 기술적 투자 지표 분석 및 시각화가 완료되었습니다!")