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
import matplotlib.dates as mdates # 시계열 X축 날짜 포맷팅을 위한 모듈

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
print("🚀 [Step 1] 데이터베이스 연결 및 커피 가격(선물) 데이터 로드 시작...")

con = duckdb.connect('my_database.duckdb')
coffee_price_investing = con.execute("SELECT 날짜, 종가, 변동 FROM coffee_price_investing;").df()

print("\n---- coffee_price_investing 데이터 정보 ----")
print(coffee_price_investing.info())
# print(coffee_price_investing.head())

con.close()
print("✅ 데이터 로드 및 DB 연결 종료 완료.")

# 시계열 분석을 위해 '날짜' 컬럼을 인덱스로 설정
coffee_price_investing = coffee_price_investing.set_index('날짜')


# =====================================================================
# [3] 기초 가격 추세 및 이동평균선(MA / EMA) 분석
# =====================================================================
print("\n📈 [Step 2] 장기 가격 추세 및 이동평균선 분석 중...")

# --- 시각화 1: 단순 커피 가격 선그래프 ---
coffee_price_investing.plot(kind='line', y='종가', title='Coffee Price Investing (Overall Trend)', figsize=(10, 6))
plt.xlabel('Date (Day)')
plt.ylabel('Price')
plt.grid(True)
plt.savefig('커피 가격 기본 추세.png', dpi=300, bbox_inches='tight') # 저장 기능 추가
plt.show()


# --- 파생변수 생성: 50일 & 200일 단순이동평균(SMA) ---
# 단기(50일)와 장기(200일) 추세를 비교하여 골든크로스/데드크로스 모멘텀을 파악합니다.
coffee_price_investing['MA_50'] = coffee_price_investing['종가'].rolling(window=50).mean()
coffee_price_investing['MA_200'] = coffee_price_investing['종가'].rolling(window=200).mean()

# --- 시각화 2: 50일 & 200일 단순이동평균선 ---
plt.figure(figsize=(10, 6))
plt.plot(coffee_price_investing.index, coffee_price_investing['종가'], label='Origin Price', alpha=0.3, color='gray')
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_50'], label='50-Day SMA', color='orange', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_200'], label='200-Day SMA', color='red', linewidth=2.5)
plt.title('A Movement Average (50-Day & 200-Day)')
plt.xlabel('Date (Day)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('단순이동평균선_50_200.png', dpi=300, bbox_inches='tight')
plt.show()


# --- 파생변수 생성: 50일 & 200일 지수이동평균(EMA) ---
# 최근 가격에 더 큰 가중치를 부여하여 추세 전환에 더 민감하게 반응하는 지표입니다.
coffee_price_investing['EMA_50'] = coffee_price_investing['종가'].ewm(span=50, adjust=False).mean()
coffee_price_investing['EMA_200'] = coffee_price_investing['종가'].ewm(span=200, adjust=False).mean()

# --- 시각화 3: 단순(SMA) vs 지수(EMA) 이동평균 비교 ---
plt.figure(figsize=(10, 6))
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_50'], label='50-Day SMA', color='gray', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['EMA_50'], label='50-Day EMA', color='orange', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['EMA_200'], label='200-Day EMA', color='red', linewidth=2.5)
plt.title('An Exponential Movement Average (50-Day & 200-Day)')
plt.xlabel('Date (Day)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('지수이동평균선_비교.png', dpi=300, bbox_inches='tight')
plt.show()


# =====================================================================
# [4] 1차 선형 회귀(Linear Regression)를 통한 장기 추세선 도출
# =====================================================================
print("\n📏 [Step 3] np.polyfit 기반 장기 추세선 및 기울기 도출...")

# 결측치(NaN) 제거 후 유효한 데이터로만 추세선 계산 (polyfit 사용)
valid_data = coffee_price_investing['종가'].dropna()
x_trend = np.arange(len(valid_data))
y_trend = valid_data.values

# 1차 방정식(직선)의 계수(기울기와 절편) 탐색
coef = np.polyfit(x_trend, y_trend, 1) 
p = np.poly1d(coef)
trend_line = p(x_trend)

# --- 시각화 4: 가격, 이동평균선, 그리고 선형 추세선 ---
plt.figure(figsize=(10, 6))
plt.plot(coffee_price_investing.index, coffee_price_investing['종가'], label='Origin Price', alpha=0.3, color='gray')
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_50'], label='50-Day SMA', color='orange', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_200'], label='200-Day SMA', color='red', linewidth=2.5)

# 추세선은 valid_data 기준으로 계산되었으므로, 해당 데이터의 인덱스(날짜)를 사용하여 매핑
plt.plot(valid_data.index, trend_line, label='Trend Line (Linear)', color='blue', linestyle='--', linewidth=2)

plt.title('커피 가격 이동평균(50일 & 200일) 및 추세선')
plt.xlabel('Date (Day)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('커피 가격 이동평균(50일 & 200일).png', dpi=300, bbox_inches='tight')
plt.show()

# 추세선의 기울기 기반 인사이트 출력
slope = coef[0]
print(f"-> 추세선의 기울기: {slope:.10f}")
if slope > 0:
    print("-> 결론: 데이터는 장기적으로 '상승'하는 추세를 보입니다.")
elif slope < 0:
    print("-> 결론: 데이터는 장기적으로 '하락'하는 추세를 보입니다.")
else:
    print("-> 결론: 데이터는 장기적으로 뚜렷한 추세가 없습니다.")


# =====================================================================
# [5] 전문 기술적 지표: MACD 및 볼린저 밴드(Bollinger Bands)
# =====================================================================
print("\n📊 [Step 4] 볼린저 밴드 및 MACD 지표 계산 및 시각화...")

# --- 1. MACD (이동평균 수렴확산) 계산 ---
# MACD 선: 단기(12일) EMA - 장기(26일) EMA
ema_12 = coffee_price_investing['종가'].ewm(span=12, adjust=False).mean()
ema_26 = coffee_price_investing['종가'].ewm(span=26, adjust=False).mean()
coffee_price_investing['MACD'] = ema_12 - ema_26

# Signal 선: MACD의 9일 EMA (매수/매도 타이밍 포착용)
coffee_price_investing['Signal'] = coffee_price_investing['MACD'].ewm(span=9, adjust=False).mean()

# MACD 히스토그램: MACD - Signal (추세의 강도)
coffee_price_investing['Histogram'] = coffee_price_investing['MACD'] - coffee_price_investing['Signal']

# --- 2. 볼린저 밴드 (Bollinger Bands) 계산 ---
# 가격의 변동성에 따라 상하단 밴드의 폭이 확장/수축하는 변동성 지표
coffee_price_investing['MA_20'] = coffee_price_investing['종가'].rolling(window=20).mean() # 중심선
coffee_price_investing['StdDev'] = coffee_price_investing['종가'].rolling(window=20).std()   # 표준편차
coffee_price_investing['UpperBand'] = coffee_price_investing['MA_20'] + (coffee_price_investing['StdDev'] * 2) # 상단 (저항)
coffee_price_investing['LowerBand'] = coffee_price_investing['MA_20'] - (coffee_price_investing['StdDev'] * 2) # 하단 (지지)

# --- 시각화 5: 볼린저 밴드 & MACD 서브플롯 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# 상단 플롯: 가격 + 볼린저 밴드
ax1.plot(coffee_price_investing.index, coffee_price_investing['종가'], color='gray', label='Price', alpha=0.7)
ax1.plot(coffee_price_investing.index, coffee_price_investing['MA_20'], color='k', linestyle='--', label='Middle (20-Day SMA)')
ax1.plot(coffee_price_investing.index, coffee_price_investing['UpperBand'], color='r', label='Upper Band')
ax1.plot(coffee_price_investing.index, coffee_price_investing['LowerBand'], color='b', label='Lower Band')
ax1.fill_between(coffee_price_investing.index, coffee_price_investing['UpperBand'], coffee_price_investing['LowerBand'], color='lightgray', alpha=0.4)
ax1.set_title('커피 가격과 볼린저 밴드 (Volatility)', fontsize=16)
ax1.set_ylabel('Price')
ax1.legend(loc='upper left')
ax1.grid(True)

# 하단 플롯: MACD 지표
ax2.plot(coffee_price_investing.index, coffee_price_investing['MACD'], color='blue', label='MACD Line')
ax2.plot(coffee_price_investing.index, coffee_price_investing['Signal'], color='red', linestyle='--', label='Signal Line')
ax2.bar(coffee_price_investing.index, coffee_price_investing['Histogram'], color='g', alpha=0.5, label='Histogram')
ax2.axhline(0, color='k', linestyle='-') # MACD 0 기준선
ax2.set_title('MACD (이동평균 수렴확산지수)', fontsize=16)
ax2.set_xlabel('Date')
ax2.set_ylabel('MACD Value')
ax2.legend(loc='upper left')
ax2.grid(True)

plt.tight_layout()
plt.savefig('볼린저밴드_및_MACD_지표.png', dpi=300, bbox_inches='tight')
plt.show()


# =====================================================================
# [6] 머신러닝 기반 통계적 구조 변화점 탐지 (Ruptures)
# =====================================================================
print("\n🚨 [Step 5] Ruptures 라이브러리를 활용한 구조 변화점(Change Point) 탐지...")

try:
    import ruptures as rpt
    
    # 분석 데이터 준비 (NaN 제거 및 numpy 배열 변환)
    valid_data_series = coffee_price_investing['종가'].dropna()
    points = valid_data_series.values

    if len(points) < 10:
        print("-> 분석할 데이터가 충분하지 않습니다.")
    else:
        # 변화점 탐지 모델 설정 (Pelt 알고리즘, l2 비용함수 적용)
        # 평균의 급격한 변화(가격의 급등락 레벨 시프트)를 감지합니다.
        algo = rpt.Pelt(model="l2").fit(points)

        # BIC(Bayesian Information Criterion) 기반 페널티 값 설정
        n = len(points)
        sigma = valid_data_series.std()
        if pd.isna(sigma) or sigma == 0:
            sigma = 1.0 
            
        penalty = 3 * np.log(n) * (sigma**2)
        print(f"-> 데이터 포인트: {n}개 / 표준편차: {sigma:.2f} / 페널티: {penalty:.2f}")

        # 변화점 예측 (.predict는 경계 인덱스 목록을 반환)
        bkps_indices = algo.predict(pen=penalty)

        # --- 시각화 6: 구조 변화점 매핑 ---
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(valid_data_series.index, valid_data_series, label='종가 (결측치 제외)', color='gray', alpha=0.8, linewidth=1)
        
        change_point_dates = []
        for i, idx in enumerate(bkps_indices):
            if idx < n: # 마지막 인덱스(n)는 실제 변화점이 아니므로 제외
                date = valid_data_series.index[idx]
                change_point_dates.append(date)
                label = f'변화점 {date.date()}' if i == 0 else None
                ax.axvline(date, color='r', linestyle='--', linewidth=2, label=label)

        if change_point_dates:
            ax.legend(loc='upper left')
        
        ax.set_title('커피 가격 구조 변화점 탐지 (Pelt 알고리즘)', fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True)
        plt.tight_layout()
        plt.suptitle('커피 가격 구조 변화점 분석', fontsize=20, y=1.03)
        plt.savefig('구조변화점_Pelt_탐지.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n--- 구조 변화점 탐지 결과 요약 ---")
        if not change_point_dates:
            print("탐지된 주요 구조 변화점이 없습니다.")
        else:
            print(f"총 {len(change_point_dates)}개의 주요 레벨 시프트(구조 변화점)가 탐지되었습니다:")
            for i, date in enumerate(change_point_dates):
                print(f"  {i+1}. {date.date()}")

except ImportError:
    print("--- 경고 ---")
    print("'ruptures' 라이브러리가 설치되지 않았습니다.")
    print("구조 변화점 탐지를 실행하려면 터미널에서 'pip install ruptures'를 실행해 주세요.")


# =====================================================================
# [7] 역사적 주요 이벤트와 가격 변동 매핑
# =====================================================================
print("\n📜 [Step 6] 커피 시장의 역사적 주요 이벤트와 가격 시각화...")

events_dict = {
    '1989-07-06': 'ICA 협정 붕괴\n(가격 자유화)',
    '1994-06-10': "브라질 '검은 서리'\n(공급 충격)",
    '1997-09-19': '아시아 금융 위기\n(수요 감소)',
    '2000-08-08': '베트남 쇼크\n(공급 과잉)',
    '2010-12-16': '가격 급등 (기후)', 
    '2014-02-20': '브라질 대가뭄\n(공급 충격)',
    '2021-07-20': 'COVID & 브라질 서리\n(복합 위기)', 
    '2024-04-05': "기후 위기\n('New Normal')"
}

# --- 시각화 7: 마스터 타임라인 차트 ---
fig, ax = plt.subplots(figsize=(20, 10))

# 원본 가격 선 그래프
ax.plot(coffee_price_investing.index, coffee_price_investing['종가'], label='커피 가격 (종가)', color='blue', linewidth=2)
# 범례용 더미 플롯
ax.plot([], [], color='red', linestyle='--', linewidth=2, label='주요 역사적 이벤트')

for date_str, event_label in events_dict.items():
    try:
        event_date = pd.to_datetime(date_str)
        if event_date < coffee_price_investing.index.min() or event_date > coffee_price_investing.index.max():
            print(f"알림: {event_date.date()} 이벤트는 차트 범위 밖에 있어 생략됩니다.")
            continue

        # 수직선 및 이벤트 텍스트 추가 (축 하단 비율 기준)
        ax.axvline(x=event_date, color='red', linestyle='--', linewidth=2)
        ax.text(x=event_date, y=0.01, s=event_label, 
                transform=ax.get_xaxis_transform(), 
                color='red', fontsize=10, 
                horizontalalignment='center', 
                verticalalignment='bottom')
    except Exception as e:
        print(f"이벤트 날짜 {date_str} 처리 중 오류: {e}")

ax.set_title('커피 가격 추이 및 주요 매크로 이벤트 (1980-2025)', fontsize=20, pad=20)
ax.set_xlabel('연도 (Year)', fontsize=12)
ax.set_ylabel('가격 (Price)', fontsize=12)
ax.set_ylim(bottom=0)
ax.legend(loc='upper left', fontsize=12)
ax.grid(True)

# X축 눈금 포맷팅: 5년 단위 메이저 눈금, 1년 단위 마이너 눈금
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.YearLocator(1)) 
ax.set_xlim(coffee_price_investing.index.min(), coffee_price_investing.index.max())

plt.tight_layout()
plt.savefig('커피 가격 추이 및 주요 이벤트.png', dpi=300, bbox_inches='tight')
plt.show()

print("🎉 --- 가격 분석 및 이벤트 매핑 시각화가 완벽히 종료되었습니다 ---")