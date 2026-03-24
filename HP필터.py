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
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# 2. 운영체제별 글꼴 동적 설정 (실무 표준 방식)
# platform.system()을 통해 현재 코드가 실행되는 환경의 OS를 정확히 파악합니다.
os_name = platform.system()

if os_name == 'Windows':
    # 윈도우 환경: 기본적으로 내장된 맑은 고딕 사용
    plt.rc('font', family='Malgun Gothic')
elif os_name == 'Darwin':
    # Mac OS 환경: 애플 고딕 사용 (platform.system()에서 Mac은 'Darwin'으로 반환됨)
    plt.rc('font', family='AppleGothic')
elif os_name == 'Linux':
    # Linux 환경: 앞서 bash로 설치한 나눔고딕으로 설정
    plt.rc('font', family='NanumGothic')
else:
    # 기타 OS의 경우 에러 방지를 위해 pass 처리 (필요시 영문 기본 폰트 등 적용 가능)
    pass

# 3. 마이너스 기호 깨짐 방지
# 한글 폰트를 설정하면 폰트 자체의 특성상 마이너스(-) 기호가 깨지는 현상이 발생하므로, 이를 방지하기 위한 필수 설정입니다.
plt.rc('axes', unicode_minus=False)

# 4. 테스트용 그래프 그리기
# 위 설정이 정상적으로 적용되었는지 확인하기 위한 간단한 시각화 코드입니다.
plt.figure(figsize=(8, 5))
plt.plot([1, 2, 3], [10, 20, 15])
plt.title('한글 제목 테스트')
plt.xlabel('X축 라벨')
plt.ylabel('Y축 라벨')
plt.grid(True)
plt.show()


# =====================================================================
# [2] 데이터베이스 연결 및 데이터 로드
# =====================================================================
print("🚀 [Step 1] 데이터베이스 연결 및 커피 가격 데이터 로드 시작...")

con = duckdb.connect('my_database.duckdb')

print("\n--- DB 내 저장된 테이블 목록 ---")
all_tables_info = con.execute("SELECT * FROM duckdb_tables()").df()
print(all_tables_info)

# 커피 가격 데이터 로드
coffee_price_investing = con.execute("SELECT 날짜, 종가, 변동 FROM coffee_price_investing;").df()

print("\n---- coffee_price_investing 데이터 정보 ----")
print(coffee_price_investing.info())
print(coffee_price_investing.head())

con.close()
print("✅ 데이터 로드 및 DB 연결 종료 완료.")


# =====================================================================
# [3] 시계열 데이터 리샘플링 및 계절성 제거 (Deseasonalization)
# =====================================================================
print("\n🛠️ [Step 2] 월별 리샘플링 및 계절성 변동 제거 진행...")

# 시계열 분석을 위해 '날짜'를 인덱스로 설정
coffee_price_investing.set_index('날짜', inplace=True)

# 1. 월별 리샘플링 (Monthly Resampling)
# 일간 데이터의 노이즈를 줄이고 거시적인 흐름을 보기 위해 월초(MS) 기준으로 평균을 구합니다.
coffee_price_monthly = coffee_price_investing['종가'].resample('MS').mean().to_frame()

print("--- 월별 리샘플링 데이터 ---")
print(coffee_price_monthly.head())

# 2. 계절성 성분 분해 (Seasonal Decompose)
# HP 필터는 계절성에 의해 왜곡될 수 있으므로, 덧셈 모델(additive)을 통해 계절성을 분리합니다.
decomposition = seasonal_decompose(coffee_price_monthly['종가'], model='additive', period=12)

# 3. 계절성 제거 (Deseasonalized)
# 원본 월별 데이터에서 분리해낸 계절성 패턴을 빼주어 순수한 추세+순환+오차만 남깁니다.
coffee_price_monthly['deseasonalized'] = coffee_price_monthly['종가'] - decomposition.seasonal

# seasonal_decompose 적용 시 앞뒤로 생기는 결측치(NaN)를 제거합니다.
coffee_price_monthly.dropna(inplace=True)

print("\n--- 계절성 제거 데이터 (Deseasonalized) ---")
print(coffee_price_monthly[['종가', 'deseasonalized']].head())


# =====================================================================
# [4] HP 필터(Hodrick-Prescott Filter) 적용
# =====================================================================
print("\n🔍 [Step 3] HP 필터를 통한 장기 추세 및 순환 변동 추출...")

# HP 필터 파라미터 (Lambda) 설정
# 월별 데이터(Monthly)의 거시경제 표준 페널티 파라미터 람다 값은 14,400을 사용합니다.
# (참고: 분기별은 1600, 연별은 100을 주로 사용합니다.)
lamb_monthly = 14400

# HP 필터 적용: 계절성이 제거된 데이터에서 장기 추세(Trend)와 단기 순환(Cycle)을 분리합니다.
hp_cycle, hp_trend = sm.tsa.filters.hpfilter(coffee_price_monthly['deseasonalized'], lamb=lamb_monthly)

# 결과를 데이터프레임에 파생변수로 추가
coffee_price_monthly['hp_trend'] = hp_trend
coffee_price_monthly['hp_cycle'] = hp_cycle

print("\n--- HP 필터 적용 완료 결과 ---")
print(coffee_price_monthly[['deseasonalized', 'hp_trend', 'hp_cycle']].head())


# =====================================================================
# [5] 거시경제 사이클 시각화 (Visualization)
# =====================================================================
print("\n🎨 [Step 4] HP 필터 분석 결과 시각화 및 저장...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
fig.suptitle('커피 가격 HP 필터 분석 결과 (거시경제 사이클)', fontsize=18, y=1.02)

# --- 상단 그래프: 원본 vs 계절성 제거 vs 장기 추세 ---
ax1.plot(coffee_price_monthly.index, coffee_price_monthly['종가'], label='원본 월별 가격', color='skyblue', alpha=0.6)
ax1.plot(coffee_price_monthly.index, coffee_price_monthly['deseasonalized'], label='계절성 제거 가격', color='gray', alpha=0.8, linestyle=':')
ax1.plot(coffee_price_monthly.index, coffee_price_monthly['hp_trend'], label=f'HP 필터 장기 추세 (λ={lamb_monthly})', color='red', linestyle='--', linewidth=2)

ax1.set_title('커피 가격 장기 추세선 (Trend)', fontsize=14)
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_ylabel('가격 (종가)')

# --- 하단 그래프: 순환 변동 (Cycle) ---
ax2.plot(coffee_price_monthly.index, coffee_price_monthly['hp_cycle'], label='HP 필터 순환 변동 (Cycle)', color='blue')
ax2.axhline(0, color='black', linestyle='--', linewidth=1) # 0 기준선 (추세 평균)

ax2.set_title('커피 가격 순환 변동 (비계절성, 추세 제외)', fontsize=14)
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.set_ylabel('변동폭 (장기 추세와의 이격도)')

plt.xlabel('연도 (Year)')
plt.tight_layout()

# 고해상도 옵션 및 확장자 명시하여 파일 저장
plt.savefig('커피_가격_hp_필터_분석_결과.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 모든 분석 및 시각화가 성공적으로 완료되었습니다!")