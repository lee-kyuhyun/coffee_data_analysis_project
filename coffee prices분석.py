#!/usr/bin/env python
# coding: utf-8

# In[1]:


import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import platform

# In[2]:


# 한글 글꼴 다운로드

# 1. 운영체제별 글꼴 동적 설정 (실무 표준 방식)
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

# 2. 마이너스 기호 깨짐 방지
# 한글 폰트를 설정하면 폰트 자체의 특성상 마이너스(-) 기호가 깨지는 현상이 발생하므로, 이를 방지하기 위한 필수 설정입니다.
plt.rc('axes', unicode_minus=False)

# 3. 테스트용 그래프 그리기
# 위 설정이 정상적으로 적용되었는지 확인하기 위한 간단한 시각화 코드입니다.
plt.figure(figsize=(8, 5))
plt.plot([1, 2, 3], [10, 20, 15])
plt.title('한글 제목 테스트')
plt.xlabel('X축 라벨')
plt.ylabel('Y축 라벨')
plt.grid(True)
plt.show()


# In[3]:


# 'my_database.duckdb'라는 이름의 파일을 열거나 생성합니다.
con = duckdb.connect('my_database.duckdb')


# In[4]:


print("db내 저장된 테이블 목록")
all_tables_info = con.execute("SELECT * FROM duckdb_tables()").df()
print(all_tables_info)


# In[5]:


# coffee price 데이터 df로 불러오기
coffee_price = con.execute("SELECT * FROM coffee_price WHERE Date >= '1979-12-27';").df()
print("----coffee_price의 정보----")
print(coffee_price.info())
print(coffee_price.head())

# coffee_price_investing  df로 불러오기
coffee_price_investing = con.execute("SELECT 날짜, 종가, 변동 FROM coffee_price_investing;").df()
print("\n----coffee_price의 정보----")
print(coffee_price_investing.info())
print(coffee_price_investing.head())

# coffee_price_investing  df로 불러오기
coffee_price_investing1 = con.execute("SELECT * FROM coffee_price_investing;").df()
print("\n----coffee_price의 정보----")
print(coffee_price_investing1.info())
print(coffee_price_investing1.head())


# In[6]:


# db 종료
con.close()


# In[7]:


# 커피 가격 선그래프로 전체 추세 확인

# Date열을 인덱스로 설정
coffee_price_index = coffee_price.set_index('Date')

# 선그래프 생성
coffee_price_index.plot(kind='line', y='Price', title = 'coffee_price_day', figsize=(10, 6))


plt.xlabel('day')
plt.ylabel('price')
plt.show()


# In[8]:


# 커피 가격 investing 선그래프로 전체 추세 확인

# Date열을 인덱스로 설정
coffee_price_investing_index = coffee_price_investing.set_index('날짜')

# 선그래프 생성
coffee_price_investing_index.plot(kind='line', y='종가', title = 'coffee_price_investing_day', figsize=(10, 6))

plt.xlabel('day')
plt.ylabel('price')
plt.show()


# In[9]:


# 표준화
from sklearn.preprocessing import MinMaxScaler
scaler_origin = MinMaxScaler()
scaler_new = MinMaxScaler()

coffee_price['price_scaled'] = scaler_origin.fit_transform(coffee_price[['Price']])
coffee_price_investing['종가_scaled'] = scaler_new.fit_transform(coffee_price_investing[['종가']])

# 두 가지의 커피 가격 데이터를 비교하는 그래프
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(coffee_price.index, coffee_price['price_scaled'], color='blue', label="기존 커피 가격")
ax.plot(coffee_price_investing.index, coffee_price_investing['종가_scaled'], color='red', label="새로 수집한 커피 가격", linestyle='--')

ax.set_title('커피 종류별 가격 추이 (날짜 범위 다름)', fontsize=16)
ax.set_xlabel('날짜')
ax.set_ylabel('가격')
ax.legend() # label로 지정한 이름을 범례로 표시
ax.grid(True) # 그리드 표시
plt.show()


# In[10]:


# coffee_price 최저가와 최고가
max_cp = coffee_price[coffee_price['Price'] == coffee_price['Price'].max()]
min_cp = coffee_price[coffee_price['Price'] == coffee_price['Price'].min()]
print('coffee_price')
print('max:', max_cp)
print('min:', min_cp)

# coffee_price_investing 최저가와 최고가
max_cpi = coffee_price_investing[coffee_price_investing['종가'] == coffee_price_investing['종가'].max()]
min_cpi = coffee_price_investing[coffee_price_investing['종가'] == coffee_price_investing['종가'].min()]
print('\n--------------------------')
print('coffee_price_investing')
print('max:', max_cpi)
print('min:', min_cpi)


# In[14]:


# 50일 이동평균
coffee_price_investing['MA_50'] = coffee_price_investing['종가'].rolling(window=50).mean()
# 200일 이동평균
coffee_price_investing['MA_200'] = coffee_price_investing['종가'].rolling(window=200).mean()



plt.figure(figsize=(10, 6))
plt.plot(coffee_price_investing.index, coffee_price_investing['종가'], label='origin_price', alpha=0.3, color='gray')

plt.plot(coffee_price_investing.index, coffee_price_investing['MA_50'], label='50day', color='orange', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_200'], label='200day', color='red', linewidth=2.5)

plt.title('a movement average (50day & 200day)')
plt.xlabel('Day')
plt.ylabel('price')
plt.legend()
plt.grid(True)
plt.show()


# In[15]:


# 50일 지수이동평균
coffee_price_investing['EMA_50'] = coffee_price_investing['종가'].ewm(span=50, adjust=False).mean()
# 200일 지수이동평균
coffee_price_investing['EMA_200'] = coffee_price_investing['종가'].ewm(span=200, adjust=False).mean()

plt.figure(figsize=(10, 6))
plt.plot(coffee_price_investing.index, coffee_price_investing['MA_50'], label='50day', color='gray', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['EMA_50'], label='50day^2', color='orange', linewidth=2.5)
plt.plot(coffee_price_investing.index, coffee_price_investing['EMA_200'], label='200day^2', color='red', linewidth=2.5)

plt.title('a exp movement average (50day & 200day)')
plt.xlabel('Day')
plt.ylabel('price')
plt.legend()
plt.grid(True)
plt.show()


# In[16]:


# 4. 추세선 계산
# x축은 시간의 흐름을 나타내는 숫자여야 하므로, 0부터 데이터 개수만큼의 숫자를 생성합니다.
x = np.arange(len(coffee_price_investing.index))
# y축은 가격 데이터입니다. 결측치가 있을 수 있으므로 .dropna()로 제거합니다.
y = coffee_price_investing['종가'].dropna()
# x축도 y와 길이를 맞춰줍니다.
x = x[:len(y)]

# polyfit을 이용해 1차 방정식(직선)의 계수(기울기와 절편)를 찾습니다.
coefficients = np.polyfit(x, y, 1) 

# 계수를 이용해 추세선 함수(다항식)를 만듭니다.
polynomial = np.poly1d(coefficients)

# x값에 대한 추세선 y값을 계산합니다.
trend_line = polynomial(x)

# 5. 추세선 시각화
plt.figure(figsize=(10, 6))
plt.plot(coffee_price_investing.index, coffee_price_investing['종가'], label='원본 가격', alpha=0.3, color='gray')
# 추세선은 계산된 y값과 y값에 해당하는 날짜 인덱스를 사용해 그립니다. 
plt.plot(coffee_price_investing.index[:len(y)], trend_line, label='추세선 (Trend Line)', color='dodgerblue', linestyle='--', linewidth=3)

plt.title('장기 추세 분석')
plt.xlabel('날짜')
plt.ylabel('가격')
plt.legend()
plt.grid(True)
plt.show()

# 추세선의 기울기 출력
slope = coefficients[0]
print(f"\n추세선의 기울기: {slope:.10f}")
if slope > 0:
    print("결론: 데이터는 장기적으로 '상승'하는 추세를 보입니다.")
elif slope < 0:
    print("결론: 데이터는 장기적으로 '하락'하는 추세를 보입니다.")
else:
    print("결론: 데이터는 장기적으로 뚜렷한 추세가 없습니다.")



