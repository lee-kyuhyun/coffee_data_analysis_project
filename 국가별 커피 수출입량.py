#!/usr/bin/env python
# coding: utf-8

# =====================================================================
# 1. 라이브러리 임포트 및 전역 환경 설정
# (주피터 노트북 여러 셀에 흩어져 있던 임포트와 폰트 설정을 최상단으로 통합했습니다.)
# =====================================================================
import os
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import platform

from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# 경고 무시 및 멀티스레드 충돌 방지 설정 (In[52]에 있던 코드 통합)
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"

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

# pandas 출력 설정 (행이 많을 때 대비)
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)


# =====================================================================
# 2. 데이터 로드
# =====================================================================
# 'my_database.duckdb'라는 이름의 파일을 열거나 생성합니다.
con = duckdb.connect('my_database.duckdb')

print("db내 저장된 테이블 목록")
all_tables_info = con.execute("SELECT * FROM duckdb_tables()").df()
print(all_tables_info)

# coffee_production df로 불러오기
df = con.execute("SELECT * FROM coffee_production;").df()
print("\n----coffee_production 의 정보----")
print(df.info())

# 실무 팁: 전체 데이터를 메모리(df)로 올렸다면 db 연결은 일찍 닫아주는 것이 좋습니다.
con.close() 

# 분석 공통 기준 연도 파악
latest_year = df['Market_Year'].max()


# =====================================================================
# 3. 비즈니스 지표 분석 및 시각화 로직 (원본 100% 유지)
# =====================================================================

# ----------------------------------------------------
# [분석 1] 국가별 생산/소비/수입량 및 생산 부족분 파악
# ----------------------------------------------------
attributes_needed = [
    'Production', 'Arabica Production', 'Robusta Production',
    'Domestic Consumption', 'Rst,Ground Dom. Consum', 'Soluble Dom. Cons.',
    'Imports', 'Bean Imports', 'Roast & Ground Imports', 'Soluble Imports'
]

df_filtered = df[df['Attribute_Description'].isin(attributes_needed)].copy()
df_filtered = df_filtered[['Country_Name', 'Market_Year', 'Attribute_Description', 'Total_Value']]

print(f"분석 기준 연도 = {latest_year}")
df_recent = df_filtered[df_filtered['Market_Year'] == latest_year]

df_pivot = df_recent.pivot_table(
    index='Country_Name',
    columns='Attribute_Description',
    values='Total_Value',
    aggfunc='sum'
).fillna(0)

df_pivot['Total_Production'] = df_pivot.get('Production', 0)
df_pivot['Total_Consumption'] = df_pivot.get('Domestic Consumption', 0)
df_pivot['Total_Imports'] = df_pivot.get('Bean Imports', 0) + df_pivot.get('Roast & Ground Imports', 0) + df_pivot.get('Soluble Imports', 0)
df_pivot['Production_Shortfall_kg'] = df_pivot['Total_Consumption'] - df_pivot['Total_Production']

target_countries = df_pivot[df_pivot['Production_Shortfall_kg'] > 0].copy()
final_report = target_countries[['Total_Consumption', 'Total_Production', 'Production_Shortfall_kg', 'Total_Imports']]

report_sorted_by_shortfall = final_report.sort_values(by='Production_Shortfall_kg', ascending=False)
report_sorted_by_imports = final_report.sort_values(by='Total_Imports', ascending=False)

report_sorted_by_shortfall['Shortfall_Million_kg'] = report_sorted_by_shortfall['Production_Shortfall_kg'] / 1e6
report_sorted_by_imports['Imports_Million_kg'] = report_sorted_by_imports['Total_Imports'] / 1e6

# 시각화 1: 생산 부족분 Top 10
top_10_shortfall = report_sorted_by_shortfall.head(10).reset_index()
plt.figure(figsize=(12, 7))
sns.barplot(data=top_10_shortfall, x='Shortfall_Million_kg', y='Country_Name', palette='Reds_r')
plt.title(f'커피 생산 부족분(수요-공급 갭) Top 10 (기준: {latest_year}년)', fontsize=16)
plt.xlabel('생산 부족분 (백만 kg)')
plt.ylabel('국가명')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('Coffee_production_shortage_top10')
plt.show()

# 시각화 2: 총 수입량 Top 10
top_10_imports = report_sorted_by_imports.head(10).reset_index()
plt.figure(figsize=(12, 7))
sns.barplot(data=top_10_imports, x='Imports_Million_kg', y='Country_Name', palette='Blues_r')
plt.title(f'커피 총 수입량 Top 10 (기준: {latest_year}년)', fontsize=16)
plt.xlabel('총 수입량 (백만 kg)')
plt.ylabel('국가명')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('Total_coffee_import_top10')
plt.show()

# ----------------------------------------------------
# [분석 1-2] 5년 평균 수요-공급 갭 Top 10
# ----------------------------------------------------
start_year = 2021
end_year = 2025
df_filtered_5yr = df[
    (df['Attribute_Description'].isin(attributes_needed)) &
    (df['Market_Year'] >= start_year) & (df['Market_Year'] <= end_year)
].copy()

df_avg_pivot = df_filtered_5yr.groupby(['Country_Name', 'Attribute_Description'])['Total_Value'].mean().unstack(fill_value=0)
df_avg_pivot['Avg_Total_Production'] = df_avg_pivot.get('Production', 0)
df_avg_pivot['Avg_Total_Consumption'] = df_avg_pivot.get('Domestic Consumption', 0)
df_avg_pivot['Avg_Production_Shortfall_kg'] = df_avg_pivot['Avg_Total_Consumption'] - df_avg_pivot['Avg_Total_Production']

target_countries_avg = df_avg_pivot[df_avg_pivot['Avg_Production_Shortfall_kg'] > 0].copy()
report_avg_shortfall = target_countries_avg.sort_values(by='Avg_Production_Shortfall_kg', ascending=False)
report_avg_shortfall['Shortfall_Million_kg'] = report_avg_shortfall['Avg_Production_Shortfall_kg'] / 1e6

# 시각화 3: 5년 평균 생산 부족분 Top 10
top_10_avg_shortfall = report_avg_shortfall.head(10).reset_index()
plt.figure(figsize=(12, 7))
sns.barplot(data=top_10_avg_shortfall, x='Shortfall_Million_kg', y='Country_Name', palette='Reds_r')
plt.title(f'커피 생산 부족분(수요-공급 갭) Top 10 (기준: {start_year}-{end_year}년 평균)', fontsize=16)
plt.xlabel('5년 평균 생산 부족분 (백만 kg)')
plt.ylabel('국가명')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('Coffee_production_shortage_top10_5yr_avg.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [분석 1-3] 상위 5개국 생산 부족분 시계열 추이
# ----------------------------------------------------
top_5_countries = report_sorted_by_shortfall.head(5).index.tolist()
df_pivot_all_years = df_filtered.pivot_table(
    index=['Country_Name', 'Market_Year'], columns='Attribute_Description', values='Total_Value', aggfunc='sum'
).fillna(0)

df_pivot_all_years['Total_Consumption'] = df_pivot_all_years.get('Domestic Consumption', 0)
df_pivot_all_years['Total_Production'] = df_pivot_all_years.get('Production', 0)
df_pivot_all_years['Production_Shortfall_kg'] = df_pivot_all_years['Total_Consumption'] - df_pivot_all_years['Total_Production']

df_trend = df_pivot_all_years.loc[top_5_countries].reset_index()
df_trend = df_trend[df_trend['Market_Year'] > 2003]

# 시각화 4: Top 5 시계열 추이
plt.figure(figsize=(14, 7))
sns.lineplot(data=df_trend, x='Market_Year', y='Production_Shortfall_kg', hue='Country_Name', style='Country_Name', markers=True, dashes=False, linewidth=2)
plt.title('Top 5 국가별 커피 생산 부족분 (수요-공급 갭) 시계열 추이', fontsize=16)
plt.ylabel('생산 부족분 (kg)')
plt.xlabel('마케팅 연도 (Market_Year)')
plt.legend(title='Country')
plt.grid(True)
plt.savefig('Top5 Trends in Coffee Production Shortage by Country')
plt.show()

# ----------------------------------------------------
# [분석 2] 수출/수입 핵심 시장 분석
# ----------------------------------------------------
attributes_to_find = ['Exports', 'Imports']
df_filtered_trade = df[df['Attribute_Description'].isin(attributes_to_find)].copy()
df_recent_trade = df_filtered_trade[df_filtered_trade['Market_Year'] == latest_year]
df_analysis = df_recent_trade[['Country_Name', 'Attribute_Description', 'Total_Value']]

# 시각화 5: 핵심 공급처 (Top 10 수출국)
df_exports = df_analysis[df_analysis['Attribute_Description'] == 'Exports'].copy()
top_10_exporters = df_exports.sort_values(by='Total_Value', ascending=False).head(10)
top_10_exporters['Exports_Million_kg'] = top_10_exporters['Total_Value'] / 1e6

plt.figure(figsize=(12, 7))
sns.barplot(data=top_10_exporters, x='Exports_Million_kg', y='Country_Name', palette='Greens_r')
plt.title(f'커피 핵심 공급처 (Top 10 수출국) (기준: {latest_year}년)', fontsize=16)
plt.xlabel('총 수출량 (백만 kg)')
plt.ylabel('국가명')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('top_10_coffee_exporters.png', dpi=300, bbox_inches='tight')
plt.show()

# 시각화 6: 핵심 공급처 (5년 평균 Top 10 수출국)
df_exports_5yr = df[(df['Attribute_Description'] == 'Exports') & (df['Market_Year'] >= start_year) & (df['Market_Year'] <= end_year)].copy()
df_avg_exports = df_exports_5yr.groupby('Country_Name')['Total_Value'].mean().reset_index()
top_10_exporters_avg = df_avg_exports.sort_values(by='Total_Value', ascending=False).head(10)
top_10_exporters_avg['Exports_Avg_Million_kg'] = top_10_exporters_avg['Total_Value'] / 1e6

plt.figure(figsize=(12, 7))
sns.barplot(data=top_10_exporters_avg, x='Exports_Avg_Million_kg', y='Country_Name', palette='Greens_r')
plt.title(f'커피 핵심 공급처 (Top 10 수출국) (기준: {start_year}-{end_year}년 평균)', fontsize=16)
plt.xlabel('평균 총 수출량 (백만 kg)')
plt.ylabel('국가명')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('top_10_coffee_exporters_5yr_avg.png', dpi=300, bbox_inches='tight')
plt.show()

# 시각화 7: 핵심 시장 (Top 10 수입국)
df_imports = df_analysis[df_analysis['Attribute_Description'] == 'Imports'].copy()
top_10_importers = df_imports.sort_values(by='Total_Value', ascending=False).head(10)
top_10_importers['Imports_Million_kg'] = top_10_importers['Total_Value'] / 1e6

plt.figure(figsize=(12, 7))
sns.barplot(data=top_10_importers, x='Imports_Million_kg', y='Country_Name', palette='Blues_r')
plt.title(f'커피 핵심 시장 (Top 10 수입국) (기준: {latest_year}년)', fontsize=16)
plt.xlabel('총 수입량 (백만 kg)')
plt.ylabel('국가명')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('top_10_coffee_importers.png', dpi=300, bbox_inches='tight')
plt.show()

# 시각화 8: 핵심 시장 (5년 평균 수입국 Top 10)
df_imports_5yr = df[(df['Attribute_Description'] == 'Imports') & (df['Market_Year'] >= start_year) & (df['Market_Year'] <= end_year)].copy()
df_avg_imports = df_imports_5yr.groupby('Country_Name')['Total_Value'].mean().reset_index()
top_10_importers_avg = df_avg_imports.sort_values(by='Total_Value', ascending=False).head(10)
top_10_importers_avg['Imports_Avg_Million_kg'] = top_10_importers_avg['Total_Value'] / 1e6

plt.figure(figsize=(12, 7))
sns.barplot(data=top_10_importers_avg, x='Imports_Avg_Million_kg', y='Country_Name', palette='Blues_r')
plt.title(f'커피 핵심 시장 (Top 10 수입국) (기준: {start_year}-{end_year}년 평균)', fontsize=16)
plt.xlabel('5년 평균 총 수입량 (백만 kg)')
plt.ylabel('국가명')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('top_10_coffee_importers_5yr_avg.png', dpi=300, bbox_inches='tight')
plt.show()

# 시각화 9: 전 세계 커피 시장 포지셔닝 맵 (Scatter Plot)
df_pivot_trade = df_analysis.pivot_table(index='Country_Name', columns='Attribute_Description', values='Total_Value', aggfunc='sum').fillna(0)
df_pivot_trade['Exports_Million_kg'] = df_pivot_trade.get('Exports', 0) / 1e6
df_pivot_trade['Imports_Million_kg'] = df_pivot_trade.get('Imports', 0) / 1e6

plt.figure(figsize=(14, 10))
sns.scatterplot(data=df_pivot_trade, x='Exports_Million_kg', y='Imports_Million_kg', alpha=0.7)

mean_export = df_pivot_trade['Exports_Million_kg'].mean()
mean_import = df_pivot_trade['Imports_Million_kg'].mean()
plt.axvline(x=mean_export, color='grey', linestyle='--', linewidth=1)
plt.axhline(y=mean_import, color='grey', linestyle='--', linewidth=1)

top_countries_names = list(top_10_exporters.head(5)['Country_Name']) + list(top_10_importers.head(5)['Country_Name'])
df_labels = df_pivot_trade[df_pivot_trade.index.isin(top_countries_names)]

for country, row in df_labels.iterrows():
    plt.text(row['Exports_Million_kg'] + 5, row['Imports_Million_kg'] + 5, country, fontsize=9)

plt.text(df_pivot_trade['Exports_Million_kg'].max(), mean_import, "  [핵심 공급처]\n (수출↑, 수입↓)", ha='right', va='bottom', color='green', fontsize=12)
plt.text(mean_export, df_pivot_trade['Imports_Million_kg'].max(), " [핵심 시장]\n (수출↓, 수입↑)", ha='left', va='top', color='blue', fontsize=12)
plt.text(df_pivot_trade['Exports_Million_kg'].max(), df_pivot_trade['Imports_Million_kg'].max(), " [허브/재수출 국가]\n (수출↑, 수입↑)", ha='right', va='top', color='orange', fontsize=12)
plt.text(mean_export, mean_import, " [저관여 국가]\n (수출↓, 수입↓)", ha='right', va='top', color='grey', fontsize=12)

plt.title(f'전 세계 커피 시장 포지셔닝 맵 (기준: {latest_year}년)', fontsize=16)
plt.xlabel('총 수출량 (백만 kg)')
plt.ylabel('총 수입량 (백만 kg)')
plt.grid(True)
plt.savefig('coffee_market_positioning_map.png', dpi=300, bbox_inches='tight')
plt.show()

# 시각화 10: 커피 교역량 Top 20 (Stacked Bar)
df_pivot_trade['Exports_Million_kg'] = df_pivot_trade['Exports_Million_kg'].clip(lower=0)
df_pivot_trade['Imports_Million_kg'] = df_pivot_trade['Imports_Million_kg'].clip(lower=0)
df_pivot_trade['Total_Trade_Million_kg'] = df_pivot_trade['Exports_Million_kg'] + df_pivot_trade['Imports_Million_kg']
df_top_trade = df_pivot_trade.sort_values(by='Total_Trade_Million_kg', ascending=False).head(20)

df_melted_trade = df_top_trade.reset_index().melt(
    id_vars='Country_Name', value_vars=['Imports_Million_kg', 'Exports_Million_kg'], var_name='Trade_Type', value_name='Volume'
)
df_melted_trade['Country_Name'] = pd.Categorical(df_melted_trade['Country_Name'], categories=df_top_trade.index, ordered=True)

plt.figure(figsize=(14, 10))
sns.barplot(data=df_melted_trade, y='Country_Name', x='Volume', hue='Trade_Type', palette={'Imports_Million_kg': 'royalblue', 'Exports_Million_kg': 'forestgreen'}, dodge=False, hue_order=['Imports_Million_kg', 'Exports_Million_kg'])
plt.title(f'커피 교역량 Top 20 국가 및 시장 포지션 (기준: {latest_year}년)', fontsize=16)
plt.xlabel('총 교역량 (백만 kg)')
plt.ylabel('국가명')
plt.legend(title='교역 유형', loc='lower right', labels=['수입량 (시장)', '수출량 (공급)'])
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('top_20_trade_stacked_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# 시각화 11: 5년 평균 커피 교역량 포지션 Top 20
df_filtered_5yr_trade = df[(df['Attribute_Description'].isin(attributes_to_find)) & (df['Market_Year'] >= start_year) & (df['Market_Year'] <= end_year)].copy()
df_avg_pivot_trade = df_filtered_5yr_trade.groupby(['Country_Name', 'Attribute_Description'])['Total_Value'].mean().unstack(fill_value=0)
df_avg_pivot_trade['Exports_Million_kg'] = df_avg_pivot_trade.get('Exports', 0) / 1e6
df_avg_pivot_trade['Imports_Million_kg'] = df_avg_pivot_trade.get('Imports', 0) / 1e6
df_avg_pivot_trade['Exports_Million_kg'] = df_avg_pivot_trade['Exports_Million_kg'].clip(lower=0)
df_avg_pivot_trade['Imports_Million_kg'] = df_avg_pivot_trade['Imports_Million_kg'].clip(lower=0)
df_avg_pivot_trade['Total_Trade_Million_kg'] = df_avg_pivot_trade['Exports_Million_kg'] + df_avg_pivot_trade['Imports_Million_kg']

df_top_trade_avg = df_avg_pivot_trade.sort_values(by='Total_Trade_Million_kg', ascending=False).head(20)
df_melted_avg = df_top_trade_avg.reset_index().melt(
    id_vars='Country_Name', value_vars=['Imports_Million_kg', 'Exports_Million_kg'], var_name='Trade_Type', value_name='Volume'
)
df_melted_avg['Country_Name'] = pd.Categorical(df_melted_avg['Country_Name'], categories=df_top_trade_avg.index, ordered=True)

plt.figure(figsize=(14, 10))
sns.barplot(data=df_melted_avg, y='Country_Name', x='Volume', hue='Trade_Type', palette={'Imports_Million_kg': 'royalblue', 'Exports_Million_kg': 'forestgreen'}, dodge=False, hue_order=['Imports_Million_kg', 'Exports_Million_kg'])
plt.title(f'커피 교역량 Top 20 국가 및 시장 포지션 (기준: {start_year}-{end_year}년 평균)', fontsize=16)
plt.xlabel('5년 평균 총 교역량 (백만 kg)')
plt.ylabel('국가명')
plt.legend(title='교역 유형', loc='lower right', labels=['수입량 (시장)', '수출량 (공급)'])
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('top_20_trade_stacked_bar_5yr_avg.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [분석 3] 순생산량 분석 (생산량 vs 소비량)
# ----------------------------------------------------
attributes_prod_cons = ['Production', 'Domestic Consumption']
df_prod_cons_all = df[df['Attribute_Description'].isin(attributes_prod_cons)].copy()
df_recent_prod_cons = df_prod_cons_all[df_prod_cons_all['Market_Year'] == latest_year]

df_pivot_pc = df_recent_prod_cons.pivot_table(index='Country_Name', columns='Attribute_Description', values='Total_Value', aggfunc='sum').fillna(0)
if 'Production' not in df_pivot_pc.columns: df_pivot_pc['Production'] = 0
if 'Domestic Consumption' not in df_pivot_pc.columns: df_pivot_pc['Domestic Consumption'] = 0

df_pivot_pc['Net_Production_kg'] = df_pivot_pc['Production'] - df_pivot_pc['Domestic Consumption']
df_pivot_pc['Net_Production_Million_kg'] = df_pivot_pc['Net_Production_kg'] / 1e6
df_pivot_pc['Status'] = np.where(df_pivot_pc['Net_Production_kg'] > 0, 'Surplus (공급 과잉)', 'Shortfall (공급 부족)')

df_sorted_pc = df_pivot_pc.sort_values(by='Net_Production_Million_kg', ascending=False)
top_10_surplus = df_sorted_pc[df_sorted_pc['Net_Production_Million_kg'] > 0].head(10)
top_10_shortfall_pc = df_sorted_pc[df_sorted_pc['Net_Production_Million_kg'] < 0].sort_values(by='Net_Production_Million_kg', ascending=True).head(10)

df_viz_pc = pd.concat([top_10_surplus, top_10_shortfall_pc]).reset_index()
palette_pc = {'Surplus (공급 과잉)': 'seagreen', 'Shortfall (공급 부족)': 'firebrick'}

# 시각화 12: 국가별 순생산량 (Diverging Bar Chart)
plt.figure(figsize=(14, 10))
sns.barplot(data=df_viz_pc, x='Net_Production_Million_kg', y='Country_Name', hue='Status', palette=palette_pc, dodge=False)
plt.title(f'국가별 순생산량 (생산량 - 소비량) Top 10 (기준: {latest_year}년)', fontsize=16)
plt.xlabel('순생산량 (백만 kg) [ +: 공급 과잉  |  -: 공급 부족 ]')
plt.ylabel('국가명')
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='유형', loc='best')
plt.savefig('net_production_surplus_shortfall_top10.png', dpi=300, bbox_inches='tight')
plt.show()

# 시각화 13: 5년 평균 순생산량 Top 10
df_filtered_5yr_pc = df_prod_cons_all[(df_prod_cons_all['Market_Year'] >= start_year) & (df_prod_cons_all['Market_Year'] <= end_year)].copy()
df_avg_pivot_pc = df_filtered_5yr_pc.groupby(['Country_Name', 'Attribute_Description'])['Total_Value'].mean().unstack(fill_value=0)
if 'Production' not in df_avg_pivot_pc.columns: df_avg_pivot_pc['Production'] = 0
if 'Domestic Consumption' not in df_avg_pivot_pc.columns: df_avg_pivot_pc['Domestic Consumption'] = 0

df_avg_pivot_pc['Avg_Net_Production_kg'] = df_avg_pivot_pc['Production'] - df_avg_pivot_pc['Domestic Consumption']
df_avg_pivot_pc['Avg_Net_Production_Million_kg'] = df_avg_pivot_pc['Avg_Net_Production_kg'] / 1e6
df_avg_pivot_pc['Status'] = np.where(df_avg_pivot_pc['Avg_Net_Production_kg'] > 0, 'Surplus (공급 과잉)', 'Shortfall (공급 부족)')

df_sorted_avg_pc = df_avg_pivot_pc.sort_values(by='Avg_Net_Production_Million_kg', ascending=False)
top_10_surplus_avg = df_sorted_avg_pc[df_sorted_avg_pc['Avg_Net_Production_Million_kg'] > 0].head(10)
top_10_shortfall_avg = df_sorted_avg_pc[df_sorted_avg_pc['Avg_Net_Production_Million_kg'] < 0].sort_values(by='Avg_Net_Production_Million_kg', ascending=True).head(10)

df_viz_avg_pc = pd.concat([top_10_surplus_avg, top_10_shortfall_avg]).reset_index()
plt.figure(figsize=(14, 10))
sns.barplot(data=df_viz_avg_pc, x='Avg_Net_Production_Million_kg', y='Country_Name', hue='Status', palette=palette_pc, dodge=False)
plt.title(f'국가별 순생산량 Top 10 (기준: {start_year}-{end_year}년 평균)', fontsize=16)
plt.xlabel('5년 평균 순생산량 (백만 kg) [ +: 공급 과잉  |  -: 공급 부족 ]')
plt.ylabel('국가명')
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='유형', loc='best')
plt.savefig('net_production_surplus_shortfall_top10_5yr_avg.png', dpi=300, bbox_inches='tight')
plt.show()

# 시각화 14: 국가별 생산량 vs 소비량 관계도 (Log Scale Scatter Plot)
df_plot_scatter = df_pivot_pc[(df_pivot_pc['Production'] > 1e6) & (df_pivot_pc['Domestic Consumption'] > 1e6)].copy()
df_plot_scatter['Production_Million_kg'] = df_plot_scatter['Production'] / 1e6
df_plot_scatter['Consumption_Million_kg'] = df_plot_scatter['Domestic Consumption'] / 1e6

plt.figure(figsize=(14, 14))
sns.scatterplot(data=df_plot_scatter, x='Production_Million_kg', y='Consumption_Million_kg', hue='Status', palette=palette_pc, alpha=0.7, s=100)

min_val = min(df_plot_scatter['Production_Million_kg'].min(), df_plot_scatter['Consumption_Million_kg'].min())
max_val = max(df_plot_scatter['Production_Million_kg'].max(), df_plot_scatter['Consumption_Million_kg'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--', label='생산=소비 (자급자족선)')
plt.xscale('log')
plt.yscale('log')

countries_to_label = df_viz_pc['Country_Name'].tolist() 
for country, row in df_plot_scatter.iterrows():
    if country in countries_to_label:
        label_color, label_fontsize, label_fontweight = 'darkred', 9, 'bold'
    else:
        label_color, label_fontsize, label_fontweight = 'grey', 7, 'normal'
    plt.text(x=row['Production_Million_kg'], y=row['Consumption_Million_kg'], s=country, fontsize=label_fontsize, color=label_color, fontweight=label_fontweight, alpha=0.7)

plt.title(f'국가별 생산량 vs 소비량 관계도 (Log Scale) (기준: {latest_year}년)', fontsize=16)
plt.xlabel('총 생산량 (백만 kg) [Log Scale]')
plt.ylabel('국내 소비량 (백만 kg) [Log Scale]')
plt.legend(title='포지션')
plt.grid(True, which="both", ls="--", linewidth=0.5) 
plt.savefig('production_vs_consumption_scatterplot_all_labels.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [분석 4] 주요 국가 시계열 추이 (lmplot)
# ----------------------------------------------------
key_countries_trend = ['United States', 'Vietnam', 'Brazil', 'Japan', 'European Union', 'China', 'Colombia', 'Switzerland']
attributes_trend = ['Exports', 'Imports', 'Domestic Consumption']
df_filtered_trend = df[(df['Country_Name'].isin(key_countries_trend)) & (df['Attribute_Description'].isin(attributes_trend))].copy()

df_pivot_trend = df_filtered_trend.pivot_table(index=['Country_Name', 'Market_Year'], columns='Attribute_Description', values='Total_Value', aggfunc='sum').fillna(0)
df_pivot_trend = df_pivot_trend / 1e6
df_melted_trend = df_pivot_trend.reset_index().melt(id_vars=['Country_Name', 'Market_Year'], var_name='Attribute', value_name='Value_Million_kg')

# 시각화 15: 주요 국가별 수출/수입/소비 시계열 추이
g = sns.lmplot(data=df_melted_trend, x='Market_Year', y='Value_Million_kg', hue='Country_Name', col='Attribute', col_wrap=1, height=6, aspect=2, sharey=False, ci=None, scatter_kws={'s': 50, 'alpha': 0.7}, line_kws={'linewidth': 3})
g.fig.suptitle('주요 국가별 커피 수출/수입/소비 시계열 추이 분석', y=1.03, fontsize=20)
g.set_axis_labels('마케팅 연도 (Market_Year)', '수량 (백만 kg)')
g.set_titles(col_template="{col_name} Trends")
plt.savefig('key_countries_trends_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [분석 5] 커피 종류별 포트폴리오 (아라비카 vs 로부스타 등)
# ----------------------------------------------------
top_producer_names = top_10_exporters['Country_Name'].tolist()
attributes_portfolio = ['Arabica Production', 'Robusta Production', 'Other Production']
df_filtered_port = df[(df['Country_Name'].isin(top_producer_names)) & (df['Market_Year'] == latest_year) & (df['Attribute_Description'].isin(attributes_portfolio))]

df_pivot_port = df_filtered_port.pivot_table(index='Country_Name', columns='Attribute_Description', values='Total_Value', aggfunc='sum').fillna(0)
df_pivot_port['Total_Production'] = df_pivot_port.sum(axis=1)
df_pivot_port = df_pivot_port[df_pivot_port['Total_Production'] > 0] 

df_pivot_port['Arabica (%)'] = (df_pivot_port['Arabica Production'] / df_pivot_port['Total_Production']) * 100
df_pivot_port['Robusta (%)'] = (df_pivot_port['Robusta Production'] / df_pivot_port['Total_Production']) * 100
df_pivot_port['Other (%)'] = (df_pivot_port['Other Production'] / df_pivot_port['Total_Production']) * 100
df_plot_port = df_pivot_port[['Arabica (%)', 'Robusta (%)', 'Other (%)']].sort_values(by='Arabica (%)', ascending=False)

# 시각화 16: 주요 커피 공급처별 생산 포트폴리오
plt.figure(figsize=(12, 8))
df_plot_port.plot(kind='barh', stacked=True, figsize=(12, 10), colormap='coolwarm_r')
plt.title(f'주요 커피 공급처별 생산 포트폴리오 (아라비카 vs 로부스타) (기준: {latest_year}년)', fontsize=16)
plt.xlabel('생산 비중 (%)')
plt.ylabel('국가명')
plt.legend(title='품종', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('production_portfolio_arabica_vs_robusta.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [분석 6] 생두 vs 가공품 교역 행태 (Value Chain)
# ----------------------------------------------------
key_countries_vc = ['United States', 'Japan', 'China', 'Vietnam', 'Brazil', 'Colombia', 'European Union', 'Switzerland']
attributes_vc = ['Bean Imports', 'Roast & Ground Imports', 'Bean Exports', 'Roast & Ground Exports']
df_filtered_vc = df[(df['Country_Name'].isin(key_countries_vc)) & (df['Attribute_Description'].isin(attributes_vc))]

df_pivot_vc = df_filtered_vc.pivot_table(index=['Country_Name', 'Market_Year'], columns='Attribute_Description', values='Total_Value', aggfunc='sum').fillna(0)
df_pivot_vc = df_pivot_vc / 1e6
df_melted_vc = df_pivot_vc.reset_index().melt(id_vars=['Country_Name', 'Market_Year'], var_name='Trade_Type', value_name='Value_Million_kg')

# 시각화 17: 수입 가치사슬 분석
df_imports_viz = df_melted_vc[df_melted_vc['Trade_Type'].isin(['Bean Imports', 'Roast & Ground Imports'])]
g_imports = sns.relplot(data=df_imports_viz, x='Market_Year', y='Value_Million_kg', hue='Trade_Type', col='Country_Name', kind='line', col_wrap=3, height=4, aspect=1.5, palette={'Bean Imports':'saddlebrown', 'Roast & Ground Imports':'darkorange'}, linewidth=2.5, facet_kws={'sharey': False})
g_imports.fig.suptitle('국가별 커피 수입 유형 시계열 분석 (생두 vs 가공품)', y=1.03, fontsize=16)
g_imports.set_axis_labels('마케팅 연도', '수입량 (백만 kg)')
plt.savefig('imports_value_chain_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 시각화 18: 수출 가치사슬 분석
df_exports_viz = df_melted_vc[df_melted_vc['Trade_Type'].isin(['Bean Exports', 'Roast & Ground Exports'])]
g_exports = sns.relplot(data=df_exports_viz, x='Market_Year', y='Value_Million_kg', hue='Trade_Type', col='Country_Name', kind='line', col_wrap=3, height=4, aspect=1.5, palette={'Bean Exports':'darkgreen', 'Roast & Ground Exports':'lime'}, linewidth=2.5, facet_kws={'sharey': False})
g_exports.fig.suptitle('국가별 커피 수출 유형 시계열 분석 (생두 vs 가공품)', y=1.03, fontsize=16)
g_exports.set_axis_labels('마케팅 연도', '수출량 (백만 kg)')
plt.savefig('exports_value_chain_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [분석 7] 소비 행태 (원두 vs 인스턴트)
# ----------------------------------------------------
top_consumer_names = top_10_importers['Country_Name'].tolist()
attributes_cons_type = ['Rst,Ground Dom. Consum', 'Soluble Dom. Cons.']
df_filtered_cons = df[(df['Country_Name'].isin(top_consumer_names)) & (df['Market_Year'] == latest_year) & (df['Attribute_Description'].isin(attributes_cons_type))]

df_pivot_cons = df_filtered_cons.pivot_table(index='Country_Name', columns='Attribute_Description', values='Total_Value', aggfunc='sum').fillna(0)
df_pivot_cons['Total_Consumption'] = df_pivot_cons.sum(axis=1)
df_pivot_cons = df_pivot_cons[df_pivot_cons['Total_Consumption'] > 0] 

df_pivot_cons['Roasted (%)'] = (df_pivot_cons['Rst,Ground Dom. Consum'] / df_pivot_cons['Total_Consumption']) * 100
df_pivot_cons['Soluble (%)'] = (df_pivot_cons['Soluble Dom. Cons.'] / df_pivot_cons['Total_Consumption']) * 100
df_plot_cons = df_pivot_cons[['Roasted (%)', 'Soluble (%)']].sort_values(by='Roasted (%)', ascending=False)

# 시각화 19: 주요 커피 시장별 소비 행태
plt.figure(figsize=(12, 8))
df_plot_cons.plot(kind='barh', stacked=True, figsize=(12, 10), colormap='copper')
plt.title(f'주요 커피 시장별 소비 행태 (원두 vs 인스턴트) (기준: {latest_year}년)', fontsize=16)
plt.xlabel('소비 비중 (%)')
plt.ylabel('국가명')
plt.legend(title='소비 유형', labels=['로스팅/분쇄 (원두)', '인스턴트 (용해성)'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('consumption_behavior_roasted_vs_soluble.png', dpi=300, bbox_inches='tight') 
plt.show()

# 시각화 20: 소비 행태 5년 평균
df_filtered_C = df[(df['Attribute_Description'].isin(attributes_cons_type)) & (df['Market_Year'] >= start_year) & (df['Market_Year'] <= end_year)].copy()
df_avg_pivot_C = df_filtered_C.groupby(['Country_Name', 'Attribute_Description'])['Total_Value'].mean().unstack(fill_value=0)
df_avg_pivot_C['Total_Consumption'] = df_avg_pivot_C['Rst,Ground Dom. Consum'] + df_avg_pivot_C['Soluble Dom. Cons.']
df_avg_pivot_C = df_avg_pivot_C[df_avg_pivot_C['Total_Consumption'] > 0]
df_avg_pivot_C['Roasted (%)'] = (df_avg_pivot_C['Rst,Ground Dom. Consum'] / df_avg_pivot_C['Total_Consumption']) * 100
df_avg_pivot_C['Soluble (%)'] = (df_avg_pivot_C['Soluble Dom. Cons.'] / df_avg_pivot_C['Total_Consumption']) * 100
df_plot_C = df_avg_pivot_C.sort_values(by='Total_Consumption', ascending=False).head(10)
df_plot_C = df_plot_C[['Roasted (%)', 'Soluble (%)']].sort_values(by='Roasted (%)', ascending=False)

plt.figure(figsize=(12, 8))
df_plot_C.plot(kind='barh', stacked=True, figsize=(12, 10), colormap='copper')
plt.title(f'주요 커피 시장별 소비 행태 (기준: {start_year}-{end_year}년 평균)', fontsize=16)
plt.xlabel('5년 평균 소비 비중 (%)')
plt.ylabel('국가명')
plt.legend(title='소비 유형', labels=['로스팅/분쇄 (원두)', '인스턴트 (용해성)'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('consumption_behavior_roasted_vs_soluble_5yr_avg.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [분석 8] 주요 시장 재고 건전성 분석
# ----------------------------------------------------
key_countries_stock = ['United States', 'Japan', 'China', 'European Union', 'Canada', 'Korea, South']
attributes_stock = ['Domestic Consumption', 'Ending Stocks']
df_filtered_stock = df[(df['Country_Name'].isin(key_countries_stock)) & (df['Attribute_Description'].isin(attributes_stock))].copy()

df_pivot_stock = df_filtered_stock.pivot_table(index=['Country_Name', 'Market_Year'], columns='Attribute_Description', values='Total_Value', aggfunc='sum').fillna(0)
df_pivot_stock = df_pivot_stock[(df_pivot_stock['Domestic Consumption'] > 0) & (df_pivot_stock['Ending Stocks'] >= 0)]
df_pivot_stock['Stock_to_Consumption_Ratio'] = df_pivot_stock['Ending Stocks'] / df_pivot_stock['Domestic Consumption']
df_melted_stock = df_pivot_stock.reset_index().melt(id_vars=['Country_Name', 'Market_Year'], value_vars=['Stock_to_Consumption_Ratio'], value_name='Ratio')

# 시각화 21: 재고 건전성 추이
g_stock = sns.lmplot(data=df_melted_stock, x='Market_Year', y='Ratio', hue='Country_Name', height=7, aspect=1.8, ci=None, scatter_kws={'s': 30, 'alpha': 0.7}, line_kws={'linewidth': 2.5})
plt.title('주요 시장별 커피 재고 건전성 추이 (소비 대비 기말 재고 비율)', fontsize=16)
plt.xlabel('마케팅 연도 (Market_Year)')
plt.ylabel('재고 비율 (1.0 = 1년치 소비량)')
plt.axhline(y=0.167, color='red', linestyle='--', linewidth=1.5, label='안정 재고선 (약 2개월치)')
plt.legend(loc='best')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig('stock_to_consumption_ratio_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [분석 9] 데이터 공표 지연 시간 (Data Lag) 분석
# ----------------------------------------------------
df_lag_analysis = df[['Country_Name', 'Market_Year', 'Calendar_Year']].copy()
df_lag_analysis['Data_Lag_Years'] = df_lag_analysis['Calendar_Year'] - df_lag_analysis['Market_Year']
df_lag_analysis['Data_Lag_Years'] = df_lag_analysis['Data_Lag_Years'].clip(lower=0)

df_avg_lag = df_lag_analysis.groupby('Country_Name')['Data_Lag_Years'].mean().reset_index()
df_top_lag = df_avg_lag.sort_values(by='Data_Lag_Years', ascending=False).head(20)

# 시각화 22: 지연 시간 Top 20
plt.figure(figsize=(12, 10))
sns.barplot(data=df_top_lag, x='Data_Lag_Years', y='Country_Name', palette='Reds_r')
plt.title('데이터 공표 평균 지연 시간 Top 20 (지연 시간이 긴 국가)', fontsize=16)
plt.xlabel('평균 지연 시간 (년)')
plt.ylabel('국가명')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('data_lag_analysis_top20.png', dpi=300, bbox_inches='tight')
plt.show()

df_avg_lag_sorted = df_avg_lag.sort_values(by='Data_Lag_Years', ascending=True)
print("--- [전체 국가별 데이터 공표 평균 지연 시간 (신뢰도 높은 순)] ---")
print(df_avg_lag_sorted)

# ----------------------------------------------------
# [분석 10] 글로벌 수출 시장 점유율 (M/S) 분석
# ----------------------------------------------------
df_market_data = df[df['Attribute_Description'].isin(['Exports', 'Imports'])]
df_global_totals = df_market_data.groupby(['Market_Year', 'Attribute_Description'])['Total_Value'].sum().unstack()
df_global_totals.columns = ['Global_Exports', 'Global_Imports']
df_global_totals = (df_global_totals / 1e6).reset_index()

df_for_pivot_ms = df[df['Attribute_Description'].isin(['Exports', 'Imports'])]
df_pivot_ms = df_for_pivot_ms.pivot_table(index=['Country_Name', 'Market_Year'], columns='Attribute_Description', values='Total_Value', aggfunc='sum').fillna(0)
df_pivot_ms = (df_pivot_ms / 1e6)

if 'Exports' not in df_pivot_ms.columns: df_pivot_ms['Exports'] = 0
if 'Imports' not in df_pivot_ms.columns: df_pivot_ms['Imports'] = 0

df_merged_ms = pd.merge(df_pivot_ms.reset_index(), df_global_totals, on='Market_Year')
key_exporters_ms = ['Vietnam', 'Brazil', 'Colombia', 'Indonesia']
df_ms = df_merged_ms[df_merged_ms['Country_Name'].isin(key_exporters_ms)].copy()
df_ms = df_ms[df_ms['Global_Exports'] > 0] 
df_ms['Export_MS (%)'] = (df_ms['Exports'] / df_ms['Global_Exports']) * 100

# 시각화 23: 점유율 추이
plt.figure(figsize=(14, 7))
sns.lineplot(data=df_ms, x='Market_Year', y='Export_MS (%)', hue='Country_Name', linewidth=2.5, style='Country_Name', markers=True)
plt.title('주요 공급처별 글로벌 수출 시장 점유율(M/S) 추이', fontsize=16)
plt.ylabel('수출 시장 점유율 (%)')
plt.xlabel('마케팅 연도 (Market_Year)')
plt.legend(title='Country')
plt.grid(True)
plt.savefig('export_market_share_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [분석 11] 공급 안정성(변동성 계수, CV) 정량 분석
# ----------------------------------------------------
df_exports_data_vol = df[df['Attribute_Description'] == 'Exports'].copy()
df_pivot_vol = df_exports_data_vol.pivot_table(index=['Country_Name', 'Market_Year'], values='Total_Value', aggfunc='sum').reset_index()
df_pivot_vol.rename(columns={'Total_Value': 'Exports'}, inplace=True)
df_pivot_vol['Exports'] = df_pivot_vol['Exports'] / 1e6

key_exporters_cv = ['Vietnam', 'Brazil', 'Colombia', 'Indonesia', 'Ethiopia', 'Honduras']
df_exports_ts = df_pivot_vol[(df_pivot_vol['Country_Name'].isin(key_exporters_cv)) & (df_pivot_vol['Market_Year'] >= 2005)]
df_stats = df_exports_ts.groupby('Country_Name')['Exports'].agg(['mean', 'std']).reset_index()
df_stats = df_stats[df_stats['mean'] > 0]
df_stats['CV (Volatility)'] = (df_stats['std'] / df_stats['mean'])
df_cv_sorted = df_stats.sort_values(by='CV (Volatility)', ascending=True)

# 시각화 24: 변동성 계수
plt.figure(figsize=(10, 6))
sns.barplot(data=df_cv_sorted, x='CV (Volatility)', y='Country_Name', palette='viridis_r')
plt.title('주요 공급처별 수출 안정성 (변동성 계수, 2005-2025년)', fontsize=16)
plt.xlabel('변동성 계수 (CV) - 낮을수록 안정적')
plt.ylabel('국가명')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('supplier_volatility_cv.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [분석 12] 공급 잠재력 (수출 여력) 분석
# ----------------------------------------------------
df_pot_data = df[df['Attribute_Description'].isin(['Production', 'Domestic Consumption'])].copy()
df_pivot_pot = df_pot_data.pivot_table(index=['Country_Name', 'Market_Year'], columns='Attribute_Description', values='Total_Value', aggfunc='sum').fillna(0)
df_pivot_pot = (df_pivot_pot / 1e6).reset_index()

key_producers_pot = ['Brazil', 'Vietnam', 'Colombia', 'Indonesia', 'Ethiopia', 'India']
df_potentials = df_pivot_pot[df_pivot_pot['Country_Name'].isin(key_producers_pot)].copy()
df_potentials['Exportable_Surplus'] = df_potentials['Production'] - df_potentials['Domestic Consumption']
df_potentials['Exportable_Surplus'] = df_potentials['Exportable_Surplus'].clip(lower=0)

# 시각화 25: 수출 잠재력 추이
g_pot = sns.lmplot(data=df_potentials, x='Market_Year', y='Exportable_Surplus', hue='Country_Name', height=7, aspect=1.8, ci=None, scatter_kws={'s': 30, 'alpha': 0.7}, line_kws={'linewidth': 2.5}, facet_kws={'sharey': False})
plt.title('주요 공급처별 수출 잠재력(생산량-소비량) 추이', fontsize=16)
plt.xlabel('마케팅 연도 (Market_Year)')
plt.ylabel('수출 가능 물량 (백만 kg)')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig('exportable_surplus_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [ML 분석 1] 시장 세분화 (K-Means Clustering)
# ----------------------------------------------------
df_recent_pivot = df[df['Market_Year'] == latest_year].pivot_table(index='Country_Name', columns='Attribute_Description', values='Total_Value', aggfunc='sum').fillna(0)
df_features = pd.DataFrame(index=df_recent_pivot.index)
df_features['Production_Vol'] = df_recent_pivot.get('Production', 0)
df_features['Consumption_Vol'] = df_recent_pivot.get('Domestic Consumption', 0)
df_features['Imports_Vol'] = df_recent_pivot.get('Imports', 0)
df_features['Ending_Stocks'] = df_recent_pivot.get('Ending Stocks', 0)
df_features['Stock_Ratio'] = (df_features['Ending_Stocks'] / (df_features['Consumption_Vol'] + 1e-6)) 

df_exports_pivot = df[(df['Attribute_Description'] == 'Exports') & (df['Market_Year'] >= 2005)].pivot_table(index='Country_Name', columns='Market_Year', values='Total_Value').fillna(0)
df_cv_ml = pd.DataFrame(index=df_exports_pivot.index)
df_cv_ml['Export_Mean'] = df_exports_pivot.mean(axis=1)
df_cv_ml['Export_Std'] = df_exports_pivot.std(axis=1)
df_features['Export_CV'] = (df_cv_ml['Export_Std'] / (df_cv_ml['Export_Mean'] + 1e-6))

recent_5_year = latest_year - 5
df_imports_pivot = df[(df['Attribute_Description'] == 'Imports') & (df['Market_Year'].isin([latest_year, recent_5_year]))].pivot_table(index='Country_Name', columns='Market_Year', values='Total_Value', aggfunc='sum').fillna(0)
df_cagr = pd.DataFrame(index=df_imports_pivot.index)
df_cagr['Start_Value'] = df_imports_pivot.get(recent_5_year, 0) + 1e-6
df_cagr['End_Value'] = df_imports_pivot.get(latest_year, 0)
df_features['Import_CAGR_5Y'] = ((df_cagr['End_Value'] / df_cagr['Start_Value']) ** (1/5)) - 1

df_features.replace([np.inf, -np.inf], 0, inplace=True)
df_features.fillna(0, inplace=True)
features_to_cluster = ['Production_Vol', 'Consumption_Vol', 'Imports_Vol', 'Stock_Ratio', 'Export_CV', 'Import_CAGR_5Y']
df_features_final = df_features[features_to_cluster]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_features_final)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10) 
df_features_final['Cluster'] = kmeans.fit_predict(features_scaled)

pca = PCA(n_components=2)
pca_results = pca.fit_transform(features_scaled)

# 시각화 26: K-Means 클러스터링
plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=df_features_final['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.title('국가 시장 세분화 (K-Means Clustering via PCA)', fontsize=16)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig('market_segmentation_cluster.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [분석 13] 공급망 상관관계 히트맵
# ----------------------------------------------------
df_pot_data_hm = df[df['Attribute_Description'].isin(['Production', 'Domestic Consumption'])].copy()
df_pivot_pot_hm = df_pot_data_hm.pivot_table(index=['Country_Name', 'Market_Year'], columns='Attribute_Description', values='Total_Value', aggfunc='sum').fillna(0)
df_pivot_pot_hm['Exportable_Surplus'] = (df_pivot_pot_hm['Production'] - df_pivot_pot_hm['Domestic Consumption']).clip(lower=0)

key_exporters_corr = ['Brazil', 'Vietnam', 'Colombia', 'Indonesia', 'Ethiopia', 'Honduras']
df_corr_pivot = df_pivot_pot_hm[df_pivot_pot_hm.index.get_level_values('Country_Name').isin(key_exporters_corr)]
df_corr = df_corr_pivot.reset_index().pivot(index='Market_Year', columns='Country_Name', values='Exportable_Surplus')
df_corr_recent = df_corr[df_corr.index >= 2005].fillna(0)
corr_matrix = df_corr_recent.corr()

# 시각화 27: 상관관계 히트맵
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('주요 공급처 간 수출 잠재력 상관관계 (2005-2025년)', fontsize=16)
plt.savefig('supplier_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [ML 분석 2] 이상 징후 잔차 분석 (선형 회귀)
# ----------------------------------------------------
df_residuals = pd.DataFrame()
model = LinearRegression()

for country in df_ms['Country_Name'].unique():
    country_data = df_ms[df_ms['Country_Name'] == country].copy()
    X = country_data[['Market_Year']]
    y = country_data['Export_MS (%)']
    model.fit(X, y)
    country_data['Predicted_MS'] = model.predict(X)
    country_data['Residual'] = country_data['Export_MS (%)'] - country_data['Predicted_MS']
    df_residuals = pd.concat([df_residuals, country_data])

# 시각화 28: 잔차 분석 FacetGrid
g_resid = sns.FacetGrid(df_residuals, col="Country_Name", col_wrap=2, height=4, aspect=2, sharey=False)
g_resid.map(sns.lineplot, "Market_Year", "Residual", color='red', marker='o')
g_resid.map(plt.axhline, y=0, color='black', linestyle='--')
g_resid.fig.suptitle('시장 점유율(M/S) 잔차 분석 (이상 징후 탐지)', y=1.03, fontsize=16)
g_resid.set_axis_labels('마케팅 연도', '잔차 (실제 - 예측)')
plt.savefig('market_share_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# [분석 14] 신흥 블루오션 탐색 (최근 5년 CAGR, 버블차트)
# ----------------------------------------------------
df_imports_data_cagr = df[df['Attribute_Description'] == 'Imports'].copy()
df_cagr_calc_b = df_imports_data_cagr[df_imports_data_cagr['Market_Year'].isin([start_year, latest_year])]
df_cagr_pivot_b = df_cagr_calc_b.pivot_table(index='Country_Name', columns='Market_Year', values='Total_Value', aggfunc='sum').fillna(0)

if start_year not in df_cagr_pivot_b.columns: df_cagr_pivot_b[start_year] = 0
if latest_year not in df_cagr_pivot_b.columns: df_cagr_pivot_b[latest_year] = 0
df_cagr_pivot_b = df_cagr_pivot_b[(df_cagr_pivot_b[start_year] > 1000000)]
df_cagr_pivot_b = df_cagr_pivot_b[(df_cagr_pivot_b[latest_year] > df_cagr_pivot_b[start_year])]

df_cagr_pivot_b['CAGR_5Y'] = ((df_cagr_pivot_b[latest_year] / (df_cagr_pivot_b[start_year] + 1e-6)) ** (1/5)) - 1
df_cagr_pivot_b['Total_Imports_kg'] = df_cagr_pivot_b[latest_year]

# 시각화 29: 신흥 시장 탐색 (최신 연도 기준)
df_plot_b = df_cagr_pivot_b.reset_index()
df_plot_filtered_b = df_plot_b[(~df_plot_b['Country_Name'].isin(['United States', 'China', 'European Union']))].sort_values(by='CAGR_5Y', ascending=False).head(30)

plt.figure(figsize=(14, 9))
sns.scatterplot(data=df_plot_filtered_b, x='Total_Imports_kg', y='CAGR_5Y', size='Total_Imports_kg', sizes=(100, 2000), hue='Country_Name', legend=False, alpha=0.7)
for i, row in df_plot_filtered_b.iterrows():
    plt.text(row['Total_Imports_kg'], row['CAGR_5Y'], row['Country_Name'], fontsize=9)
plt.title('신흥 시장 탐색 (시장 규모 vs 성장 속도)', fontsize=16)
plt.xlabel('총 수입량 (kg) - 시장 규모')
plt.ylabel('최근 5년 연평균 성장률 (CAGR) - 성장 속도')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.xscale('log')
plt.savefig('emerging_markets_bubble_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# 시각화 30: 신흥 시장 탐색 (5년 평균 시장 규모 기준)
df_imports_5yr_avg_b = df_imports_data_cagr[(df_imports_data_cagr['Market_Year'] >= start_year) & (df_imports_data_cagr['Market_Year'] <= end_year)].groupby('Country_Name')['Total_Value'].mean().reset_index()
df_imports_5yr_avg_b.rename(columns={'Total_Value': 'Avg_Total_Imports_kg'}, inplace=True)
df_plot_data_b = pd.merge(df_cagr_pivot_b.reset_index(), df_imports_5yr_avg_b, on='Country_Name')
df_plot_filtered_b2 = df_plot_data_b.sort_values(by='CAGR_5Y', ascending=False).head(30)

plt.figure(figsize=(14, 9))
sns.scatterplot(data=df_plot_filtered_b2, x='Avg_Total_Imports_kg', y='CAGR_5Y', size='Avg_Total_Imports_kg', sizes=(100, 2000), hue='Country_Name', legend=False, alpha=0.7)
for i, row in df_plot_filtered_b2.iterrows():
    plt.text(row['Avg_Total_Imports_kg'], row['CAGR_5Y'], row['Country_Name'], fontsize=9)
plt.title('신흥 시장 탐색 (5년 평균 시장 규모 vs 5년 성장 속도)', fontsize=16)
plt.xlabel('5년 평균 총 수입량 (kg) - 시장 규모 (Log Scale)')
plt.ylabel('최근 5년 연평균 성장률 (CAGR) - 성장 속도')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.xscale('log')
plt.savefig('emerging_markets_bubble_chart_5yr_avg.png', dpi=300, bbox_inches='tight')
plt.show()