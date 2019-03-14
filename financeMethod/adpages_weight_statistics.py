import  pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

conn = pymysql.Connect(host='192.168.101.70', port=3306, user='scm', passwd='dugoohoo', db='adm_files',
                       charset='utf8')
cursor = conn.cursor()
sql_part = "select * from compare_statistic_df"
df_adpages_part = pd.read_sql(sql=sql_part, con=conn)
sql_except_part = "select * from compare_statistic_except_df"
df_adpages_except_part = pd.read_sql(sql=sql_except_part, con=conn)

sql_part_1 = "select * from compare_statistic_df_1"
df_adpages_part_1 = pd.read_sql(sql=sql_part_1, con=conn)
sql_except_part_1 = "select * from compare_statistic_except_df_1"
df_adpages_except_part_1 = pd.read_sql(sql=sql_except_part_1, con=conn)

cursor.close()

df_adpages_stat = df_adpages_part.append(df_adpages_part_1)
df_adpages_stat.set_index('campaign_id',inplace=True)
df_adpages_stat = df_adpages_stat[~df_adpages_stat.index.duplicated()].sort_values(by='diff_cum_crwa_crrwo',ascending=False)[['equal_rate','diff_wl_rate','diff_cum_crwa_crrwo']]
df_adpages_stat.dropna()

df_adpages_stat.dropna()['diff_wl_rate'].describe()
df_adpages_stat.dropna()['diff_wl_rate'].hist(bins=150)
stats.ttest_1samp(df_adpages_stat.dropna()['diff_wl_rate'],0.0)

df_adpages_stat.dropna()['diff_cum_crwa_crrwo'].describe()
df_adpages_stat.dropna()['diff_cum_crwa_crrwo'].sum()
df_adpages_stat.dropna()['diff_cum_crwa_crrwo'].hist(bins=60)
df_adpages_stat.dropna()['diff_cum_crwa_crrwo']
stats.ttest_1samp(df_adpages_stat.dropna()['diff_cum_crwa_crrwo'],0.0)


df_adpages_stat_nonna = df_adpages_stat.dropna()
len(df_adpages_stat_nonna[df_adpages_stat_nonna['diff_cum_crwa_crrwo']>0]) / len(df_adpages_stat_nonna)
len(df_adpages_stat_nonna[df_adpages_stat_nonna['diff_wl_rate']>0]) / len(df_adpages_stat_nonna)

##################################################################################################################
df_adpages_except_stat = df_adpages_except_part.append(df_adpages_except_part_1)
df_adpages_except_stat.set_index('campaign_id',inplace=True)
df_adpages_except_stat = df_adpages_except_stat[~df_adpages_except_stat.index.duplicated()].sort_values(by='diff_cum_crwa_crrwo',ascending=False)[['equal_rate','diff_wl_rate','diff_cum_crwa_crrwo']]
df_adpages_except_stat

df_adpages_except_stat.dropna()['diff_wl_rate'].describe()
df_adpages_except_stat.dropna()['diff_wl_rate'].hist(bins=150)
stats.ttest_1samp(df_adpages_except_stat.dropna()['diff_wl_rate'],0.0)

df_adpages_except_stat.dropna()['diff_cum_crwa_crrwo'].describe()
df_adpages_except_stat.dropna()['diff_cum_crwa_crrwo'].sum()
df_adpages_except_stat.dropna()['diff_cum_crwa_crrwo'].hist(bins=60)
df_adpages_except_stat.dropna()['diff_cum_crwa_crrwo']
stats.ttest_1samp(df_adpages_except_stat.dropna()['diff_cum_crwa_crrwo'],0.0)

df_adpages_except_stat_nonna = df_adpages_except_stat.dropna()
len(df_adpages_except_stat_nonna[df_adpages_except_stat_nonna['diff_cum_crwa_crrwo']>0]) / len(df_adpages_except_stat_nonna)
len(df_adpages_except_stat_nonna[df_adpages_except_stat_nonna['diff_wl_rate']>0]) / len(df_adpages_except_stat_nonna)


