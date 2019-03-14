import rqalpha
from rqalpha.api import *
import os

from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
plt.rcParams['axes.unicode_minus'] = False

from rqdbs import rqdbs
from datetime import datetime
# from datetime import date, timedelta
ods = rqdbs()
pds = rqdbs("e:/data/rqalpha")

szzz = pds.get_price('000001.XSHG', start = '2005-01-04' , end = '2018-07-22',fields =['high','open','low','close','volume'])
# print(szzz)


def z_score(series, win=0):
    if win == 0:
        zs = (series - series.mean()) / series.std()
    else:
        zs = (series - series.rolling(win).mean()) / series.rolling(win).std()
    return zs

def RSRS(df_ohlcv, n=40, m=100, threshold_up = 0,threshold_down = 0,adjusted=True):
    from sklearn.linear_model import LinearRegression
    '''
    RSRS - timing method from 银河证券
    :param n:
    :param m:
    :param adjusted:
    :return:
    '''
    betas = []
    r2s = []
    for i in range(len(df_ohlcv)):
        if i < n - 1:
            betas.append(np.NAN)
            r2s.append(np.NAN)
            continue

        Y = np.matrix(df_ohlcv['high'][i - n + 1:i + 1].values).T
        X = np.matrix(df_ohlcv['low'][i - n + 1:i + 1].values).T
        regr = LinearRegression()
        regr.fit(X, Y)
        r2 = regr.score(X, Y)
        beta = regr.coef_[0]

        betas.append(float(beta))
        r2s.append(float(r2))

        # print('beta={}: r2 = {}'.format(beta, r2))

    data = pd.DataFrame({'beta': betas, 'r2': r2s}, index=df_ohlcv.index)
    z = z_score(data['beta'],m)

    if adjusted:
        z = z * data['r2']

    signal = pd.DataFrame(1 * (z>threshold_up) + (-1) *(z<threshold_down),columns=['signal'])

    return signal

def RSRS_right(df_ohlcv, n=40, m=100, l=20,threshold_up = 0,threshold_down = 0,adjusted=True,right_adjusted=True):
    from sklearn.linear_model import LinearRegression
    import talib as ta
    '''
    RSRS - timing method from 银河证券
    :param n:
    :param m:
    :param adjusted:
    :return:
    '''
    ma20 = ta.SMA(np.array(df_ohlcv.close),l)
    ma_signal = []
    for i in range(len(df_ohlcv)):
        if i>=l+3:
            if ma20[i-1]>ma20[i-3]:
                ma_signal.append(1)
            else:
                ma_signal.append(0)
        else:
            ma_signal.append(np.NAN)
    betas = []
    r2s = []
    for i in range(len(df_ohlcv)):
        if i < n - 1:
            betas.append(np.NAN)
            r2s.append(np.NAN)
            continue

        Y = np.matrix(df_ohlcv['high'][i - n + 1:i + 1].values).T
        X = np.matrix(df_ohlcv['low'][i - n + 1:i + 1].values).T
        regr = LinearRegression()
        regr.fit(X, Y)
        r2 = regr.score(X, Y)
        beta = regr.coef_[0]

        betas.append(float(beta))
        r2s.append(float(r2))

        # print('beta={}: r2 = {}'.format(beta, r2))

    data = pd.DataFrame({'beta': betas, 'r2': r2s,'ma_signal':ma_signal}, index = df_ohlcv.index)
    z = z_score(data['beta'],m)
    ma_sig = data['ma_signal']
    if adjusted:
        z = z * data['r2']
    if right_adjusted:
        z = z * data['beta']

    signal = pd.DataFrame(1 * ((z>threshold_up)&(ma_sig==1)) + (-1) *(z<threshold_down),columns=['signal'])

    return signal

def calc_beta_range(windows, df_industry, ds_benchmark):
    '''
    windows：用多少个周期的价格计算   df_industry: 各行业指数 pandas.DataFrame     ds_benchmark: 基准指数 pandas.Series
    注：数据日期频率不固定，以输入的频率为准

    '''
    df = pd.DataFrame(np.nan, index=ds_benchmark.index[windows + 1:], columns=df_industry.columns)
    for date in df.index:
        beta_ds = pd.Series(np.nan, index=df_industry.columns)
        for symbol in df_industry.columns:
            date = pd.to_datetime(date)
            ds = df_industry[symbol][:date][-windows - 1:]
            ds_bm1 = ds_benchmark[:date][-windows - 1:]
            ret_bm = ds_bm1.pct_change()[1:]
            ret = ds.pct_change()[1:]
            beta = np.cov(ret_bm, ret) / np.var(ret_bm)
            beta_ds[symbol] = beta[0][1]
        df.loc[date, :] = beta_ds
    return df

def rescale(serie, out_max, out_min=0, win=0):
    output_range = out_max - out_min

    if win == 0:
        values_min = serie.min()
        input_range = serie.max() - values_min
    else:
        values_min = serie.rolling(win).min()
        input_range = serie.rolling(win).max() - values_min
    values = (serie - values_min) * output_range / input_range + out_min
    return values


def percentrank(series, win=100):
    import scipy.stats as stats
    if win != 0:
        rank = series.rolling(win).apply(lambda x: stats.percentileofscore(x, x[-1]))
    else:
        rank = series.rolling(1).apply(lambda x: stats.percentileofscore(series, x))

    rank = rescale(rank, out_max=100, out_min=0, win=win)
    # rank = rank.dropna().astype(int)
    return rank


def volatility_difference(df_holcv, winpr=100, threshold_up=0, threshold_down=0):
    def walkwincal(RPS):
        walkwin = 1
        if RPS >= 0 and RPS < 10:
            walkwin = 100
        if RPS >= 10 and RPS < 40:
            walkwin = 80
        if RPS >= 40 and RPS < 60:
            walkwin = 60
        if RPS >= 60 and RPS < 90:
            walkwin = 80
        if RPS >= 90 and RPS <= 100:
            walkwin = 100
        return walkwin

    vol_diff = pd.Series(np.nan, index=df_holcv.index)
    vol_diff_m = pd.Series(0, index=df_holcv.index)
    vol_diff = (df_holcv.high + df_holcv.low) / df_holcv.open - 2

    RPSv = percentrank(df_holcv.close, winpr)

    for i in range((winpr - 1) * 2, len(vol_diff)):
        walkv = int(walkwincal(float(RPSv[i])) / 100 * winpr)
        vol_diff_m.iloc[i] = vol_diff.iloc[i - walkv + 1:i + 1].mean()

    signal = pd.DataFrame((vol_diff_m > threshold_up) * 1 + (vol_diff_m <= threshold_down) * (-1), columns=['signal'])

    return signal

def calc_up_and_down_std(close,s,N,ma_N):
    start_date = close.index[0]
    end_date = close.index[-1]
#     close = get_price(target,start_date = start_date, end_date = end_date,fields=['close'])
    df = pd.DataFrame(close, index = close.index, columns = ['close'])
    df['ret' ]= df['close'].pct_change()
    df['minus'] = pd.Series(0,index = close.index)
    df['signal'] = pd.Series(0,index = close.index)
    for date in close.index[N+1:]:
        ret_i =  df['ret'][:date][-N:]
        up = ret_i[ret_i > 0]
        down = ret_i[ret_i < 0]
        if len(up) == 0 :
            up_var = 0
        else:
            up_var = sum([(r-s)**2 for r in up])#/len(up)
        if len(down) == 0 :
            down_var = 0
        else:
            down_var = sum([(r-s)**2 for r in down])#/len(down)
#         print(up_var - down_var)
        df.loc[date,'minus'] = up_var - down_var
#     df['MA'] = pd.rolling_mean(df['minus'],ma_N)
    df['MA'] = df['minus'].rolling(ma_N).mean()
    df['signal'][df['MA']>0] = 1
    return pd.DataFrame(df['signal'])

def LLT(df_holcv, alpha=float(2 / (12 + 1)), threshold_up=0, threshold_down=0):
    Lltval = []

    n = len(df_holcv.close)
    for i in range(n):
        if i < 2:
            Lltval.append(df_holcv.close.values[i])
        else:
            newlltval = ((alpha - alpha ** 2 / float(4)) * df_holcv.close.iloc[i]
                         + alpha ** 2 / float(2) * df_holcv.close.iloc[i - 1] - (alpha - alpha ** 2 * 3 / float(4)) *
                         df_holcv.close.iloc[i - 2]
                         + 2 * (1 - alpha) * Lltval[i - 1] - (1 - alpha) ** 2 * (Lltval[i - 2]))
            Lltval.append(newlltval)
    llt = pd.Series(Lltval, index=df_holcv.close.index)
    strength = llt / llt.shift(1) - 1

    signal = pd.DataFrame((strength > threshold_up) * 1 + (strength <= threshold_down) * (-1), columns=['signal'])

    return signal

def volume_quantile(df_holcv, N=40, S=0.5):
    #index should not be default
    index = df_holcv.index
    ds = pd.Series(np.nan, index=index)
    for date in ds.index:
        end_date = date
        vol_N = df_holcv.volume[:end_date][-N - 1:]
        if len(vol_N) >= N + 1:
            rank_N = vol_N.rank()
            std_rank_N = (2 * rank_N - N - 2) / N
            ds[date] = list(std_rank_N)[-1]
        else:
            pass

    signal = pd.DataFrame((ds > S) * 1 + (ds <= S) * (-1), columns=['signal'])

    return signal

# def day_dream_sig(df_holcv,T_ini = 20,T_rolling = 5,threshold = 0.03):
#     # 窗口放缩，区间突破
#     times = int(T_ini/T_rolling)
#     df_close = pd.DataFrame(np.array(df_holcv.close),index= df_holcv.index,columns=['close'])
#     df_close['trade_if'] = 0
#     df_close['signal'] = 0
#
#     length = T_ini+T_rolling*times+2
#     for (s,value) in enumerate(df_close.index[length:]):
#         if s%2 == 0:
#             close = df_holcv.close[s:s+length]
#             Min = min(close[ -T_ini:])
#             Max = max(close[ -T_ini:])
#             MtM = Min/Max
#
#             for j in range(times):
#                 if MtM > 1 - threshold:
#                     T_ini += T_rolling
#                     Min = min(close[ -T_ini:])
#                     Max = max(close[ -T_ini:])
#                     MtM = Min/Max
#                 else:
#                     pass
#             if list(close)[-1]<= Min:
#                 trade_signal = 'sell'
#             elif list(close)[-1]>= Max*(1-threshold/2):
#                 trade_signal = 'buy'
#             else:
#                 trade_signal = 'keep'
#         else:
#             trade_signal = 'keep'
#         df_close.loc[df_close.index[s+length],'trade_if'] = trade_signal
#
#     for i in range(1,len(df_close)):
# #         print(i)
#         s = df_close.ix[df_close.index[i-1],'trade_if']
#         if   s == 0 or s == 'keep':
#             for j in reversed(range(1,i-1)):
# #             print(j)
#                 if df_close.loc[df_close.index[j],'trade_if'] == 'buy':
#                     df_close.loc[df_close.index[i],'signal'] = 1
#                     break
#                 elif df_close.loc[df_close.index[j],'trade_if'] == 'sell':
#                     df_close.loc[df_close.index[i],'signal'] =  -1
#                     break
#                 else:
#                     pass
#         elif s == 'buy':
#             df_close.loc[df_close.index[i],'signal'] = 1
#         elif s == 'sell':
#             df_close.loc[df_close.index[i],'signal'] = -1
#     signal = pd.DataFrame(df_close['signal'],columns = ['signal'])
#     return signal

def day_dream_sig(df_holcv,T_ini = 20,T_rolling = 5,times = 4,threshold = 0.03):
    df_close = pd.DataFrame(np.array(df_holcv.close),index= df_holcv.index,columns=['close'])
    df_close['trade_if'] = 0
    df_close['signal'] = 0

    length = T_ini+T_rolling*times+2
    for (s,value) in enumerate(df_close.index[length:]):
        if s%2 == 0:
            T_ini_ = T_ini
            close = df_holcv.close[s:s+length]
            Min = min(close[ -T_ini_-1:-1])
            Max = max(close[ -T_ini_-1:-1])
            MtM = Min/Max
            for j in range(times):
                if MtM > 1 - threshold:
                    T_ini_+= T_rolling
                    Min = min(close[ -T_ini_-1:-1])
                    Max = max(close[ -T_ini_-1:-1])
                    MtM = Min/Max
                else:
                    pass
            if close[-1]<= Min:
                trade_signal = 'sell'
            elif close[-1]>= Max*(1-threshold/2):
                trade_signal = 'buy'
            else:
                trade_signal = 'keep'
        else:
            trade_signal = 'keep'
        df_close.ix[df_close.index[s+length],'trade_if'] = trade_signal

    for i in range(1,len(df_close)):
#         print(i)
        s = df_close.ix[df_close.index[i-1],'trade_if']
        if   s == 0 or s == 'keep':
            for j in reversed(range(1,i-1)):
#             print(j)
                if df_close.ix[df_close.index[j],'trade_if'] == 'buy':
                    df_close.ix[df_close.index[i],'signal'] = 1
                    break
                elif df_close.ix[df_close.index[j],'trade_if'] == 'sell':
                    df_close.ix[df_close.index[i],'signal'] =  0
                    break
                else:
                    pass
        elif s == 'buy':
            df_close.ix[df_close.index[i],'signal'] = 1
        elif s == 'sell':
            df_close.ix[df_close.index[i],'signal'] = 0
    signal = pd.DataFrame(df_close['signal'],columns = ['signal'])
    return signal

# from WindPy import w
# w.start()
# from datetime import date
# today_date = date.today().strftime("%Y-%m-%d")
# symbol = "150019.OF"#"000300.SH"，"399330.SZ"
# raw_data = w.wsd(symbol, "close", "1996-12-01", today_date)
# pd_data = pd.DataFrame(raw_data.Data,index=raw_data.Fields,columns=raw_data.Times).T
# yhrj = pd_data
#
# symbol = "000300.SH"#"000300.SH"，"399330.SZ"
# raw_data = w.wsd(symbol, "close", "1996-12-01", today_date)
# pd_data = pd.DataFrame(raw_data.Data,index=raw_data.Fields,columns=raw_data.Times).T
# hs300 = pd_data
#
# symbol = "399330.SZ"#"000300.SH"，"399330.SZ"
# raw_data = w.wsd(symbol, "close", "1996-12-01", today_date)
# pd_data = pd.DataFrame(raw_data.Data,index=raw_data.Fields,columns=raw_data.Times).T
# sz100 = pd_data

def structured_fund_and_base_index(structured_fund, base_index, obj_timing, signal_which=0, threshold_up1=2.5,
                                   threshold_down1=-2.5, threshold_up2=1.2, threshold_down2=-1.2):
    df_join1 = structured_fund.join(base_index, lsuffix='_structured_fund', rsuffix='_market_index').dropna()
    df_join1_norm = df_join1 / df_join1.iloc[0,]
    df_join1_ch = df_join1.pct_change()
    relative_value1 = df_join1_ch.iloc[:, 0] / df_join1_ch.iloc[:, 1]
    signal1 = pd.DataFrame((relative_value1 > threshold_up1) * 1 + (relative_value1 <= threshold_down1) * (-1),
                           columns=['signal'])

    df_join2 = pd.DataFrame(obj_timing).join(df_join1).dropna().iloc[:, [0, 2]]
    df_join2_norm = df_join2 / df_join2.iloc[0,]
    df_join2_ch = df_join2.pct_change()
    relative_value2 = df_join2_ch.iloc[:, 0] / df_join2_ch.iloc[:, 1]
    signal2 = pd.DataFrame((relative_value2 > threshold_up2) * 1 + (relative_value2 <= threshold_down2) * (-1),
                           columns=['signal'])

    if signal_which == 1:
        signal = signal1
    elif signal_which == 2:
        signal = signal2
    else:
        signal = pd.DataFrame((((signal1 == 1) * 1 + (signal2 == 1) * 1) >= 1) * 1, columns=['signal'])

    return signal

"""基于走势分散度的择时"""
###数据准备
# import matplotlib.dates as mdates
# import statsmodels.api as sm
# SW_2_LEVEL = pd.read_csv('SW_2_LEVEL.csv',index_col = 0)
# SW_2_LEVEL['index'] = SW_2_LEVEL['2_level'].apply(lambda x:str(x) +'.INDX')
# INDEX_LIST  = list(SW_2_LEVEL['index'])
# SW_A = pd.read_csv('SW_A.csv',index_col = 0)
# SW_A['date'] = SW_A.index
# SW_A['date'] = SW_A['date'].apply(lambda x:pd.to_datetime(x))
# SW_A.index = SW_A['date']
# del SW_A['date']
#
# T = 60
# start_date = '2005-01-04'
# end_date = '2018-09-05'
# index_name = 'beta_std'
#
# df_sw_level2 = pd.DataFrame()
# for k, v in enumerate(INDEX_LIST):
#     df_sw_level2_one = pds.get_price(v, start=start_date, end=end_date, fields=['datetime', 'close'])
#     df_sw_level2_one["date"] = df_sw_level2_one["datetime"].map(lambda x: parse(str(x)))
#     df_sw_level2_one = df_sw_level2_one.set_index('date')
#
#     df_sw_level2.insert

def trend_differentiation(df_sw_level2, SW_A, T=60, start_date='2005-01-04', end_date='2018-09-05', name='beta_std',
                          m_s=11, n_s=5, m_l=16, n_l=15, stop_loss=False):
    import statsmodels.api as sm
    def calc_index(df_sw_level2, SW_A, T, start_date, end_date, name):
        df_bm = SW_A
        df_sw_level2 = df_sw_level2.dropna(axis=1)
        date_list = list(df_sw_level2.index)
        df_index = pd.DataFrame(0, index=date_list, columns=[name])

        for date in df_index.index[T + 1:]:
            df = df_sw_level2[:date][-T - 1:].pct_change()[1:]
            bm = df_bm[:date][-T - 1:].pct_change()[1:]
            X = np.array(df)
            y = np.array(bm)
            Beta = []
            R2 = 0
            Corr = []
            for j in range(len(df.columns)):
                model = sm.OLS(y, sm.add_constant(X[:, j]))
                results = model.fit()
                Beta.append(results.params[1])
                R2 += results.rsquared
                corr = np.corrcoef(y[:, 0], X[:, j])[0][1]
                Corr.append(corr)

            if name == 'beta_std':
                df_index.loc[date, 'beta_std'] = np.std(Beta)
            elif name == '1-R2':
                df_index.loc[date, '1-R2'] = 1 - R2 / X.shape[1]
            elif name == 'corr_std':
                df_index.loc[date, 'corr_std'] = np.std(Corr)
        return df_index

    df0 = calc_index(df_sw_level2, SW_A, T, start_date, end_date, name)
    bm_close = SW_A.close.copy()
    df = df0.copy()
    df['close'] = bm_close
    df['signal'] = 0
    df['signal2'] = 0
    df['positions'] = 0
    df['buy_close'] = np.nan
    df['stop_loss'] = np.nan
    for (i, value) in enumerate(df.index[2 + max(m_l, n_l):]):
        df_part = df[:value][-2 - max(m_l, n_l):]
        if df.loc[df_part.index[-2], 'close'] < df.loc[df_part.index[-2 - m_l], 'close'] and df.loc[
            df_part.index[-2], name] < df.loc[df_part.index[-2 - n_l], name]:
            df.loc[value, 'signal'] = 'sell'
        elif df.loc[df_part.index[-2], 'close'] > df.loc[df_part.index[-2 - m_s], 'close'] and df.loc[
            df_part.index[-2], name] > df.loc[df_part.index[-2 - n_s], name]:
            df.loc[value, 'signal'] = 'buy'
        if df.loc[df_part.index[-2], 'signal'] == 'buy' and df.loc[df_part.index[-1], 'signal'] == 'buy':
            df.loc[value, 'signal2'] = 'buy'
            df.loc[value, 'buy_close'] = df.loc[value, 'close']
        elif df.loc[df_part.index[-2], 'signal'] == 'sell' and df.loc[df_part.index[-1], 'signal'] == 'sell':
            df.loc[value, 'signal2'] = 'sell'
        else:
            df.loc[value, 'signal2'] = 'keep'
    if stop_loss == True:
        for (i, value) in enumerate(df.index[2 + max(m_l, n_l):]):
            if len(df[:value][:-1]['buy_close'].dropna()) > 1:
                thres = df[:value]['buy_close'][:-1].dropna()[-1]
            else:
                thres = 1
            if df.loc[value, 'close'] / thres < 0.95:  # 5%止损
                df.loc[value, 'signal2'] = 'sell'
                df.loc[value, 'stop_loss'] = 1

    for (i, value) in enumerate(df.index[2 + max(m_l, n_l):]):
        df_part2 = df[:value]
        for j in range(2, len(df_part2)):
            if df_part2['signal2'][-j] == 'buy':
                df.loc[value, 'positions'] = 1
                break
            elif df_part2['signal2'][-j] == 'sell':
                df.loc[value, 'positions'] = -1
                break
            else:
                pass
    signal = pd.DataFrame(df['positions'])
    return signal
"""基于宏观变量的择时（月度）"""

# from WindPy import w
# w.start()
# M1_M2.index = pd.to_datetime(M1_M2.index)
# M1_M2.index = M1_M2.index.strftime('%Y-%m')
# M1_M2.plot()
# today_date = date.today().strftime("%Y-%m-%d")
# symbol = "M0001385" #"M0001227"(PPI),"M0017133"(PMI进口),"M0001383"(M1),"M0001385"(M2)
# raw_data = w.edb(symbol, "1996-12-01", today_date,"Fill=Previous")
# M2 = pd.DataFrame(raw_data.Data,index=raw_data.Fields,columns=raw_data.Times).T
# symbol = "M0001383" #"M0001227"(PPI),"M0017133"(PMI进口),"M0001383"(M1),"M0001385"(M2)
# raw_data = w.edb(symbol, "1996-12-01", today_date,"Fill=Previous")
# M1 = pd.DataFrame(raw_data.Data,index=raw_data.Fields,columns=raw_data.Times).T
# M1_M2 = M1-M2

# from WindPy import w
# w.start()
# today_date = date.today().strftime("%Y-%m-%d")
# symbol = "M0001227" #"M0001227"(PPI),"M0017133"(PMI进口),"M0001383"(M1),"M0001385"(M2)
# raw_data = w.edb(symbol, "1996-12-01", today_date,"Fill=Previous")
# ppi_m = pd.DataFrame(raw_data.Data,index=raw_data.Fields,columns=raw_data.Times).T
#
# ppi_m.index = pd.to_datetime(ppi_m.index)
# ppi_m.index = ppi_m.index.strftime('%Y-%m')
# ppi_m.plot()

# from WindPy import w
# w.start()
# today_date = date.today().strftime("%Y-%m-%d")
# symbol = "M0017133" #"M0001227"(PPI),"M0017133"(PMI进口),"M0001383"(M1),"M0001385"(M2)
# raw_data = w.edb(symbol, "1996-12-01", today_date,"Fill=Previous")
# pmi_m = pd.DataFrame(raw_data.Data,index=raw_data.Fields,columns=raw_data.Times).T
#
# pmi_m.index = pd.to_datetime(pmi_m.index)
# pmi_m.index = pmi_m.index.strftime('%Y-%m')
# pmi_m.plot()
# indicator3 = indicator_trend(mac_indicator,j=1,k=1)

# from WindPy import w
# w.start()
# symbol = "000300.SH"
# today_date = date.today().strftime("%Y-%m-%d")
# raw_data = w.wsd(symbol, "close", "1996-12-01", today_date, "Period=M")
# hs300_m = pd.DataFrame(raw_data.Data,index=raw_data.Fields,columns=raw_data.Times).T
#
# hs300_m.index = pd.to_datetime(hs300_m.index)
# hs300_m.index = hs300_m.index.strftime('%Y-%m')

# mac_indicator = M1-M2
# mac_indicator.index = pd.to_datetime(mac_indicator.index)
# mac_indicator.index = mac_indicator.index.strftime('%Y-%m')
# indicator1 = indicator_trend(mac_indicator,j=2,k=2)
#
# mac_indicator = ppi_m
# mac_indicator.index = pd.to_datetime(mac_indicator.index)
# mac_indicator.index = mac_indicator.index.strftime('%Y-%m')
# indicator2 = indicator_trend(mac_indicator,j=3,k=2)
#
# mac_indicator = pmi_m
# mac_indicator.index = pd.to_datetime(mac_indicator.index)
# mac_indicator.index = mac_indicator.index.strftime('%Y-%m')
# indicator3 = indicator_trend(mac_indicator,j=1,k=1)
#
# indicator = pd.DataFrame(((indicator1['signal']==1)*1 + (indicator2['signal']==-1)*1+ ((indicator3['signal']==1)*1)>=2)*1,columns=['signal'])


def indicator_trend(mac_indicator, j=1, k=-2):
    mac_indicator_trend = pd.Series(np.nan, index=mac_indicator.index)
    pre_trend = 0
    for i in range(j, len(mac_indicator)):
        if mac_indicator.iloc[:, 0].iloc[i] > mac_indicator.iloc[:, 0].iloc[i - 1]:
            trend = 1
        elif mac_indicator.iloc[:, 0].iloc[i] < mac_indicator.iloc[:, 0].iloc[i - 1]:
            trend = -1
        else:
            trend = 0

        if j > 1:
            for k in range(2, j + 1):
                if mac_indicator.iloc[:, 0].iloc[i] > mac_indicator.iloc[:, 0].iloc[i - k] and (trend == 1):
                    trend = 1
                elif mac_indicator.iloc[:, 0].iloc[i] <= mac_indicator.iloc[:, 0].iloc[i - k] and (trend == 1):
                    trend = 0
                elif mac_indicator.iloc[:, 0].iloc[i] < mac_indicator.iloc[:, 0].iloc[i - k] and (trend == -1):
                    trend = -1
                elif mac_indicator.iloc[:, 0].iloc[i] >= mac_indicator.iloc[:, 0].iloc[i - k] and (trend == -1):
                    trend = 0
                else:
                    trend = 0
        if trend == 0:
            trend = pre_trend
        mac_indicator_trend.iloc[i] = trend
        pre_trend = trend

    signal = pd.DataFrame(mac_indicator_trend.shift(k), columns=['signal']).fillna(0)

    return signal


#######################################################################################################################
def backtest(prices, indicator, feein=0.0, feeout=0.0, ini_principle=1.0, shortable=False):
    df_1 = pd.DataFrame(prices).join(indicator)
    df_1 = df_1.fillna(method='pad').dropna()
    pricesbt = df_1.iloc[:, 0]
    indicators = df_1.iloc[:, 1]
    rets = pricesbt.diff()
    netv = []
    netv.append(ini_principle)
    holdy = []
    holdy.append(0)
    position = 0
    Pos = pd.DataFrame(columns={'Pos_rec', 'Pos_retr'}, index=[indicators.index], dtype=float)
    for i in range(len(indicators) - 1):
        if indicators[i] == 1:
            if position == 0:
                netv_fee = netv[i] / (1 + feein)
                netv[i] = netv_fee
                Pos['Pos_rec'].iloc[i] = 1
                pricein = pricesbt[i]
                Pos['Pos_retr'].iloc[i] = 0
                position = 1
                holdy.append(rets[i + 1])
                newnetv = netv[i] * (rets[i + 1] / pricesbt[i] + 1.0)
                netv.append(newnetv)
            elif position == 1:
                holdy.append(rets[i + 1])
                newnetv = netv[i] * (rets[i + 1] / pricesbt[i] + 1.0)
                netv.append(newnetv)
                Pos['Pos_rec'].iloc[i] = 0
                Pos['Pos_retr'].iloc[i] = 0
            else:
                if shortable == True:
                    netv_fee = netv[i] * (1 - feeout)
                    netv[i] = netv_fee
                    Pos['Pos_retr'].iloc[i] = 1 - pricesbt[i] / float(pricein)

                    netv_fee = netv[i] / (1 + feein)
                    Pos['Pos_rec'].iloc[i] = 2
                    netv[i] = netv_fee
                    pricein = pricesbt[i]
                    position = 1
                    holdy.append(rets[i + 1])
                    newnetv = netv[i] * (rets[i + 1] / pricesbt[i] + 1.0)
                    netv.append(newnetv)
                else:
                    pass
        elif indicators[i] == -1:
            if shortable == True:
                if position == 0:
                    netv_fee = netv[i] / (1 - feein)
                    Pos['Pos_rec'].iloc[i] = -1
                    Pos['Pos_retr'].iloc[i] = 0
                    netv[i] = netv_fee
                    pricein = pricesbt[i]
                    position = -1
                    holdy.append(-1 * rets[i + 1])
                    newnetv = netv[i] * (-1 * rets[i + 1] / pricesbt[i] + 1.0)
                    netv.append(newnetv)
                elif position == 1:
                    netv_fee = netv[i] * (1 - feeout)
                    netv[i] = netv_fee
                    Pos['Pos_retr'].iloc[i] = pricesbt[i] / float(pricein) - 1

                    netv_fee = netv[i] / (1 - feein)
                    Pos['Pos_rec'].iloc[i] = -2
                    netv[i] = netv_fee
                    pricein = pricesbt[i]
                    position = -1
                    holdy.append(-1 * rets[i + 1])
                    newnetv = netv[i] * (-1 * rets[i + 1] / pricesbt[i] + 1.0)
                    netv.append(newnetv)
                else:
                    holdy.append(-1 * rets[i + 1])
                    newnetv = netv[i] * (-1 * rets[i + 1] / pricesbt[i] + 1.0)
                    netv.append(newnetv)
                    Pos['Pos_rec'].iloc[i] = 0
                    Pos['Pos_retr'].iloc[i] = 0
            else:
                if position == 0:
                    Pos['Pos_rec'].iloc[i] = 0
                    Pos['Pos_retr'].iloc[i] = 0
                #                     holdy.append(0)
                #                     newnetv = netv[i]
                #                     netv.append(newnetv)
                else:
                    netv_fee = netv[i] * (1 - feeout)
                    netv[i] = netv_fee
                    Pos['Pos_rec'].iloc[i] = -1
                    Pos['Pos_retr'].iloc[i] = pricesbt[i] / float(pricein) - 1
                    position = 0
                holdy.append(0)
                newnetv = netv[i]
                netv.append(newnetv)
        else:
            if position == 1:
                netv_fee = netv[i] * (1 - feeout)
                netv[i] = netv_fee
                Pos['Pos_rec'].iloc[i] = -1
                Pos['Pos_retr'].iloc[i] = pricesbt[i] / float(pricein) - 1
            elif position == -1:
                netv_fee = netv[i] * (1 - feeout)
                netv[i] = netv_fee
                Pos['Pos_rec'].iloc[i] = 1
                Pos['Pos_retr'].iloc[i] = 1 - pricesbt[i] / float(pricein)
            else:
                Pos['Pos_rec'].iloc[i] = 0
                Pos['Pos_retr'].iloc[i] = 0
            position = 0
            holdy.append(0)
            newnetv = netv[i]
            netv.append(newnetv)

    i = len(indicators) - 1
    if position == 1:
        Pos['Pos_rec'].iloc[i] = -1
        Pos['Pos_retr'].iloc[i] = pricesbt[i] / float(pricein) - 1
    elif position == -1:
        Pos['Pos_rec'].iloc[i] = 1
        Pos['Pos_retr'].iloc[i] = 1 - pricesbt[i] / float(pricein)
    else:
        Pos['Pos_rec'].iloc[i] = 0
        Pos['Pos_retr'].iloc[i] = 0
    holdy = pd.DataFrame(holdy, index=[indicators.index])
    netv = pd.DataFrame(netv, index=[indicators.index])
    return [netv, holdy, Pos]


# def backtest(prices,indicator,feein=0.0,feeout=0.0,ini_pri=1.0,short = False):
#     indicators = indicator.signal
#     pricesbt = prices[len(prices)-len(indicators):]
#     rets = pricesbt.diff()
#     netv = []
#     netv.append(ini_pri)
#     holdy = []
#     holdy.append(0)
#     position = 0
#     Pos = pd.DataFrame(columns= {'Pos_rec','Pos_retr'},index=[indicators.index],dtype = float)
#     for i in range(len(indicators)-1):
#         if indicators[i] == 1:
#             if position == 0:
#                 netv_fee = netv[i] / (1 + feein)
#                 netv[i] = netv_fee
#                 Pos['Pos_rec'].iloc[i] = 1
#                 pricein = pricesbt[i]
#                 Pos['Pos_retr'].iloc[i] = 0
#             else:
#                 Pos['Pos_rec'].iloc[i] = 0
#                 Pos['Pos_retr'].iloc[i] = 0
#             position = 1
#             holdy.append(rets[i+1])
#             newnetv = netv[i] * (rets[i+1] / pricesbt[i] + 1.0)
#             netv.append(newnetv)
#         elif position == 1 and indicators[i] ==1 :
#             holdy.append(rets[i+1])
#             newnetv = netv[i] * (rets[i+1] / pricesbt[i] + 1.0)
#             netv.append(newnetv)
#             Pos['Pos_rec'].iloc[i] = 0
#             Pos['Pos_retr'].iloc[i] = 0
#         else:
#             if position == 1:
#                 netv_fee = netv[i] * (1 - feeout)
#                 netv[i] = netv_fee
#                 Pos['Pos_rec'].iloc[i] = -1
#                 Pos['Pos_retr'].iloc[i] = pricesbt[i]/float(pricein) - 1
#             else:
#                 Pos['Pos_rec'].iloc[i] = 0
#                 Pos['Pos_retr'].iloc[i] = 0
#             position = 0
#             holdy.append(0)
#             newnetv = netv[i]
#             netv.append(newnetv)
    i = len(indicators)-1
    # if position == 1:
    #     Pos['Pos_rec'].iloc[i] = -1
    #     Pos['Pos_retr'].iloc[i] = pricesbt[i]/float(pricein) - 1
    # else:
    #     Pos['Pos_rec'].iloc[i] = 0
    #     Pos['Pos_retr'].iloc[i] = 0
    # holdy = pd.DataFrame(holdy,index = [indicators.index])
    # netv = pd.DataFrame(netv,index = [indicators.index])
    # return [holdy,netv,Pos]




