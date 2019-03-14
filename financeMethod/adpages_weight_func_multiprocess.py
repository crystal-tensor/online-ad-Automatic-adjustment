import  pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
import scipy.optimize as opt
import copy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import multiprocessing
import timeit

def equal_weights(nr_asset):
    weights = np.ones(nr_asset) / nr_asset
    return weights

def calc_port_var(sigma, W):
    w = np.matrix(W).T
    return float(w.T.dot(sigma).dot(w))

def calc_port_vol(sigma, W):
    return np.sqrt(calc_port_var(sigma, W))

def calc_port_mean(mu, W):
    return mu.dot(W)

def optimize_max_sharpe(mu, sigma, rf=0.0, sum_one=True, constraints=None):
    def fitness(W):
        mean_p = calc_port_mean(mu, W)
        vol_p = calc_port_vol(sigma, W)
        util = (mean_p - rf) / vol_p
        util = -util
        return util

    n = len(mu)
    W = equal_weights(n) # start with equal weights
    b_ = [(0., 1.) for i in range(n)]  # weights between 0%..100%.
    # No leverage, no shorting
    if sum_one:
        c_port_weights = {'type': 'eq', 'fun': lambda W: sum(W) - 1.}
    else:
        c_port_weights = {'type': 'ineq', 'fun': lambda W: 1. - sum(W)}

    if constraints is None:
        c_ = [c_port_weights]
    else:
        c_ = [c_port_weights] + constraints

    optimized = opt.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_,
                             tol=1e-5, options={'maxiter': 10000})
    if not optimized.success:
        raise BaseException(optimized.message)
    weights = pd.Series(optimized.x,index = mu.index)
    return weights

def optimize_max_ret(mu, sum_one=False, constraints=None):
    n = len(mu)

    def fitness(W):
        mean = calc_port_mean(mu, W)
        util = -mean
        return util

    W = equal_weights(n) # start with equal weights
    b_ = [(0., 1.) for i in range(n)]  # weights between 0%..100%.
    # No leverage, no shorting
    if sum_one:
        c_port_weights = {'type': 'eq', 'fun': lambda W: sum(W) - 1.}
    else:
        c_port_weights = {'type': 'ineq', 'fun': lambda W: 1. - sum(W)}

    if constraints is None:
        c_ = [c_port_weights]
    else:
        c_ = [c_port_weights] + constraints

    optimized = opt.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_, tol=1e-10)
    if not optimized.success:
        raise BaseException(optimized.message)
    weights = pd.Series(optimized.x, index=mu.index)
    return weights

def constraint_generator(i_th,low_bound):
    return {'type': 'ineq', 'fun': lambda W: W[i_th] - low_bound}


def get_raw_camp(campaign_one):
    conn = pymysql.Connect(host='192.168.101.71',port=3306,user='aipredict',passwd='63170a532f1847bc',db='aidata',charset='utf8')
    cursor = conn.cursor()
    sql_camp_old = "select * from ai_weight_20181126 where campaign_id=" + str(campaign_one) +" and report_date < '2018-11-01' ORDER BY ai_weight_20181126. report_date "
    df_adpages_old = pd.read_sql(sql=sql_camp_old, con=conn)

    sql_camp = "select * from ai_weight where campaign_id="+ str(campaign_one) +" ORDER BY ai_weight. report_date "
    df_adpages = pd.read_sql(sql=sql_camp, con=conn)

    df_adpages_raw = df_adpages_old.append(df_adpages)
    cursor.close()
    return (df_adpages_raw )

def adpages_allo_weight(campaign_one,lookback_days_init = 3,next_days = 2,alpha = 0.1,low_bound = 0.01):
    df_adpages_processed = get_raw_camp(campaign_one)
    df_adpages_processed['return'] = df_adpages_processed.revenue - df_adpages_processed.cost
    df_adpages_processed['return_ratio'] = df_adpages_processed['return'] / df_adpages_processed.cost
    camp_date = df_adpages_processed['report_date'].sort_values().drop_duplicates()
    df_weight_alloc = pd.DataFrame()

    for camp_th, date_one in enumerate(
            camp_date[lookback_days_init:((len(camp_date) - next_days) + 1)]):  # (len(camp_date) - next_days)
        """
        df_adpages_part = df_adpages_processed[
            df_adpages_processed['report_date'].isin(camp_date[camp_date < date_one].tolist()[-lookback_days_init:])]
        if len(df_adpages_part.adpage_id.unique()) <= lookback_days_init:
            lookback_days = lookback_days_init
        else:
            lookback_days = len(df_adpages_part.adpage_id.unique())
            df_adpages_part = df_adpages_processed[
                df_adpages_processed['report_date'].isin(camp_date[camp_date < date_one].tolist()[-lookback_days:])]

        df_adpages_part.set_index(['report_date'], inplace=True)
        adpages_part_df = pd.DataFrame()
        weight_constraints = []
        i_th = 0

        for adpage_one in df_adpages_part.adpage_id.unique():
            # print(df_adpages[df_adpages.adpage_id == adpage_one])
            df_adpages_one = df_adpages_part[df_adpages_part.adpage_id == adpage_one]
            adpage_one_s = df_adpages_one[
                'return_ratio']  # (df_adpages_one.rpi - df_adpages_one.cpc) / df_adpages_one.cpc
            if np.any(adpage_one_s.isnull()) or len(adpage_one_s) < lookback_days:
                continue
            adpage_one_s.name = adpage_one
            adpages_part_df = pd.concat([adpages_part_df, adpage_one_s], axis=1, join='outer', sort=True)
            # print(adpages_part_df)
            if df_adpages_one.clicks[-1] > 0:
                # print('i_th:',i_th)
                # print(adpage_one,df_adpages_one.clicks[-1])
                weight_constraints = weight_constraints + [constraint_generator(i_th, low_bound)]
            i_th += 1
"""
        adpages_part_df = pd.DataFrame()
        weight_constraints = []
        i_th = 0
        adpages = df_adpages_processed[df_adpages_processed['report_date'] == date_one].adpage_id
        lookback_days = len(adpages)
        for adpage_one in adpages.values:
            df_adpages_processed1 = df_adpages_processed[df_adpages_processed['report_date'] <= date_one]
            df_adpages_processed1_adpage = df_adpages_processed1[df_adpages_processed1['adpage_id'] == adpage_one]
            adpage_one_rr = df_adpages_processed1_adpage['return_ratio'].dropna()[-lookback_days:]

            if len(df_adpages_processed1_adpage['return_ratio'].dropna()) >= lookback_days:
                j_count = 1
                while len(adpage_one_rr) < lookback_days:
                    adpage_one_rr = df_adpages_processed1[df_adpages_processed1['adpage_id'] == adpage_one][
                                        'return_ratio'].dropna()[-(lookback_days + j_count):]
                    j_count += 1
            else:
                continue

            if len(adpage_one_rr) < lookback_days:
                continue
            adpage_one_s = pd.DataFrame(adpage_one_rr.values, columns=[adpage_one])
            adpages_part_df = pd.concat([adpages_part_df, adpage_one_s], axis=1)
            if df_adpages_processed1_adpage.clicks.iloc[-1] > 0:
                # print('i_th:',i_th)
                # print(adpage_one,df_adpages_one.clicks[-1])
                weight_constraints = weight_constraints + [constraint_generator(i_th, low_bound)]
            i_th += 1

        mu = adpages_part_df.mean()
        sigma = adpages_part_df.cov()
        if len(weight_constraints) == 0:
            weight_constraints = None
        try:
            weight_sharp = optimize_max_sharpe(mu, sigma, rf=0.0, sum_one=True, constraints=weight_constraints)
            weight_ret = optimize_max_ret(mu, sum_one=True, constraints=weight_constraints)
            weight_one = alpha * weight_sharp + (1 - alpha) * weight_ret
        except:
            if len(adpages_part_df) > 0 or not ('weight_one' in dir()):
                weight_one = pd.Series(0.0, index=mu.index)

        weight_one.name = 'weight_alloc'
        weight_df = pd.DataFrame(weight_one)
        weight_df.index.name = 'adpage_id'
        weight_df.reset_index(inplace=True)
        weight_df['report_date'] = camp_date.iloc[lookback_days_init + camp_th + next_days - 1]
        df_weight_alloc = df_weight_alloc.append(weight_df)

    df_weight_alloc['adpage_id'] = df_weight_alloc['adpage_id'].astype(np.int64)
    df_adpages_processed['adpage_id'] = df_adpages_processed['adpage_id'].astype(np.int64)
    df_adpages_processed = pd.merge(df_adpages_processed, df_weight_alloc, right_index=True,
                                    on=['report_date', 'adpage_id'], how='outer')

    return df_adpages_processed,df_weight_alloc

def calc_camp_ret_ratio(df_adpages_processed):
    camp_date = df_adpages_processed['report_date'].sort_values().drop_duplicates()
    camp_ret_ratio = pd.DataFrame()
    for camp_th, date_one in enumerate(camp_date):
        df_adpages_day = df_adpages_processed[df_adpages_processed['report_date'] == date_one]
        df_adpages_day_1 = df_adpages_day[df_adpages_day.weight_alloc.notnull()]
        ret_real = df_adpages_day.revenue.sum() - df_adpages_day.cost.sum()

        if df_adpages_day.cost.sum() > 0:
            ret_ratio_real = df_adpages_day.revenue.sum() / df_adpages_day.cost.sum() - 1
            if len(df_adpages_day_1) > 0:
                ret_ratio_weight_alloc = df_adpages_day_1.weight_alloc.dot(df_adpages_day_1.return_ratio)
                ret_ratio_real_weight_old = (df_adpages_day_1.weight_old / df_adpages_day_1.weight_old.sum()).dot(
                    df_adpages_day_1.return_ratio)
                ret_weight_alloc = ret_ratio_weight_alloc * df_adpages_day_1.cost.sum()
                ret_real_weight_old = ret_ratio_real_weight_old * df_adpages_day_1.cost.sum()
            else:
                ret_ratio_weight_alloc = 0
                ret_ratio_real_weight_old = 0
                ret_weight_alloc = 0
                ret_real_weight_old = 0
        else:
            ret_ratio_real = 0
            ret_ratio_weight_alloc = 0
            ret_ratio_real_weight_old = 0
            ret_weight_alloc = 0
            ret_real_weight_old = 0

        camp_ret_ratio = camp_ret_ratio.append(pd.DataFrame({'ret_real': ret_real, 'ret_ratio_real': ret_ratio_real, \
                                                             'ret_ratio_real_weight_old': ret_ratio_real_weight_old,
                                                             'ret_real_weight_old': ret_real_weight_old, \
                                                             'ret_ratio_weight_alloc': ret_ratio_weight_alloc,
                                                             'ret_weight_alloc': ret_weight_alloc,}, index=[date_one]))

    camp_ret_ratio.fillna(0, inplace=True)
    camp_ret_ratio['cum_ret_real'] = camp_ret_ratio['ret_real'].cumsum()
    camp_ret_ratio['cum_ret_real_weight_old'] = camp_ret_ratio['ret_real_weight_old'].cumsum()
    camp_ret_ratio['cum_ret_weight_alloc'] = camp_ret_ratio['ret_weight_alloc'].cumsum()

    return camp_ret_ratio

def camp_ret_ratio_statistics(df_adpages_processed,campaign_one):
    camp_ret_ratio = calc_camp_ret_ratio(df_adpages_processed)

    win_rate = len(
        camp_ret_ratio[camp_ret_ratio['ret_ratio_weight_alloc'] > camp_ret_ratio['ret_ratio_real_weight_old']]) / len(
        camp_ret_ratio)
    loss_rate = len(
        camp_ret_ratio[camp_ret_ratio['ret_ratio_weight_alloc'] < camp_ret_ratio['ret_ratio_real_weight_old']]) / len(
        camp_ret_ratio)
    equal_rate = len(
        camp_ret_ratio[camp_ret_ratio['ret_ratio_weight_alloc'] == camp_ret_ratio['ret_ratio_real_weight_old']]) / len(
        camp_ret_ratio)
    mean_weight_alloc = camp_ret_ratio['ret_ratio_weight_alloc'].mean()
    mean_weight_old = camp_ret_ratio['ret_ratio_real_weight_old'].mean()
    mean_real = camp_ret_ratio['ret_ratio_real'].mean()
    # sharp_ratio_weight_alloc = mean_weight_alloc / camp_ret_ratio['ret_ratio_weight_alloc'].std()
    # sharp_ratio_weight_old = mean_weight_old / camp_ret_ratio['ret_ratio_real_weight_old'].std()
    # sharp_ratio_real = mean_real / camp_ret_ratio['ret_ratio_real'].std()
    crwa_crr = camp_ret_ratio['cum_ret_weight_alloc'][-1] / camp_ret_ratio['cum_ret_real'][-1]
    crwa_crrwo = camp_ret_ratio['cum_ret_weight_alloc'][-1] / camp_ret_ratio['cum_ret_real_weight_old'][-1]
    crwa_crrwo_diff = camp_ret_ratio['cum_ret_weight_alloc'][-1] - camp_ret_ratio['cum_ret_real_weight_old'][-1]
    # diff_crwa_crr = sharp_ratio_weight_alloc - sharp_ratio_real
    # diff_crwa_crrwo = sharp_ratio_weight_alloc - sharp_ratio_weight_old
    statistics_df = pd.DataFrame(
        {'campaign_id':campaign_one,'equal_rate':equal_rate,'win_rate': win_rate, 'loss_rate': loss_rate, 'diff_wl_rate': win_rate - loss_rate,\
         'cum_crwa_crr': crwa_crr,'cum_crwa_crrwo': crwa_crrwo, 'diff_cum_crwa_crrwo': crwa_crrwo_diff,\
         'mean_weight_alloc': mean_weight_alloc, 'mean_weight_old': mean_weight_old, 'mean_real': mean_real}, index=[campaign_one])

    return statistics_df,equal_rate


# campaign_one = 329634732
# #329634732,351373541,351325304,357181647,274704543,274868904,274833205,357040945,355171186
# df_adpages_processed,_ = adpages_allo_weight(campaign_one,lookback_days_init = 3,next_days = 2,alpha = 0.1,low_bound = 0.01)
# df_adpages_processed.head(50)
# camp_ret_ratio = calc_camp_ret_ratio(df_adpages_processed)
# # camp_ret_ratio.head()
# camp_ret_ratio[['cum_ret_real', 'cum_ret_real_weight_old', 'cum_ret_weight_alloc']].plot()
# camp_ret_ratio[['ret_ratio_real_weight_old', 'ret_ratio_weight_alloc']].plot()
# camp_ret_ratio[['ret_real_weight_old', 'ret_weight_alloc']].plot()
# plt.show()
# compare_df,equal_rate = camp_ret_ratio_statistics(df_adpages_processed)
# # df_adpages_processed.to_csv("E:\df_adpages_processed_1.csv")
# # camp_ret_ratio.to_csv("E:\camp_ret_ratio.csv")
#
# print(compare_df[['diff_wl_rate','cum_crwa_crrwo','diff_crwa_crrwo']])
# print('equal_rate:',equal_rate)
#
# campaign_one = 1647216632
# #329634732,351373541,351325304,357181647,274704543,274868904,274833205,357040945,355171186
# df_adpages_processed,_ = adpages_allo_weight(campaign_one,lookback_days_init = 3,next_days = 2,alpha = 0.1,low_bound = 0.01)
# compare_df,equal_rate = camp_ret_ratio_statistics(df_adpages_processed)
# print(compare_df[['diff_wl_rate','cum_crwa_crrwo','diff_crwa_crrwo']])
# print('equal_rate:',equal_rate)
compare_statistic_df = pd.DataFrame()
compare_statistic_except_df = pd.DataFrame()
campaign_wrong_list = [ ]
 # campaign_list = []
def campaign_one_work(campaign_one):

    # print(np.where(campaign_list == campaign_one)[0])
    print('Now ',campaign_one)
    # engine = create_engine('mysql+pymysql://root:dugoohoo@192.168.101.70:3306/adm_files')
    engine = create_engine('mysql+pymysql://scm:dugoohoo@192.168.101.70:3306/adm_files')
    try:
        df_adpages_processed, _ = adpages_allo_weight(campaign_one, lookback_days_init=3, next_days=2, alpha=0.1,
                                                      low_bound=0.01)
        compare_df, equal_rate = camp_ret_ratio_statistics(df_adpages_processed,campaign_one)
        if equal_rate < 0.8:
            # compare_statistic_df = compare_statistic_df.append(compare_df)
            compare_df.to_sql('compare_statistic_df_1',con=engine,schema='adm_files', index=False, index_label=False,if_exists='append')
        else:
            # compare_statistic_except_df = compare_statistic_except_df.append(compare_df)
            compare_df.to_sql('compare_statistic_except_df_1', con=engine,schema='adm_files', index=False, index_label=False,if_exists='append')
    except:
        campaign_wrong_df = pd.DataFrame({'campaignid': campaign_one}, index=[campaign_one])
        campaign_wrong_df.to_sql('campaign_wrong_df_1', con=engine, schema='adm_files', index=False, index_label=False,
                          if_exists='append')
    engine.dispose()

    # return compare_statistic_df,compare_statistic_except_df,campaign_wrong_list

if __name__ == '__main__':
    print('main function is beginning')
    conn = pymysql.Connect(host='192.168.101.71',port=3306,user='aipredict',passwd='63170a532f1847bc',db='aidata',charset='utf8')
    cursor = conn.cursor()
    # sql = "select report_date,campaign_id,adpage_id,cost,revenue,clicks,rpi,cpc from ai_weight where report_date >= '2019-01-01' and clicks>10 ORDER BY ai_weight. report_date"
    sql = "select report_date,campaign_id,adpage_id,cost,revenue,clicks,rpi,cpc from ai_weight where report_date >= '2019-01-01' and clicks<=10 ORDER BY ai_weight. report_date"
    df = pd.read_sql(sql=sql, con=conn)
    df_group = df.groupby('campaign_id')
    df['return'] = df.revenue - df.cost

    df_group_sum = df['return'].groupby(df['campaign_id']).sum()
    camp_list_index = df_group_sum.sort_values(ascending=False).index
    campaign_list = camp_list_index.values

    cursor.close()
    print('The length of campaign_list is: ',len(campaign_list))

    items = [campaign_one for campaign_one in campaign_list[43045:]]
    p = multiprocessing.Pool(20)
    start = timeit.default_timer()
    p.map(campaign_one_work, items)
    p.close()
    p.join()
    end = timeit.default_timer()
    print('multi processing time:', str(end - start))
    # for th_i,campaign_one in enumerate(campaign_list[:5]):
    #     campaign_one_work(campaign_one)
    # compare_statistic_df.to_csv("E:\compare_statistic_df.csv")
    # compare_statistic_except_df.to_csv("E:\compare_statistic_except_df.csv")

    # campaign_one = 329543295
    # df_adpages_processed, _ = adpages_allo_weight(campaign_one, lookback_days_init=3, next_days=2, alpha=0.1,
    #                                               low_bound=0.01)
    # compare_df, equal_rate = camp_ret_ratio_statistics(df_adpages_processed,campaign_one)
    # camp_ret_ratio = calc_camp_ret_ratio(df_adpages_processed)
    # # camp_ret_ratio.head()
    # camp_ret_ratio[['cum_ret_real', 'cum_ret_real_weight_old', 'cum_ret_weight_alloc']].plot()
    # camp_ret_ratio[['ret_ratio_real_weight_old', 'ret_ratio_weight_alloc']].plot()
    # camp_ret_ratio[['ret_real_weight_old', 'ret_weight_alloc']].plot()
    # plt.show()