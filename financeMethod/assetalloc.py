#encoding=utf-8

import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize as opt
from sklearn import mixture


from pquant.analysis.backtest import WalkForwardAnalysis
from pquant.analysis.tools.indicators import IndicatorSeries

def equal_weights(nr_asset):
    weights = np.ones(nr_asset)/nr_asset
    return weights

################################
# risk measure
################################
def calc_halflife_vol_by_asset(df_rets, n=4, lamb=0.9):
    len = df_rets.shape[0]
    win_n = int(len/n)
    weights = [pow(lamb, i) for i in range(n)]
    weights = weights / np.sum(weights)

    sigma = None
    for i in range(n):
        if i < n-1:
            data = df_rets[i*win_n:(i+1)*win_n]
        else:
            data = df_rets[i*win_n:]

        if sigma is None:
            sigma = np.std(data) * weights[i]
        else :
            sigma = sigma + np.std(data) * weights[i]
    return sigma


def calc_vol_by_asset(df_rets, halflife=False, n=4,  lamb=0.9):
    if halflife:
        return calc_halflife_vol_by_asset(df_rets, n, lamb)
    else :
        return np.std(df_rets)

def calc_corr_by_asset(df_rets) :
    return df_rets.corr()

def calc_vcv_by_asset(df_rets, halflife=False, n=4, lamb=0.9):
    '''
    calc variance-covariance matrix (VCV)
    :param df_rets:
    :return:
    '''
    if halflife:
        corr = calc_corr_by_asset(df_rets)
        vol = calc_halflife_vol_by_asset(df_rets, n, lamb)
        vol_d = np.diag(vol)
        vcv = vol_d.dot(corr).dot(vol_d)
    else:
        vcv = np.cov(df_rets.values.T)
    return vcv

def calc_mean_by_asset(df_rets):
    R = df_rets.mean().values.T
    return R

# def calc_mean_vcv_by_asset(df_rets):
#     R = calc_mean_by_asset(df_rets)
#     C = calc_vcv_by_asset(df_rets)
#     return R, C

####################################################
def calc_port_var(sigma, W):
    w = np.matrix(W).T
    return float(w.T * sigma * w)


def calc_port_vol(sigma, W):
    return np.sqrt(calc_port_var(sigma, W))


def calc_port_mean(mu, W):
    return mu.dot(W)


def calc_port_ret(ret, W):
    return ret.dot(W)


def calc_vol(sigma) :
    return [sigma[x, x] for x in range(sigma.shape[0])]

#####################################################

def calc_mean_var_by_portfolio(df_rets, weights=None, **kwargs):
    if weights is None:
        weights = equal_weights(df_rets.shape[1])

    R = calc_mean_by_asset(df_rets)
    C = calc_vcv_by_asset(df_rets, **kwargs)
    W = np.matrix(weights).T
    mean_p = R * W
    var_p = W.T * C * W
    return float(mean_p), float(var_p)


def calc_var_by_portfolio(df_rets, weights=None, **kwargs):
    if weights is None:
        weights = equal_weights(df_rets.shape[1])

    C = calc_vcv_by_asset(df_rets, **kwargs)
    W = np.matrix(weights).T
    var_p = W.T * C * W
    return float(var_p)

def calc_ret_by_portfolio(df_rets, weights=None):
    if weights is None:
        weights = equal_weights(df_rets.shape[1])

    ret_port = df_rets.dot(weights.T)

    return ret_port


def calc_conditional_vol_by_portfolio(df_rets, weights=None, **kwargs):
    if weights is None:
        weights = equal_weights(df_rets.shape[1])

    C = calc_vcv_by_asset(df_rets, **kwargs)
    mean_p, var_p = calc_mean_var_by_portfolio(df_rets, weights)
    vol_p = np.sqrt(var_p)

    vol_con = weights.dot(C) / vol_p
    return IndicatorSeries(pd.Series(vol_con, index=df_rets.columns))

def calc_conditional_vol(mu, sigma, weights=None):
    if weights is None:
        weights = equal_weights(len(mu))

    R, C = mu, sigma
    vol_p = calc_port_vol(sigma, weights)

    vol_con = weights.T.dot(C) / vol_p
    return np.array(vol_con)

def calc_hist_expected_loss(df_rets, weights=None, target_prob=0.05):
    n = df_rets.shape[1]
    weights = equal_weights(n) if weights is None else weights
    port_ret = calc_ret_by_portfolio(df_rets, weights)

    threshold = port_ret.quantile(target_prob)
    mean_loss = port_ret[port_ret < threshold].mean()
    return float(mean_loss)

def calc_hist_VaR(df_rets, weights=None, target_prob=0.05):
    n = df_rets.shape[1]
    weights = equal_weights(n) if weights is None else weights
    port_ret = calc_ret_by_portfolio(df_rets, weights)

    threshold = port_ret.quantile(target_prob)
    return float(threshold)

#####################################################
## optimization
#####################################################
# from cvxopt import solvers, matrix

def optimize_equal_weights(mu):
    n = len(mu)
    return equal_weights(n)


def optimize_target_vol(mu, sigma, target_vol=0.0, sum_one=False, constraints=None):

    def fitness(W):
        ret_p = calc_port_ret(mu, W)
        util = -ret_p
        return util

    n = len(mu)
    W = equal_weights(n)  # start with equal weights
    b_ = [(0., 1.) for i in range(n)]  # weights between 0%..100%.
    # No leverage, no shorting
    if sum_one:
        c_port_weights = {'type': 'eq', 'fun': lambda W: sum(W) - 1.}
    else :
        c_port_weights = {'type': 'ineq', 'fun': lambda W: 1. - sum(W)}
    c_target_vol = {'type': 'eq', 'fun': lambda W: np.square(target_vol - calc_port_vol(sigma, W))}

    if constraints is None:
        c_ = [c_port_weights, c_target_vol]  # Sum of weights = 100%
    else:
        c_ = [c_port_weights, c_target_vol] + constraints

    optimized = opt.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_, options={'maxiter': 10000})
    if not optimized.success:
        raise BaseException(optimized.message)
    return optimized.x

def optimize_historical_target_loss(df_rets, target_loss=0.0, target_prob=0.0, sum_one=True):
    def fitness(W):
        ret_p = calc_ret_by_portfolio(df_rets, W)
        util = ret_p.sum()
        return -util

    n = df_rets.shape[1]
    W = equal_weights(n)  # start with equal weights
    b_ = [(0., 1.) for i in range(n)]  # weights between 0%..100%.
    # No leverage, no shorting
    if sum_one:
        c_port_weights = {'type': 'eq', 'fun': lambda W: sum(W) - 1.}
    else:
        c_port_weights = {'type': 'ineq', 'fun': lambda W: 1. - sum(W)}
    c_var = {'type': 'ineq', 'fun': lambda W: calc_hist_VaR(df_rets, W, target_prob) - target_loss}
    c_ = (c_port_weights, c_var )  # Sum of weights = 100%

    optimized = opt.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_, options={'maxiter': 2000})
    if not optimized.success:
        raise BaseException(optimized.message)
    return pd.Series(optimized.x, index=df_rets.columns)  # Return opW, timized weights

def optimize_historical_target_expected_loss(df_rets, target_loss=0.0, target_prob=0.0, sum_one=True):
    def fitness(W):
        ret_p = calc_port_ret(df_rets, W)
        util = ret_p.sum()
        return -util

    n = df_rets.shape[1]
    W = equal_weights(n)  # start with equal weights
    b_ = [(0., 1.) for i in range(n)]  # weights between 0%..100%.
    # No leverage, no shorting
    if sum_one:
        c_port_weights = {'type': 'eq', 'fun': lambda W: sum(W) - 1.}
    else :
        c_port_weights = {'type': 'ineq', 'fun': lambda W: 1. - sum(W)}
    c_cvar = {'type': 'ineq', 'fun':
              lambda W: calc_hist_expected_loss(df_rets, W, target_prob) - target_loss}

    c_ = (c_port_weights, c_cvar)


          # Sum of weights = 100%
    optimized = opt.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_, options={'maxiter': 2000})
    if not optimized.success:
        raise BaseException(optimized.message)
    return pd.Series(optimized.x, index=df_rets.columns)  # Return opW, timized weights

# def optimize_target_ret(mu, sigma, target_ret=0.0):
#     n = len(mu)
#
#     G = -matrix(np.eye(n))  # negative n x n identity matrix
#     h = matrix(0.0, (n, 1))
#     A = matrix(np.array([mu, np.ones(n)]))
#     b = matrix([target_ret, 1.0])
#
#     P = matrix(sigma)
#     q = -matrix(0.0, (n, 1))
#
#     solvers.options['show_progress'] = False
#     sol = solvers.qp(P, q, G, h, A, b)
#     return np.array(list(sol['x']))
#     # return pd.Series(list(sol['x']), index=df_rets.columns)

# def optimize_target_ret(mu, sigma, target_ret=0.0, sum_one=False):
#     n = len(mu)
#
#     def fitness(W):
#         mean_p = calc_port_mean(mu, W)
#         util = np.square(mean_p - target_ret)
#         return util
#
#     W = equal_weights(n)  # start with equal weights
#     b_ = [(0., 1.) for i in range(n)]  # weights between 0%..100%.
#     # No leverage, no shorting
#     if sum_one:
#         c_port_weights = {'type': 'eq', 'fun': lambda W: sum(W) - 1.}
#     else:
#         c_port_weights = {'type': 'ineq', 'fun': lambda W: 1. - sum(W)}
#     c_ = (c_port_weights)
#
#     optimized = opt.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_, tol=1e-10)
#     if not optimized.success:
#         raise BaseException(optimized.message)
#     return optimized.x


def optimize_max_sharpe(mu, sigma, rf=0.0, sum_one=False, constraints=None):
    n = len(mu)

    def fitness(W):
        mean_p = calc_port_mean(mu, W)
        vol_p = calc_port_vol(sigma, W)
        util = (mean_p - rf) / vol_p
        util = -util
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

    optimized = opt.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_,
                             tol=1e-5, options={'maxiter': 10000})
    if not optimized.success:
        raise BaseException(optimized.message)
    return optimized.x


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
    return optimized.x


def optimize_min_vol(mu, sigma, sum_one=False, constraints=None):
    n = len(mu)

    def fitness(W):
        util = calc_port_vol(sigma, W)
        return util

    W = equal_weights(n) # start with equal weights
    b_ = [(0., 1.) for i in range(n)]  # weights between 0%..100%.
    # No leverage, no shorting
    if sum_one:
        c_port_weights = {'type': 'eq', 'fun': lambda W: sum(W) - 1.}
    else:
        c_port_weights = {'type': 'ineq', 'fun': lambda W: 1. - sum(W)}

    if constraints is None:
        c_ = [c_port_weights]  # Sum of weights = 100%
    else:
        c_ = [c_port_weights] + constraints

    optimized = opt.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success:
        raise BaseException(optimized.message)
    return optimized.x

def optimize_target_ret(mu, sigma, target_ret=0.0, sum_one=False, constraints=None):
    n = len(mu)

    def fitness(W):
        util = calc_port_vol(sigma, W)
        return util

    W = equal_weights(n) # start with equal weights
    b_ = [(0., 1.) for i in range(n)]  # weights between 0%..100%.
    # No leverage, no shorting
    if sum_one:
        c_port_weights = {'type': 'eq', 'fun': lambda W: sum(W) - 1.}
    else:
        c_port_weights = {'type': 'ineq', 'fun': lambda W: 1. - sum(W)}

    c_target_ret = {'type': 'eq', 'fun': lambda w: calc_port_mean(mu, w) - target_ret}

    if constraints is None:
        c_ = [c_port_weights, c_target_ret]  # Sum of weights = 100%
    else:
        c_ = [c_port_weights, c_target_ret] + constraints
    optimized = opt.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success:
        raise BaseException(optimized.message)
    return optimized.x

# def optimize_target_ret(mu, sigma, target_ret=0.0, sum_one=False):
#     n = len(mu)
#
#     def fitness(W):
#         mean_p = calc_port_mean(mu, W)
#         util = np.square(mean_p - target_ret)
#         return util
#
#     W = equal_weights(n)  # start with equal weights
#     b_ = [(0., 1.) for i in range(n)]  # weights between 0%..100%.
#     # No leverage, no shorting
#     if sum_one:
#         c_port_weights = {'type': 'eq', 'fun': lambda W: sum(W) - 1.}
#     else:
#         c_port_weights = {'type': 'ineq', 'fun': lambda W: 1. - sum(W)}
#     c_ = (c_port_weights)
#
#     optimized = opt.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_, tol=1e-10)
#     if not optimized.success:
#         raise BaseException(optimized.message)
#     return optimized.x



def optimize_risk_parity(mu, sigma, sum_one=False, constraints=None):
    n = len(mu)

    def fitness(W):
        cvol = calc_conditional_vol(mu, sigma, W)
        vol_p = calc_port_vol(sigma, W)

        a = cvol * W / vol_p - 1.0 / n
        a2 = a * a
        util = a2.sum()
        return util

    W = equal_weights(n)  # start with equal weights
    b_ = [(0., 1.) for i in range(n)]  # weights between 0%..100%.
    # No leverage, no shorting
    if sum_one:
        c_port_weights = {'type': 'eq', 'fun': lambda W: sum(W) - 1.}
    else:
        c_port_weights = {'type': 'ineq', 'fun': lambda W: 1. - sum(W)}

    if constraints is None:
        c_ = [c_port_weights]  # Sum of weights = 100%
    else:
        c_ = [c_port_weights] + constraints

    optimized = opt.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_, tol=1e-10, options={'maxiter': 10000})
    # optimized = opt.minimize(fitness, W, method='SLSQP',
    #                          tol=1e-20, options={'maxiter': 2000})
    if not optimized.success:
        raise BaseException(optimized.message)
    return optimized.x


def optimize_max_vol_dispersion(mu, sigma, sum_one=False, constraints=None):
    n = len(mu)

    def fitness(W):
        vol_p = calc_port_vol(sigma, W)
        vol_a = calc_vol(sigma)
        util = - (vol_a * W).sum() / vol_p
        return util

    W = equal_weights(n)  # start with equal weights
    b_ = [(0., 1.) for i in range(n)]  # weights between 0%..100%.
    # No leverage, no shorting
    if sum_one:
        c_port_weights = {'type': 'eq', 'fun': lambda W: sum(W) - 1.}
    else:
        c_port_weights = {'type': 'ineq', 'fun': lambda W: 1. - sum(W)}
    c_ = (c_port_weights)  # Sum of weights = 100%

    if constraints is None:
        c_ = [c_port_weights]  # Sum of weights = 100%
    else:
        c_ = [c_port_weights] + constraints

    optimized = opt.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success:
        raise BaseException(optimized.message)
    return optimized.x


##########################################################

##########################################################
def best_GMM_model(X, max_n_components=5):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, max_n_components)
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    cv_types = ['spherical']
    for cv_type in cv_types:
        for n_components in n_components_range:
            #             print('fitting n_component = {} : cv_type = {}'.format(n_components, cv_types))
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            gmm_bic = gmm.bic(X)
            bic.append(gmm_bic)
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    print('#####################################')
    print('best_gmm: bic = {} : n_component = {} : cv_type = {}'.
          format(best_gmm.bic(X), best_gmm.n_components,best_gmm.covariance_type))
    print('#####################################')
    return best_gmm


def optimize_gmm(ret, max_n_components=5, best_gmm=True, eq_weight=False, n_components=2,  covariance_type='full', **kwargs):
    if best_gmm:
        gmm = best_GMM_model(ret, max_n_components=max_n_components)
    else:
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
        gmm.fit(ret)

    dict_weights = {}
    probs = gmm.weights_
    for i in range(gmm.n_components):
        mean = np.array(gmm.means_[i, :])
        if gmm.covariance_type == 'diag':
            sigma = np.diag(gmm.covariances_[i])
        elif gmm.covariance_type == 'tied':
            sigma = gmm.covariances_
        elif gmm.covariance_type == 'spherical':
            sigma = np.eye(gmm.means_.shape[1])*gmm.covariances_[i]
        else:
            sigma = gmm.covariances_[i]

        ## asset allocation
        weights = optimize_by_model(mean, sigma, **kwargs)
        dict_weights[i] = weights

    ## aggregated weights
    pd_weights = pd.DataFrame(dict_weights, index=ret.columns)
    print(pd_weights)
    print(probs)
    if eq_weight:
        agg_weights = pd_weights.mean(axis=1)
    else :
        agg_weights = pd_weights.dot(probs)

    # output
    output = {
        'weights': agg_weights,
        'weights_by_regime': pd_weights,
        'nr_regimes': gmm.n_components,
        'prob_regimes': probs,
        'eq_weight': eq_weight
    }
    return output


from hmmlearn.hmm import GaussianHMM
def optimize_hmm(ret, n_components=5, covariance_type='full', eq_weight=False, **kwargs):
    hmm = GaussianHMM(n_components=n_components,
                      covariance_type=covariance_type,
                      n_iter=1000)
    hmm.fit(ret.values)

    dict_weights = {}
    transmat = hmm.transmat_
    dict_means = {}
    dict_sigmas = {}
    for i in range(hmm.n_components):
        mean = np.array(hmm.means_[i, :])
        if hmm.covariance_type == 'diag':
            sigma = hmm.covars_[i]
        elif hmm.covariance_type == 'tied':
            sigma = hmm.covars_
        elif hmm.covariance_type == 'spherical':
            sigma = np.eye(hmm.means_.shape[1])*hmm.covariances_[i]
        else:
            sigma = hmm.covars_[i]

        dict_means[i] = mean
        dict_sigmas[i] = sigma

        ## asset allocation
        weights = optimize_by_model(mean, sigma, **kwargs)
        dict_weights[i] = weights

    ## aggregated weights
    pd_weights = pd.DataFrame(dict_weights, index=ret.columns)
    ## use transmit matrix to find the right weights for the future
    if eq_weight:
        pd_weights_future = pd_weights.mean(axis=1)
    else:
        pd_weights_future = pd_weights.dot(transmat.T)

    ## current state
    last_state = hmm.predict(ret.values)[-1]
    weights = pd_weights_future[last_state]

    print('#####################################')
    print('hmm: last_state={} : mean={} :  transmat={}'.
          format(last_state, dict_means[i], transmat[i,:]))

    print('#####################################')

    return weights


####################################################################################33

AA_OPTIMIZE_MODEL_EQUAL_WEIGHTS = 'Equal Weights'
AA_OPTIMIZE_MODEL_TARGET_VOL = 'Target Vol'
AA_OPTIMIZE_MODEL_TARGET_RET = 'Target Return'
AA_OPTIMIZE_MODEL_MAX_SHARPE = 'Max Sharpe Ratio'
AA_OPTIMIZE_MODEL_MAX_RET = 'Max Return'
AA_OPTIMIZE_MODEL_MIN_VOL = 'Min Vol'
AA_OPTIMIZE_MODEL_RISK_PARITY = 'Risk Parity'
AA_OPTIMIZE_MODEL_MAX_VOL_DISPERSION = 'Max Vol Dispersion'
AA_OPTIMIZE_MODEL_TARGET_EXPECTED_LOSS = 'Target Expected Loss'
AA_OPTIMIZE_MODEL_TARGET_LOSS = 'Target Loss'

AA_OPTIMIZE_MODELS = [
    AA_OPTIMIZE_MODEL_EQUAL_WEIGHTS, AA_OPTIMIZE_MODEL_TARGET_VOL, AA_OPTIMIZE_MODEL_TARGET_RET,
    AA_OPTIMIZE_MODEL_MAX_SHARPE, AA_OPTIMIZE_MODEL_MAX_RET, AA_OPTIMIZE_MODEL_MIN_VOL,
    AA_OPTIMIZE_MODEL_RISK_PARITY, AA_OPTIMIZE_MODEL_MAX_VOL_DISPERSION,
    AA_OPTIMIZE_MODEL_TARGET_EXPECTED_LOSS, AA_OPTIMIZE_MODEL_TARGET_LOSS
]


def optimize_by_rets(rets, **options):
    risk_measure = options.get('risk_measure', None)
    if risk_measure is None:
        raise Exception('Risk measure should be none or empty')

    # call optimize_by_historical methods
    if risk_measure in [AA_OPTIMIZE_MODEL_TARGET_EXPECTED_LOSS, AA_OPTIMIZE_MODEL_TARGET_LOSS]:
        target_loss = options.get('target_loss', -0.05)
        target_prob = options.get('target_prob', 0.05)
        sum_one = options.get('sum_one', False)
        if risk_measure == 'target_expected_loss':
            return optimize_historical_target_expected_loss(rets, target_loss, target_prob, sum_one)
        else :
            return optimize_historical_target_loss(rets, target_loss, target_prob, sum_one)
    else:
        robust_est = options.get('robust_est', {})
        half_life = robust_est.get('halflife', False)
        # using robust estimator
        if half_life:
            mu = calc_mean_by_asset(rets)
            n = robust_est.get('n', 4)
            lamb = robust_est.get('lamb', 0.9)
            sigma = calc_vcv_by_asset(rets, half_life, n=n, lamb=lamb)
        else:
            mu = calc_mean_by_asset(rets)
            sigma = calc_vcv_by_asset(rets)

        ## call the optimization
        return optimize_by_model(mu, sigma, **options)


def optimize_by_model(mu, sigma, **options) :
    rf = options.get('rf', 0.0)
    risk_measure = options.get('risk_measure', AA_OPTIMIZE_MODEL_MAX_SHARPE)
    sum_one = options.get('sum_one', False)
    constraints = options.get('constraints', None)

    weights = None
    if risk_measure == AA_OPTIMIZE_MODEL_EQUAL_WEIGHTS:
        weights = optimize_equal_weights(mu)
    if risk_measure == AA_OPTIMIZE_MODEL_TARGET_VOL:
        target_vol = options.get('target_vol', 0.0)
        weights = optimize_target_vol(mu, sigma, target_vol=target_vol, sum_one=sum_one, constraints=constraints)
    if risk_measure == AA_OPTIMIZE_MODEL_TARGET_RET:
        target_ret = options.get('target_ret', 0.0)
        weights = optimize_target_ret(mu, sigma, target_ret=target_ret, sum_one=sum_one, constraints=constraints)
    if risk_measure == AA_OPTIMIZE_MODEL_MAX_SHARPE:
        weights = optimize_max_sharpe(mu, sigma, rf=rf, sum_one=sum_one, constraints=constraints)
    if risk_measure == AA_OPTIMIZE_MODEL_MAX_RET:
        weights = optimize_max_ret(mu, sum_one=sum_one, constraints=constraints)
    if risk_measure == AA_OPTIMIZE_MODEL_MIN_VOL:
        weights = optimize_min_vol(mu, sigma, sum_one=sum_one, constraints=constraints)
    if risk_measure == AA_OPTIMIZE_MODEL_RISK_PARITY:
        weights = optimize_risk_parity(mu, sigma, sum_one=sum_one, constraints=constraints)
    if risk_measure == AA_OPTIMIZE_MODEL_MAX_VOL_DISPERSION:
        weights = optimize_max_vol_dispersion(mu, sigma, sum_one=sum_one, constraints=constraints)

    return weights


################################################################################################
# Elastic Asset Allocation
################################################################################################
AA_EAA_TYPE_GOLD_OFFENSIVE = 'Golden Offensive EAA'
AA_EAA_TYPE_GOLD_DEFENSIVE = 'Golden Defensive EAA'
AA_EAA_TYPE_EQUAL_WEIGHTED_RETURN = 'Equal Weighted Return'
AA_EAA_TYPE_EQUAL_WEIGHTED_HEDGED = 'Equal Weighted Hedged'
AA_EAA_TYPE_SCORING_FUNCTION_TEST = 'Scoring Function Test'
AA_EAA_TYPES = [AA_EAA_TYPE_GOLD_DEFENSIVE, AA_EAA_TYPE_GOLD_OFFENSIVE,
               AA_EAA_TYPE_EQUAL_WEIGHTED_RETURN, AA_EAA_TYPE_EQUAL_WEIGHTED_HEDGED,
               AA_EAA_TYPE_SCORING_FUNCTION_TEST]

def elastic_asset_allocation(df_assets, df_cash=None, score_weights=(2.0, 1.0, 0.0, 1.0, 1e-6)):
    # nr of assets
    n = df_assets.shape[1]
    # names
    asset_names = list(df_assets.columns)
    nav_assets = (1 + df_assets).cumprod()
    nav_assets_m = nav_assets.resample('M').last()
    ret_assets_m = nav_assets_m.pct_change()

    if df_cash is not None:
        cash_name = list(df_cash.columns)[0]
        nav_cash = (1 + df_cash).cumprod()
        nav_cash_m = nav_cash.resample('M').last()
        ret_cash_m = nav_cash_m.pct_change()

        # calc the momentum
        ex_ret1 = nav_assets_m.iloc[-1] / nav_assets_m.iloc[-2] - (nav_cash_m.iloc[-1] / nav_cash_m.iloc[-2]).values
        ex_ret2 = nav_assets_m.iloc[-1] / nav_assets_m.iloc[-4] - (nav_cash_m.iloc[-1] / nav_cash_m.iloc[-4]).values
        ex_ret3 = nav_assets_m.iloc[-1] / nav_assets_m.iloc[-7] - (nav_cash_m.iloc[-1] / nav_cash_m.iloc[-7]).values
        ex_ret4 = nav_assets_m.iloc[-1] / nav_assets_m.iloc[-12] - (nav_cash_m.iloc[-1] / nav_cash_m.iloc[-12]).values
    else:
        # calc the momentum
        ex_ret1 = nav_assets_m.iloc[-1] / nav_assets_m.iloc[-2] - 1
        ex_ret2 = nav_assets_m.iloc[-1] / nav_assets_m.iloc[-4] - 1
        ex_ret3 = nav_assets_m.iloc[-1] / nav_assets_m.iloc[-7] - 1
        ex_ret4 = nav_assets_m.iloc[-1] / nav_assets_m.iloc[-12] - 1
    mom = (ex_ret1 + ex_ret2 + ex_ret3 + ex_ret4) / 21
    mom[mom < 0] = 0

    # nominal return correlation to equi-weight portfolio
    ew_index = ret_assets_m.mean(axis=1)
    corr = pd.Series([ret_assets_m[x].corr(ew_index) for x in ret_assets_m.columns], index=ret_assets_m.columns)

    # std
    vol = ret_assets_m.std()

    # Generalized Momentum
    #
    # wi ~ zi = ( ri^wR * (1-ci)^wC / vi^wV )^wS
    wR = score_weights[0]
    wC = score_weights[1]
    wV = score_weights[2]
    wS = score_weights[3]
    eps = score_weights[4]

    z = ((mom ** wR) * ((1 - corr) ** wC) / (vol ** wV)) ** (wS + eps)

    # Crash Protection
    #
    num_neg = mom[mom <= 0].count()
    cpf = float(num_neg) / n
    print("cash protection % = {}".format(cpf))

    # top N assets to invest
    top_n = min(int(np.ceil(np.sqrt(n)) + 1), int(n / 2))

    #
    # Allocation
    #
    top_z = z.sort_values().index[-top_n:]
    print("top_z = {}".format(top_z.values))
    w_z = ((1 - cpf) * z[top_z] / z[top_z].sum()).dropna()

    # Total weights
    if df_cash is not None:
        w = {cash_name: cpf}
    else:
        w = {}
    for name in asset_names:
        if name in w_z.index:
            w.update({name: w_z.loc[name]})
        else:
            w.update({name: 0.0})
    w = pd.Series(w)

    return w



def elastic_asset_allocation_by_type(df_rets, df_cash=None, type=None):
    # Weights:
    #   [wR, wC, wV, wS, eps]
    #   Golden Offensive EAA: wi ~ zi = (1-ci) * ri^2
    # context.score_weights = (2.0, 1.0, 0.0, 1.0, 1e-6)
    #   Golden Defensive EAA: wi ~ zi = squareroot( ri * (1-ci) )
    # context.score_weights = (1.0, 1.0, 0.0, 0.5, 1e-6)
    #   Equal Weighted Return: wi ~ zi = ri ^ eps
    # context.score_weights = (1.0, 0.0, 0.0, 0.0, 1e-6)
    #   Equal Weighted Hedged: wi ~ zi = ( ri * (1-ci) )^eps
    # context.score_weights = (1.0, 1.0, 0.0, 0.0, 1e-6)
    #   Scoring Function Test:
    # context.score_weights = (1.0, 1.0, 1.0, 1.0, 0.0)
    if type == AA_EAA_TYPE_GOLD_OFFENSIVE:
        score_weights = (2.0, 1.0, 0.0, 1.0, 1e-6)
    elif type == AA_EAA_TYPE_GOLD_DEFENSIVE:
        score_weights = (1.0, 1.0, 0.0, 0.5, 1e-6)
    elif type == AA_EAA_TYPE_EQUAL_WEIGHTED_RETURN:  # wi ~ zi = ri ^ eps
        score_weights = (1.0, 0.0, 0.0, 0.0, 1e-6)
    elif type == AA_EAA_TYPE_EQUAL_WEIGHTED_HEDGED:  # wi ~ zi = ( ri * (1-ci) )^eps
        score_weights = (1.0, 1.0, 0.0, 0.0, 1e-6)
    elif type == AA_EAA_TYPE_SCORING_FUNCTION_TEST:
        score_weights = (1.0, 1.0, 1.0, 1.0, 0.0)
    else:
        score_weights = (1.0, 1.0, 1.0, 1.0, 0.0)
    return elastic_asset_allocation(df_rets, df_cash, score_weights)


################################################################################################
# Asset Allocation based on LLT
################################################################################################
def LLT_asset_allocation(df_assets, llt_threshold=0.0, cash_protection=False) :
    def LLT(close, alpha=float(2 / (12 + 1))):
        llt_val = []

        n = len(close)
        for i in range(n):
            if i < 2:
                llt_val.append(close.values[i])
            else:
                newlltval = ((alpha - alpha ** 2 / float(4)) * close.iloc[i]
                             + alpha ** 2 / float(2) * close.iloc[i - 1] - (alpha - alpha ** 2 * 3 / float(4)) *
                             close.iloc[i - 2]
                             + 2 * (1 - alpha) * llt_val[i - 1] - (1 - alpha) ** 2 * (llt_val[i - 2]))
                llt_val.append(newlltval)
        llt = pd.Series(llt_val, index=close.index)
        return IndicatorSeries(llt)

    df_assets_nav = (1+df_assets).cumprod()
    nr_assets = df_assets.shape[1]
    name_assets = list(df_assets.columns)

    strengths = pd.Series(0., index=name_assets)
    for i in range(nr_assets):
        nav = df_assets_nav.iloc[:, i]
        llt_value = LLT(nav)
        llt_strength = (llt_value/llt_value.shift(1) -1 )[-1]
        strengths[i] = llt_strength if llt_strength > llt_threshold else 0

    ## set strengths as zero, if below the threshold
    total_strength = sum(strengths)
    if total_strength == 0.0:
        weights = pd.Series(0.0, index=name_assets)
    else:
        weights = strengths / total_strength
        if cash_protection:
            cash_weight = 1-sum(np.sign(strengths)) / len(strengths)
            weights = (1-cash_weight) * weights

            weights = weights.append(pd.Series(cash_weight, index=['cash']))
    return weights


def BL_asset_allocation(df_ret, tau=0.03, p=None, q=None, optim_setting={}):
    nr_asset = df_ret.columns.shape[0]

    # prior estimation:
    sigma = df_ret.cov()
    # w_mkt = pd.Series(1.0 / nr_asset, index=df_ret.columns)
    # pi = delta * sigma.dot(w_mkt)  # equilibrium returns
    pi = df_ret.mean()

    # opinioin
    if p is None or q is None:
        pi_t = pi
        sigma_t = sigma
    else:
        assert p.shape[1] == nr_asset, 'p has dimension {} not matching nr of assets {}'.format(p.shape, nr_asset)
        assert p.shape[0] <= p.shape[1], 'p has {} conditions, more than {} nr of assets'.format(p.shape[0], p.shape[1])
        assert q.shape[0] == p.shape[0], 'q has {} conditons, different {} in p'.format(q.shape[0], p.shape[0])

        P = pd.DataFrame(p.T, index=df_ret.columns)
        Q = pd.Series(q)
        omega = pd.DataFrame(np.diag(np.diag(tau * P.T.dot(sigma.dot(P)))))

        # posterior estimation:
        pi_p_0 = tau * sigma.dot(P)
        pi_p_1 = omega + tau * P.T.dot(sigma.dot(P))

        pi_p_1 = pd.DataFrame(sp.linalg.inv(pi_p_1))
        pi_p_2 = Q - P.T.dot(pi)

        pi_t = pi + pi_p_0.dot(pi_p_1.dot(pi_p_2))  # posterior equibrium return

        M_p_0 = tau * sigma
        M_p_1 = tau * sigma.dot(P)
        M_p_2 = pi_p_1
        M_p_3 = tau * P.T.dot(sigma)
        M = M_p_0 - M_p_1.dot(M_p_2.dot(M_p_3))
        sigma_t = sigma + M  # posterior sigma

    mu = np.array(pi_t.values)
    sigma = np.matrix(sigma_t)
    weights = optimize_by_model(mu=mu, sigma=sigma, **optim_setting)

    # generate output
    output = {
        'weights': pd.Series(weights, index=df_ret.columns),
        'pi': pi,
        'sigma': sigma,
        'pi_t': pi_t,
        'sigma_t': sigma_t,
        'tau': tau,
        'P': P,
        'Q': Q,
        'omega': omega

    }
    return output

#######################################################################################

AA_MODEL_LLT = 'LLT'
AA_LLT_TYPES = [AA_MODEL_LLT]

AA_MODEL_BL = 'Black-Litterman'
AA_BL_TYPES = [AA_MODEL_BL]

AA_MIXTURE_TYPE_GMM = 'GMM'
AA_MIXTURE_TYPE_HMM = 'HMM'
AA_MODEL_MIXTURE = 'Mixture'
AA_MIXTURE_TYPES = [AA_MIXTURE_TYPE_GMM, AA_MIXTURE_TYPE_HMM]

AA_MODEL_EAA = 'EAA'
AA_MODEL_OPTIMIZE = 'Optimization'

AA_ALL_MODELS = {
    AA_MODEL_OPTIMIZE: AA_OPTIMIZE_MODELS,
    AA_MODEL_EAA: AA_EAA_TYPES,
    AA_MODEL_BL: AA_BL_TYPES,
    AA_MODEL_LLT: AA_LLT_TYPES,
    AA_MODEL_MIXTURE: AA_MIXTURE_TYPES
}

AA_ALL_MODELS_REVERSE = {
    **{i: AA_MODEL_OPTIMIZE for i in AA_OPTIMIZE_MODELS},
    **{i: AA_EAA_TYPES for i in AA_MODEL_EAA},
    **{i: AA_BL_TYPES for i in AA_MODEL_BL},
    **{i: AA_LLT_TYPES for i in AA_MODEL_LLT},
    **{i: AA_MIXTURE_TYPES for i in AA_MODEL_MIXTURE}
}

    # '''
    #
    #
    # :param df_assets:
    # :param df_cash:
    # :param settings:
    # :return:
    #
    # #####################################3
    # Example 1:
    # settings = {
    #     'asset_allocate': 'optimize',
    #     'optimize':{
    #         'risk_measure': 'risk_parity',
    #         'rf': 0.0,
    #     }
    # }
    #
    # Example 2:
    # settings = {
    #     'asset_allocate': 'mixture',
    #     'mixture': {
    #         'model': 'gmm',
    #         'max_n_components': 5,
    #         'best_gmm': False
    #     },
    #     'optimize':{
    #         'risk_measure': 'risk_parity',
    #         'rf': 0.0,
    #     }
    # }
    #
    # Example 3:
    # settings = {
    #     'asset_allocate': 'EAA',
    #     'EAA': {
    #         'eaa_type': 'Golden Offensive EAA',
    #         'score_weights': (2.0, 1.0, 0.0, 1.0, 1e-6)
    #     }
    # }
    #
    # Example 4:
    # settings = {
    #     'asset_allocate': 'BL',
    #     'BL': {
    #         'tao': 0.03,
    #         'delta': 13.0,
    #         'p': np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]]),
    #         'q': np.array([0.01, -0.01]),
    #         'omega': np.array([0.5, 0.9])
    #     },
    #     'optimize': {
    #         'risk_measure': 'max_sharpe',
    #         'rf': 0.0
    #     }
    # }
    # '''

def efficient_frontier(df_assets, **setting):
    pass


def asset_allocation(df_assets, df_cash=None, **settings):    
    '''


    :param df_assets:
    :param df_cash:
    :param settings:
    :return:

    #####################################3
    Example 1:
    settings = {
        'asset_allocate': 'optimize',
        'optimize':{
            'risk_measure': 'risk_parity',
            'rf': 0.0,
        }
    }

    Example 2:
    settings = {
        'asset_allocate': 'mixture',
        'mixture': {
            'model': 'gmm',
            'max_n_components': 5,
            'best_gmm': False
        },
        'optimize':{
            'risk_measure': 'risk_parity',
            'rf': 0.0,
        }
    }

    Example 3:
    settings = {
        'asset_allocate': 'EAA',
        'EAA': {
            'eaa_type': 'Golden Offensive EAA',
            'score_weights': (2.0, 1.0, 0.0, 1.0, 1e-6)
        }
    }

    '''
    import copy
    
    asset_alloc_mode = settings.get('asset_allocate', None)

    if asset_alloc_mode is None:
        raise Exception('Asset Allocation not specified. Exit')

    elif asset_alloc_mode == AA_MODEL_OPTIMIZE:
        opt_setting = settings.get(AA_MODEL_OPTIMIZE, {})
        weights =  optimize_by_rets(df_assets, **opt_setting)
        ds_weights = pd.Series(weights, index=df_assets.columns)
        output = {
            'weights': ds_weights
        }
#         output = ds_weights
    elif asset_alloc_mode == AA_MODEL_MIXTURE:
        mixture_setting = settings.get(AA_MODEL_MIXTURE, {})
        opt_setting = settings.get(AA_MODEL_OPTIMIZE, {})
        mixture_model = mixture_setting.get('model', None)

        if mixture_model == AA_MIXTURE_TYPE_GMM:
            max_n_components = mixture_setting.get('max_n_components', 5)
            best_gmm = mixture_setting.get('best_gmm', True)
            eq_weight = mixture_setting.get('eq_weight', False)
            n_components = mixture_setting.get('n_components', 2)
            covariance_type = mixture_setting.get('covariance_type', 'full')
            output = optimize_gmm(df_assets, max_n_components=max_n_components,
                                  best_gmm=best_gmm, eq_weight=eq_weight, n_components=n_components,
                                  covariance_type=covariance_type, **opt_setting)
#             output = output0['weights']
        elif mixture_model == AA_MIXTURE_TYPE_HMM:
            n_components = mixture_setting.get('n_components', 5)
            covariance_type = mixture_setting.get('covariance_type', 'full')
            eq_weight = mixture_setting.get('eq_weight', False)
            weights = optimize_hmm(df_assets, n_components=n_components,
                                  covariance_type=covariance_type, eq_weight=eq_weight, **opt_setting)
            ds_weights = pd.Series(weights, index=df_assets.columns)
            output = {
                'weights': ds_weights
            }
    #             output = pd.Series(output0, index=df_assets.columns)

        else:
            print('mixture model [mode] not set. Exit!')
            raise Exception('Mixture Model not set')

    elif asset_alloc_mode == AA_MODEL_EAA:
        eaa_setting = settings.get(AA_MODEL_EAA, {})
        eaa_type = eaa_setting.get('eaa_type', AA_EAA_TYPE_EQUAL_WEIGHTED_RETURN)
        score_weights = eaa_setting.get('score_weights', None)
        if eaa_type is None:
            output = elastic_asset_allocation(df_assets, df_cash, score_weights)
        else:
            output = elastic_asset_allocation_by_type(df_assets, df_cash, eaa_type)

    elif asset_alloc_mode == AA_MODEL_LLT:
        llt_setting = settings.get(AA_MODEL_LLT, {})
        llt_threshold = llt_setting.get('llt_threshold', 0.0)
        cash_protection = llt_setting.get('cash_protection', False)
        output = LLT_asset_allocation(df_assets, llt_threshold=llt_threshold,
                                    cash_protection=cash_protection)

    elif asset_alloc_mode == AA_MODEL_BL:
        bl_setting = copy.deepcopy(settings.get(AA_MODEL_BL, {}))
        optim_setting = settings.get(AA_MODEL_OPTIMIZE, {})
        # tau
        bl_setting['tau'] = 1.0 / (df_assets.mean().shape[0] - 1)
        # p and q
        opinions = bl_setting.pop('opinions',{})
        ops = pd.DataFrame(index=df_assets.mean().index)
        ops['1'] = df_assets.mean() - 2 * df_assets.std()
        ops['2'] = df_assets.mean() - df_assets.std()
        ops['3'] = df_assets.mean()
        ops['4'] = df_assets.mean() + df_assets.std()
        ops['5'] = df_assets.mean() + 2 * df_assets.std()
        p = pd.DataFrame({k: [0] * len(opinions) for k in df_assets.mean().index.tolist()})
        q = []
        c = 0
        for k, v in opinions.items():
            p.loc[c,k] = 1
            q.append(ops.loc[k,str(v)])
            c += 1
        bl_setting['p'] = p.as_matrix()
        bl_setting['q'] = np.array(q)
    
        
        output = BL_asset_allocation(df_assets, optim_setting=optim_setting, **bl_setting)

    else:
        print('asset alloc model not set. Exit!')
        raise Exception('Asset Alloc Model not set')

    return output


def asset_allocation_by_walkforward(df_assets, df_cash=None, cash_rate=None, **setting) :
    wf_setting = setting.get('walkforward', {})
    start_time = pd.to_datetime(wf_setting.get('start_time', df_assets.index[0]))
    end_time = pd.to_datetime(wf_setting.get('end_time', df_assets.index[-1]))
    span_is = wf_setting.get('span_is', '1Y')
    span_os = wf_setting.get('span_os', '3M')
    trans_cost = wf_setting.get('trans_cost', 0.0)

    ## walk forward windows splitting
    windows = WalkForwardAnalysis.get_walkforward_windows_by_time(span_is, span_os, start_time, end_time)
    wf_weights = {}
    pd_rets = pd.DataFrame()

    for nr_walk in range(len(windows)):
        start_is = windows.loc[nr_walk, 'start_is']
        df_assets_is = df_assets[windows.loc[nr_walk, 'start_is']:windows.loc[nr_walk, 'end_is']]
        df_assets_os = df_assets[windows.loc[nr_walk, 'start_os']:windows.loc[nr_walk, 'end_os']]

        if df_cash is None:
            if cash_rate is None:
                df_cash_is = None
                df_cash_os = None
            else:
                df_cash_is = pd.DataFrame({'cash': cash_rate}, index=df_assets_is.index)
                df_cash_os = pd.DataFrame({'cash': cash_rate}, index=df_assets_os.index)
        else :
            df_cash_is = df_cash[windows.loc[nr_walk, 'start_is']:windows.loc[nr_walk, 'end_is']]
            df_cash_os = df_cash[windows.loc[nr_walk, 'start_os']:windows.loc[nr_walk, 'end_os']]

        # calc the optimal weights
        weights = asset_allocation(df_assets_is, df_cash_is, **setting)

        # outsample rets
        pd_ret_wf = pd.DataFrame()
        if df_cash_os is not None:
            df_assets_os = df_assets_os.join(df_cash_os, how='outer')

        pd_ret_wf['ret'] = calc_ret_by_portfolio(df_assets_os, weights)
        pd_ret_wf['nr_walk'] = nr_walk
        # transaction cost
        if nr_walk == 0:
            cost = np.sum(np.abs(weights * trans_cost))
        else:
            cost = np.sum(np.abs(weights - wf_weights[nr_walk - 1]) * trans_cost)

        pd_ret_wf['cost'] = 0.0
        pd_ret_wf['cost'][0] = cost
        pd_ret_wf['ret_wt_cost'] = pd_ret_wf['ret'] - pd_ret_wf['cost']

        # put all together
        wf_weights[nr_walk] = weights
        pd_rets = pd_rets.append(pd_ret_wf)

    pd_weights = pd.DataFrame(wf_weights).T
    pd_rets = pd_rets.reset_index().drop_duplicates(subset='date').set_index('date')
    return pd_weights, pd_rets


def asset_allocation_by_walkforward1(df_assets, df_cash=None, cash_rate=None, **setting) :
    wf_setting = setting.get('walkforward', {})
    wf_mode = wf_setting.get('mode', None)
    if wf_mode is None:
        raise Exception('Walkforward mode [wf_mode] not set')
    elif wf_mode == 'index':
        start_index = wf_setting.get('start_index', 0)
        end_index = wf_setting.get('end_index', len(df_assets) -1)
        nr_is = wf_setting.get('nr_is', 100)
        nr_os = wf_setting.get('nr_os', 10)
        windows = WalkForwardAnalysis.get_walkforward_windows_by_number(nr_is, nr_os, start_index, end_index)
    elif wf_mode == 'date':
        start_time = pd.to_datetime(wf_setting.get('start_time', df_assets.index[0]))
        end_time = pd.to_datetime(wf_setting.get('end_time', df_assets.index[-1]))
        span_is = wf_setting.get('span_is', '1Y')
        span_os = wf_setting.get('span_os', '3M')
        windows = WalkForwardAnalysis.get_walkforward_windows_by_time(span_is, span_os, start_time, end_time)

    else:
        raise Exception('Walkforward mode [{}] not recoginzed'.format(wf_mode))

    trans_cost = wf_setting.get('trans_cost', 0.0)

    ## walk forward windows splitting
    wf_weights = {}
    pd_rets = pd.DataFrame()

    for nr_walk in range(len(windows)):
        start_is = windows.loc[nr_walk, 'start_is']
        df_assets_is = df_assets[windows.loc[nr_walk, 'start_is']:windows.loc[nr_walk, 'end_is']]
        df_assets_os = df_assets[windows.loc[nr_walk, 'start_os']:windows.loc[nr_walk, 'end_os']]

        if df_cash is None:
            if cash_rate is None:
                df_cash_is = None
                df_cash_os = None
            else:
                df_cash_is = pd.DataFrame({'cash': cash_rate}, index=df_assets_is.index)
                df_cash_os = pd.DataFrame({'cash': cash_rate}, index=df_assets_os.index)
        else :
            df_cash_is = df_cash[windows.loc[nr_walk, 'start_is']:windows.loc[nr_walk, 'end_is']]
            df_cash_os = df_cash[windows.loc[nr_walk, 'start_os']:windows.loc[nr_walk, 'end_os']]

        # calc the optimal weights
        aa_output = asset_allocation(df_assets_is, df_cash_is, **setting)
        weights = aa_output['weights']

        # outsample rets
        pd_ret_wf = pd.DataFrame()
        if df_cash_os is not None:
            df_assets_os = df_assets_os.join(df_cash_os, how='outer')

        pd_ret_wf['ret'] = calc_ret_by_portfolio(df_assets_os, weights)
        pd_ret_wf['nr_walk'] = nr_walk
        # transaction cost
        if nr_walk == 0:
            cost = np.sum(np.abs(weights * trans_cost))
        else:
            cost = np.sum(np.abs(weights - wf_weights[nr_walk - 1]) * trans_cost)

        pd_ret_wf['cost'] = 0.0
        pd_ret_wf['cost'][0] = cost
        pd_ret_wf['ret_wt_cost'] = pd_ret_wf['ret'] - pd_ret_wf['cost']

        # put all together
        wf_weights[nr_walk] = weights
        pd_rets = pd_rets.append(pd_ret_wf)

    pd_weights = pd.DataFrame(wf_weights).T
    pd_rets = pd_rets.reset_index().drop_duplicates(subset='date').set_index('date')

    # generate output
    output = {
        'wf_weights': pd_weights,
        'port_ret': pd_rets
    }

#     return output
    return pd_weights, pd_rets
