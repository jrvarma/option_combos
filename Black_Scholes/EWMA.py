import numpy as np


def EWMA(df, input_name='Price', vol_name='Sigma', ret_name='lnret',
         input_is_price=True, Lambda=0.94, burn_in=20,
         sigma_init=None, sigma_init_sample=20, inplace=True):
    r"""Compute time varying historical volatility using EWMA method.

    EWMA (Exponentially Weighted Moving Average) is also sometimes called the
    RiskMetrics method (because it was first popularized by Riskmetrics) or
    the IGarch (Integrated Garch) method (because it is similar to the Garch
    method except that the volatility is assumed to follow a unit root process
    instead of being a stationary process).


    Parameters
    ----------
    df : Pandas DataFrame
          One column must contain prices/returns and the index must be the date
    input_name : string
          Column name  of df that contains price/return
    vol_name : string
          Column name  of df in which to store estimated volatitlity
    ret_name : string
          Column name  of df in which to store returns (if input is price)
    input_is_price : boolean
          Whether the input column is price (True) or return (False)
    Lambda : float
          Smoothing parameter of exponential moving average
    burn_in : int
          Number of initial values of volatility to be set to nan
    sigma_init : float or None
          Initial volatility (if None sample volatility is used)
    sigma_init_sample : int
          Length of initial sample to compute initial volatility
          (used only if sigma_init is None)
    inplace : boolean
          If true df is modified in place and returned. Else copy is made

    Returns
    -------
    Pandas DataFrame
       Input DataFrame with added column(s) containing the estimated volatility
       and if necessary the log return


    """
    if not inplace:
        df = df.copy()
    # compute days in the year for annualizing volatility
    df.sort_index(inplace=True)
    ndays = len(df.index)
    nyears = (df.index[ndays-1].date().toordinal()
              - df.index[0].date().toordinal()) / 365.25
    days_in_year = ndays / nyears
    if input_name not in df.columns:
        raise Exception("Column {:} not found in DataFrame {:}".format(
            input_name, df))
    if input_is_price:
        df[ret_name] = np.log(df[input_name]).diff()
    else:
        ret_name = input_name
    # compute starting value of daily volatility from annualized value
    if sigma_init is None:
        sigma_init_sample = max(sigma_init_sample, len(df[ret_name]))
        curr_variance = df[ret_name].iloc[0:sigma_init_sample].std()**2
    else:
        curr_variance = (sigma_init/100)**2 / days_in_year
    for dt in df.index:
        logret = df.loc[dt, ret_name]
        if not np.isnan(logret):
            curr_variance = (Lambda * curr_variance + (1 - Lambda) * logret**2)
        df.loc[dt, vol_name] = np.sqrt(curr_variance)
    df[vol_name] *= np.sqrt(days_in_year) * 100  # annualized and in percent
    df.iloc[0:burn_in, df.columns.get_loc(vol_name)] = np.nan
    return df
