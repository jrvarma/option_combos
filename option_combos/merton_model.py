# Copyright (C) 2016, 2020, 2023  Jayanth R Varma and Vineet Virmani

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from option_combos import option_type, GBS
from scipy.optimize import brentq
import numpy as np
import pandas as pd
from collections import OrderedDict

# parameters used to bracket the root while using brentq
# interval for sigmaA
sigma_low = 1e-4  # 0.01%
sigma_high = 100  # 10,000%
# interval for assets when DebtMV
# assets_by_debt_low = 1 (theoretical lower bound as equity goes to 0)
assets_by_debt_high = 1000  # market value leverage of at least 0.1%


def ccrate(y, frequency):
    r"""Convert interest rate into equivalent continuously compounded rate

    Parameters
    ----------
    y : float
        Interest rate compounded at frequency
    frequency : int or np.inf
        Compounding frequency
    Returns
    -------
    Float: continuously compounded interest rate.
    """
    if frequency == np.inf:
        return y/100
    return frequency * np.log(1 + y/100/frequency)


def equiv(y, frequency):
    r"""Convert continuously compounded interest rate into equivalent rate

    Parameters
    ----------
    y : float
        Interest rate continuously compounded
    frequency : int or np.inf
        Compounding frequency

    Returns
    -------
    Float: Interest rate compounded at frequency
    """
    if frequency == np.inf:
        return y * 100
    return 100 * frequency * (np.exp(y/frequency) - 1)


def onerow(row):
    r"""Compute Merton model on one row of input DataFrame

    Parameters
    ----------
    row : Pandas Series representing one row of input DataFrame

    Returns
    -------
    Pandas Series representing one row of DataFrame with model values added

    Notes
    -----
    Intended to be called internally by merton using DataFrame apply
    """
    if np.isnan(row.frequency):
        row.frequency = np.inf
    row['rho'] = ccrate(row.r, row.frequency)
    row['q'] = ccrate(row.DivYld, row.frequency)
    if np.isnan(row.coupon):
        row.coupon = row.rho
    else:
        row.coupon = ccrate(row.coupon, row.frequency)
    row['ZeroFV'] = row.DebtBV * np.exp(row.coupon * row.maturity)
    if not np.isnan(row.DebtYTM):
        row.DebtMV = row.ZeroFV * np.exp(-ccrate(
            row.DebtYTM, row.frequency)*row.maturity)
    if not np.isnan(row.CreditSpread):
        row.DebtMV = row.ZeroFV * np.exp(-ccrate(
            row.r+row.CreditSpread/1e2, row.frequency)*row.maturity)
    if not np.isnan(row.DebtMV) and not np.isnan(row.EquityMV):
        row.Assets = row.DebtMV + row.EquityMV
    if np.isnan(row.Assets) and np.isnan(row.sigmaA):
        raise Exception(
            'When neither Assets nor sigmaA provided, ' +
            'both EquityMV and DebtMV/DebtYTM/CreditSpread are required')
    if np.isnan(row.Assets) or np.isnan(row.sigmaA):
        if np.isnan(row.EquityMV) and np.isnan(row.DebtMV):
            raise Exception('Please supply either EquityMV or ' +
                            'DebtMV/DebtYTM/CreditSpread')
    if np.isnan(row.sigmaA):
        if not np.isnan(row.EquityMV):
            row.sigmaA = 100 * brentq(
                lambda sigma:
                GBS(S=row.Assets, K=row.ZeroFV, sigma=sigma,
                    ttm=row.maturity, r=row.rho, q=row.q,
                    optType=option_type.call).NPV() - row.EquityMV,
                a=sigma_low, b=sigma_high)
        else:  # not np.isnan(DebtMV)
            putvalue = (row.ZeroFV * np.exp(-row.rho * row.maturity)
                        - row.DebtMV)
            row.sigmaA = 100 * brentq(
                lambda sigma:
                GBS(S=row.Assets, K=row.ZeroFV, sigma=sigma,
                    ttm=row.maturity, r=row.rho, q=row.q,
                    optType=option_type.put).NPV() - putvalue,
                a=sigma_low, b=sigma_high)
    if np.isnan(row.Assets):
        if not np.isnan(row.EquityMV):
            row.Assets = brentq(
                lambda assets:
                GBS(S=assets, K=row.ZeroFV, sigma=row.sigmaA/100,
                    ttm=row.maturity, r=row.rho, q=row.q,
                    optType=option_type.call).NPV()
                - row.EquityMV,
                a=row.EquityMV, b=row.EquityMV+row.ZeroFV)
        else:  # not np.isnan(DebtMV)
            putvalue = (row.ZeroFV * np.exp(-row.rho * row.maturity)
                        - row.DebtMV)
            row.Assets = brentq(
                lambda assets:
                GBS(S=assets, K=row.ZeroFV, sigma=row.sigmaA/100,
                    ttm=row.maturity, r=row.rho, q=row.q,
                    optType=option_type.put).NPV() - putvalue,
                a=row.DebtMV, b=row.DebtMV*assets_by_debt_high)
    gbs = GBS(S=row.Assets, K=row.ZeroFV, sigma=row.sigmaA/100,
              ttm=row.maturity, r=row.rho, q=row.q,
              optType=option_type.put)
    row.DebtMV = (row.ZeroFV * np.exp(-row.rho * row.maturity) - gbs.NPV())
    row.EquityMV = row.Assets - row.DebtMV
    row.DebtYTM = equiv(-np.log(row.DebtMV/row.ZeroFV)/row.maturity,
                        row.frequency)
    row.CreditSpread = (row.DebtYTM - row.r) * 100
    row['RNPD_pct'] = gbs.N_optType_d2 * 100
    row['IntensityDefault_pct'] = -100*np.log(
        1-row.RNPD_pct/100)/row.maturity
    row['Call_Delta'] = 1 + gbs.Delta()  # call delta = 1 + put delta
    return row


def makedf(d, index_from_Series=False):
    r"""Construct Pandas DataFrame out of input values

    Parameters
    ----------
    d : dict
        Containing all input arguments to merton

    Returns
    -------
    Pandas DataFrame containing one column for each input argument

    Notes
    -----
    Intended only to be called internally from merton

    """
    if index_from_Series:
        index = None
        for y in d.values():
            if hasattr(y, 'index'):
                index = y.index
                break
        if index is None:
            raise Exception("No series found. index_from_Series failed")
    else:
        lengths = set([len(y) for y in d.values() if hasattr(y, "__len__")])
        if len(lengths) == 0:
            index = [0]
        elif len(lengths) == 1:
            index = range(lengths.pop())
        else:
            raise Exception("Input contains vectors of unequal lengths: {:}"
                            .format(str(lengths)))
    df = pd.DataFrame(index=index)
    for k, v in d.items():
        if hasattr(v, 'values'):
            df[k] = v.values
        else:
            df[k] = v
    return df


def merton(DebtBV, maturity, r, DivYld=0, frequency=np.inf, Assets=np.nan,
           sigmaA=np.nan, EquityMV=np.nan, DebtMV=np.nan, DebtYTM=np.nan,
           CreditSpread=np.nan, coupon=np.nan, index_from_Series=False):
    r"""Merton (structural) credit model.

    Given data on two of the following variables, compute the third
       #. debt market value, YTM and credit spread, or
       #. equity market value, or
       #. asset volatility

    Parameters
    ----------
    DebtBV : None or float or sequence
        The book value of the Debt
    maturity : float or sequence
        Maturity of zero coupon debt in years
    r  : float or sequence
        The risk free rate
        This is annualized, and in percent
    DivYld : float or sequence
        The dividend yield
        This is annualized, and in percent
    frequency: integer or np.inf or sequence
        The frequency of compounding of interest rates and spreads
    Assets : None or float or sequence
        The current market price of the assets of the firm
    sigmaA : None or float or sequence
        The annualized asset volatility in percent
    EquityMV : None or float or sequence
        The market value of the equity
    DebtMV : None or float or sequence
        The market value of the debt
    DebtYTM : None or float or sequence
        The yield to maturity of the debt
    CreditSpread : None or float or sequence
        The credit spread of the debt in basis points
    coupon: None or float or sequence
        The coupon rate on the debt annualized, and in percent.
        If None, the risk free rate is assumed.
        Face value of zero coupon debt is computed by assuming that the coupon
        is compounded and paid at maturity. If DebtBV is already the face value
        of zero coupon debt, provide an explicit coupon rate if 0.
    index_from_Series: boolean
        if true the returned DataFrame is indexed with the index of the
        first Pandas Series among the input arguments


    Returns
    ------
    Pandas DataFrame
        Input variables and model values of other variables


    Notes
    -----
    If both debt market and equity market information are given,
    sigmaA is calibrated from the given data.
    If sigmaA and information from either debt or equity market is given,
    the asset value is calibrated from the given information and then
    data for the missing market is computed.
    Equity market information is given as EquityMV
    Debt market information can be given as DebtMV, DebtYTM or CreditSpread

    """
    df = makedf(OrderedDict(
        DebtBV=DebtBV, maturity=maturity, r=r, DivYld=DivYld,
        frequency=frequency, Assets=Assets, sigmaA=sigmaA,
        EquityMV=EquityMV, DebtMV=DebtMV, DebtYTM=DebtYTM,
        CreditSpread=CreditSpread, coupon=coupon),
                index_from_Series=index_from_Series)
    # print(df)
    res = df.apply(onerow, axis=1)
    # print("In Main\n", res)
    return res
