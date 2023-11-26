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

from Black_Scholes import GBSx, instrument
from scipy.stats import norm, uniform
from math import exp

greeks = [
    (GBSx.Delta, GBSx.value, 'S', 1),
    (GBSx.DeltaDual, GBSx.value, 'K', 1),
    (GBSx.Gamma, GBSx.Delta, 'S', 1),
    (GBSx.GammaDual, GBSx.DeltaDual, 'K', 1),
    (GBSx.Theta, GBSx.value, 'ttm', -1),
    (GBSx.Theta_daily, GBSx.value, 'ttm', -1/365),
    (GBSx.Vega, GBSx.value, 'sigma', 1),
    (GBSx.Vega_percent, GBSx.value, 'sigma', 1/100),
    (GBSx.RhoD, GBSx.value, 'r', 1),
    (GBSx.RhoF, GBSx.value, 'q', 1),
    (GBSx.Charm, GBSx.Delta, 'ttm', -1),
    (GBSx.Vanna, GBSx.Vega, 'S', 1),
    (GBSx.Color, GBSx.Gamma, 'ttm', 1),
    ]


def check(S, K, sigma, ttm, r, q, optType,
          delta=1e-6, tolerance=5e-5, verbose=True):
    r"""Check analytic greeks of a specific option by numerical differentiation

    Parameters
    ----------
    S : float
        The current market price of the underlying
    K : float
        The strike price of the option
    sigma : float
        The annualized volatility in decimal (0.25 for 25%)
    ttm : float
        Time to maturity in years
    r : float
        The (domestic) risk free rate
        This is continuously compounded annualized, and in decimal
    q : float
        The dividend yield or foreign risk free rate
        This is continuously compounded annualized, and in decimal
    optType :
        instrument.call, instrument.put, instrument.forward, instrument.bond
    delta : float
        numerical differentiation uses values delta apart
    tolerance : float
        tolerance for discrepancy between analytic and numerical derivative
        (discrepancy is regarded as significant only if both absolute and
        relative error must exceed this tolerance)
    verbose: boolean
        if true error details are printed
    Returns
    -------
    int: error count (number of greeks with discrepancy > tolerance)

    Examples
    --------
    >>> check(S=93.112533, K=106.693664, sigma=0.260802,
    ...       ttm=108.625926, r=0.072272, q=0.073218, optType=instrument.put,
    ...       delta=1e-6, tolerance=5e-5, verbose=True)
    ... # doctest: +NORMALIZE_WHITESPACE
    RhoD=-4.180261. Numl=-4.180501. abserr=0.000240. relerr=0.000057
    1 errors in excess of 5e-05
    1

    """
    deltas = [-delta/2, delta/2]
    g = GBSx(S, K, sigma, ttm, r, q, optType)
    err_count = 0
    for deriv, func, var, factor in greeks:
        f = [None] * 2
        v = getattr(g, var)
        for i in [0, 1]:
            setattr(g, var, v + deltas[i])
            g.pre_compute()
            f[i] = func(g)
        setattr(g, var, v)
        numdiff = factor * (f[1] - f[0]) / delta
        greek = deriv(g)
        abserr = abs(numdiff-greek)
        relerr = abserr / max(1e-6, abs(greek))
        if abserr > tolerance and relerr > tolerance:
            err_count += 1
            verbose and print(f"{deriv.__name__}={greek:.6f}.",
                              f"Numl={numdiff:.6f}.",
                              f"{abserr=:.6f}. {relerr=:.6f}")
    if err_count > 0 and verbose:
        print(f"{err_count} errors in excess of {tolerance}")
    return err_count


def test_GBS_greeks_by_numerical_differentiation(
        delta=1e-6, tolerance=5e-5, samplesize=50, verbose=True):
    r"""Check analytic Greeks of a random sample of options
    by numerical differentiation

    Parameters
    ----------
    delta : float
        numerical differentiation uses values delta apart
    tolerance : float
        tolerance for discrepancy between analytic and numerical derivative
    samplesize : int
        number of options (random sample) to be checked
    verbose: boolean
        if true error details are printed

    Returns
    -------

    tuple of two ints:
      first element is the number of options which had at least one greek with
      discrepancy exceeding tolerance, second element is the total number of
      greeks across all options that had discrepancy exceeding tolerance

    Notes
    -----
    * S and K are sampled from normal distribution with mean=100 and sd=10
    * r and q are sampled from normal distribution with mean=5% and sd=3%
    * ln(sigma) is sampled from normal distribution with mean=-1.5 and sd=0.5
    * ln(ttm) is sampled from normal distribution with mean=0 and sd=2
    * the instrument type is sampled from [call, put, forward, bond]
      with probabilities [35%, 35%, 15%, 15%].


    """
    options_err_count = 0
    greeks_err_count = 0
    for i in range(samplesize):
        S = norm.rvs(loc=100, scale=10)
        K = norm.rvs(loc=100, scale=10)
        sigma = exp(norm.rvs(loc=-1.5, scale=0.5))
        ttm = exp(norm.rvs(loc=0, scale=2))
        r = norm.rvs(loc=0.05, scale=0.03)
        q = norm.rvs(loc=0.05, scale=0.03)
        u = uniform.rvs()
        if u < 0.35:
            optType = instrument.call
        elif u < 0.7:
            optType = instrument.put
        elif u < 0.85:
            optType = instrument.forward
        else:  # note the numerical method does not work for exposure
            optType = instrument.bond
        err_count = check(S, K, sigma, ttm, r, q, optType,
                          delta, tolerance, verbose)
        if err_count > 0:
            options_err_count += 1
            greeks_err_count += err_count
            verbose and print(
                f"Instrument Details: {instrument.name(optType)} with "
                f"{S=:.6f}, {K=:.6f}, {sigma=:.6f}",
                f"\n{ttm=:.6f}, {r=:.6f}, {q=:.6f}\n")
    return options_err_count, greeks_err_count
