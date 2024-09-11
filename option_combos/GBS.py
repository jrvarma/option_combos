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

import numpy  # noqa:F401
from numpy import log, exp, sqrt, isnan, maximum, where
from scipy.stats import norm
from scipy.optimize import brentq
from option_combos import defaults
from option_combos import instrument  # noqa:F401 (used in doctest)

import warnings
sigma_low = 1e-4  # 0.01%
sigma_high = 100  # 10,000%
# All the code works with np.Inf as a value and so we ignore this error
warnings.filterwarnings(action="ignore", message='divide by zero')
# Where we know what to do with np.nan, we filter those warnings
# also using a context manager: "with warnings.catch_warnings():"


def mywhere(condition, x, y):
    r"""same as numpy.where except it works with scalars as well

    This function uses numpy.where if numpy array is received
    else it uses a simple if else

    """
    if hasattr(condition, 'shape') and condition.shape:
        return where(condition, x, y)    # use numpy.where on numpy arrays
    else:
        if condition:
            return x
        else:
            return y
    pass


class GBS:
    r"""A Black Scholes Option class

    Important methods include 'value' for option value as well as
    various Greeks like 'delta', 'gamma', 'volga', 'vanna'

    Parameters
    ----------
    S : float or numpy array
        The current market price of the underlying
        This can be changed subsequently using the set_S method
    K : float or numpy array
        The strike price of the option
    sigma : float or numpy array
        The annualized volatility in decimal (0.25 for 25%)
    ttm : float or numpy array
        Time to maturity in years
    r : float or numpy array
        The (domestic) risk free rate
        This is continuously compounded annualized, and in decimal
    q : float or numpy array, optional
        The dividend yield or foreign risk free rate
        This is continuously compounded annualized, and in decimal
    optType : instrument.call [+1]  or instrument.put [-1], optional
        Whether call or put option

    Notes
    -----
    If any of the parameters is a numpy array, the class instance behaves
    like an array of options and all its methods return a numpy array
    of values.
    """

    def __init__(self, S=None, K=None, sigma=None, ttm=None,
                 r=None, q=None, optType=None):
        S = defaults.S if S is None else S
        K = defaults.K if K is None else K
        sigma = defaults.sigma if sigma is None else sigma
        ttm = defaults.ttm if ttm is None else ttm
        r = defaults.r if r is None else r
        q = defaults.q if q is None else q
        optType = defaults.instrumentType if optType is None else optType
        self.S = S
        self.K = K
        self.sigma = sigma
        self.ttm = ttm
        self.r = r
        self.q = q
        self.optType = optType
        self.pre_compute()

    def pre_compute(self):
        r"""Precompute elements of option pricing and greek formulas

        Called from __init__, set_S and set_X.
        Should be called after changing any option parameter.
        The following are precomputed:
        d1, d2, N(+/-d1), N(+/-d2)
        DFr and DFq (the discount factors)
        Fwd (forward price)
        """
        self.DFr = exp(-self.r * self.ttm)
        self.DFq = exp(-self.q * self.ttm)
        self.Fwd = self.S * self.DFq/self.DFr
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message='divide by zero')
            d2n = log(self.Fwd/self.K) - 0.5 * self.sigma ** 2 * self.ttm
        d2d = self.sigma * sqrt(self.ttm)
        # the division below can be 0/0=NaN if Fwd=K and (ttm=0 or sigma=0)
        # we set d2 to 0 in this case
        # since numerator is of higher order in ttm and sigma
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message='invalid value')
            self.d2 = d2n/d2d
            self.d2 = mywhere(isnan(self.d2), 0.0, self.d2)
        self.d1 = self.d2 + self.sigma * sqrt(self.ttm)
        self.N_optType_d1 = norm.cdf(self.optType * self.d1)
        self.N_optType_d2 = norm.cdf(self.optType * self.d2)

    def set_S(self, S):
        r"""Change market price of underlying. Then call pre_compute
        """
        self.S = S
        self.pre_compute()
        return self

    def set_K(self, K):
        r"""Change strike price of option. Then call pre_compute
        """
        self.K = K
        self.pre_compute()
        return self

    def set_ttm(self, ttm):
        r"""Change maturity of option. Then call pre_compute
        """
        self.ttm = ttm
        self.pre_compute()
        return self

    def set_sigma(self, sigma):
        r"""Change volatility of option. Then call pre_compute
        """
        self.sigma = sigma
        self.pre_compute()
        return self

    def set_r(self, r):
        r"""Change risk free rate of option. Then call pre_compute
        """
        self.r = r
        self.pre_compute()
        return self

    def set_q(self, q):
        r"""Change dividend yield of option. Then call pre_compute
        """
        self.q = q
        self.pre_compute()
        return self

    def value(self):
        r"""Compute the option value (NPV is an alias for value)

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS().value().round(10).item()
        2.423056836

        >>> GBS(S=101, ttm=1).NPV().round(10).item()
        9.823259516

        >>> sigmas = numpy.array([15, 20, 25]) / 100
        >>> GBS(S=101, sigma=sigmas, ttm=1).value().round(8)
        ... # doctest: +NORMALIZE_WHITESPACE
        array([  7.94559796,   9.82325952,  11.71625574])

        """
        value = self.optType * self.DFr * (self.Fwd * self.N_optType_d1
                                           - self.K * self.N_optType_d2)
        return value

    # make NPV an alias for value
    NPV = value

    def payoff(self, ST=None):
        r"""Compute the option payoff

        Parameters
        ----------
        ST : float or numpy array or None (for self.S)
             The market price of the underlying at maturity

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> prices = numpy.array([99, 100, 101])
        >>> GBS(S=101, ttm=1).payoff(prices)
        array([0, 0, 1])
        >>> GBS(S=101, ttm=1, optType=instrument.put).payoff(prices)
        array([1, 0, 0])

        """
        if ST is None:
            ST = self.S
        payoff = maximum(0, self.optType * (ST - self.K))
        return payoff

    def profit(self, ST=None):
        r"""Compute the option profit

        Parameters
        ----------
        ST : float or numpy array
             The market price of the underlying at maturity

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> prices = numpy.array([99, 100, 101])
        >>> GBS(S=101, ttm=1).profit(prices).round(6)
        array([-9.82326, -9.82326, -8.82326])
        >>> GBS(S=101, ttm=1, optType=instrument.put).profit(prices).round(6)
        array([-4.946136, -5.946136, -5.946136])

        """
        if ST is None:
            ST = self.S
        profit = self.payoff(ST) - self.value()
        return profit

    # ############# First Order Greeks ########################

    def Delta(self):
        r"""Compute the option delta (dV/dS)

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS(S=101, ttm=1).Delta().round(8).item()
        0.60558311

        """
        delta = self.optType * self.DFq * self.N_optType_d1
        return delta

    def DeltaFwd(self):
        r"""Compute the option delta dV/dF where F is the forward price

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS(S=101, ttm=1).DeltaFwd().round(8).item()
        0.58768543

        """
        deltaFwd = self.optType * self.DFr * self.N_optType_d1
        return deltaFwd

    def DeltaDriftless(self):
        r"""Compute the option delta (dV/dS) without drift

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS(S=101, ttm=1).DeltaDriftless().round(8).item()
        0.6178167

        """
        deltaDriftless = self.optType * self.N_optType_d1
        return deltaDriftless

    def DeltaDual(self):
        r"""Compute the option dual delta (dV/dK)

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS(S=101, ttm=1).DeltaDual().round(8).item()
        -0.51340635

        """
        deltaDual = -self.optType * self.DFr * self.N_optType_d2
        return deltaDual

    def Theta(self):
        r"""Compute the option theta (dV/dt).
        Divide by 365 to get daily theta.  Note t = -ttm

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS(S=101, ttm=1).Theta().round(8).item()
        -5.11977695

        """
        theta1 = (-self.DFq * norm.pdf(self.d1) * self.S * self.sigma
                  / (2 * sqrt(self.ttm)))
        theta2 = self.optType * (
            self.q * self.S * self.DFq * self.N_optType_d1
            - self.r * self.K * self.DFr * self.N_optType_d2)
        theta = theta1 + theta2
        return theta

    def Theta_daily(self):
        r"""Compute the option theta (dV/dt) with t in days not years.
        Note t = -ttm

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS(S=101, ttm=1).Theta_daily().round(8).item()
        -0.01402679

        """
        return self.Theta() / 365

    def Vega(self):
        r"""Compute the option vega (dV/dsigma). Divide by 100 to get
        vega per 1% change in sigma

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS().Vega().round(8).item()
        11.4673916

        """
        vega = self.S * self.DFq * sqrt(self.ttm) * norm.pdf(self.d1)
        return vega

    def Vega_percent(self):
        r"""Compute the option vega (dV/dsigma) per 1% change in sigma

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS().Vega_percent().round(8).item()
        0.11467392

        """
        return self.Vega() / 100

    def RhoD(self):
        r"""Compute the option rho (dV/dr). Divide by 100 to get RhoD
        per 1% change in r

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS().RhoD().round(8).item()
        4.19712579

        """
        rhoD = self.optType * self.K * self.ttm * self.DFr * self.N_optType_d2
        return rhoD

    def RhoF(self):
        r"""Compute the option foreign rho or psi (dV/dq).
        Divide by 100 to get RhoF per 1% change in q.

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS().RhoF().round(8).item()
        -4.39904719

        """
        rhoF = -(self.optType * self.S * self.ttm
                 * self.DFq * self.N_optType_d1)
        return rhoF

    # ############# Second Order Greeks ########################

    def Gamma(self):
        r"""Compute the option gamma (d^2V/dS^2)

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS(S=101, ttm=1).Gamma().round(8).item()
        0.0185081

        """
        gammaN = self.DFq * norm.pdf(self.d1)
        gammaD = self.S * self.sigma * sqrt(self.ttm)
        # the division below can be 0/0=NaN if Fwd<>K and (ttm=0 or sigma=0)
        # but norm.pdf(+/-1/x) = exp(-1/x^2) goes to 0 faster than x
        # so we set the result to 0 in this case
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message='invalid value')
            gamma = gammaN/gammaD
        gamma = mywhere(isnan(gamma), 0, gamma)
        return gamma

    def GammaDual(self):
        r"""Compute the option dual gamma (d^2V/dK^2)

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS(S=101, ttm=1).GammaDual().round(8).item()
        0.01888012

        """
        gammaDual = self.DFr * norm.pdf(self.d2) / (
            self.K * self.sigma * sqrt(self.ttm))
        return gammaDual

    def Charm(self):
        r"""Compute the option charm (d^2V / dS dt)
        or (d delta / dt) or (d theta / dS). Note t = -ttm

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS(S=101, ttm=1).Charm().round(8).item()
        -0.02532113

        """
        charmN1 = self.optType * self.q * self.DFq * self.N_optType_d1
        charmN2 = - self.DFq * norm.pdf(self.d1) * (
            2 * (self.r-self.q) * self.ttm
            - self.d2 * self.sigma * sqrt(self.ttm))
        charmD2 = 2 * self.ttm * self.sigma * sqrt(self.ttm)
        # do not know how to handle 0/0 here, will leave it as NaN
        charm = charmN1 + charmN2/charmD2
        return charm

    def Vanna(self):
        r"""Compute the option vanna (d^2V / dsigma dS)
        or (d vega / d dS) or (d delta / d sigma)

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS(S=101, ttm=1).Vanna().round(8).item()
        -0.1864676

        """
        # do not know how to handle 0/0 here, will leave it as NaN
        vanna = -self.DFq * norm.pdf(self.d1) * (self.d2/self.sigma)
        return vanna

    def Volga(self):
        r"""Compute the option volga (d^2V/dsigma^2)

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS(S=101, ttm=1).Volga().round(8).item()
        5.6452911

        """
        vega = self.Vega()
        # do not know how to handle 0/0 here, will leave it as NaN
        volga = vega * self.d1 * self.d2/self.sigma
        return volga

    # ############# Third Order Greeks ########################

    def Color(self):
        r"""Compute the option color (d^3V / dttm dS^2)
        or (d gamma / d ttm). Note this is ttm not t.

        Returns
        -------
        float or numpy array (depending on the GBS inputs)

        Examples
        --------
        >>> GBS().Color().round(8).item()
        -0.41635232

        """
        color1 = -self.DFq * norm.pdf(self.d1) / (
            2 * self.S * self.ttm * self.sigma * sqrt(self.ttm))
        color2 = 2 * self.q * self.ttm + 1
        color3n = (2 * (self.r - self.q) * self.ttm
                   - self.d2 * self.sigma * sqrt(self.ttm)) * self.d1
        color3d = self.sigma * sqrt(self.ttm)
        # do not know how to handle 0/0 here, will leave it as NaN
        color = color1 * (color2 + color3n/color3d)
        return color


def GBSImplied(
        price,
        S=None,
        K=None,
        ttm=None,
        r=None,
        q=None,
        optType=None):
    r"""Compute Black Scholes implied volatility

    Parameters
    ----------
    P : option price (premium)
    S : float or numpy array
        The current market price of the underlying
        This can be changed subsequently using the set_S method
    K : float or numpy array
        The strike price of the option
    ttm : float or numpy array
        Time to maturity in years
    r : float or numpy array
        The (domestic) risk free rate
        This is continuously compounded annualized, and in decimal
    q : float or numpy array, optional
        The dividend yield or foreign risk free rate
        This is continuously compounded annualized, and in decimal
    optType : instrument.call [+1]  or instrument.put [-1], optional
        Whether call or put option

    Returns
    -------
    float

    Examples
    --------
    >>> 100 * round(GBSImplied(price=3), 6)
    25.0308
"""
    S = defaults.S if S is None else S
    K = defaults.K if K is None else K
    ttm = defaults.ttm if ttm is None else ttm
    r = defaults.r if r is None else r
    q = defaults.q if q is None else q
    optType = defaults.instrumentType if optType is None else optType
    sigma = brentq(
        lambda sigma:
        GBS(S=S, K=K, sigma=sigma, ttm=ttm,
            r=r, q=q, optType=optType).value()-price,
        a=sigma_low, b=sigma_high)
    return sigma
