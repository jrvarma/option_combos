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

from numpy import dot, array
from option_combos import GBSx, instrument


class option_portfolio(GBSx):
    r"""A Black Scholes Option Portfolio class

    Modifies methods like 'NPV', 'delta', 'gamma', 'volga', 'vanna'
    inherited from GBSx to compute the value for the portfolio

    Parameters
    ----------
    S : float or numpy array
        The current market price of the underlying
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
    instrumentType : instrument.call, instrument.put etc.
        Whether call or put option or other instrument
    weight: float or numpy array
        Number of options in the portfolio
        (long if positive, short if negative)

    Notes
    -----
    It is intended that one or all of the parameters is a numpy array.
    Otherwise, the base class GBSx should be adequate

    Examples
    --------
    >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
    ...     ttm=1, r=0.01, q=0, weight=[2, 1])

    This portfolio is used in all the examples below for this class
    """
    def __init__(self, S, K, sigma, ttm, r, q=0,
                 instrumentType=instrument.call, weight=1):
        r"""

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
        weight : float or numpy array
            size of position in each option (negative for short position)

        Returns
        -------
        option_portfolio:

        """
        GBSx.__init__(self, S, K, sigma, ttm, r, q, instrumentType)
        self.weight = weight

    def value(self):
        r"""Compute the portfolio value (NPV is an alias for value)

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.value().round(6)
        25.804862

        Verify by taking each option separately

        >>> p.weight
        [2, 1]
        >>> GBSx.value(p).round(6)
        array([8.918504, 7.967853])
        """
        return dot(GBSx.NPV(self), self.weight)

    # make NPV an alias for value
    NPV = value

    def payoff(self, ST=None):
        r"""Compute the portfolio payoff

        Parameters
        ----------
        ST : float or numpy array or None (for self.S)
             The market price of the underlying at maturity

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.payoff()
        2

        Verify by taking each option separately

        >>> p.weight
        [2, 1]
        >>> GBSx.payoff(p)
        array([1, 0])
        """
        try:
            res = array([dot(GBSx.payoff(self, st), self.weight) for st in ST])
        except TypeError:
            res = dot(GBSx.payoff(self, ST), self.weight)
        return res

    def profit(self, ST=None):
        r"""Compute the portfolio profit

        Parameters
        ----------
        ST : float or numpy array or None (for self.S)
             The market price of the underlying at maturity
        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.profit(ST=110).round(6)
        5.195138

        Verify by taking each option separately

        >>> p.weight
        [2, 1]
        >>> (GBSx.payoff(p, ST=110) - GBSx.value(p)).round(6)
        array([2.081496, 1.032147])
        """
        return self.payoff(ST) - self.NPV()

    def negated_payoff(self, ST):
        return -self.payoff(ST)

    def negated_profit(self, ST):
        return -self.profit(ST)

    def negated_value(self, ST):
        return -self.value(ST)

    negated_NPV = negated_value

    # ############# First Order Greeks ########################

    def Delta(self):
        r"""Compute the portfolio delta (dV/dS)

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.Delta().round(6)
        1.698643

        Verify by taking each option separately

        >>> p.weight
        [2, 1]
        >>> GBSx.Delta(p).round(6)
        array([0.579358, 0.539926])
        """
        return dot(GBSx.Delta(self), self.weight)

    def DeltaFwd(self):
        r"""Compute the portfolio delta dV/dF where F is the forward price

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.DeltaFwd().round(6)
        1.681741

        """
        return dot(GBSx.DeltaFwd(self), self.weight)

    def DeltaDriftless(self):
        r"""Compute the portfolio delta (dV/dS) without drift

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.DeltaDriftless().round(6)
        1.698643

        """
        return dot(GBSx.DeltaDriftless(self), self.weight)

    def DeltaDual(self):
        r"""Compute the portfolio dual delta (dV/dK)

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.DeltaDual().round(6)
        -1.44594

        """
        return dot(GBSx.DeltaDual(self), self.weight)

    def Theta(self):
        r"""Compute the portfolio theta (dV/dt).

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.Theta().round(6)
        -13.230481

        """
        return dot(GBSx.Theta(self), self.weight)

    def Theta_daily(self):
        r"""Compute the portfolio theta (dV/dt). t is in days not years

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.Theta_daily().round(6)
        -0.036248

        """
        return self.Theta() / 365

    def Vega(self):
        r"""Compute the option vega (dV/dsigma). Divide by 100 to get
        vega per 1% change in sigma

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.Vega().round(6)
        117.898867

        """

        return dot(GBSx.Vega(self), self.weight)

    def Vega_percent(self):
        r"""Compute the option vega (dV/dsigma) per 1% change in sigma

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.Vega_percent().round(6)
        1.178989

        """
        return self.Vega() / 100

    def RhoD(self):
        r"""Compute the option rho (dV/dr). Divide by 100 to get RhoD
        per 1% change in r

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.RhoD().round(6)
        144.059404

        """
        return dot(GBSx.RhoD(self), self.weight)

    def RhoF(self):
        r"""Compute the portfolio foreign rho or psi (dV/dq).

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.RhoF().round(6)
        -169.864267

        """
        return dot(GBSx.RhoF(self), self.weight)

    # ############# Second Order Greeks ########################

    def Gamma(self):
        r"""Compute the portfolio gamma (d^2V/dS^2)

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.Gamma().round(6)
        0.058949

        """
        return dot(GBSx.Gamma(self), self.weight)

    def GammaDual(self):
        r"""Compute the portfolio dual gamma (d^2V/dK^2)

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.GammaDual().round(6)
        0.059352

        """
        return dot(GBSx.GammaDual(self), self.weight)

    def Charm(self):
        r"""Compute the portfolio charm (d^2V / dS dt)
        or (d delta / dt) or (d theta / dS). Note t = -ttm

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.Charm().round(6)
        -0.078649

       """
        return dot(GBSx.Charm(self), self.weight)

    def Vanna(self):
        r"""Compute the portfolio vanna (d^2V / dsigma dS)
        or (d vega / d dS) or (d delta / d sigma)

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.Vanna().round(6)
        0.196994

        """
        return dot(GBSx.Vanna(self), self.weight)

    def Volga(self):
        r"""Compute the portfolio volga (d^2V/dsigma^2)

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.Volga().round(6)
        -5.835487

       """
        return dot(GBSx.Volga(self), self.weight)

    # ############# Third Order Greeks ########################

    def Color(self):
        r"""Compute the portfolio color (d^3V / dttm dS^2)
        or (d gamma / d ttm). Note this is ttm not t.

        Examples
        --------
        >>> p = option_portfolio(S=100, K=array([99, 101]), sigma=0.2,
        ...     ttm=1, r=0.01, q=0, weight=[2, 1])
        >>> p.Color().round(6)
        -0.030064

        """
        return dot(GBSx.Color(self), self.weight)

    pass   # end of class option_portfolio
