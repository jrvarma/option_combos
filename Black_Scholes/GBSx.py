import numpy  # noqa:F401
from numpy import exp
from Black_Scholes import GBS, instrument, mywhere, defaults


class GBSx(GBS):
    r"""An extended Black Scholes class that includes bond and forward/futures

    Modifies methods like 'NPV', 'delta', 'gamma', 'volga', 'vanna'
    inherited from GBS to handle bond and forward/futures

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
    instrumentType : int
        * instrument.call: call option
        * instrument.put: put option
        * instrument.bond: zero coupon face value=K, maturity=ttm
        * instrument.forward: forward contract to buy at K
        * instrument.exposure: an unhedged exposure
                             (for example input purchase)
                             this is a long or short zero strike call
                             except that no premium has been received
                             only profit() method is modified for this

    """
    def __init__(self, S=None, K=None, sigma=None, ttm=None,
                 r=None, q=None, instrumentType=None):
        S = defaults.S if S is None else S
        K = defaults.K if K is None else K
        sigma = defaults.sigma if sigma is None else sigma
        ttm = defaults.ttm if ttm is None else ttm
        r = defaults.r if r is None else r
        q = defaults.q if q is None else q
        instrumentType = (defaults.instrumentType if instrumentType is None
                          else instrumentType)
        self.isoption = mywhere(
            instrumentType == instrument.call, True,
            mywhere(instrumentType == instrument.put,
                    True, False))
        self.isbond = instrumentType == instrument.bond
        self.isforward = instrumentType == instrument.forward
        self.isexposure = instrumentType == instrument.exposure
        GBS.__init__(self, S=S, K=K, sigma=sigma, ttm=ttm, r=r, q=q,
                     optType=mywhere(self.isoption, instrumentType,
                                     instrument.call))

    def value(self):
        r"""Compute the instrument value (NPV is an alias for value)

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx(K=100, ttm=1, r=numpy.array([0, 0.05, 0.20]),
        ...      instrumentType=instrument.bond
        ...      ).value().round(6) # doctest: +NORMALIZE_WHITESPACE
        array([100.      ,  95.122942,  81.873075])

        >>> GBSx(K=numpy.array([90, 100, 110]), ttm=1,
        ...      instrumentType=instrument.forward
        ...      ).value().round(6) # doctest: +NORMALIZE_WHITESPACE
        array([12.409219,  2.896925, -6.615369])

        >>> GBSx(K=numpy.array([90, 100, 110]), ttm=1,
        ...      instrumentType=instrument.exposure
        ...      ).value().round(6) # doctest: +NORMALIZE_WHITESPACE
        array([0,  0, 0])

        """
        return mywhere(
            self.isoption,
            GBS.NPV(self),
            mywhere(self.isbond, self.K * exp(-self.r * self.ttm),
                    mywhere(self.isexposure,
                            self.K - self.K,  # zero
                            self.DFr * (self.Fwd - self.K))))

    # make NPV an alias for value
    NPV = value

    def payoff(self, ST=None):
        r"""Compute the portfolio payoff

        Parameters
        ----------
        ST : float or numpy array or None (for self.S)
             The market price of the underlying at maturity

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> prices = numpy.array([99, 100, 101])
        >>> GBSx(S=101, ttm=1).payoff(prices)
        array([0, 0, 1])

        >>> GBSx(instrumentType=instrument.bond).payoff()
        100

        >>> GBSx(instrumentType=instrument.forward).payoff(110)
        10
        """
        if ST is None:
            ST = self.S
        return mywhere(self.isoption, GBS.payoff(self, ST),
                       mywhere(self.isbond, self.K, ST - self.K))

    def profit(self, ST=None):
        r"""Compute the portfolio profit

        Parameters
        ----------
        ST : float or numpy array or None (for self.S)
             The market price of the underlying at maturity

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> prices = numpy.array([99, 100, 101])
        >>> GBSx(S=101, ttm=1).profit(prices).round(6)
        array([-9.82326, -9.82326, -8.82326])
        >>> GBSx(S=101, ttm=1, instrumentType=instrument.put).profit(
        ...      prices).round(6)
        array([-4.946136, -5.946136, -5.946136])

        """
        return self.payoff(ST) - self.value()

    # ############# First Order Greeks ########################

    def Delta(self):
        r"""Compute the portfolio delta (dV/dS)

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx(S=101, ttm=1).Delta().round(8)
        0.60558311
        >>> GBSx(instrumentType=instrument.forward).Delta().round(6)
        0.998335

        Bond has zero delta
        >>> GBSx(instrumentType=instrument.bond).Delta()
        0
        """
        return mywhere(self.isoption, GBS.Delta(self),
                       mywhere(self.isbond,
                               self.K - self.K,  # zero
                               self.DFq))

    def DeltaFwd(self):
        r"""Compute the portfolio delta dV/dF where F is the forward price

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx(instrumentType=instrument.forward).DeltaFwd().round(6)
        0.995842
        >>> GBSx(S=101, ttm=1).DeltaFwd().round(8)
        0.58768543
        """
        return mywhere(self.isoption, GBS.DeltaFwd(self),
                       mywhere(self.isbond,
                               self.K - self.K,  # zero
                               self.DFr))

    def DeltaDriftless(self):
        r"""Compute the portfolio delta (dV/dS) without drift

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx(S=101, ttm=1).DeltaDriftless().round(8)
        0.6178167

        Driftless delta of forward is exactly 1

        >>> GBSx(instrumentType=instrument.forward).DeltaDriftless()
        1

        """
        return mywhere(self.isoption, GBS.DeltaDriftless(self),
                       mywhere(self.isbond,
                               self.K - self.K,  # zero
                               1 + self.K - self.K  # one
                               ))

    def DeltaDual(self):
        r"""Compute the portfolio dual delta (dV/dK)

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx(S=101, ttm=1).DeltaDual().round(8)
        -0.51340635
        """
        return mywhere(self.isoption, GBS.DeltaDual(self),
                       mywhere(self.isbond, self.DFr, -self.DFr))

    def Theta(self):
        r"""Compute the portfolio theta (dV/dt).

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx(S=101, ttm=1).Theta().round(8)
        -5.11977695

        Most Greeks of a bond are zero, but theta is non zero

        >>> GBSx(instrumentType=instrument.bond).Theta().round(6)
        4.97921

        """
        # thetafwd = (self.q * self.S * self.DFq *
        #             (norm.cdf(self.d1) + norm.cdf(-self.d1))
        #             - self.r * self.K * self.DFr *
        #             (norm.cdf(self.d2) + norm.cdf(-self.d2)))
        thetafwd = (self.q * self.S * self.DFq
                    - self.r * self.K * self.DFr)
        return mywhere(self.isoption, GBS.Theta(self),
                       mywhere(self.isbond, self.r*self.DFr*self.K,
                               thetafwd))

    def Theta_daily(self):
        r"""Compute the option theta (dV/dt) with t in days not years.
        Note t = -ttm

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx(S=101, ttm=1).Theta_daily().round(8)
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
        >>> GBS().Vega().round(8)
        11.4673916

        Forward has zero vega

        >>> GBSx(instrumentType=instrument.forward).Vega()
        0
        """
        return mywhere(self.isoption, GBS.Vega(self),
                       self.K - self.K  # zero
                       )

    def Vega_percent(self):
        r"""Compute the option vega (dV/dsigma) per 1% change in sigma

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBS().Vega_percent().round(8)
        0.11467392

        """
        return self.Vega() / 100

    def RhoD(self):
        r"""Compute the option rho (dV/dr). Divide by 100 to get RhoD
        per 1% change in r

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx().RhoD().round(8)
        4.19712579
        """
        # rhoDfwd = self.K * self.ttm * self.DFr * (
        #     norm.cdf(self.d2) + norm.cdf(-self.d2))
        rhoDfwd = self.K * self.ttm * self.DFr
        return mywhere(self.isoption, GBS.RhoD(self),
                       mywhere(self.isbond, -self.ttm*self.DFr*self.K,
                               rhoDfwd))

    def RhoF(self):
        r"""Compute the option foreign rho or psi (dV/dq).
        Divide by 100 to get RhoF per 1% change in q.

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx().RhoF().round(8)
        -4.39904719
        """
        # rhoFfwd = -self.S * self.ttm * self.DFq * (
        #     norm.cdf(self.d1) + norm.cdf(-self.d1))
        rhoFfwd = -self.S * self.ttm * self.DFq
        return mywhere(self.isoption, GBS.RhoF(self),
                       mywhere(self.isbond,
                               self.K - self.K,  # zero
                               rhoFfwd))

    # ############# Second Order Greeks ########################

    def Gamma(self):
        r"""Compute the portfolio gamma (d^2V/dS^2)

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx(S=101, ttm=1).Gamma().round(8)
        0.0185081

        Forward has zero gamma

        >>> GBSx(instrumentType=instrument.forward).Gamma()
        0
        """
        return mywhere(self.isoption, GBS.Gamma(self),
                       self.K - self.K  # zero
                       )

    def GammaDual(self):
        r"""Compute the portfolio dual gamma (d^2V/dK^2)

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx(S=101, ttm=1).GammaDual().round(8)
        0.01888012
        """
        return mywhere(self.isoption, GBS.GammaDual(self), 0)

    def Charm(self):
        r"""Compute the portfolio charm (d^2V / dS dt)
        or (d delta / dt) or (d theta / dS). Note t = -ttm

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx(S=101, ttm=1).Charm().round(8)
        -0.02532113
       """
        return mywhere(self.isoption, GBS.Charm(self),
                       mywhere(self.isbond,
                               self.K - self.K,  # zero
                               self.q * self.DFq))

    def Vanna(self):
        r"""Compute the portfolio vanna (d^2V / dsigma dS)
        or (d vega / d dS) or (d delta / d sigma)

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx(S=101, ttm=1).Vanna().round(8)
        -0.1864676
        """
        return mywhere(self.isoption, GBS.Vanna(self),
                       self.K - self.K  # zero
                       )

    def Volga(self):
        r"""Compute the portfolio volga (d^2V/dsigma^2)

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx(S=101, ttm=1).Volga().round(8)
        5.6452911
       """
        return mywhere(self.isoption, GBS.Volga(self),
                       self.K - self.K  # zero
                       )

    # ############# Third Order Greeks ########################

    def Color(self):
        r"""Compute the portfolio color (d^3V / dttm dS^2)
        or (d gamma / d ttm). Note this is ttm not t.

        Returns
        -------
        float or numpy array (depending on the GBSx inputs)

        Examples
        --------
        >>> GBSx().Color().round(8)
        -0.41635232
        >>> GBSx(instrumentType=instrument.forward).Color()
        0
        """
        return mywhere(self.isoption, GBS.Color(self),
                       self.K - self.K  # zero
                       )
