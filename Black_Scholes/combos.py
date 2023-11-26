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

from numpy import sqrt, linspace, sort, inf
from Black_Scholes import option_portfolio, instrument, defaults
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from math import ceil


def _mylen(x):
    r"""Return length of array and return 1 for scalars

    Parameters
    ----------
    x : object

    Returns
    -------
    int

    Examples
    --------
    >>> _mylen([10, 20, 30])
    3

    >>> _mylen(10)
    1

    """
    try:
        return len(x)
    except TypeError:
        return 1
    pass


def _fmt(x):
    r"""Format float as string after rounding to 2 digits

    Parameters
    ----------
    x : float

    Returns
    -------
    string

    Examples
    --------
    >>> _fmt(sqrt(2))
    '1.41'
    """
    return str(round(x, 2))


def _make_layout(layout=None, n=None):
    r"""Return layout (rows,  columns) for displaying n plots in a figure

    Parameters
    ----------
    layout : tuple of two ints or None
        nr and nc arguments to the matplotlib add_subplot command
        If None, an automatic choice is made
    n : int

    Returns
    -------
    tuple of int:
        nr and nc arguments to the matplotlib add_subplot command
        If layout if not None, this is layout itself
        Else a three column layoout is generated

    Examples
    --------
    >>> _make_layout(n=11)
    (4, 3)

    >>> _make_layout(layout=(3, 7))
    (3, 7)

    >>> _make_layout((3, 7), 100)
    (3, 7)

    """
    if layout is None:
        nr = ceil(n / 3)
        nc = ceil(n / nr)
        layout = (nr, nc)
    return layout


class combo(option_portfolio):
    r"""A Black Scholes Option Combo class

    Inherits methods like 'NPV', 'delta', 'gamma', 'volga', 'vanna'
    from option_portfolio (ultimately from GBS)

    Instance Variables
    ------------------
    df  :     Pandas DataFrame containing combo data
              (see documentation of __init__)
    name :    string
              Combo name used in __repr__ and in the plots
    ksigmas : float
          Width of x-axis (in standard deviations) in plots

    Constructors
    ------------
    * The __init__ constructor is not intended to be used directly.
      The following static methods are easier to use:
    * general constructor
       - combo.combo(K, [instrumentType], [weight], [ttm], [name])
    * specific constructors
       - combo.call(K, [ttm], [name]): call option
       - combo.put(K, [ttm], [name]): put option
       - combo.forward(K, [ttm], [name]): forward contract
       - combo.underlying(): call option with zero strike
       - combo.exposure(): same as underlying but with value set to 0
       - ZC_Bond(FV, [ttm], [name]): zero coupon bond

    Overloaded Operators
    --------------------
    * '+' create new combo containing the positions of two combos
    * '-' (unary minus) new combo is short the first combo
    * '-' (subtraction) new combo is long first combo and short second combo
    * '*' (scalar multiply) left term is a float and second term is a combo
          creates new combo with positions multiplied by the number

    New Methods
    -----------
    The new methods (not inherited from option_portfolio) are:
      * set_name: set (change) the name of the combo
      * set_one_strike: change one of the strikes
        used to change striked interactively
      * plot_payoff: plots payoff and optionally profit and value
        This is a wrapper around plot_any
      * plot_any: plots any set of quantities (value, greeks)
      * plot_interactive: plots_any with sliders
    """

    def __init__(self, df, name='', ksigmas=None,
                 S0=None, sigma0=None):
        r"""This constructor is not intended to be called directly
        The staticmethods like combo, call, put are easier

        Parameters
        ----------
        df : pandas DataFrame with the following columns
            df.K : float
               strike price
            df.ttm : float
               option maturity
            df.instrumentType: int
               type of instrument (e.g. instrument.put)
            df.weight: float
               number of options (negative number for short)
            df.component_names : string
               name of the component
        name: string
            name of the combo
        ksigmas : float
            Width of x-axis (in standard deviations) in plots
        Returns
        -------
        combo

        Examples
        --------

        """
        # self.data = df.sort_values('K')
        self.data = df
        self.name = name
        self.tick_pct = 1
        self.ksigmas = defaults.ksigmas if ksigmas is None else ksigmas
        self.nlinspace = defaults.nlinspace
        S0 = defaults.S if S0 is None else S0
        self.S0 = S0
        sigma0 = defaults.sigma if sigma0 is None else sigma0
        self.sigma0 = sigma0
        option_portfolio.__init__(self, S=S0, K=df.K,
                                  sigma=sigma0,
                                  ttm=df.ttm, r=defaults.r, q=defaults.q,
                                  instrumentType=df.instrumentType,
                                  weight=df.weight)
        return

    def set_name(self, newname):
        r"""Change name of combo

        Parameters
        ----------
        newname : string
            new name of combo

        Returns
        -------
        combo

        Examples
        --------

        """
        self.name = newname
        return self

    def set_one_strike(self, i, newk):
        r"""Change strike of one instrument in the combo

        Parameters
        ----------
        i : int
            row number of instrument to change
        newk : float
            new strike

        Returns
        -------
        combo

        Examples
        --------

        """
        self.data.loc[i, 'K'] = newk
        option_portfolio.set_K(self, self.data.K)
        return self

    @staticmethod
    def combo(K, instrumentType=instrument.call, weight=1, ttm=None,
              name='', component_names=None):
        r"""Convenience constructor without DataFrame as argument

        Parameters
        ----------
        K : float or array of floats
            strike price
        instrumentType : int
            type of instrument (e.g. instrument.put)
        weight : float
            number of options (negative number for ttm)
        ttm : float
            time to maturity in years
        name : string
            name of the combo
        component_names : string
            name of the component

        Returns
        -------
        combo

        Examples
        --------

       """
        # ttm = ttm or combo.ttm0
        n = max(_mylen(K), _mylen(instrumentType),
                _mylen(weight), _mylen(ttm))
        df = pd.DataFrame(
            index=range(n),
            columns=['K', 'instrumentType', 'weight', 'ttm',
                     'component_names'])
        if component_names is None:
            if n == 1:
                component_names = name
            else:
                component_names = [
                    name + ':' + str(i) for i in range(n)]
        ttm = defaults.ttm if ttm is None else ttm
        df.K, df.instrumentType, df.weight, df.ttm, df.component_names = (
            K, instrumentType, weight, ttm, component_names)
        return combo(df, name=name)

    @staticmethod
    def call(K=None, ttm=None, name=None, instrumentType=instrument.call):
        r"""Construct combo with only one call option

        Parameters
        ----------
        K : float
            strike price
        ttm : float
            time to maturity in years
        name : string or None
            name of combo
        instrumentType : int
            type of instrument (e.g. instrument.call or instrument.exposure)

        Returns
        -------
        combo

        Examples
        --------
        >>> combo.call(120)
        combo: Call@120
        """
        K = defaults.K if K is None else K
        ttm = defaults.ttm if ttm is None else ttm
        if name is None:
            name = 'Call@' + _fmt(K)
        return combo.combo(K=K, instrumentType=instrumentType, weight=1,
                           ttm=ttm, name=name)

    @staticmethod
    def put(K=None, ttm=None, name=None):
        r"""Construct combo with only one put option

        Parameters
        ----------
        K : float
            strike price
        ttm : float
            time to maturity in years
        name : string or None
            name of combo

        Returns
        -------
        combo

        Examples
        --------
        >>> combo.put(90)
        combo: Put@90

        """
        K = defaults.K if K is None else K
        ttm = defaults.ttm if ttm is None else ttm
        if name is None:
            name = 'Put@' + _fmt(K)
        return combo.combo(K=K, instrumentType=instrument.put, weight=1,
                           ttm=ttm, name=name)

    @staticmethod
    def underlying(name=None):
        r"""Construct combo of only one zero strike forward (underlying)

        Parameters
        ----------
        name : string or None
            name of combo

        Returns
        -------
        combo

        Examples
        --------
        >>> combo.underlying()
        combo: Underlying

        """
        if name is None:
            name = 'Underlying'
        return combo.call(K=0, name=name, instrumentType=instrument.forward)

    @staticmethod
    def exposure(name=None):
        r"""Construct combo of only one exposure to underlying

        Parameters
        ----------
        name : string or None

        Returns
        -------
        combo

        Examples
        --------
        >>> combo.exposure()
        combo: Exposure

        """
        if name is None:
            name = 'Exposure'
        return combo.call(K=0, name=name,
                          instrumentType=instrument.exposure)

    @staticmethod
    def ZC_Bond(FV=None, ttm=None, name=None):
        r"""Construct combo of only one zero coupon bond

        Parameters
        ----------
        FV :   float
               face value of bond
        ttm :  float
               maturity in years
        name : string
               name of combo

        Returns
        -------
        combo

        Examples
        --------
        >>> combo.ZC_Bond(FV=100,ttm=3)
        combo: 3Y Bond (100)

        """
        FV = defaults.K if FV is None else FV
        ttm = defaults.ttm if ttm is None else ttm
        if name is None:
            name = _fmt(ttm) + 'Y Bond (' + _fmt(FV) + ')'
        return combo.combo(K=FV, instrumentType=instrument.bond,
                           ttm=ttm, name=name)

    @staticmethod
    def forward(K=None, ttm=None, name=None):
        r"""Construct combo with only one forward contract

        Parameters
        ----------
        K : float
            strike price of forward
        ttm : float
            time to maturity in years
        name : string or None
            name of combo
        Returns
        -------
        combo

        Examples
        --------
        >>> combo.forward(105)
        combo: Forward@105

        """
        K = defaults.K if K is None else K
        ttm = defaults.ttm if ttm is None else ttm
        if name is None:
            name = 'Forward@' + _fmt(K)
        return combo.combo(K=K, instrumentType=instrument.forward,
                           ttm=ttm, name=name)

    def __add__(self, other):
        r"""Add another combo to this combo

        Parameters
        ----------
        other : combo

        Returns
        -------
        combo

        Examples
        --------
        >>> combo.call(100) + combo.put(100)
        combo: Call@100+Put@100

        >>> (combo.underlying() + combo.put(100)).data
        ... # doctest: +NORMALIZE_WHITESPACE
             K  instrumentType  weight       ttm component_names
        0    0               3       1  0.083333        Underlying
        1  100              -1       1  0.083333        Put@100
        """
        if not isinstance(other, combo):
            raise RuntimeError('Can add only combo to combo')
        else:
            A = combo(
                pd.concat([self.data, other.data], ignore_index=True))
            if other.name.startswith('-'):
                op = ''
            else:
                op = '+'
            A.name = self.name + op + other.name
            return A

    def __neg__(self):
        r"""Return combo which is a short position in this combo

        Returns
        -------
        combo

        Examples
        --------
        >>> -combo.call(100)
        combo: -Call@100

        """
        def rename(old):
            op = '-'
            new = op + old
            if new.startswith('--'):
                new = new.replace('--', '', 1)
            return new
        df = self.data.copy()
        df.weight = -df.weight
        df.component_names = [
            rename(s) for s in df.component_names.values]
        A = combo(df, rename(self.name))
        return A

    def __sub__(self, other):
        r"""Add short position in another combo to this combo

        Parameters
        ----------
        other : combo

        Returns
        -------
        combo

        Examples
        --------
        >>> combo.call(100) - combo.put(100)
        combo: Call@100-Put@100


        """
        if not isinstance(other, combo):
            raise RuntimeError('Can subtract only combo from combo')
        else:
            return self + -other

    def __rmul__(self, other):
        r"""Multiply this combo by a scalar float

        Parameters
        ----------
        other : float

        Returns
        -------
        combo

        Examples
        --------
        >>> 3 * combo.call(100)
        combo: 3*Call@100
        """
        def rename(old):
            op = '*'
            if '+' in old or '-' in old:
                left, right = '(', ')'
            else:
                left, right = '', ''
            return _fmt(other) + op + left + old + right
        try:
            float(other)
        except ValueError:
            raise RuntimeError('Can multiply combo only by a number')
        df = self.data.copy()
        df.weight = other * df.weight
        df.component_names = [
            rename(s) for s in df.component_names.values]
        A = combo(df, rename(self.name))
        return A

    def __repr__(self):
        r"""Return string representation of combo

        Returns
        -------
        string

        Examples
        --------
        >>> print(combo.call(100) - combo.put(100))
        combo: Call@100-Put@100

        """
        return "combo: " + self.name

    def decomposition(self):
        r"""Return list of combos consisting of this combo and its components

        Returns
        -------
        list of combos

        Examples
        --------
        >>> (combo.call(100) - combo.put(100)).decomposition()
        [combo: Call@100, combo: -Put@100, combo: Call@100-Put@100]
        """
        n = len(self.data)
        return [combo(self.data.loc[[i], :],
                      self.data.loc[i, 'component_names'])
                for i in range(n)] + [self]

    def _make_S_array(self, tick_pct=1, ksigmas=None, nlinspace=1,
                     extra_strikes=[]):
        r"""create Pandas Series of underlying prices based on minimum and
        maximum strike in combo.

        Parameters
        ----------
        tick_pct: float
            ST includes points at strikes ± tick_pct% * S0
        ksigmas: float
            ST includes points at S0 ± ksigmas σ√t
        nlinspace: int
            ST includes 2*nlinspace+1 points between S0 ± ksigmas σ√t
        extra_strikes: list of float
            additional dummy strikes to include while setting range of S

        Returns
        -------
        Pandas Series

        Examples
        --------
        >>> (combo.call(120) - combo.put(80))._make_S_array().round(2)
        ... # doctest: +NORMALIZE_WHITESPACE
        array([ 79. , 80. , 81. , 82.68, 100. , 117.32, 119. , 120. , 121. ])

        """
        ksigmas = ksigmas or self.ksigmas
        tick = tick_pct * 0.01 * self.S0
        maxt = max(self.data.ttm)
        bigmove = ksigmas * self.sigma0 * sqrt(maxt) * self.S0
        # The underlying is a zero strike call (strike = 0)
        # We exclude this value from the strikes
        strikes = pd.Series(
            self.data.K[
                self.data.instrumentType == instrument.call].values.tolist()
            + self.data.K[
                self.data.instrumentType == instrument.put].values.tolist()
            + extra_strikes,
            dtype='float64')
        return sort(pd.concat(
            [strikes, strikes - tick, strikes + tick,
             strikes.rolling(window=2, center=False).mean(),
             pd.Series(linspace(self.S0 - bigmove, self.S0 + bigmove,
                                2*nlinspace+1))]).dropna().unique())

    def compute(self, flist, extra_strikes=[]):
        r"""

        Parameters
        ----------
        flist: list of strings
            each string is the name of a combo method
            For example, ['combo.NPV', 'comb.payoff', 'combo.Delta']
            Values returned by this method are plotted
        extra_strikes: list of float
            additional dummy strikes to include while setting x axis limits
            These are passed on to the _make_S_array method

        Returns
        -------
        Pandas DataFrame
            This DataFrame has one column for each string in flist
            Each column contains the payoff, profit, value or greek of combo
            for the set of set of underlying prices returned by _make_S_array

        Examples
        --------
        >>> combo.call(100).compute(['payoff']).round(2)
        ... # doctest: +NORMALIZE_WHITESPACE
                    payoff
        82.679492     0.00
        99.000000     0.00
        100.000000    0.00
        101.000000    1.00
        117.320508   17.32

        """
        def call_method(s, ST=None):
            f = getattr(combo, s)
            if ST is None:
                return f(self)
            else:
                return f(self, ST)

        flist = set(flist)
        payoff_etc = set('payoff profit negated_payoff negated_profit'.split())
        only_payoff = flist <= payoff_etc
        nlinspace = self.nlinspace if not only_payoff else 1
        ST = self._make_S_array(self.tick_pct, self.ksigmas,
                               nlinspace, extra_strikes=extra_strikes)
        df = pd.DataFrame(index=ST, columns=[f for f in flist])
        for f in flist.intersection(payoff_etc):
            df[f] = call_method(f, ST)
        if not only_payoff:
            oldS = self.S
            for s in ST:
                self.set_S(s)
                for f in flist - payoff_etc:
                    df.loc[s, f] = call_method(f)
            self.set_S(oldS)
        return df

    def plot_any(self, flist, axis=None, title=None,
                 xlabel='Asset Price', ylabel=None, legend=True,
                 strikes=True, spot=True,
                 name_mapping=None, extra_strikes=[], ylim=None):
        r"""Plot one graph (payoff, premium or greek) for one combo

        Parameters
        ----------
        flist: list of strings
            each string is the name of a combo method
            For example, ['combo.NPV', 'comb.payoff', 'combo.Delta']
            Values returned by this method are plotted
        axis: matplotlib Axis object or None
            matplotlib Axis object to be used for plotting.
            If None, plt.gca() is used
        title: string or None
            If not None, this is used as the title for plot
        xlabel: string
            X axis label
        ylabel: string
            Y axis label
        legend : boolean
            If True, plot includes legend
        strikes: boolean
            whether to draw vertical lines at the strikes
        spot: boolean
            whether to draw vertical lines at the current spot
        name_mapping: Dictionary
            maps method names of this class to label in plots
        extra_strikes: list of float
            additional dummy strikes to include while setting x axis limits
            These are passed on to the compute method
        ylim: tuple of two floats
            lower and upper bounds on the y-axis to override defaults
            useful to force same axes in different plots

        Returns
        -------
        Pandas DataFrame
            This DataFrame has one column for each string in flist
            Each column contains the payoff, profit, value or greek of combo
            for the set of set of underlying prices returned by _make_S_array

        """

        def newname(x):
            return (
                name_mapping[x] if x in name_mapping else x)

        if name_mapping is None:
            name_mapping = defaults.name_mapping
        if title is None:
            title = self.name
        if axis is None:
            Ax = plt.gca()
        else:
            Ax = axis
        df = self.compute(flist, extra_strikes=extra_strikes)
        for f in flist:
            Ax.plot(df.index, df[f], lw=3, label=newname(f))
        if strikes:
            first_strike = True
            K_list = self.data.K[
                self.data.instrumentType == instrument.call].values.tolist()
            K_list += self.data.K[
                self.data.instrumentType == instrument.put].values.tolist()
            for K in K_list:
                # we want only one legend entry for all the strikes
                # so we set label only while plotting the first strike
                if first_strike:
                    Ax.axvline(K, color='0.7', label='Strike',)
                    first_strike = False
                else:
                    Ax.axvline(K, color='0.7')
        if spot:
            Ax.axvline(self.S0, color='0.4', linestyle='dotted',
                       label='Current Spot')
        legend and Ax.legend()
        xlabel and Ax.set_xlabel(xlabel)
        ylabel and Ax.set_ylabel(ylabel)
        ylim and Ax.set_ybound(lower=ylim[0], upper=ylim[1])
        title and Ax.set_title(title)
        if axis is None:
            plt.show()
        return df

    def plot_payoff(self, profit=True, value=False, axis=None,
                    title=None, xlabel='Asset Price', ylabel=None,
                    legend=True, strikes=True, spot=True,
                    name_mapping=None, extra_strikes=[], ylim=None):
        r"""Plot payoff (and optionally profit and value) of combo

        Parameters
        ----------
        profit : boolean
                 if True, profit is also plotted
        value : boolean
                 if True, value is also plotted
        axis: matplotlib Axis object or None
            matplotlib Axis object to be used for plotting.
            If None, plt.gca() is used
        title: string or None
            If not None, this is used as the title for plot
        xlabel: string
            X axis label
        ylabel: string
            Y axis label
        strikes: boolean
            whether to draw vertical lines at the strikes
        spot: boolean
            whether to draw vertical lines at the current spot
        legend : boolean
            If True, plot includes legend
        name_mapping: Dictionary
            maps method names of this class to label in plots
        extra_strikes: list of float
            additional dummy strikes to include while setting x axis limits
            These are passed on to the compute method
        ylim: tuple of two floats
            lower and upper bounds on the y-axis to override defaults
            useful to force same axes in different plots

        Returns
        -------
        Pandas DataFrame
            This DataFrame has one column for each string in flist
            Each column of this DataFrame contains payoff/profit/value
            for the set of underlying prices returned by _make_S_array

        """
        if name_mapping is None:
            name_mapping = defaults.name_mapping
        flist = (['payoff']
                 + (['profit'] if profit else [])
                 + (['value'] if value else []))
        return self.plot_any(
            flist=flist, axis=axis, title=title, xlabel=xlabel,
            ylabel=ylabel, legend=legend, strikes=strikes, spot=spot,
            name_mapping=name_mapping, extra_strikes=extra_strikes, ylim=ylim)

    def plot_many(Combo, fl_list=None, layout=None,
                  title=None, xlabel='Asset Price', ylabel=None,
                  legend=True, strikes=True, spot=True,
                  name_mapping=None, extra_strikes=[], ylim=None):
        r"""Plot many graphs for same combo
            for different variables (payoff, premium, greeks)

        Parameters
        ----------
        Combo : instrument or bundle
        fl_list: list
            This is a list of sublists of function names.
            Each sublist is plotted in a separate subplot
            by calling plot_any method. Each function is a combo method.
        layout : tuple of two ints or None
            nr and nc arguments to the matplotlib add_subplot command
            If None, an automatic choice is made
        title: string or None
            If not None, this is used as the title for plot
        xlabel: string
            X axis label
        ylabel: string
            Y axis label
        strikes: boolean
            whether to draw vertical lines at the strikes
        spot: boolean
            whether to draw vertical lines at the current spot
        legend : boolean
            If True, plot includes legend
        name_mapping: Dictionary
            maps method names of this class to label in plots

        """
        if name_mapping is None:
            name_mapping = defaults.name_mapping
        if fl_list is None:
            fl_list = defaults.all_list
        fig = plt.figure()
        nr, nc = _make_layout(layout, len(fl_list))
        for i, flist in enumerate(fl_list):
            axis = fig.add_subplot(nr, nc, i+1)
            Combo.plot_any(flist, axis=axis, title=flist[0],
                           xlabel=xlabel, ylabel=ylabel,
                           legend=legend, strikes=strikes, spot=spot,
                           name_mapping=name_mapping,
                           extra_strikes=extra_strikes, ylim=ylim)
        plt.suptitle(Combo.name)
        plt.tight_layout()
        plt.show()
        return

    def interactive_plot(self, flist, title=None, xlabel='Asset Price',
                         ylabel=None, inplace=False, legend=True,
                         strikes=True, spot=True,
                         name_mapping=None, extra_strikes=[], ylim=None):
        r""" Interactive plot with sliders to change strikes

        Parameters
        ----------
        flist: list of strings
            each string is the name of a combo method
            For example, ['NPV', 'payoff', 'Delta']
            Values returned by this method are plotted
        title: string or None
            If not None, this is used as the title for plot
        xlabel: string
            X axis label
        ylabel: string
            Y axis label
        inplace : boolean
            if False, a copy of the combo is used so that the interactive
            sliders do not alter the original combo
        legend : boolean
            If True, plot includes legend
        strikes: boolean
            whether to draw vertical lines at the strikes
        spot: boolean
            whether to draw vertical lines at the current spot
        name_mapping: Dictionary
            maps method names of this class to label in plots
        extra_strikes: list of float
            additional dummy strikes to include while setting x axis limits
            These are passed on to the compute method
        ylim: tuple of two floats
            lower and upper bounds on the y-axis to override defaults
            useful to force same axes in different plots
        inplace : boolean
            if False, a copy of the combo is used so that the interactive
            sliders do not alter the original combo
        Returns
        -------
        combo
            This is original combo if inplace is False
            Else it is the interactively altered combo

        """
        def get_bounds():
            r"""
            Each strike is allowed to move in the range between the strike
            on the left and the strike on the right. For the extreme strikes
            the bounds are more extreme of one sigma from strike and
            ksigmas from the spot.
            These ranges are dynamic. When one strike is moved, the ranges
            of its neighbouring strikes adjust automatically. So this function
            is called each time the plot is updated (see make_update_plot_fn)

            Returns
            tuple of two dicts of float (lower and upper bounds)
            first dict maps the index (of self.data) to lower bounds
            second dict maps the index (of self.data) to upper bounds
            """
            # To prevent strike moving past neighboura due to rounding errors
            # the bounds are narrowed by a factor of 1 +/- toler
            toler = 1e-4
            lb = {}
            ub = {}
            bounds = [low] + sorted(list(set(self.data.K[nz_K]))) + [high]
            for Ki in self.data.index[nz_K]:
                k = bounds.index(self.data.K[Ki])
                lb[Ki] = bounds[k-1] * (1 + toler)
                ub[Ki] = bounds[k+1] * (1 - toler)
            return lb, ub

        def make_update_slider_fn(ki):
            r""" Returns function to be called when a slider is moved.
            We use a function factory so that returned function can 'remember'
            the value of ki and update_plot_fn which are in scope here but
            will not be in scope when matplotlib calls the update function
            The update function
            (a) changes the strike referenced by ki
            (b) updates plot by calling update_plot_fn
            Parameters
            ----------
            ki: int
                index (of self.data)
            """
            def f(val):
                self.set_one_strike(ki, val)
                update_plot_fn()
                return
            return f

        def make_update_plot_fn():
            r""" Return function to be called to update plot.
            We use a function factory so that returned function can 'remember'
            self, sliders, get_bounds, ax which are in scope here but
            will not be in scope when matplotlib calls the update function
            The update function
            (a) updates the slider ranges
            (b) updates plot by calling the plot_any method
            """
            def f():
                # Clear the existing plot
                ax.clear()
                # Get new dynamically updated bounds for each strike
                lb, ub = get_bounds()
                # Set updated bounds for each slider
                for ki in nz_K:
                    sliders[ki].valmin = lb[ki]
                    sliders[ki].valmax = ub[ki]
                    sliders[ki].ax.set_xlim(sliders[ki].valmin,
                                            sliders[ki].valmax)
                # The strike has already been updated (see _update_slider_fn)
                # So we just plot afresh
                self.plot_any(flist=flist, axis=ax, title=title,
                              xlabel=xlabel, legend=legend,
                              strikes=strikes, spot=spot,
                              extra_strikes=extra_strikes, ylim=ylim)
                return
            return f

        if name_mapping is None:
            name_mapping = defaults.name_mapping
        if not inplace:
            # to prevent interactive plotting from change the strikes
            # of the original combo, we plot a copy of the combo
            cloned_copy = combo(self.data, self.name)
            return cloned_copy.interactive_plot(
                flist=flist, title=title, xlabel=xlabel, ylabel=ylabel,
                inplace=True, legend=legend, strikes=strikes, spot=spot,
                name_mapping=name_mapping, extra_strikes=extra_strikes,
                ylim=ylim)
        if sum(self.data.K > 0) == 0:
            # since there is no strike to be changed interactively
            # we do a non interactive plot
            self.plot_any(
                flist=flist, axis=None, title=title, xlabel=xlabel,
                ylabel=ylabel, legend=legend, strikes=strikes,
                spot=spot, name_mapping=name_mapping,
                extra_strikes=extra_strikes, ylim=ylim)
            return self
        instrument_short = {
            instrument.call: 'Call',
            instrument.put: 'Put',
            instrument.bond: 'Bond',
            instrument.forward: 'Fwd',
        }
        # we ignore artificial strike of 0 used to represent exposures
        nz_K = self.data[self.data.K > 0].copy().sort_values('K').index
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        one_sigma_move = self.sigma0 * sqrt(self.ttm.max()) * self.S0
        bigmove = self.ksigmas * one_sigma_move
        # lower limit of x axis is lower of ksigmas below spot and
        # one sigma below lowest strike
        low = min(self.S0 - bigmove, min(self.data.K[nz_K]) - one_sigma_move)
        # upper limit of x axis is higher of ksigmas above spot and
        # one sigma above highest strike
        high = max(self.S0 + bigmove, max(self.data.K[nz_K]) + one_sigma_move)
        # We use a function factory so that
        # self, sliders, get_bounds, ax which are in scope here
        # are remembered when matplotlib calls the update function
        update_plot_fn = make_update_plot_fn()
        sliders = {}
        left = 0.1
        sep = 0.15
        bottom = 0.1
        height = 0.03
        width_plus_sep = (1 - left) / len(self.data.index[nz_K])
        width = width_plus_sep - sep
        lb, ub = get_bounds()
        for Ki in self.data.index[nz_K]:
            label = (
                ("+" if self.data.weight[Ki] >= 0 else "-")
                + (str(abs(self.data.weight[Ki]))
                   if abs(self.data.weight[Ki]) != 1 else "")
                + instrument_short[self.data.instrumentType[Ki]])
            axslider = plt.axes([left, bottom, width, height])
            left += width_plus_sep
            sliders[Ki] = Slider(ax=axslider, label=label,
                                 valmin=lb[Ki], valmax=ub[Ki],
                                 valinit=self.data.K[Ki])
            # We use a function factory so that Ki which is in scope here
            # is remembered when matplotlib calls the update function
            sliders[Ki].on_changed(make_update_slider_fn(Ki))
        update_plot_fn()
        plt.show()
        return self

    def plot_decomposition(self, flist=None, layout=None, name_mapping=None):
        r"""Plot graphs of same variables (payoff, premium, greeks)
            for combo and its components

        Parameters
        ----------
        flist: list of string
            Each string is name of a GBSx method whose result is plotted.
        layout : tuple of two ints or None
            nr and nc arguments to the matplotlib add_subplot command
            If None, an automatic choice is made
        name_mapping: Dictionary
            maps method names of this class to label in plots

        """
        plot_many_combos(self, flist=flist, layout=layout, decompose=True,
                         name_mapping=name_mapping)
    pass


def plot_many_combos(combos, flist=None, layout=None, decompose=True,
                     title=None, xlabel='Asset Price', ylabel=None,
                     legend=True, strikes=True, spot=True,
                     name_mapping=None):
    r"""Plot graphs of same variables (payoff, premium, greeks)
        for different combos

    Parameters
    ----------
    combos : combo or list of combos
    flist: list of string
        Each string is name of a GBSx method whose result is plotted.
    layout : tuple of two ints or None
        nr and nc arguments to the matplotlib add_subplot command
        If None, an automatic choice is made
    decompose : boolean
        If True and combos is a single combo (not a list), the combo is
        decomposed into components and each component is plotted
    title: string or None
        If not None, this is used as the title for plot
    xlabel: string
        X axis label
    ylabel: string
        Y axis label
    strikes: boolean
        whether to draw vertical lines at the strikes
    spot: boolean
        whether to draw vertical lines at the current spot
    legend : boolean
        If True, plot includes legend
    name_mapping: Dictionary
        maps method names of this class to label in plots

    """

    if name_mapping is None:
        name_mapping = defaults.name_mapping
    if flist is None:
        flist = defaults.payoff
    if decompose and isinstance(combos, combo):
        combos = combos.decomposition()
    # We want all the plots to have the same axes. So we consolidate all the
    # strikes across all the combos into extra_strikes
    # Next we find the max and min of the y axis across all combos
    # These will be passed on to plot_any when it is called at the end
    extra_strikes = []
    miny, maxy = inf, -inf
    for Combo in combos:
        extra_strikes += Combo.data.K[
            Combo.data.instrumentType == instrument.call].values.tolist()
        extra_strikes += Combo.data.K[
            Combo.data.instrumentType == instrument.put].values.tolist()
        df = Combo.compute(flist)
        miny = min(miny, df.min(numeric_only=True).min())
        maxy = max(maxy, df.max(numeric_only=True).max())
    # make a rectangular layout of subplots to plot all combos
    nr, nc = _make_layout(layout, len(combos))
    fig = plt.figure()
    # Plot each combo in a separate subplot using plot_any
    # Passing on the extra_strikes and min/max y computed earlier
    for i, Combo in enumerate(combos):
        axis = fig.add_subplot(nr, nc, i+1)
        Combo.plot_any(flist, axis=axis, title=Combo.name,
                       xlabel=xlabel, ylabel=ylabel,
                       legend=legend, strikes=strikes, spot=spot,
                       name_mapping=name_mapping, extra_strikes=extra_strikes,
                       ylim=(miny, maxy))
    plt.tight_layout()
    plt.show()
