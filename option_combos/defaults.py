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

from option_combos import instrument


class defaults:
    """Default values for GBS/GBSx constructors and combo plots

    * The first set of defaults allow a GBS or GBSx class to be instantiated
      without giving values for any arguments.
    * The second set of defaults provide default settings for plotting a combo
    """
    # ########## default arguments for GBS/GBSx #############
    #: default value for current market price of the underlying
    S = 100
    #: default value for Strike price
    K = 100
    #: Annualized volatility in decimal (0.25 for 25%)
    sigma = 20e-2
    #: convert months into years in option maturity
    months = 1.0 / 12.0
    #: convert days into years in option maturity
    days = 1.0 / 365
    #: Default time to maturity in years
    ttm = 1 * months
    #: (Domestic) risk free rate, decimal annlzd cont. compounded
    r = 5e-2
    #: Dividend yield or foreign risk free rate dec annlzd cont comp
    q = 2e-2
    #: default instrument type
    instrumentType = instrument.call
    #: default instrument type by name
    instrumentType_name = instrument.name(instrumentType)

    # ########## default arguments for combo #############
    #: default Width of x-axis (in standard deviations) in plots
    ksigmas = 3
    #: by default x-axis includes 2*nlinspace+1 points between end points
    nlinspace = 20
    #: Dictionary to rename method names to label in plots
    name_mapping = {}
    # ########## ready made lists for plot_many #############
    #
    # Each is a list of sublists of strings. Each sublist is plotted in
    # a separate subplot.  Each string is the name of a GBS method.
    # A sublist of this list (for example pay_pro[0]) can be used
    # as flist argument for plot_one
    #: fl_list argument for plot_many to plot only payoff
    payoff_only = "payoff".split()
    #: fl_list argument for plot_many to plot payoff and profit
    payoff = "payoff profit".split()
    #: fl_list argument for plot_many to plot payoff and value
    value = "value payoff".split()
    #: fl_list argument for plot_many to plot payoff profit and value
    profit_value = "payoff profit value".split()
    #: fl_list argument for plot_many to plot greeks
    greeks = [[x] for x in "Delta Gamma Vega Theta".split()]
    #: fl_list argument for plot_many to plot payoff value and greeks
    all_list = [payoff] + [value] + greeks
    #: fl_list argument for plot_many to plot payoff and greeks
    all_list2 = [payoff_only] + [value] + greeks
