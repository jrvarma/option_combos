Full documentation is available at <https://option-combos.readthedocs.io/>

This package computes Black Scholes options values and Greeks for options and option combos with a number of valuable features:

* The functions cover a comprehensive set of Greeks:
  - The first order Greeks include (a) several variants of delta including delta driftless, delta forward and delta dual, (b) rho's with respect to domestic and foreign interest rates, (c) theta per day and vega per percentage point change in volatility.
  - Important second order Greeks including gamma, gamma dual, vanna, volga and charm
  - The most useful third order Greek, color
* Since the asset is allowed to pay a continuous dividend yield, foreign currency options, options on futures and options on stock indices are covered.
* Values and Greeks can be computed for an array of options (using a Pandas DataFrame).
* The package handles a portfolio of options (option combinations) where different options are held long or short positions with different weights. The aggregate value and Greeks can be computed for the entire portfolio.
* Option portfolios can contain forward contracts and zero coupon bonds. For example, a portfolio might include self financing strategies like buying a call option and investing the present value of the exercise price in a zero coupon bond.
* It is very easy to create common option combinations. For example, a straddle can be created in a single line as `straddle = combo.call(K=100) + combo.put(K=100)` and a butterfly can be created as `butterfly = combo.call(K=90) + combo.call(K=110) - 2 * combo.call(K=100)`. This illustrates the following features enabled by operator overloading:
  - Option combinations can be created by "adding" two options (combos)
  - Short options are created by using a negative sign
  - Weights can be assigned by simply multiplying an option (combo) by the weight.
* Plotting functions are provided to plot payoffs, profits, values and Greeks of various options (combos).
  - Multiple Greeks of a single combo can be plotted on a single graph. For example, the gamma, vega and theta of a butterfly can be overlaid in a single plot.
  - Different things can be plotted in different plots in a grid in the same figure. For example, the payoff and profit of a strangle can be shown in one plot, and the delta in a separate plot by the side.
  - The same Greek can be plotted for different combos in a single plot. This allows, for example, the gamma of a straddle and a strangle to be compared in a single plot.
* Interactive plot that includes sliders for changing the strikes of each option. For example, an interactive plot of a butterfly can help choose the high, mid and low strikes to achieve a desired option price or gamma/vega/theta profile.
