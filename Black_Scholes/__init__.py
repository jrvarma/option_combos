from Black_Scholes.instruments import instrument  # noqa: F401
# For backward compatibility with older code
from Black_Scholes.instruments import instrument as Option  # noqa: F401
from Black_Scholes.instruments import instrument as option_type  # noqa: F401
from Black_Scholes.defaults import defaults  # noqa: F401

from Black_Scholes.GBS import GBS, mywhere, GBSImplied  # noqa: F401
from Black_Scholes.GBSx import GBSx  # noqa: F401
from Black_Scholes.portfolio import option_portfolio  # noqa: F401
from Black_Scholes.combos import combo, plot_many_combos  # noqa: F401

from Black_Scholes.merton_model import merton  # noqa: F401
from Black_Scholes.EWMA import EWMA  # noqa: F401
