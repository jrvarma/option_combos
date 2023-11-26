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
