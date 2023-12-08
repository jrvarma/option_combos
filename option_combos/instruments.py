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

class instrument:
    r"""Defines constants to denote call and put options.
    Useful as the optType parameter for the GBS class
    and instrument parameter for GBSx, option_portfolio and combo classes

    Attributes
    ----------
    call : int
        Equals +1. Denotes a call option
    put : int
        Equals -1. Denotes a put option
    bond : int
        Equals 2. Zero coupon bond
    forward : int
        Equals 3. Forward contract to buy
    exposure: int
        Equals 4. Unhedged exposure (for example input purchase)
            this is a long or short zero strike call
            except that no premium has been received
            only profit() method of GBS is modified for this
    """
    call = 1
    put = -1
    bond = 2
    forward = 3
    exposure = 4

    @staticmethod
    def name(optType):
        if optType == 1:
            return 'call'
        if optType == -1:
            return 'put'
        if optType == 2:
            return 'bond'
        if optType == 3:
            return 'forward'
        if optType == 4:
            return 'exposure'
