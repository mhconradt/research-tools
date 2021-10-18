import numpy as np
import pandas as pd

from backtesting.analysis import plot_cost_proceeds, plot_holdings, \
    plot_performance
from backtesting.report import Report
from backtesting.simulation import simulate


def main() -> None:
    from string import ascii_uppercase
    np.random.seed(42)
    markets = list(ascii_uppercase[:5])
    m = len(markets)
    n = 86400
    fee = 0.0001
    expiration = 10
    times = pd.date_range('2000-01-01', freq='S', periods=n)
    bf_ = (1 - np.random.rand(n, m) ** (1 / 30))
    buy_fraction = pd.DataFrame(bf_, index=times, columns=markets)
    sf_ = 1 - (1 - np.random.rand(n, m)) ** (1 / 300)
    sell_fraction = pd.DataFrame(sf_, index=times, columns=markets)
    _prices = np.random.lognormal(1e-7, 1e-4, size=(n, m)).cumprod(axis=0)
    price = pd.DataFrame(_prices, index=times, columns=markets)
    drop = np.random.permutation(np.arange(price.size).reshape(*price.shape))
    price[drop % 7 == 0] = np.nan
    _report = simulate(100_000., buy_fraction, sell_fraction, price, fee,
                       expiration, expiration, single_trade=True)
    plot_holdings(_report)
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    main()

__all__ = ['simulate', 'plot_holdings', 'plot_cost_proceeds',
           'plot_performance']
