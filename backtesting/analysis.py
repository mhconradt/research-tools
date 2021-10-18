import typing as t

import numpy as np
import pandas as pd

from .report import Report


def plot_performance(freq: str = '1h', **kwargs: t.Union[pd.Series, Report]) -> None:
    comparison = pd.DataFrame(dtype=np.float64)
    price = min([x.initial_aum for x in kwargs.values() if isinstance(x, Report)])
    report_count = 0
    for name, arg in kwargs.items():
        if isinstance(arg, Report):
            report_count += 1
            aum = arg.holdings.resample(freq).asfreq().sum(axis=1)
            comparison[name] = aum
        if isinstance(arg, pd.Series):
            factor = price / arg.dropna().iloc[0]
            comparison[name] = (arg.resample(freq).ffill() * factor)
    comparison.plot()


def plot_holdings(report: Report, *, freq: str = '1h', **kwargs) -> None:
    holding_sample = report.holdings.resample(freq).asfreq()
    holding_sample.where(holding_sample > 0., 0.).plot.area(**kwargs)


def plot_cost_proceeds(report: Report, **kwargs) -> None:
    df = pd.DataFrame({'Cost': report.costs, 'Proceeds': report.proceeds})
    df.plot.scatter(x='Cost', y='Proceeds', **kwargs)
