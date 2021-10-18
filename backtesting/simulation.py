import functools as ft
import typing as t

import numpy as np
import pandas as pd
from numba import njit

from .report import Report


def simulate(available_fiat: float, buy_fraction: pd.DataFrame,
             sell_fraction: pd.DataFrame, price: pd.DataFrame, fee,
             buy_expiration: int = 30, sell_expiration: int = 30,
             single_trade: bool = False) -> Report:
    buy_fraction = _limit_row_sum(buy_fraction)
    buy_fraction, sell_fraction, price = _align_labels(buy_fraction,
                                                       sell_fraction, price)
    costs, proceeds, holdings = _simulate(available_fiat,
                                          buy_fraction.values,
                                          sell_fraction.values,
                                          price.values,
                                          fee,
                                          buy_expiration=buy_expiration,
                                          sell_expiration=sell_expiration,
                                          single_trade=single_trade)
    markets = buy_fraction.columns
    holdings = pd.DataFrame(holdings, buy_fraction.index, ['$', *markets])
    return Report(available_fiat, holdings.iloc[-1].sum(), holdings,
                  pd.Series(costs, markets),
                  pd.Series(proceeds, markets))


Inputs = t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def _limit_row_sum(df: pd.DataFrame, to: float = 1.) -> pd.DataFrame:
    """
    Ensure each row sums to no more than one.
    :param df: the DataFrame to limit
    :return: the limited DataFrame
    """
    denominator = np.maximum(df.sum(axis=1), to)
    limited_fraction = (df.transpose() / denominator * to).transpose()
    return limited_fraction


def _align_labels(buy_fraction: pd.DataFrame,
                  sell_fraction: pd.DataFrame,
                  price: pd.DataFrame) -> Inputs:
    buy_fraction, sell_fraction, price = _align_index(buy_fraction,
                                                      sell_fraction, price)
    buy_fraction, sell_fraction, price = _align_columns(buy_fraction,
                                                        sell_fraction, price)
    return buy_fraction, sell_fraction, price


def _align_columns(buy_fraction: pd.DataFrame,
                   sell_fraction: pd.DataFrame,
                   price: pd.DataFrame) -> Inputs:
    columns = ft.reduce(lambda a, v: a.intersection(v),
                        (buy_fraction.columns, sell_fraction.columns,
                         price.columns))
    buy_fraction = buy_fraction[columns]
    sell_fraction = sell_fraction[columns]
    price = price[columns]
    return buy_fraction, sell_fraction, price


def _align_index(buy_fraction: pd.DataFrame,
                 sell_fraction: pd.DataFrame,
                 price: pd.DataFrame) -> Inputs:
    buy_fraction = buy_fraction.resample('S').ffill()
    sell_fraction = sell_fraction.resample('S').ffill()
    price = price.resample('S').asfreq()
    max_min = max(buy_fraction.index.min(), sell_fraction.index.min(),
                  price.index.min())
    min_max = min(buy_fraction.index.max(), sell_fraction.index.max(),
                  price.index.max())
    buy_fraction = buy_fraction.loc[max_min:min_max]
    sell_fraction = sell_fraction.loc[max_min:min_max]
    price = price.loc[max_min:min_max]
    return buy_fraction, sell_fraction, price


@njit
def _simulate(starting_fiat: float, buy_fraction: np.array,
              sell_fraction: np.array,
              price: np.array, fee: float, buy_expiration: int = 30,
              sell_expiration: int = 30,
              single_trade: bool = False) -> np.array:
    """
    Simulate a portfolio's trading activity + returns
    :param single_trade: do not buy more once a position is on
    :param starting_fiat: the starting fiat balance
    :param buy_fraction: [t, m] = fraction of fiat to spend on m at t
    :param sell_fraction: [t, m] = fraction of balance[m] to sell at t
    :param price: [t, m] = price of m at t
    :param fee: fee paid to exchange as fraction of spending amount
    :param buy_expiration: the number of periods after which the buy expires
    :param sell_expiration: the number of periods after which the sell expires
    :return: the final market value of the portfolio
    """
    m = buy_fraction.shape[1]
    buy_sizes = np.zeros((buy_expiration, m))
    buy_prices = np.zeros((buy_expiration, m))
    sell_sizes = np.zeros((sell_expiration, m))
    sell_prices = np.zeros((sell_expiration, m))
    available_balance = np.zeros(m)
    total_balance = np.zeros(m)
    pending_buy_size = np.zeros(m)
    available_fiat = starting_fiat
    total_fiat = starting_fiat
    most_recent_price = price[0]
    cost_tracker, proceeds_tracker = np.zeros(m), np.zeros(m)
    holdings_tracker = np.zeros((buy_fraction.shape[0], m + 1))
    for i in range(buy_fraction.shape[0]):
        most_recent_price = np.where(np.isnan(price[i]),
                                     most_recent_price, price[i])
        # holds -> balance
        buy_fills = buy_prices > price[i]  # filled if market moves below price
        pw_fills_cost = (buy_sizes * buy_fills * buy_prices).sum(axis=0)
        fiat_fill_total = pw_fills_cost.sum()
        available_fiat -= fiat_fill_total * fee
        cost_tracker += pw_fills_cost * (1 + fee)
        total_fiat -= fiat_fill_total * (1 + fee)
        filled_size = (buy_sizes * buy_fills).sum(axis=0)
        pending_buy_size -= filled_size
        available_balance += filled_size
        total_balance += filled_size
        buy_sizes = np.where(buy_fills, 0., buy_sizes)
        # holds -> fiat
        sell_fills = sell_prices < price[i]
        total_balance -= (sell_sizes * sell_fills).sum(axis=0)
        pw_proceeds = (sell_sizes * sell_fills * sell_prices).sum(axis=0)
        proceeds = pw_proceeds.sum()
        proceeds_tracker += pw_proceeds * (1 - fee)
        net_proceeds = proceeds * (1 - fee)
        available_fiat += net_proceeds
        total_fiat += net_proceeds
        sell_sizes = np.where(sell_fills, 0., sell_sizes)
        # expiration
        retry = sell_fraction[i] > 0.
        retry_base_amount = retry * sell_sizes[i % sell_expiration]
        available_balance += sell_sizes[
                                 i % sell_expiration] - retry_base_amount
        available_fiat += buy_sizes[i % buy_expiration] @ buy_prices[
            i % buy_expiration]
        pending_buy_size -= buy_sizes[i % buy_expiration]
        # buys -> holds
        buy_fraction_t = np.where(np.isnan(most_recent_price), 0.,
                                  buy_fraction[i])
        if single_trade:
            buy_fraction_t *= total_balance + pending_buy_size == 0.
        buy_quote_amount = available_fiat * buy_fraction_t
        available_fiat -= buy_quote_amount.sum()
        buy_base_amount = np.where(np.isnan(most_recent_price), 0.,
                                   buy_quote_amount / most_recent_price)
        buy_sizes[i % buy_expiration] = buy_base_amount
        pending_buy_size += buy_base_amount
        buy_prices[i % buy_expiration] = np.where(np.isnan(most_recent_price),
                                                  0., most_recent_price)
        # sells -> holds
        sell_fraction_t = np.where(np.isnan(most_recent_price), 0.,
                                   sell_fraction[i])
        sell_base_amount = available_balance * sell_fraction_t
        available_balance -= sell_base_amount
        sell_sizes[i % sell_expiration] = sell_base_amount + retry_base_amount
        sell_prices[i % sell_expiration] = np.where(
            np.isnan(most_recent_price), 0., most_recent_price)
        holdings_tracker[i, 0] = total_fiat
        holdings_tracker[i, 1:] = total_balance * most_recent_price
        continue
    m2m_final = total_balance * most_recent_price
    proceeds_tracker += np.where(np.isnan(m2m_final), 0., m2m_final)
    return cost_tracker, proceeds_tracker, holdings_tracker
