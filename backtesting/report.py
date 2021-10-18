from dataclasses import dataclass

import pandas as pd


@dataclass
class Report:
    initial_aum: float
    final_aum: float

    holdings: pd.DataFrame

    costs: pd.Series
    proceeds: pd.Series
