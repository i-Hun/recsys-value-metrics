import typing as tp
import warnings

import attr
import pandas as pd

from .names import Columns

ExternalItemId = tp.Union[str, int]
Catalog = tp.Collection[ExternalItemId]

class MetricAtK:
    """
    Base class of metrics that depends on `k` -
    a number of top recommendations used to calculate a metric.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    """

    k: int

    @classmethod
    def _check(
        cls,
        recommendations: pd.DataFrame,
        interactions: tp.Optional[pd.DataFrame] = None
    ) -> None:
        cls._check_columns(recommendations, "recommendations", (Columns.Query, Columns.Item, Columns.Score))
        cls._check_columns(interactions, "interactions", (Columns.Query, Columns.Item))

        if not pd.api.types.is_float_dtype(recommendations[Columns.Score]):
            warnings.warn(f"Expected integer dtype of '{Columns.Score}' column in 'recommendations' dataframe.")

    @staticmethod
    def _check_columns(df: tp.Optional[pd.DataFrame], name: str, required_columns: tp.Iterable[str]) -> None:
        if df is None:
            return
        required_columns = set(required_columns)
        actual_columns = set(df.columns)
        if not actual_columns >= required_columns:
            raise KeyError(f"Missed columns {required_columns - actual_columns} in '{name}' dataframe")


class _RankingMetric(MetricAtK):
    """
    Simple classification metric base class.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    """

    def calc(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> float:
        """
        Calculate metric value.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user(reco, interactions)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        raise NotImplementedError()