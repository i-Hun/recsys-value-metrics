import typing as tp
import warnings

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

    def __init__(self, k: int) -> None:
        self.k = k

    @classmethod
    def _check(
        cls,
        recommendations: tp.Optional[pd.DataFrame],
        interactions: tp.Optional[pd.DataFrame] = None,
    ) -> None:
        cls._check_columns(
            recommendations,
            "recommendations",
            (Columns.Query, Columns.Item, Columns.Score),
        )
        cls._check_columns(interactions, "interactions", (Columns.Query, Columns.Item))

        if not pd.api.types.is_float_dtype(recommendations[Columns.Score]):
            warnings.warn(
                f"Expected integer dtype of '{Columns.Score}' column in 'recommendations' dataframe."
            )

    @staticmethod
    def _check_columns(
        df: tp.Optional[pd.DataFrame], name: str, required_columns: tp.Iterable[str]
    ) -> None:
        if df is None:
            return
        required_columns = set(required_columns)
        actual_columns = set(df.columns)
        if not actual_columns >= required_columns:
            raise KeyError(
                f"Missed columns {required_columns - actual_columns} in '{name}' dataframe"
            )


class _RankingMetric(MetricAtK):
    def calc(
        self,
        recommendations: tp.Optional[pd.DataFrame] = None,
        interactions: tp.Optional[pd.DataFrame] = None,
        value_by_item: tp.Optional[tp.Union[tp.Dict[tp.Any, tp.SupportsComplex], pd.Series]] = None,
        merged_reco: tp.Optional[pd.DataFrame] = None,
    ) -> float:
        """
        The calc function takes in a recommendations dataframe and an interactions dataframe.
        It then calculates the metric value by calling calc_per_user on each user's recommendation list,
        and returns the mean of all these values.

        :param self: Represent the instance of the class
        :param recommendations: pd.DataFrame: Pass the recommendations dataframe to the function
        :param interactions: pd.DataFrame: Calculate the metric value
        :return: The average of the metric value per user
        :doc-author: Trelent
        """
        if merged_reco is None:
            per_query = self.calc_per_query(recommendations, interactions, value_by_item)
        else:
            per_query = self.calc_per_query(merged_reco=merged_reco)

        return per_query.sum()

    def calc_per_query(
        self,
        recommendations: tp.Optional[pd.DataFrame] = None,
        interactions: tp.Optional[pd.DataFrame] = None,
        value_by_item: tp.Optional[tp.Union[tp.Dict[tp.Any, tp.SupportsComplex], pd.Series]] = None,
        merged_reco: tp.Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        recommendations : pd.DataFrame
            Recommendations table with columns `Columns.Query`, `Columns.Item`, `Columns.Score`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.Query`, `Columns.Item`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        raise NotImplementedError()


def get_top_k_recommendations(recommendations: pd.DataFrame, k: int) -> pd.DataFrame:
    rnk = recommendations.groupby(Columns.Query)[Columns.Score].rank(
        method="first", ascending=False
    )
    at_k_indeces = rnk[rnk <= k].index
    return recommendations[recommendations.index.isin(at_k_indeces)]


def merge_recommendations_interactions(
    recommendations: pd.DataFrame, interactions: pd.DataFrame
) -> pd.DataFrame:
    merged = pd.merge(
        recommendations,
        interactions.assign(relevant=True),
        on=Columns.QueryItem,
        how="left",
    )
    return merged
