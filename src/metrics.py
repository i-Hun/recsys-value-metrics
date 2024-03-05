import pandas as pd
import numpy as np
import typing as tp
from .names import Columns
from .base import (
    MetricAtK,
    merge_recommendations_interactions,
    get_top_k_recommendations,
)


def prepare_dataframe(
    recommendations: pd.DataFrame,
    interactions: pd.DataFrame,
    value_by_item: tp.Union[tp.Dict[tp.Any, tp.SupportsComplex], pd.Series],
    k: int,
) -> pd.DataFrame:
    recommendations = get_top_k_recommendations(recommendations, k=k)
    merged_reco = merge_recommendations_interactions(recommendations, interactions)
    merged_reco["value"] = merged_reco["relevant"] * merged_reco[Columns.Item].map(value_by_item)
    return merged_reco


class ValueAtK(MetricAtK):
    def calc_per_query(
        self,
        recommendations: tp.Optional[pd.DataFrame] = None,
        interactions: tp.Optional[pd.DataFrame] = None,
        value_by_item: tp.Optional[tp.Union[tp.Dict[tp.Any, tp.SupportsComplex], pd.Series]] = None,
        merged_reco: tp.Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        if merged_reco is None:
            merged_reco = prepare_dataframe(recommendations, interactions, value_by_item, self.k)
        value_per_query = merged_reco.groupby(Columns.Query)["value"].sum()
        return value_per_query

    def calc(
        self,
        recommendations: tp.Optional[pd.DataFrame] = None,
        interactions: tp.Optional[pd.DataFrame] = None,
        value_by_item: tp.Optional[tp.Union[tp.Dict[tp.Any, tp.SupportsComplex], pd.Series]] = None,
        merged_reco: tp.Optional[pd.DataFrame] = None,
    ) -> float:
        if merged_reco is None:
            per_query = self.calc_per_query(recommendations, interactions, value_by_item)
        else:
            per_query = self.calc_per_query(merged_reco=merged_reco)

        return per_query.sum()


class PAHAtK(MetricAtK):
    def calc(
        self,
        recommendations: tp.Optional[pd.DataFrame] = None,
        interactions: tp.Optional[pd.DataFrame] = None,
        value_by_item: tp.Optional[tp.Union[tp.Dict[tp.Any, tp.SupportsComplex], pd.Series]] = None,
        merged_reco: tp.Optional[pd.DataFrame] = None,
    ) -> float:
        if merged_reco is None:
            merged_reco = prepare_dataframe(recommendations, interactions, value_by_item, self.k)
        value_at_k = ValueAtK(k=self.k).calc(merged_reco=merged_reco)

        profit_at_hit = (
            value_at_k / merged_reco["relevant"].sum() / merged_reco[Columns.Query].nunique()
        )
        return profit_at_hit


class EPAtK(MetricAtK):
    def __init__(self, k: int, normalize=True) -> None:
        super().__init__(k)
        self.normalize = normalize

    def calc_per_query(
        self,
        recommendations: pd.DataFrame,
        value_by_item: tp.Union[tp.Dict[tp.Any, tp.SupportsComplex], pd.Series],
    ) -> pd.Series:
        recommendations = get_top_k_recommendations(recommendations, k=self.k)
        if self.normalize:
            scores = recommendations.groupby(Columns.Query)[Columns.Score].transform(
                lambda x: x / x.sum()
            )
        else:
            scores = recommendations[Columns.Score]
        ep = scores * recommendations[Columns.Item].map(value_by_item)
        return ep

    def calc(
        self,
        recommendations: pd.DataFrame,
        value_by_item: tp.Union[tp.Dict[tp.Any, tp.SupportsComplex], pd.Series],
    ) -> float:
        per_query = self.calc_per_query(recommendations, value_by_item)

        return per_query.sum()


class PNDCGAtK(MetricAtK):
    def calc_per_query(
        self,
        recommendations: tp.Optional[pd.DataFrame] = None,
        interactions: tp.Optional[pd.DataFrame] = None,
        value_by_item: tp.Optional[tp.Union[tp.Dict[tp.Any, tp.SupportsComplex], pd.Series]] = None,
        merged_reco: tp.Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        if merged_reco is None:
            merged_reco = prepare_dataframe(recommendations, interactions, value_by_item, self.k)
        rnk = merged_reco.groupby(Columns.Query)[Columns.Score].rank(
            method="first", ascending=False
        )
        dcg_u = merged_reco.groupby(Columns.Query)["value"].apply(
            lambda value: (value / np.log2(rnk + 1)).sum()
        )
        relevant_recs = merged_reco.query("relevant == True")
        rnk_ideal = relevant_recs.groupby(Columns.Query)["value"].rank(
            method="first", ascending=False
        )
        idcg_u = relevant_recs.groupby(Columns.Query)["value"].apply(
            lambda value: (value / np.log2(rnk_ideal + 1)).sum()
        )
        return dcg_u / idcg_u

    def calc(
        self,
        recommendations: tp.Optional[pd.DataFrame] = None,
        interactions: tp.Optional[pd.DataFrame] = None,
        value_by_item: tp.Optional[tp.Union[tp.Dict[tp.Any, tp.SupportsComplex], pd.Series]] = None,
        merged_reco: tp.Optional[pd.DataFrame] = None,
    ) -> float:
        if merged_reco is None:
            per_query = self.calc_per_query(recommendations, interactions, value_by_item)
        else:
            per_query = self.calc_per_query(merged_reco=merged_reco)

        return per_query.mean()
