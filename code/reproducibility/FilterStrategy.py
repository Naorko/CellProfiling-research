from abc import ABC, abstractmethod

import pandas as pd
from typing import List


class FilterStrategy(ABC):

    @abstractmethod
    def filter_dataframe(self, compounds: pd.DataFrame, field: str):
        pass

    @abstractmethod
    def get_filter_threshold(self) -> float:
        pass


class AbsoluteFilterStrategy(FilterStrategy):
    def __init__(self, filter_threshold):
        self._filter_threshold = filter_threshold

    def filter_dataframe(self, compounds: pd.DataFrame, field: str):
        df_zero = compounds.query(f'{field} >= 0').index.get_level_values(1).value_counts()
        df_zero.name = 'zero'
        df_filter = compounds.query(f'{field} >= {self._filter_threshold}').index.get_level_values(1).value_counts()
        df_filter.name = 'filter'
        counts = pd.concat([df_zero, df_filter], axis=1).fillna(0).astype(int)
        comp_names = counts.query('zero > 1 and filter > 0').index
        return compounds[compounds.index.isin(comp_names, 1)]

    def get_filter_threshold(self) -> float:
        return self._filter_threshold


class TopKFilterStrategy(FilterStrategy):
    def __init__(self, top_k):
        self._top_k = top_k
        self._filter_threshold = None

    def filter_dataframe(self, compounds: pd.DataFrame, field: str):
        # return top_k compounds
        topk_df = compounds.nlargest(self._top_k, field)
        self._filter_threshold = topk_df[field].min()
        return topk_df

    def get_filter_threshold(self) -> float:
        return self._filter_threshold


class TopKFromDupFilterStrategy(FilterStrategy):
    def __init__(self, top_k):
        self._top_k = top_k
        self._filter_threshold = None

    def filter_dataframe(self, compounds: pd.DataFrame, field: str):
        # return top_k compounds
        compounds_cnt = compounds.query(f'{field} >= 0').index.get_level_values(1).value_counts()
        dup_compounds = (compounds_cnt[compounds_cnt > 1]).index
        topk_df = compounds[compounds.index.isin(dup_compounds, 1)].nlargest(self._top_k, field)
        self._filter_threshold = topk_df[field].min()
        return topk_df

    def get_filter_threshold(self) -> float:
        return self._filter_threshold


class TopKFromDupWithMatchFilterStrategy(TopKFromDupFilterStrategy):
    def __init__(self, top_k):
        super().__init__(top_k)

    def filter_dataframe(self, compounds: pd.DataFrame, field: str):
        # return top_k compounds
        topk_df = super().filter_dataframe(compounds, field)
        comp_names = topk_df.index.get_level_values(1).unique()
        topk_df = compounds[compounds.index.isin(comp_names, 1)]
        self._filter_threshold = topk_df[field].min()
        return topk_df


class MatchFilterDecorator(FilterStrategy):
    def __init__(self, filter_strategy: FilterStrategy, compounds: pd.DataFrame, field: str):
        self._filter_strategy = filter_strategy
        self._filtered_base = filter_strategy.filter_dataframe(compounds, field)

    def filter_dataframe(self, compounds: pd.DataFrame, field: str):
        cur_filtered = self._filter_strategy.filter_dataframe(compounds, field)
        cur_filtered = cur_filtered.loc[[x for x in cur_filtered.index if x in self._filtered_base.index]]
        return cur_filtered

    def get_filter_threshold(self) -> float:
        return self._filter_strategy.get_filter_threshold()


class DerivedFilterDecorator(FilterStrategy):
    def __init__(self, filter_strategy: FilterStrategy, compounds: pd.DataFrame, field: str):
        self._filter_strategy = filter_strategy
        self._filtered_base = filter_strategy.filter_dataframe(compounds, field)

    def filter_dataframe(self, compounds: pd.DataFrame, field: str):
        cur_filtered = compounds.loc[[x for x in compounds.index if x in self._filtered_base.index]]
        return cur_filtered

    def get_filter_threshold(self) -> float:
        return self._filter_strategy.get_filter_threshold()


class UnionFilterDecorator(FilterStrategy):
    def __init__(self, filter_strategy: FilterStrategy, compounds: List[pd.DataFrame], fields: List[str]):
        self._filter_strategy = filter_strategy
        self._filtered_base = [filter_strategy.filter_dataframe(cs, f) for cs, f in zip(compounds, fields)]
        self._base_index = {idx for df in self._filtered_base for idx in df.index}
        del self._filtered_base

    def filter_dataframe(self, compounds: pd.DataFrame, field: str):
        cur_filtered = compounds.loc[[x for x in compounds.index if x in self._base_index]]
        return cur_filtered

    def get_filter_threshold(self) -> float:
        return self._filter_strategy.get_filter_threshold()


class IntersectFilterDecorator(FilterStrategy):
    def __init__(self, filter_strategy: FilterStrategy, compounds: List[pd.DataFrame], fields: List[str]):
        self._filter_strategy = filter_strategy
        self._filtered_base = [filter_strategy.filter_dataframe(cs, f) for cs, f in zip(compounds, fields)]
        self._base_index = self._filtered_base[0].index
        for df in self._filtered_base[1:]:
            self._base_index = self._base_index.intersection(df.index)

        del self._filtered_base

    def filter_dataframe(self, compounds: pd.DataFrame, field: str):
        cur_filtered = compounds.loc[[x for x in compounds.index if x in self._base_index]]
        return cur_filtered

    def get_filter_threshold(self) -> float:
        return self._filter_strategy.get_filter_threshold()