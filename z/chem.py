from typing import List, Iterable
from pandas import DataFrame
from .utils import split_df
from .smi import RXN


class t5chem(object):
    def __init__(self):
        pass

    def rxn2product_df(self, rxns: Iterable[RXN]) -> DataFrame:
        """将反应转化为 t5chem 的 product 预测任务"""
        products = [rxn.product for rxn in rxns]
        reactants = [rxn.reactant for rxn in rxns]
        return DataFrame({'products': products, 'reactants': reactants})
