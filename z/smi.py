from abc import ABC, abstractmethod
from typing import Optional, Iterable
from rdkit import Chem


def canonize_smi(smi: str):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
    except:
        return None


def canonize_rxn(rxn: str):
    """标准化反应"""
    try:
        new_rxn = rxn.split('>>')
        r = new_rxn[0]
        p = new_rxn[1]
        new_r = '.'.join([canonize_smi(x) for x in r.split('.')])
        new_p = canonize_smi(p)
        return new_r + '>>' + new_p
    except:
        return None


def canonize_rxns(rxns: Iterable[str]):
    """批量标准化反应"""
    return [canonize_rxn(rxn) for rxn in rxns]


class SMI(object):
    def __init__(self, smi_str):
        self.smi_str = smi_str

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.smi_str


class RXN(object):
    def __init__(self, rxn_str: str, yield_: Optional[float] = None, canonize: bool = True):
        if canonize:
            self.rxn_str = canonize_rxn(rxn_str)
        else:
            self.rxn_str = rxn_str
        if yield_:
            self.yield_ = yield_

    @property
    def products(self):
        return self.rxn_str.split('>>')[1].split('.')

    @property
    def reactants(self):
        return self.rxn_str.split('>>')[0].split('.')

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.rxn_str
