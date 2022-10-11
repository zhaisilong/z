import random

import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm.auto import tqdm
import re
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, track
import pandas as pd

def pep_gen(n:int=1000, length:int=12):
    """随机生成固定长度的多肽
    """
    peps = []
    for i in range(n):
        peps.append(''.join([random.choice(list('AGVLIFWPDEKRHSTNQYCM')) for j in range(length)]))
    return peps

def seq2record(seq_str, idx, description):
    id = str(idx) + '.' + str(description)
    return SeqRecord(Seq(seq_str), id=id)
def record_iters(seqs, idxs, descriptions):
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), "Elapsed:", TimeElapsedColumn()) as progress:
        for seq, idx, description in progress.track(zip(seqs, idxs, descriptions), description='Processing: ', total=len(seqs)):
            yield seq2record(seq, idx, description)

pattern = r".*>(\d+)\.\d"
prog = re.compile(pattern)
def idx_from_line(line, prog):
    return int(prog.match(line).group(1))
def get_cluster(path, no_cluster:int = 0):
    cluster = []
    with open(path) as f:
        flag = False
        for line in f:
            if line.startswith(f">Cluster") and flag:
                break
            elif line.startswith(f">Cluster {no_cluster}"):
                    flag = True  # Startup cluster Counter
                    continue
            elif flag:
                cluster.append(idx_from_line(line, prog))
    return cluster

def cluster2df(cluster, refer_df):
    idx = pd.Index(cluster)
    return refer_df.iloc[idx].copy().reset_index(drop=True)