import os
import random
from itertools import takewhile, repeat
from pathlib import Path
from typing import Union, List, Optional, Iterable
import numpy as np
import pandas as pd
import torch
from rich.table import Table
from sklearn.model_selection import KFold  # 交叉验证
import logging
from rich.logging import RichHandler
from logging import FileHandler
from typing import Optional
from sklearn.utils import shuffle


def get_logger(name: Optional[str] = None, filename: Optional[str] = None, level: str = 'NOTSET') -> logging.Logger:
    """获取一个 Rich 美化的 Logger"""
    name = name if name else __name__

    handlers = [RichHandler(
        rich_tracebacks=True,
    )]
    if filename:
        handlers.append(FileHandler(filename))

    logging.basicConfig(
        level=level,
        format='%(name)s: %(message)s',
        handlers=handlers)
    return logging.getLogger(name)


log = get_logger()


def read_excel(paths: Union[Path, List[Path]], drop_by: Optional[str] = None) -> pd.DataFrame:
    """读取 excel 保存为 pandas.DataFrame"""
    if isinstance(paths, List):
        # use openpyxl for better excel
        df = pd.concat([pd.read_excel(path, engine='openpyxl') for path in paths])
    elif isinstance(paths, Path):
        df = pd.read_excel(paths, engine='openpyxl')
    else:
        raise NotImplementedError
    # xlsx 的最后面往往都是空的
    # drop 保证顺序
    if drop_by:
        df.dropna(subset=[drop_by], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def iter_count(file_name):
    """获取文本行数"""
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


def df_to_table(
        pandas_dataframe: pd.DataFrame,
        rich_table: Table,
        show_index: bool = True,
        index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def show_ratio(df: pd.DataFrame, label='label', sort=None, n=5) -> None:
    """df 的标签中的各类比值
        Args:
            sort: 'value' or 'label'
    """
    n_all = len(df)
    if sort == 'value':
        n_classes = df[label].value_counts().reset_index().sort_values(by=label, ascending=False)
    elif sort == 'label':
        n_classes = df[label].value_counts().reset_index().sort_values(by='index')
    else:
        n_classes = df[label].value_counts().reset_index()

    n_classes = n_classes[:n]

    for i in n_classes.index:
        log.info(
            f'标签 {n_classes.at[i, "index"]} 比例为: {n_classes.at[i, label] / n_all * 100:.2f}%, 个数为: {n_classes.at[i, label]}')


def split_df(df: pd.DataFrame, shuf=True, val=True, random_state=42):
    """Split df into train/val/test set and write into files
    ratio: 8:1:1

    Args：
        - df (DataFrame)： some data
        - shuf (bool, default=True): shuffle the DataFrame
        - val (bool, default=True): split into three set, train/val/test
    """
    if shuf:
        df = shuffle(df, random_state=random_state)

    sep = int(len(df) * 0.1)

    if val:
        test_df = df.iloc[:sep]
        val_df = df.iloc[sep:sep * 2]
        train_df = df.iloc[sep * 2:]
        return train_df, val_df, test_df
    else:
        test_df = df.iloc[:sep]
        train_df = df.iloc[sep:]
        return train_df, test_df


def kfold(df: pd.DataFrame, n_splits=5, shuffle=True, random_state=42) -> pd.DataFrame:
    """
    :param df: 输入的索引必须要对,否则会出错
    :param n_splits:
    :param shuffle:
    :param random_state:
    :return:
    """
    _df = df.copy()
    if shuffle:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    else:
        kf = KFold(n_splits=n_splits)

    for fold in range(n_splits):
        _df[f'fold{fold}'] = False
    fold = 0
    for train_idxs, test_idxs in kf.split(_df):
        print(train_idxs, test_idxs)
        for i in test_idxs:
            _df.loc[i, f'fold{fold}'] = True
        fold += 1
    return _df


def get_CV(df: pd.DataFrame, n_splits=5, dir: Path = Path('CV'), header=True, index=True, cols=None):
    os.makedirs(dir, exist_ok=True)
    for fold in range(n_splits):
        _df = df.copy()
        df_fold_test = _df[_df[f'fold{fold}']]
        df_fold_train = _df[~_df[f'fold{fold}']]
        if cols:
            df_fold_test = df_fold_test[cols]
            df_fold_train = df_fold_train[cols]
            _df = _df[cols]
        fold_dir = dir / f'fold{fold}'
        os.makedirs(fold_dir, exist_ok=True)
        df_fold_test.to_csv(fold_dir / 'test.csv', header=header, index=index)
        df_fold_train.to_csv(fold_dir / 'train.csv', header=header, index=index)
        _df.to_csv(fold_dir / 'all.csv', header=header, index=index)


if __name__ == '__main__':
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 1, 1], 'b': [4, 5, 6, 7, 8, 9, 10, 2, 1]})
    print(kfold(df))
