"""
Sampling schemes.
"""

import itertools

from typing import List, Tuple, Optional, Dict, Iterator

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def split_strata(df_match: pd.DataFrame, n_train_strata: int = 5,
                 n_draws: Optional[int] = None) \
        -> Tuple[pd.DataFrame, Dict[str, List]]:
    """Generate train/test splits of strata.

    Parameters
    ----------
    df_match: pd.DataFrame
        DataFrame with matched subjects and strata.
    n_train_strata: int, default=5
        Number of strata to use for training (remaining for test).
    n_draws: int, default=None
        If ``n_draws > 0``, generate `n_draws` contiguous draws (i.e.,
        training strata are contiguous). Generate `n_draws` diverse draws if
        ``n_draws < 0`` (i.e., training strata are the most diverse).
        If None, generate all possible draws.

    Returns
    -------
    df_draws: pd.DataFrame
        Dataframe where each row represent a train/test split of strata.
        Each row also contains the difference in propensity scores within
        the training strata ('intra_ps' column) and between the training strata
        and each of the held-out stratum ('inter_ps_strata' column).
    dict_strata: dict
        Dictionary with test and train strata ids for each split.
    """

    keys = ['ps']

    dist = {}
    for k in keys:
        d = (df_match[[k]].to_numpy() - df_match[k].to_numpy())
        d.flat[::d.shape[1] + 1] = np.nan
        dist[k] = pd.DataFrame(d, columns=df_match.strata).abs()

    splits = ['train', 'test']
    strata = np.unique(df_match.strata)
    comb = np.vstack(list(itertools.combinations(strata, n_train_strata)))

    if n_draws is not None:
        comb_diff = np.diff(comb)
        if n_draws > 0:  # contiguous strata for training
            comb = comb[np.all(comb_diff == 1, axis=1)]
        elif n_draws < 0:  # diverse strata for training
            cost = np.array([pdist(c[:, None]).sum() for c in comb])
            idx = np.argpartition(cost, n_draws)[n_draws:]
            comb = comb[idx]

    nc = comb.shape[0]
    dict_strata = {'train': comb, 'test': [None]*nc}

    cols1 = [f'intra_{k}' for k in keys]
    cols2 = [f'inter_{k}_strata' for k in keys]
    mi = pd.MultiIndex.from_product([splits, cols1])
    mi = mi.append(pd.MultiIndex.from_product([splits[1:], cols2]))
    df_strata = pd.DataFrame(columns=mi, index=range(nc))

    for i, rw in df_strata.iterrows():
        st_train = comb[i]
        dict_strata['test'][i] = np.setdiff1d(strata, st_train)

        m = df_match.strata.isin(st_train)
        idx_train, idx_test = df_match[m].index, df_match[~m].index

        for k in keys:
            d = np.nanmean(dist[k].to_numpy()[idx_train][:, idx_train])
            rw['train', f'intra_{k}'] = d

            xd = dist[k].iloc[idx_train, idx_test]
            d = [v.to_numpy().mean() for _, v in xd.groupby('strata', axis=1)]
            rw['test', f'inter_{k}_strata' % k] = d

    dict_strata['test'] = np.vstack(dict_strata['test'])
    for k, c in df_strata.items():
        try:
            df_strata[k] = c.astype(float)
        except ValueError:
            pass

    if n_draws is not None and abs(n_draws) < df_strata.shape[0] and n_draws < 0:
        idx = np.argpartition(df_strata.train.intra_ps.values, n_draws)[n_draws:]
        df_strata = df_strata.iloc[idx].reset_index(drop=True)
        dict_strata['train'] = dict_strata['train'][idx]
        dict_strata['test'] = dict_strata['test'][idx]

    return df_strata, dict_strata


def get_indices(df_match: pd.DataFrame, dict_strata: Dict[str, List],
                join: bool = False) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generate train/test indices based on train/test strata for each split.

    This is use for both Contiguous/Diverse sampling schemes.

    Parameters
    ----------
    df_match: pd.DataFrame
        Dataframe with matched subjects.
    dict_strata: dict
        Dictionary holding lists train/test strata.
    join: bool, default=False
        Ir False, yield one array of indices for all test strata. Otherwise,
        a list of arrays for each test stratum is returned.

    Yields
    -------
    idx_train: np.ndarray
        The training set indices for that split.
    idx_test: np.ndarray
        The testing set indices for that split.

    """

    g = df_match.groupby('strata').groups
    for i, st_train in enumerate(dict_strata['train']):
        idx_train = np.concatenate([g[s].to_numpy() for s in st_train])
        idx_test = [g[s].to_numpy() for s in dict_strata['test'][i]]
        if join:
            idx_test = np.concatenate(idx_test)
        yield idx_train, idx_test


def get_indices_null(df_match: pd.DataFrame, n_train_strata: int = 2,
                     n_draws: int = 20) \
        -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generate random train/test splits.

    This is the Random sampling scheme.

    Parameters
    ----------
    df_match: pd.DataFrame
        Dataframe with matched subjects.
    n_train_strata: int, default=2
        Number of strata to use for training.
    n_draws: int: default=20
        Number of splits.

    Yields
    -------
    idx_train: np.ndarray
        The training set indices for that split.
    idx_test: np.ndarray
        The testing set indices for that split.

    """

    train_size = n_train_strata
    if n_train_strata > 1:
        train_size /= df_match.strata.nunique()

    kf = StratifiedShuffleSplit(n_splits=n_draws, train_size=train_size,
                                random_state=0)

    y = (df_match[['group']] == 'Pos').astype(int)
    idx = df_match.index.to_numpy()
    for idx_train, idx_test in kf.split(y, y):
        yield idx[idx_train], idx[idx_test]


def get_indices_cv(df_match: pd.DataFrame, strata_train: list,
                   is_strata: bool = True, n_splits: int = 10) \
        -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generate train/test splits from a set of strata.

    Uses sklearn's StratifiedKFold.
    This is used to asses within-distribution performance.

    Parameters
    ----------
    df_match: pd.DataFrame
        Dataframe with matched subjects.
    strata_train: array-like
        Strata ids used for training.
    is_strata: bool, default=True
        If True, `strata_train` contains strata ids. Otherwise, it contains
        subject indices.
    n_splits: int: default=10
        Number of splits.

    Yields
    -------
    idx_train: np.ndarray
        The training set indices for that split.
    idx_test: np.ndarray
        The testing set indices for that split.

    """

    kf = StratifiedKFold(n_splits=n_splits, random_state=0)

    for st in strata_train:
        if is_strata:
            df = df_match[df_match.strata.isin(st)]
        else:  # is index
            df = df_match.iloc[st]

        y = (df[['group']] == 'Pos').astype(int)
        idx = df.index.to_numpy()
        for idx_train, idx_test in kf.split(y, y):
            yield idx[idx_train], idx[idx_test]


