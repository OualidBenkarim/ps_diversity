"""
Cross-validation functions.
"""

from typing import Union, List, Tuple, Optional, Any, Dict, Iterable

from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.base import clone

from joblib import Parallel, delayed, cpu_count


METRICS = ['auc', 'f1', 'tn', 'fp', 'fn', 'tp']


def _eval_classification(y_test: np.ndarray, y_prob: np.ndarray) -> Tuple:
    if y_test.size == 0:
        return (np.nan,) * len(METRICS)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = np.nan

    y_pred = (y_prob > .5).astype(int)
    try:
        # tn, fp, fn, tp
        cm = confusion_matrix(y_test, y_pred, normalize='all',
                              labels=[0, 1]).ravel()
    except:
        cm = [np.nan] * 4

    f1 = f1_score(y_test, y_pred, zero_division=0)

    return (auc, f1, *cm)


def _eval(y_test: np.ndarray, y_prob: np.ndarray, to_df: bool = False) \
        -> Union[Tuple, pd.Series]:

    res = _eval_classification(y_test, y_prob)
    if to_df:
        return pd.Series(res, index=pd.Index(METRICS, name='metric'))
    return res


def _predict_eval(clf: Any, x: np.ndarray, y: np.ndarray, idx_test: np.ndarray,
                  mult: bool = False) \
        -> Tuple[Union[list, np.ndarray], pd.DataFrame]:

    if not mult:
        x_test, y_test = x[idx_test], y[idx_test]
        if hasattr(clf, 'predict_proba'):
            pred = clf.predict_proba(x_test)[:, 1]
        else:
            pred = clf.predict(x_test)

        score = _eval(y_test, pred, to_df=True).to_frame().T
        score = pd.concat([score], keys=['total'])
        score.index.names = ['set', 'chunk']
        return pred, score

    x_test, y_test = zip(*[(x[idx], y[idx]) for idx in idx_test])
    if hasattr(clf, 'predict_proba'):
        pred = [clf.predict_proba(x)[:, 1] for x in x_test]
    else:
        pred = [clf.predict(x) for x in x_test]

    score = [_eval(y1, p1) for y1, p1 in zip(y_test, pred)]
    score = pd.DataFrame(score, columns=pd.Index(METRICS, name='metric'))

    y_test, pred = np.concatenate(y_test), np.concatenate(pred)
    score_total = _eval(y_test, pred, to_df=True).to_frame().T

    score = pd.concat([score, score_total], keys=['strata', 'total'])
    score.index.names = ['set', 'chunk']
    return pred, score


def _fit_predict_chunk(clf: Any, x: np.ndarray, y: np.ndarray,
                       df_match: pd.DataFrame, indices: Iterable,
                       keys_dissect: Optional[List[str]] = None) \
        -> Tuple[np.ndarray, dict]:

    dissect = len(keys_dissect) > 0

    if isinstance(y, str):
        y = df_match[y].to_numpy()

    if dissect:
        df_match = df_match.copy()
        df_match['y'], df_match['p'] = y, 0
        for k in keys_dissect:
            df_match[k] = df_match[k].astype('category')

    coef = []
    score = {'default': defaultdict(list)}
    if dissect:
        score.update({k: defaultdict(list) for k in keys_dissect})

    for i, (idx_train, idx_test) in enumerate(indices):
        x_train, y_train = x[idx_train], y[idx_train]
        clf_fit = clone(clf).fit(x_train, y_train)

        # save coefficients
        est = clf_fit[-1] if isinstance(clf_fit, Pipeline) else clf_fit

        try:
            c = est.coef_
            if c.size != x.shape[1]:
                raise ValueError
            coef.append(c)

        except (ValueError, AttributeError):
            coef.append(np.full(x.shape[1], np.nan))

        # predict test
        idx_test = {'test': idx_test}
        for s, idx in idx_test.items():
            mult = isinstance(idx, list)
            p, sc = _predict_eval(clf_fit, x, y, idx, mult=mult)
            score['default'][s].append(sc)

            if not dissect:
                continue

            if mult:
                idx = np.concatenate(idx)

            # Dissect performance for categorical confounds
            df_match.loc[idx, 'p'] = p
            df = df_match.iloc[idx].copy()
            df['strata'] = df['strata'].astype('category')
            for k in keys_dissect:
                sc = df.groupby(k).apply(
                    lambda a: _eval(a.y, a.p, to_df=True))
                sc = [sc.unstack(0).to_frame().T]
                sets = ['total']

                if mult:
                    sc_strata = df.groupby([k, 'strata']).apply(
                        lambda a: _eval(a.y, a.p, to_df=True))
                    sc += [sc_strata.unstack(0)]
                    sets += ['strata']

                sc = pd.concat(sc, keys=sets)
                sc.index.names = ['set', 'chunk']
                score[k][s].append(sc)

    return np.vstack(coef), score


def _check_n_jobs(n_jobs):
    if n_jobs == 0:  # invalid according to joblib's conventions
        raise ValueError("'n_jobs == 0' is not a valid choice.")
    if n_jobs < 0:
        return max(1, cpu_count() - int(n_jobs) + 1)
    return min(n_jobs, cpu_count())


def _get_n_chunks(n_perm, n_jobs):
    if n_jobs >= n_perm:
        c = np.ones(n_jobs, dtype=int)
        c[n_perm:] = 0
    else:
        c = np.full(n_jobs, n_perm//n_jobs, dtype=int)
        c[:n_perm % n_jobs] += 1
    return c


def _get_chunked_pairs(n_perm, n_chunks):
    c = _get_n_chunks(n_perm, n_chunks)
    c = np.insert(np.cumsum(c), 0, 0)
    c = np.c_[c[:-1], c[1:]]
    c = c[:min(n_chunks, n_perm)]
    return c


def cross_validate(clf: Any, x: np.ndarray, y: np.ndarray,
                   df_match: pd.DataFrame, indices: Iterable,
                   keys_dissect: Optional[List[str]] = None,
                   n_jobs: int = 1) -> Tuple[np.ndarray, dict]:
    """ Cross validation.

    `indices` provide the train/test indices for each split.

    Parameters
    ----------
    clf: estimator object
        Sklearn-like predictive model with `coef_` attribute.
    x: np.ndarray of shape (n_subjects, n_features)
        Data to fit .
    y: np.ndarray of shape (n_subjects,)
        Target variable to predict (e.g., diagnosis)
    df_match: pd.DataFrame
        Dataframe with matched subjects.
    indices: List or iterable object
        Iterable with train/test indices for each split.
    keys_dissect: list of str, default=None
        Covariates in `df_match` used to dissect performance (e.g., subject
        sex or scan acquisition site).
    n_jobs: int, default=1
        Number of jobs to run in parallel.

    Returns
    -------
    coef: np.ndarray of shape (n_splits, n_features)
        Model coefficients
    scores: dict
        Dictionary of scores for each split.

    """
    if keys_dissect is None:
        keys_dissect = []
    elif isinstance(keys_dissect, str):
        keys_dissect = [keys_dissect]

    indices = list(indices)
    n = len(indices)
    n_jobs = _check_n_jobs(n_jobs)
    chunks = _get_chunked_pairs(n, n_jobs)

    res = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_chunk)(
            clf, x, y, df_match, [indices[k] for k in np.arange(i, j)],
            keys_dissect=keys_dissect) for (i, j) in chunks)

    coef, list_scores = list(zip(*res))
    score = defaultdict(dict)
    for k1, d in list_scores[0].items():
        for k2 in d.keys():
            sc = [s1 for s in list_scores for s1 in s[k1][k2]]
            sc = pd.concat(sc, keys=range(len(sc)), names=['draw'])
            score[k1][k2] = sc.reorder_levels([1, 0, 2]).sort_index(
                level=['set', 'draw'])

    return np.vstack(coef), score
