"""
Diversity-aware within- and Out-of-distribution performance evaluation.
"""

from time import time
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from typing import List, Tuple, Optional, Iterator, Any

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression

from propensity_score import estimate_ps, match_ps, stratify_ps
from sampling import (split_strata, get_indices, get_indices_cv,
                      get_indices_null)
from validation import cross_validate


def _add_indices_cont(df_match: pd.DataFrame, n_train_strata: int, n_add: int,
                      random_state: int = 0) \
        -> Iterator[Tuple[np.ndarray, np.ndarray]]:

    g = df_match.groupby('pair').mean().sort_values(['strata', 'ps'])
    g.reset_index(inplace=True)

    n_train_pairs = g.groupby('strata').count().min().iloc[0] * n_train_strata

    pairs_discard = g.groupby('strata').first().pair
    candidates = np.setdiff1d(g.iloc[:-n_train_pairs].pair, pairs_discard)

    rs = np.random.RandomState(random_state)
    selected = rs.permutation(candidates)[:n_add]

    # contiguous chunk of pairs for training
    idx_start = g[g.pair.isin(selected)].index
    idx_end = idx_start + n_train_pairs

    idx = df_match.index.to_numpy()
    for i, j in zip(idx_start, idx_end):
        mtrain = df_match.pair.isin(g.iloc[i:j].pair)
        yield idx[mtrain], idx[~mtrain]


def _eval_samp_one(clf: Any, x: np.ndarray, y: np.ndarray,
                   df_match: pd.DataFrame, n_train_strata: int, kind: str,
                   n_splits: int = 10, n_draws: int = 20, n_jobs: int = 1) \
        -> Tuple[np.ndarray, dict]:

    n_draws = -np.abs(n_draws) if kind == 'div' else np.abs(n_draws)
    kwds = dict(n_train_strata=n_train_strata, n_draws=n_draws)

    # Out-of-distribution
    if kind in ['cont', 'div']:
        _, dict_strata = split_strata(df_match, **kwds)

        idx_ood = get_indices(df_match, dict_strata, join=True)
        strata_train = dict_strata['train']
        is_strata = True

        # Add draws if necessary
        n_rem = n_draws - strata_train.shape[0]
        if kind == 'cont' and n_rem > 0:
            idx_ood = list(idx_ood)
            idx_ood2 = _add_indices_cont(df_match, n_train_strata, n_rem)
            idx_ood += list(idx_ood2)
            strata_train = [idx[0] for idx in idx_ood]
            is_strata = False

    else:  # random sampling scheme
        idx_ood = list(get_indices_null(df_match, **kwds))
        strata_train = [idx[0] for idx in idx_ood]
        is_strata = False

    coef, score = cross_validate(clf, x, y, df_match, idx_ood, n_jobs=n_jobs)

    # Within-distribution - CV
    idx_cv = get_indices_cv(df_match, strata_train, is_strata=is_strata,
                            n_splits=n_splits)
    sc = cross_validate(clf, x, y, df_match, idx_cv, n_jobs=n_jobs)[1]

    idx = pd.Index(np.repeat(range(len(strata_train)), n_splits),
                   name='draw')
    df = sc['default']['test'].droplevel('draw').set_index(idx, append=True)
    score['default']['cv'] = df.groupby(['set', 'draw', 'chunk']).mean()

    return coef, score


def evaluate_sampling(clf: Any, x: np.ndarray, y: np.ndarray,
                      df_match: pd.DataFrame,
                      kinds: Optional[List[str]] = None, n_draws: int = 20,
                      list_train_strata: Optional[list] = None,
                      n_splits: int = 10, n_jobs: int = 1,
                      verbose: bool = False) -> Tuple[dict, dict]:
    """Evaluate sampling schemes.

    Parameters
    ----------
    clf: estimator object
        Sklearn-like predictive model with `coef_` attribute.
    x: np.ndarray of shape (n_subjects, n_features)
        Data to use for prediction.
    y: np.ndarray of shape (n_subjects,)
        Binary class label (e.g., diagnosis).
    df_match: pd.DataFrame
        Dataframe with matched subjects.
    kinds: list of str, default=None
        List of possible sampling schemes:
        - 'cont': contiguous strata for training and remaining for test
        - 'div': (at least) one non-contiguous strata for training
        - 'null': random splitting of paired subjects in train/test
        If None, evaluate all sampling schemes.
    n_draws: int, default=20
        Number of draws (aka train/test strata splits)
    list_train_strata: list of int, default=None
        List with different numbers of strata to use for training.
        Evaluations are repeated for each number of training strata (remaining
        strata are used for test). If None, use ``range(2, n_strata-1)``.
    n_splits: int, default=10
        Number of CV splits based only on train strata (aka
        within-distribution performance).
    n_jobs: int, default=1
        Number of jobs to run in parallel.
    verbose: bool, default=False
        Verbosity.

    Returns
    -------
    coef: dict
        Dictionary of model coefficients for each sampling scheme in `kind`.
    scores: dict
        Performance scores for each sampling scheme in `kind`.
        Scores for within (CV in train strata) and out-of-distribution (test
        strata).
    """

    kwds = dict(n_splits=n_splits, n_draws=np.abs(n_draws), n_jobs=n_jobs)

    if list_train_strata is None:
        n_strata = df_match.strata.nuique()
        list_train_strata = np.arange(2, n_strata-1)

    # Compare contiguous, diverse and random (aka null) sampling schemes
    if kinds is None:
        kinds = ['cont', 'div', 'null']

    score = {k: defaultdict(list) for k in kinds}
    coef = defaultdict(list)

    for k, ns in itertools.product(kinds, list_train_strata):
        t1 = time()

        c, sc = _eval_samp_one(clf, x, y, df_match, ns, k, **kwds)
        coef[k].append(c)
        for s, v in sc['default'].items():
            score[k][s].append(v)

        if verbose:
            print('{:<14} [{:>2}]: {:.3f}s'.format(k, ns, time()-t1))

    splits = score['cont'].keys()
    for k1, k2 in itertools.product(kinds, splits):
        score[k1][k2] = pd.concat(score[k1][k2], keys=list_train_strata,
                                  names=['n_train_strata'])
    new_score = {}
    for s in splits:
        sc = pd.concat({k: v[s] for k, v in score.items()}, axis=1,
                       names=['kind'])
        new_score[s] = sc.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    return coef, new_score


def evaluate_diversity(clf: Any, x: np.ndarray, y: np.ndarray,
                       df_match: pd.DataFrame, n_train_strata: int = 5,
                       keys_dissect: Optional[List[str]] = None,
                       n_splits: int = 10, n_jobs: int = 1,
                       verbose: bool = False) \
        -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """Evaluate classification performance vs diversity.

    Parameters
    ----------
    clf: estimator object
        Sklearn-like predictive model with `coef_` attribute.
    x: np.ndarray of shape (n_subjects, n_features)
        Data to use for prediction.
    y: np.ndarray of shape (n_subjects,)
        Binary class label (e.g., diagnosis).
    df_match: pd.DataFrame
        Dataframe with matched subjects.
    n_train_strata: int, default=5
        Number of strata to use for training, remaining are used for test.
    keys_dissect: list of str, default=None
        Covariate names used to dissect performance. Only works with
        categorical covariates (e.g., scan acquisition site, sex). If
    n_splits: int, default=10
        Number of CV splits based only on train strata (aka
        within-distribution performance).
    n_jobs: int, default=1
        Number of jobs to run in parallel.
    verbose: bool, default=False
        Verbosity.

    Returns
    -------
    df_strata: pd.DataFrame
        Dataframe where each row represent a train/test split of strata.
        See split_strata.
    coef: dict
        Dictionary of model coefficients for each sampling scheme in `kind`.
    scores: dict
        Performance scores for each sampling scheme in `kind`.
        Scores for within (CV in train strata) and out-of-distribution (test
        strata).

    """

    # Generate all possible splits of n_train_strata strata for training
    # and the remaining for test
    df_strata, dict_strata = split_strata(df_match,
                                          n_train_strata=n_train_strata)

    t1 = time()

    # Within-distribution - CV
    idx = get_indices_cv(df_match, dict_strata['train'], n_splits=n_splits,
                         is_strata=True)
    score_cv = cross_validate(clf, x, y, df_match, idx, n_jobs=n_jobs)[1]

    t2 = time()
    if verbose:
        print(f'Within-distribution elapsed time: {t2-t1:.2f}s')

    # Out-of-distribution
    idx = list(get_indices(df_match, dict_strata))
    coef, score = cross_validate(clf, x, y, df_match, idx, n_jobs=n_jobs,
                                 keys_dissect=keys_dissect)

    t3 = time()
    if verbose:
        print(f'Out-of-distribution elapsed time: {t3-t2:.2f}s')

    score_cv = score_cv['default']['test']
    score_cv = score_cv.reset_index(level='draw', drop=False)
    score_cv['draw'] //= n_splits
    score_cv = score_cv.set_index('draw', append=True)
    score['default']['cv'] = score_cv.groupby(['set', 'draw', 'chunk']).mean()

    return df_strata, coef, score


def decounfound(dec: str, df_conf: pd.DataFrame, x: np.ndarray,
                site_col: Optional[str] = None,
                cat: Optional[List[str]] = None) -> np.ndarray:
    """Deconfounding.

    Parameters
    ----------
    dec: {'rout', 'combat'}
        Deconfounding approach.
    df_conf: pd.DataFrame
        Dataframe with confounds.
    x: np.ndarray of shape (n_subjects, n_features)
        Data to deconfound.
    site_col: str, default=None
        Column in `df_conf` holding acquisition site.
    cat: list of str, default=None
        Categorical covariates in `df_conf` that are not site. If None,
        assumes no categorical columns in dataframe.

    Returns
    -------
    x_dec: np.ndarray of shape (n_subjects, n_features)
        Deconfounded data.
    """

    cat = [] if cat is None else cat
    if dec == 'combat' and site_col is None:
        raise ValueError("Combat requires site information: site_col is None")

    cat_site = cat if site_col is None else cat + [site_col]
    scale_keys = np.setdiff1d(df_conf.columns, cat_site)

    if dec == 'rout':
        ct = make_column_transformer((MinMaxScaler(), scale_keys),
                                     (OneHotEncoder(drop='first'), cat_site))
        xconf = ct.fit_transform(df_conf)
        logit = LinearRegression(normalize=False)
        clf = logit.fit(xconf, x)

        return x - (xconf @ clf.coef_.T)

    if dec == 'combat':

        from neuroCombat import neuroCombat
        x = neuroCombat(pd.DataFrame(x.T), df_conf, site_col,
                        categorical_cols=cat,
                        continuous_cols=scale_keys, eb=True, parametric=True,
                        mean_only=False, ref_batch=None).T

        return x

    raise ValueError(f"Unknown deconfounding approach '{dec}'!")


def prepare_data(x: np.ndarray, y: np.ndarray, df_cov: pd.DataFrame,
                 site_col: Optional[str] = None,
                 cat: Optional[List[str]] = None, n_strata: int = 10,
                 caliper: Optional[float] = 0.2, dec: Optional[str] = None) \
        -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """ Prepare data.

    Performs propensity score estimation, matching and stratification.
    Deconfounding, if performed, is applied to the matched data (i.e., after
    discarding unmatched subjects with no matches).

    Parameters
    ----------
    x: np.ndarray of shape (n_subjects, n_features)
        Data to use for prediction.
    y: np.ndarray of shape (n_subjects,)
        Binary class label (e.g., diagnosis)
    df_cov: pd.DataFrame of shape=(n_subjects, n_covariates)
        DataFrame holding subjects covariates. These covariates are used for
        both propensity score estimation and deconfounding.
    site_col: str, default=None
        Column in `df_cov` holding acquisition site.
    cat: list of str, default=None
        Categorical covariates in `df_cov` that are not site. If None,
        assumes no categorical columns in dataframe.
    n_strata: int, default=10
        Number of strata.
    caliper: float, optional, default=0.2
        Caliper to use for imperfect matches. If None, no caliper is used.
    dec: {'rout', 'combat'}
        Deconfounding approach.

    Returns
    -------
    df_match: pd.DataFrame of shape (n_matched, n_covariates + 2)
        DataFrame with matched subjects. Paired subjects share the same value
        in the 'pair' column. An additional columns 'ps' holds the propensity
        score and 'strata' indicates the stratum of each subject.
    x_match: np.ndarray of shape (n_matched, n_features)
        Data to use for prediction after matching.
    y_match: np.ndarray of shape (n_matched,)
        Binary class label after matching.
    """

    cat = [] if cat is None else cat
    cat_site = cat if site_col is None else (cat + [site_col])

    # Compute PS
    ps = estimate_ps(df_cov, y, cat=cat_site)
    df_cov['ps'] = ps

    # Matching
    df_match, mask_sel = match_ps(y, ps, caliper=caliper)
    x, y = x[mask_sel], y[mask_sel]

    # Stratification
    df_match = stratify_ps(df_match, n_strata=n_strata)
    df_match.replace({'Pos': 1, 'Neg': 0}, inplace=True)

    # Deconfounding
    if dec is not None:
        if dec == 'combat':
            cat += ['group']
        x = decounfound(dec, df_match, x, site_col=site_col, cat=cat)

    return df_match, x, y
