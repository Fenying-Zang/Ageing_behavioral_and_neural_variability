"""
Statistical helper functions.

Includes:
    - GLM permutation test (single run and batched)
    - Bayes factor calculation (via Pearson correlation equivalence)
    - Cached results management (read/write)
"""
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from statsmodels.formula.api import glm
from pathlib import Path
import pandas as pd
import config as C


def single_permutation(i, data, permuted_label, *, formula, family_func):
    """
    Run a single GLM fit with permuted 'age_years' labels.

    Parameters
    ----------
    i : int
        Permutation index (for logging/debugging only).
    data : pd.DataFrame
        Input data containing predictors and dependent variable.
    permuted_label : array-like
        Shuffled version of 'age_years'.
    formula : str
        GLM formula, e.g. "y ~ age_years".
    family_func : statsmodels.genmod.families.Family
        GLM family (e.g. Gaussian(), Gamma()).

    Returns
    -------
    float
        The coefficient for 'age_years' (np.nan if fitting fails).
    """
    try:
        shuffled = data.copy()
        shuffled["age_years"] = permuted_label
        model = glm(formula=formula, data=shuffled, family=family_func, eval_env=0).fit()
        return model.params["age_years"] if "age_years" in model.params.index else np.nan
    except Exception:
        return np.nan


def run_permutation_test(data, age_labels, *, formula,
                         family_func, shuffling, n_permut, n_jobs,
                         random_state, plot=False, group_labels=None):
    """
    Run a permutation test for the effect of 'age_years' in a GLM.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing dependent variable and predictors.
    age_labels : array-like
        Values of the age predictor.
    formula : str
        GLM formula, e.g. "y ~ age_years".
    family_func : statsmodels.genmod.families.Family
        GLM family (e.g. Gaussian(), Gamma()).
    shuffling : str
        Shuffling mode: 'labels1_based_on_2' or 'labels1_global'.
    n_permut : int
        Number of permutations.
    n_jobs : int
        Number of parallel jobs.
    random_state : int
        Random seed for reproducibility.
    plot : bool, optional
        If True, plot permutation distribution (default False).
    group_labels : array-like, optional
        Grouping labels for 'labels1_based_on_2' shuffling.

    Returns
    -------
    observed_val : float
        Observed coefficient for 'age_years'.
    observed_val_p : float
        p-value from the GLM (Wald test).
    p_perm : float
        Permutation-based p-value.
    valid_null : np.ndarray
        Distribution of permuted coefficients.
    """
    df = data.copy()
    try:
        from scripts.utils.data_utils import shuffle_labels_perm
    except Exception:
        shuffle_labels_perm = None
    if shuffle_labels_perm is None:
        raise RuntimeError("shuffle_labels_perm not available.")

    permuted_labels, _ = shuffle_labels_perm(
        labels1=age_labels, labels2=group_labels, shuffling=shuffling,
        n_permut=n_permut, random_state=random_state, n_cores=n_jobs,
    )

    null_dist = Parallel(n_jobs=n_jobs)(
        delayed(single_permutation)(
            i, df, permuted_labels[i], formula=formula, family_func=family_func
        )
        for i in tqdm(range(n_permut))
    )
    null_dist = np.asarray(null_dist, dtype=float)
    valid_null = null_dist[np.isfinite(null_dist)]

    model_obs = glm(formula=formula, data=df, family=family_func, eval_env=0).fit()
    observed_val = model_obs.params.get("age_years", np.nan)
    observed_val_p = model_obs.pvalues.get("age_years", np.nan)

    if valid_null.size == 0 or not np.isfinite(observed_val):
        p_perm = np.nan
    else:
        p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed_val)) + 1.0) / (valid_null.size + 1.0)

    if plot and valid_null.size > 0 and np.isfinite(observed_val):
        try:
            from scripts.develop.permutation_test import plot_permut_test
            plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None)
        except Exception:
            pass

    return observed_val, observed_val_p, p_perm, valid_null


def get_bf_results(content, df, age2use, filename=None):
    """
    Read cached Bayes Factor results if available; otherwise compute via Pearson-based BF.

    Parameters
    ----------
    content : str
        Name of the analysis context (used for cache filename).
    df : pd.DataFrame
        Data containing the measure and predictor.
    age2use : str
        Name of the age column to use.
    filename : Path or None
        Optional override for cache file path.

    Returns
    -------
    BF10 : float
        Bayes factor (evidence for H1 over H0).
    BF_conclusion : str
        Textual interpretation of BF strength.
    """
    import pandas as pd
    import config as C
    from scripts.utils.data_utils import bf_gaussian_via_pearson, interpret_bayes_factor

    if filename is None:
        filename = C.RESULTSPATH / f"beyesfactor_{content}_trials.csv"  # keep current naming
    filename = Path(filename)

    if filename.exists():
        BF_dict = pd.read_csv(filename)
        BF10 = BF_dict["BF10"].values[0]
        BF_conclusion = BF_dict["BF_conclusion"].values[0]
        return BF10, BF_conclusion

    BF = bf_gaussian_via_pearson(df, "n_trials", age2use)
    BF10 = BF["BF10"]
    BF_conclusion = interpret_bayes_factor(BF10)

    filename.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_col": [content], "BF10": [BF10], "BF_conclusion": [BF_conclusion]}).to_csv(
        filename, index=False
    )
    return BF10, BF_conclusion


def get_permut_results(content, age2use, df, filename=None):
    """
    Read cached permutation test results if available; otherwise compute.

    Parameters
    ----------
    content : str
        Analysis context (used for cache filename).
    age2use : str
        Predictor column name for age.
    df : pd.DataFrame
        Input data containing dependent variable and predictor.
    filename : Path or None
        Optional override for cache file path.

    Returns
    -------
    p_perm : float
        Permutation-based p-value.
    observed_val : float
        Observed coefficient for age.
    """
    import pandas as pd
    import config as C
    from statsmodels.genmod.families import Gaussian

    n_permut = C.N_PERMUT_BEHAVIOR
    if filename is None:
        filename = C.RESULTSPATH / f"num_{content}_trials_{age2use}_{n_permut}permutation.csv"
    filename = Path(filename)

    if filename.exists():
        permut_df = pd.read_csv(filename)
        p_perm = permut_df["p_perm"].values[0]
        observed_val = permut_df["observed_val"].values[0]
        return p_perm, observed_val

    formula2use = f"n_trials ~ {age2use}"
    this_age = df[age2use].values
    family_func = Gaussian()

    observed_val, observed_val_p, p_perm, valid_null = run_permutation_test(
        data=df,
        age_labels=this_age,
        formula=formula2use,
        family_func=family_func,
        shuffling="labels1_global",
        n_permut=n_permut,
        n_jobs=6,
        random_state=C.RANDOM_STATE,
        plot=False,
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "y_col": [content],
            "n_perm": [n_permut],
            "formula": [formula2use],
            "observed_val": [observed_val],
            "observed_val_p": [observed_val_p],
            "p_perm": [p_perm],
            "ave_null_dist": [np.nanmean(valid_null) if len(valid_null) else np.nan],
        }
    ).to_csv(filename, index=False)

    return p_perm, observed_val


def get_permut_results_table(df, age_col, measures,
                             family_func, shuffling, n_permut, n_jobs,
                             random_state, filename=None):
    """
    Batch version of permutation test across multiple dependent variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    age_col : str
        Column name for age predictor.
    measures : list of str
        Dependent variables (metrics) to test.
    family_func : statsmodels.genmod.families.Family
        GLM family.
    shuffling : str
        Shuffling mode.
    n_permut : int
        Number of permutations.
    n_jobs : int
        Number of parallel jobs.
    random_state : int
        Random seed.
    filename : Path or None
        Optional cache file path.

    Returns
    -------
    pd.DataFrame
        One row per measure with permutation test statistics.
    """
    if filename.exists():
        return pd.read_csv(filename)

    work = df.copy()
    # map to the unified predictor name expected by run_permutation_test
    if age_col != 'age_years':
        work['age_years'] = work[age_col]

    rows = []
    for measure in measures:
        fit = work.loc[~np.isnan(work[measure])].copy().reset_index(drop=True)
        formula = f"{measure} ~ age_years"
        age_vals = fit['age_years'].to_numpy()

        observed, observed_p, p_perm, valid_null = run_permutation_test(
            data=fit,
            age_labels=age_vals,
            formula=formula,
            family_func=family_func,
            shuffling=shuffling,
            n_permut=n_permut,
            n_jobs=n_jobs,
            random_state=random_state,
            plot=False,
        )
        rows.append({
            'y_var': measure,
            'n_perm': n_permut,
            'formula': formula,
            'observed_val': observed,
            'observed_val_p': observed_p,
            'p_perm': p_perm,
            'ave_null_dist': float(np.nanmean(valid_null)) if len(valid_null) else np.nan,
        })

    res = pd.DataFrame(rows)
    filename.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(filename, index=False)
    return res
