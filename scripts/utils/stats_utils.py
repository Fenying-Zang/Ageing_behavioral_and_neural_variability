import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from statsmodels.formula.api import glm
from pathlib import Path
import pandas as pd
import config as C

def single_permutation(i, data, permuted_label, *, formula, family_func):
    """Fit GLM once with permuted 'age_years'; return the coefficient for 'age_years' (np.nan on failure)."""
    try:
        shuffled = data.copy()
        shuffled["age_years"] = permuted_label
        model = glm(formula=formula, data=shuffled, family=family_func, eval_env=0).fit()
        return model.params["age_years"] if "age_years" in model.params.index else np.nan
    except Exception:
        return np.nan


# def run_permutation_test(data, age_labels, *, formula,
#                          family_func, shuffling, n_permut, n_jobs,
#                          random_state, plot=False):
#     """Permutation test for 'age_years' in a GLM; returns (observed_beta, glm_p, perm_p, valid_null)."""
#     df = data.copy()

#     # lazy import to avoid circular deps
#     try:
#         from scripts.utils.data_utils import shuffle_labels_perm
#     except Exception:
#         shuffle_labels_perm = None

#     if shuffle_labels_perm is None:
#         raise RuntimeError("shuffle_labels_perm not available: please ensure scripts.utils.data_utils provides it.")

#     permuted_labels, _ = shuffle_labels_perm(
#         labels1=age_labels, labels2=None, shuffling=shuffling,
#         n_permut=n_permut, random_state=random_state, n_cores=n_jobs,
#     )

#     null_dist = Parallel(n_jobs=n_jobs)(
#         delayed(single_permutation)(
#             i, df, permuted_labels[i], formula=formula, family_func=family_func
#         )
#         for i in tqdm(range(n_permut))
#     )
#     null_dist = np.asarray(null_dist, dtype=float)
#     valid_null = null_dist[np.isfinite(null_dist)]

#     model_obs = glm(formula=formula, data=df, family=family_func).fit()
#     observed_val = model_obs.params["age_years"] if "age_years" in model_obs.params.index else np.nan
#     observed_val_p = model_obs.pvalues["age_years"] if "age_years" in model_obs.pvalues.index else np.nan

#     if valid_null.size == 0 or not np.isfinite(observed_val):
#         p_perm = np.nan
#     else:
#         p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed_val)) + 1.0) / (valid_null.size + 1.0)

#     # Optional plotting via lazy import; silently skip if unavailable
#     if plot and valid_null.size > 0 and np.isfinite(observed_val):
#         try:
#             from scripts.utils.plot_utils import plot_permut_test
#             plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None)
#         except Exception:
#             pass

#     return observed_val, observed_val_p, p_perm, valid_null

# scripts/utils/stats_utils.py 里
def run_permutation_test(data, age_labels, *, formula,
                         family_func, shuffling, n_permut, n_jobs,
                         random_state, plot=False, group_labels=None):
    """Permutation test for 'age_years' in a GLM; returns (observed_beta, glm_p, perm_p, valid_null)."""
    df = data.copy()
    try:
        from scripts.utils.data_utils import shuffle_labels_perm
    except Exception:
        shuffle_labels_perm = None
    if shuffle_labels_perm is None:
        raise RuntimeError("shuffle_labels_perm not available.")

    # 关键改动：把 group_labels 传进去（为 'labels1_based_on_2' 提供 labels2）
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
            from scripts.utils.permutation_test import plot_permut_test
            plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None)
        except Exception:
            pass

    return observed_val, observed_val_p, p_perm, valid_null


def get_bf_results(content, df, age2use, filename=None):
    """Read BF cache if exists; otherwise compute via Pearson-based BF and save. Returns (BF10, BF_conclusion)."""
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
    """Read permutation cache if exists; otherwise run unified permutation test and save. Returns (p_perm, observed_val)."""
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
    Read-if-exists else compute-and-save permutation stats for multiple measures.

    Returns a DataFrame with rows:
    ['y_var','n_perm','formula','observed_val','observed_val_p','p_perm','ave_null_dist']
    """

    # reuse the unified runner already in this module
    # from scripts.utils.stats_utils import run_permutation_test  # not needed if same file

    # default cache path mirrors your current naming
    # if filename is None:
    #     filename = C.RESULTSPATH / f"{age_col}_{n_permut}permutation.csv"
    # filename = Path(filename)

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
