import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from statsmodels.formula.api import glm

def single_permutation(i, data, permuted_label, *, formula, family_func):
    """Fit GLM once with permuted 'age_years'; return the coefficient for 'age_years' (np.nan on failure)."""
    try:
        shuffled = data.copy()
        shuffled["age_years"] = permuted_label
        model = glm(formula=formula, data=shuffled, family=family_func).fit()
        return model.params["age_years"] if "age_years" in model.params.index else np.nan
    except Exception:
        return np.nan


def run_permutation_test(data, age_labels, *, formula,
                         family_func, shuffling, n_permut, n_jobs,
                         random_state, plot=False):
    """Permutation test for 'age_years' in a GLM; returns (observed_beta, glm_p, perm_p, valid_null)."""
    df = data.copy()

    # lazy import to avoid circular deps
    try:
        from scripts.utils.data_utils import shuffle_labels_perm
    except Exception:
        shuffle_labels_perm = None

    if shuffle_labels_perm is None:
        raise RuntimeError("shuffle_labels_perm not available: please ensure scripts.utils.data_utils provides it.")

    permuted_labels, _ = shuffle_labels_perm(
        labels1=age_labels, labels2=None, shuffling=shuffling,
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

    model_obs = glm(formula=formula, data=df, family=family_func).fit()
    observed_val = model_obs.params["age_years"] if "age_years" in model_obs.params.index else np.nan
    observed_val_p = model_obs.pvalues["age_years"] if "age_years" in model_obs.pvalues.index else np.nan

    if valid_null.size == 0 or not np.isfinite(observed_val):
        p_perm = np.nan
    else:
        p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed_val)) + 1.0) / (valid_null.size + 1.0)

    # Optional plotting via lazy import; silently skip if unavailable
    if plot and valid_null.size > 0 and np.isfinite(observed_val):
        try:
            from scripts.utils.plot_utils import plot_permut_test
            plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None)
        except Exception:
            pass

    return observed_val, observed_val_p, p_perm, valid_null
