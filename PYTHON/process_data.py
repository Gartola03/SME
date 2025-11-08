################################################################################
# Script: Data Preprocessing, Metrics, and Association Analysis
# ******************************************************************************
# Description:
#   This script provides functions for:
#     1. Discretizing numeric variables (equal-width and equal-frequency)
#     2. Computing dataset metrics (entropy, variance, AUC)
#     3. Normalizing and standardizing numeric data
#     4. Filtering variables by metrics
#     5. Computing pairwise association matrices
#     6. Plotting metrics and association matrices
#
# Author: Garikoitz Artola Obando (gartola008@ikasle.ehu.eus)
# Date: 09/11/2025
################################################################################



# LOAD LIBRARIES ---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# AUXILIAR FUNCTIONS -----------------------------------------------------------
def discretizeEW(x, num_bin):
    x = np.asarray(x)

    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("discretizeEW: x must be numeric")
    if not isinstance(num_bin, int) or num_bin < 1:
        raise ValueError("discretizeEW: num_bin must be integer >= 1")

    if np.all(np.isnan(x)):
        return {"factor": pd.Categorical(x), "cut_points": np.array([])}

    finite_x = x[np.isfinite(x)]
    if len(finite_x) == 0:
        return {"factor": pd.Categorical(x), "cut_points": np.array([])}

    minx, maxx = np.min(finite_x), np.max(finite_x)
    if minx == maxx:
        return {
            "factor": pd.Categorical(["single_value"] * len(x)),
            "cut_points": np.array([])
        }

    breaks = np.linspace(minx, maxx, num_bin + 1)
    cut_points = breaks[1:-1]
    intervals = [-np.inf] + cut_points.tolist() + [np.inf]

    cats = pd.cut(x, bins=intervals, include_lowest=True, right=False)
    return {"factor": cats, "cut_points": cut_points}


def discretizeEF(x, num_bin):
    x = np.asarray(x)

    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("discretizeEF: x must be numeric")
    if not isinstance(num_bin, int) or num_bin < 1:
        raise ValueError("discretizeEF: num_bin must be integer >= 1")

    if np.all(np.isnan(x)):
        return {"factor": pd.Categorical(x), "cut_points": np.array([])}

    finite_mask = ~np.isnan(x)
    n = finite_mask.sum()
    if n == 0:
        return {"factor": pd.Categorical(x), "cut_points": np.array([])}

    num_per_bin = n // num_bin
    rest = n % num_bin

    bin_sizes = [num_per_bin + 1] * rest + [num_per_bin] * (num_bin - rest)

    sorted_idx = np.argsort(np.where(np.isnan(x), np.inf, x))
    sorted_x = x[sorted_idx]

    cum = np.cumsum(bin_sizes)
    cut_indices = cum[:-1]

    cut_points = np.unique(sorted_x[cut_indices])
    intervals = [-np.inf] + cut_points.tolist() + [np.inf]

    cats = pd.cut(x, bins=intervals, include_lowest=True, right=False)
    return {"factor": cats, "cut_points": cut_points}


def entropy(x, base=2):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return 0.0
    probs = x.value_counts(normalize=True)
    return float(-(probs * np.log(probs) / np.log(base)).sum())


def joint_entropy(x, y, base=2):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return 0.0
    probs = df.groupby(["x", "y"]).size() / len(df)
    return float(-(probs * np.log(probs) / np.log(base)).sum())


def mutual_information(x, y, base=2):
    return entropy(x, base) + entropy(y, base) - joint_entropy(x, y, base)


def compute_auc_numeric(x, class_col):
    # Convert class to boolean (0/1) if necessary
    class_col = pd.Series(class_col)
    if class_col.dtype.kind in "ifc":  # numeric or categorical
        class_col = class_col.astype("category").cat.codes
    class_col = class_col.astype(bool)

    x = pd.Series(x)

    # Remove NAs
    mask = ~(x.isna() | class_col.isna())
    x = x[mask].values
    class_col = class_col[mask].values

    P = np.sum(class_col)         # total positives
    N = np.sum(~class_col)        # total negatives

    if P == 0 or N == 0:
        raise ValueError("compute_auc_numeric: class must have both True and False")

    # All unique cutpoints
    cuts = np.unique(x)
    TPR = np.zeros(len(cuts))
    FPR = np.zeros(len(cuts))

    for i, cutoff in enumerate(cuts):
        pred = x >= cutoff

        TP = np.sum(pred & class_col)
        FP = np.sum(pred & ~class_col)
        FN = np.sum(~pred & class_col)
        TN = np.sum(~pred & ~class_col)

        TPR[i] = TP / (TP + FN)
        FPR[i] = FP / (FP + TN)

    # Add starting and ending points
    FPR = np.concatenate(([0], FPR, [1]))
    TPR = np.concatenate(([0], TPR, [1]))

    # Trapezoidal rule
    auc = np.sum(np.diff(FPR) * (TPR[:-1] + TPR[1:]) / 2)

    return auc



# FUNCTIONS --------------------------------------------------------------------


# 1. DISCRETIZATION ------------------------------------------------------------
def dis_atribute(attribute, num_bin, mode=False):
    if not np.issubdtype(np.asarray(attribute).dtype, np.number):
        raise ValueError("attribute must be numeric")
    if not isinstance(num_bin, int):
        raise ValueError("num_bin must be int")
    if not isinstance(mode, bool):
        raise ValueError("mode must be bool")

    return discretizeEW(attribute, num_bin) if not mode else discretizeEF(attribute, num_bin)


def dis_dataset(dataset, num_bin, mode=False):
    df = dataset.copy()
    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        d = dis_atribute(df[col].values, num_bin, mode)
        df[col] = d["factor"]

    return df



# 2. METRICS -------------------------------------------------------------------
def metrics_dataset(dataset, class_var=None, verbose=False):
    df = dataset.copy()
    if class_var is not None and class_var not in df.columns:
        raise ValueError("class_var not found")

    class_vec = df[class_var] if class_var else None

    results = []

    for col in df.columns:
        if col == class_var:
            continue

        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            varv = series.var(skipna=True)
            ent = np.nan
            aucv = np.nan

            if class_vec is not None:
                try:
                    aucv = compute_auc_numeric(series, class_vec)
                except Exception:
                    if verbose:
                        print("AUC failed for", col)

            results.append([col, "numeric", varv, ent, aucv])

        else:
            varv = np.nan
            ent = entropy(series)
            aucv = np.nan
            results.append([col, "categorical", varv, ent, aucv])

    return pd.DataFrame(results, columns=["variable", "type", "variance", "entropy", "auc"])



# 3. NORMALIZATION AND STANDARDIZATION -----------------------------------------
def normalize_vector(v):
    v = np.asarray(v)
    finite = v[np.isfinite(v)]
    if len(finite) == 0:
        return np.full(len(v), np.nan)
    minv, maxv = finite.min(), finite.max()
    if minv == maxv:
        return np.zeros(len(v))
    return (v - minv) / (maxv - minv)


def standardize_vector(v):
    v = np.asarray(v)
    finite = v[np.isfinite(v)]
    if len(finite) == 0:
        return np.full(len(v), np.nan)
    mean, sd = finite.mean(), finite.std()
    if sd == 0:
        return np.zeros(len(v))
    return (v - mean) / sd


def normalize_dataset(df):
    out = df.copy()
    for col in out.select_dtypes(include=np.number).columns:
        out[col] = normalize_vector(out[col])
    return out


def standardize_dataset(df):
    out = df.copy()
    for col in out.select_dtypes(include=np.number).columns:
        out[col] = standardize_vector(out[col])
    return out



# 4. FILTERING -----------------------------------------------------------------
def filter_by_metric(dataset, metric, threshold, class_var=None):
    if metric not in ["entropy", "variance", "auc"]:
        raise ValueError("metric must be 'entropy', 'variance', or 'auc'")

    met = metrics_dataset(dataset, class_var)

    if metric == "entropy":
        keep = met.loc[met["entropy"] >= threshold, "variable"]
    elif metric == "variance":
        keep = met.loc[met["variance"] >= threshold, "variable"]
    else:
        if class_var is None:
            raise ValueError("class_var required for AUC")
        keep = met.loc[met["auc"] >= threshold, "variable"]

    final_vars = [v for v in keep if v in dataset.columns]
    if class_var and class_var not in final_vars:
        final_vars.append(class_var)

    return dataset[final_vars]



# 5. PAIRWISE ASSOCIATION MATRIX -----------------------------------------------
def pairwise_assoc_matrix(dataset, num_bin=10, normalize_mi=True, mode=False):

    df = dataset.copy()
    cols = df.columns
    n = len(cols)
    mat = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

    # Discretize numeric columns once
    df_discretized = dis_dataset(df, num_bin=num_bin, mode=mode)

    for i in cols:
        for j in cols:
            if i == j:
                mat.loc[i, j] = 1
                continue

            a, b = df[i], df[j]

            # Both numeric → Pearson correlation
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                mat.loc[i, j] = a.corr(b)
            else:
                # Use discretized version for mutual information
                A, B = df_discretized[i], df_discretized[j]

                mi = mutual_information(A, B)  # Assumes you have this function

                if normalize_mi:
                    Ha = entropy(A)  # Assumes you have this function
                    Hb = entropy(B)
                    denom = np.sqrt(Ha * Hb)
                    mat.loc[i, j] = mi / denom if denom > 0 else 0
                else:
                    mat.loc[i, j] = mi

    return mat



# 6. PLOTS ---------------------------------------------------------------------
def plot_auc(metrics_df):
    df = metrics_df.dropna(subset=["auc"])

    variables = df["variable"]
    aucs = df["auc"]

    plt.bar(variables, aucs)
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.ylabel("AUC")
    plt.title("AUC per numeric attribute")
    plt.axhline(0.5, linestyle="--")
    plt.show()


def plot_assoc_matrix(mat):
    plt.imshow(mat, cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(mat.shape[1]), mat.columns, rotation=90)
    plt.yticks(range(mat.shape[0]), mat.index)
    plt.title("Association matrix")
    plt.show()




