

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # works without a display (Colab/Jupyter also fine)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
K_RANGE = range(2, 9)                      # candidate numbers of clusters
DBSCAN_MIN_SAMPLES_GRID = [3, 4, 5, 6, 8, 10]
DBSCAN_MAX_NOISE_FRACTION = 0.30           # reject DBSCAN settings that discard > 30 % of points


# --------------------------------------------------------------------------- #
# 1. Data loading, inspection, preprocessing
# --------------------------------------------------------------------------- #
def load_dataset():
    """Load the Wine dataset. Returns (features DataFrame, cultivar labels Series)."""
    wine = load_wine(as_frame=True)
    return wine.data.copy(), wine.target.copy()


def inspect_dataset(features, labels):
    """Print shape, dtypes, missing values, duplicates and class counts."""
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Records  : {features.shape[0]}")
    print(f"Features : {features.shape[1]}")
    print(f"Missing values (total)   : {int(features.isna().sum().sum())}")
    print(f"Duplicate rows           : {int(features.duplicated().sum())}")
    numeric_flags = features.dtypes.apply(pd.api.types.is_numeric_dtype)
    print(f"Non-numeric columns      : {int((~numeric_flags).sum())}")
    print("Cultivar counts (reference only, not used for fitting):")
    print(labels.value_counts().sort_index().to_string())
    print("\nFeature statistics:")
    print(features.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    return features.describe().T[["mean", "std", "min", "max"]]


def clean_and_scale(features):
    """Drop duplicates / impute missing values (none expected) and standardise."""
    cleaned = features.drop_duplicates().copy()
    if cleaned.isna().any().any():                       # edge case: handle missing values
        cleaned = cleaned.fillna(cleaned.median(numeric_only=True))
    scaler = StandardScaler()                            # z = (x - mean) / std
    scaled = scaler.fit_transform(cleaned)
    return cleaned, scaled


# --------------------------------------------------------------------------- #
# 2. Metric helper with edge-case handling
# --------------------------------------------------------------------------- #
def safe_internal_metrics(data, cluster_labels):
    """
    Silhouette and Davies-Bouldin need 2 <= n_clusters <= n_samples - 1.
    DBSCAN noise points (label -1) are excluded before scoring.
    Returns (silhouette, davies_bouldin, n_clusters, n_noise); NaN if not computable.
    """
    labels_arr = np.asarray(cluster_labels)
    mask = labels_arr != -1
    n_noise = int((~mask).sum())
    kept_labels = labels_arr[mask]
    n_clusters = len(set(kept_labels))
    if n_clusters < 2 or n_clusters > mask.sum() - 1:
        return np.nan, np.nan, n_clusters, n_noise
    return (silhouette_score(data[mask], kept_labels),
            davies_bouldin_score(data[mask], kept_labels),
            n_clusters, n_noise)


# --------------------------------------------------------------------------- #
# 3. Parameter selection
# --------------------------------------------------------------------------- #
def evaluate_k_range(scaled):
    """Inertia, silhouette and DB for K-Means and Ward for each candidate k."""
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(scaled)
        sil_km, db_km, _, _ = safe_internal_metrics(scaled, km.labels_)
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(scaled)
        sil_hc, db_hc, _, _ = safe_internal_metrics(scaled, hc.labels_)
        rows.append({"k": k, "kmeans_inertia": km.inertia_,
                     "kmeans_silhouette": sil_km, "kmeans_db": db_km,
                     "ward_silhouette": sil_hc, "ward_db": db_hc})
    return pd.DataFrame(rows)


def choose_k(k_table):
    """Pick k with the highest K-Means silhouette (elbow + DB are used as cross-checks)."""
    return int(k_table.loc[k_table["kmeans_silhouette"].idxmax(), "k"])


def k_distance_curve(scaled, k_neighbors):
    """Sorted distance to the k-th nearest neighbour (used for the DBSCAN eps heuristic)."""
    nn = NearestNeighbors(n_neighbors=k_neighbors).fit(scaled)
    distances, _ = nn.kneighbors(scaled)
    return np.sort(distances[:, -1])


def tune_dbscan(scaled):
    """
    Grid-search eps / min_samples. Keep only settings that give >= 2 clusters and
    <= 30 % noise, then choose the one with the best silhouette (non-noise points).
    """
    eps_grid = np.round(np.arange(1.0, 4.01, 0.1), 2)
    records = []
    for min_samples in DBSCAN_MIN_SAMPLES_GRID:
        for eps in eps_grid:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
            sil, db, n_clusters, n_noise = safe_internal_metrics(scaled, labels)
            records.append({"eps": eps, "min_samples": min_samples, "clusters": n_clusters,
                            "noise": n_noise, "silhouette": sil, "davies_bouldin": db})
    grid = pd.DataFrame(records)
    valid = grid[(grid["clusters"] >= 2) &
                 (grid["noise"] <= DBSCAN_MAX_NOISE_FRACTION * len(scaled))].dropna()
    if valid.empty:
        return None, grid
    best = valid.loc[valid["silhouette"].idxmax()]
    return best, grid


# --------------------------------------------------------------------------- #
# 4. Fit final models
# --------------------------------------------------------------------------- #
def fit_final_models(scaled, best_k, dbscan_params):
    models = {}
    models["K-Means"] = KMeans(n_clusters=best_k, n_init=20,
                               random_state=RANDOM_STATE).fit_predict(scaled)
    models["Hierarchical (Ward)"] = AgglomerativeClustering(
        n_clusters=best_k, linkage="ward").fit_predict(scaled)
    if dbscan_params is not None:
        models["DBSCAN"] = DBSCAN(eps=float(dbscan_params["eps"]),
                                  min_samples=int(dbscan_params["min_samples"])).fit_predict(scaled)
    return models


def build_comparison_table(scaled, models, true_labels, dbscan_params):
    rows = []
    for name, labels in models.items():
        sil, db, n_clusters, n_noise = safe_internal_metrics(scaled, labels)
        mask = np.asarray(labels) != -1
        ari = adjusted_rand_score(true_labels[mask], np.asarray(labels)[mask])
        if name == "DBSCAN":
            setting = (f"eps={float(dbscan_params['eps']):.1f}, "
                       f"min_samples={int(dbscan_params['min_samples'])}")
        else:
            setting = f"k={n_clusters}"
        rows.append({"Algorithm": name, "Parameters": setting, "Clusters": n_clusters,
                     "Noise points": n_noise, "Silhouette": round(sil, 4),
                     "Davies-Bouldin": round(db, 4), "ARI (reference)": round(ari, 4)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Visualisation
# --------------------------------------------------------------------------- #
def plot_model_selection(k_table, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(k_table["k"], k_table["kmeans_inertia"], "o-")
    axes[0].set(title="K-Means elbow (inertia)", xlabel="k", ylabel="Inertia (WCSS)")
    axes[1].plot(k_table["k"], k_table["kmeans_silhouette"], "o-", label="K-Means")
    axes[1].plot(k_table["k"], k_table["ward_silhouette"], "s--", label="Ward")
    axes[1].set(title="Silhouette vs k (higher is better)", xlabel="k", ylabel="Silhouette")
    axes[1].legend()
    axes[2].plot(k_table["k"], k_table["kmeans_db"], "o-", label="K-Means")
    axes[2].plot(k_table["k"], k_table["ward_db"], "s--", label="Ward")
    axes[2].set(title="Davies-Bouldin vs k (lower is better)", xlabel="k", ylabel="DB index")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_kdistance(curve, k_neighbors, path, chosen_eps=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve)
    if chosen_eps is not None:
        ax.axhline(chosen_eps, color="red", ls="--", label=f"selected eps = {chosen_eps:.1f}")
        ax.legend()
    ax.set(title=f"k-distance plot (k = {k_neighbors})", xlabel="Points sorted by distance",
           ylabel=f"Distance to {k_neighbors}-th nearest neighbour")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dendrogram(scaled, best_k, path):
    link = linkage(scaled, method="ward")
    fig, ax = plt.subplots(figsize=(9, 4))
    cut_height = link[-(best_k - 1), 2] - 1e-9          # cut that yields best_k clusters
    dendrogram(link, truncate_mode="lastp", p=20, ax=ax, color_threshold=cut_height,
               no_labels=True)
    ax.axhline(cut_height, color="red", ls="--", label=f"cut for k = {best_k}")
    ax.set(title="Ward dendrogram (last 20 merges)", ylabel="Merge distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cluster_scatter(scaled, models, true_labels, path):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_ * 100
    panels = dict(models)
    panels["Cultivar (reference only)"] = np.asarray(true_labels)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, labels) in zip(axes.ravel(), panels.items()):
        palette = sns.color_palette("tab10", n_colors=max(len(set(labels)), 3))
        for idx, lab in enumerate(sorted(set(labels))):
            sel = np.asarray(labels) == lab
            if lab == -1:
                ax.scatter(coords[sel, 0], coords[sel, 1], c="lightgrey", marker="x",
                           s=25, label="noise")
            else:
                ax.scatter(coords[sel, 0], coords[sel, 1], color=palette[idx % len(palette)],
                           s=28, edgecolor="k", linewidth=0.3, label=f"cluster {lab}")
        ax.set(title=name, xlabel=f"PC1 ({explained[0]:.1f}%)", ylabel=f"PC2 ({explained[1]:.1f}%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return explained


def plot_cluster_profile(cleaned, labels, path):
    """Heatmap of mean standardised feature value per K-Means cluster."""
    z = (cleaned - cleaned.mean()) / cleaned.std()
    profile = z.groupby(labels).mean().T
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(profile, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "Mean z-score"})
    ax.set(title="K-Means cluster profiles (mean z-score)", xlabel="Cluster", ylabel="")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return profile


def scaling_sensitivity(cleaned, scaled, best_k):
    """Show why scaling matters: same K-Means on raw vs standardised features."""
    raw_labels = KMeans(n_clusters=best_k, n_init=20,
                        random_state=RANDOM_STATE).fit_predict(cleaned.values)
    sil_raw = silhouette_score(cleaned.values, raw_labels)
    db_raw = davies_bouldin_score(cleaned.values, raw_labels)
    # evaluate raw-feature clustering in the standardised space for a fair comparison
    sil_raw_in_scaled, db_raw_in_scaled, _, _ = safe_internal_metrics(scaled, raw_labels)
    return {"raw_space_silhouette": sil_raw, "raw_space_db": db_raw,
            "scaled_space_silhouette_of_raw_labels": sil_raw_in_scaled,
            "scaled_space_db_of_raw_labels": db_raw_in_scaled}


def measure_runtime(scaled, best_k, dbscan_params):
    """Wall-clock time of a single fit for each algorithm (indicative only)."""
    candidates = {
        "K-Means": KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE),
        "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=best_k, linkage="ward"),
    }
    if dbscan_params is not None:
        candidates["DBSCAN"] = DBSCAN(eps=float(dbscan_params["eps"]),
                                      min_samples=int(dbscan_params["min_samples"]))
    timings = {}
    for name, estimator in candidates.items():
        start = time.perf_counter()
        estimator.fit(scaled)
        timings[name] = time.perf_counter() - start
    return timings


# --------------------------------------------------------------------------- #
# 7. Self-tests (used for the Test Cases table in the report)
# --------------------------------------------------------------------------- #
def run_self_tests(features, cleaned, scaled, models, comparison, best_k):
    """Run simple checks and print PASS / FAIL for each test case."""
    results = []

    def record(name, condition, detail):
        results.append(condition)
        print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}")

    print("\n" + "=" * 70)
    print("SELF-TESTS")
    print("=" * 70)
    record("TC1 dataset loading", features.shape == (178, 13),
           f"shape = {features.shape}")
    n_missing = int(features.isna().sum().sum())
    n_duplicates = int(features.duplicated().sum())
    record("TC2 data quality", n_missing == 0 and n_duplicates == 0,
           f"missing = {n_missing}, duplicates = {n_duplicates}")
    record("TC3 feature scaling",
           np.abs(scaled.mean(axis=0)).max() < 1e-9 and np.allclose(scaled.std(axis=0), 1.0),
           f"max |mean| = {np.abs(scaled.mean(axis=0)).max():.2e}, std range = "
           f"[{scaled.std(axis=0).min():.4f}, {scaled.std(axis=0).max():.4f}]")
    record("TC4 model execution",
           all(len(lab) == len(scaled) for lab in models.values()) and len(models) == 3,
           "labels returned by " + ", ".join(f"{n} ({len(set(l)) - (1 if -1 in l else 0)} clusters)"
                                             for n, l in models.items()))
    sil_ok = comparison["Silhouette"].between(-1, 1).all()
    db_ok = (comparison["Davies-Bouldin"] >= 0).all()
    record("TC5 metric ranges", bool(sil_ok and db_ok),
           f"silhouette in [-1, 1]: {bool(sil_ok)}; Davies-Bouldin >= 0: {bool(db_ok)}")
    one_cluster = safe_internal_metrics(scaled, np.zeros(len(scaled), dtype=int))
    all_noise = safe_internal_metrics(scaled, -np.ones(len(scaled), dtype=int))
    record("TC6 metric edge cases", np.isnan(one_cluster[0]) and np.isnan(all_noise[0]),
           f"1 cluster -> {one_cluster[0]}, all noise -> {all_noise[0]} (NaN returned, no crash)")
    repeat = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE).fit_predict(scaled)
    ari_repeat = adjusted_rand_score(models["K-Means"], repeat)
    record("TC7 reproducibility", ari_repeat == 1.0,
           f"ARI between two seeded runs = {ari_repeat:.1f}")
    expected = ["fig1_model_selection.png", "fig2_kdistance.png", "fig3_dendrogram.png",
                "fig4_cluster_scatter.png", "fig5_kmeans_profile.png"]
    present = [f for f in expected if os.path.getsize(os.path.join(OUTPUT_DIR, f)) > 0]
    record("TC8 visualisation files", len(present) == len(expected),
           f"{len(present)} of {len(expected)} figures saved")
    print(f"Summary: {sum(results)} of {len(results)} tests passed")


# --------------------------------------------------------------------------- #
# 6. Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} | "
          f"scikit-learn {sklearn.__version__} | matplotlib {matplotlib.__version__} | "
          f"seaborn {sns.__version__}")

    features, true_labels = load_dataset()
    stats = inspect_dataset(features, true_labels)
    stats.round(4).to_csv(f"{OUTPUT_DIR}/dataset_statistics.csv")

    cleaned, scaled = clean_and_scale(features)
    true_labels = true_labels.loc[cleaned.index]
    print(f"\nAfter cleaning: {cleaned.shape[0]} records, {cleaned.shape[1]} features (all scaled)")

    # ---- parameter selection ------------------------------------------------
    k_table = evaluate_k_range(scaled)
    best_k = choose_k(k_table)
    print("\nK selection table:")
    print(k_table.round(4).to_string(index=False))
    print(f"\nSelected k (max K-Means silhouette) = {best_k}")
    plot_model_selection(k_table, f"{OUTPUT_DIR}/fig1_model_selection.png")
    k_table.round(4).to_csv(f"{OUTPUT_DIR}/k_selection.csv", index=False)

    best_dbscan, dbscan_grid = tune_dbscan(scaled)
    dbscan_grid.round(4).to_csv(f"{OUTPUT_DIR}/dbscan_grid.csv", index=False)
    if best_dbscan is None:
        print("\nNo DBSCAN setting satisfied the constraints (>=2 clusters, <=30% noise).")
        curve = k_distance_curve(scaled, 5)
        plot_kdistance(curve, 5, f"{OUTPUT_DIR}/fig2_kdistance.png")
    else:
        print("\nSelected DBSCAN setting:")
        print(best_dbscan.to_string())
        ms = int(best_dbscan["min_samples"])
        plot_kdistance(k_distance_curve(scaled, ms), ms, f"{OUTPUT_DIR}/fig2_kdistance.png",
                       chosen_eps=float(best_dbscan["eps"]))

    # ---- final models and comparison ------------------------------------------
    models = fit_final_models(scaled, best_k, best_dbscan)
    comparison = build_comparison_table(scaled, models, true_labels, best_dbscan)
    print("\nCOMPARISON TABLE")
    print(comparison.to_string(index=False))
    comparison.to_csv(f"{OUTPUT_DIR}/comparison_table.csv", index=False)

    # ---- visualisation ---------------------------------------------------------
    plot_dendrogram(scaled, best_k, f"{OUTPUT_DIR}/fig3_dendrogram.png")
    explained = plot_cluster_scatter(scaled, models, true_labels,
                                     f"{OUTPUT_DIR}/fig4_cluster_scatter.png")
    print(f"\nPCA variance explained by PC1, PC2: {explained[0]:.2f}%, {explained[1]:.2f}%")
    profile = plot_cluster_profile(cleaned, models["K-Means"],
                                   f"{OUTPUT_DIR}/fig5_kmeans_profile.png")
    print("\nK-Means cluster sizes:", np.bincount(models["K-Means"]).tolist())
    print("Ward cluster sizes   :", np.bincount(models["Hierarchical (Ward)"]).tolist())
    if "DBSCAN" in models:
        lab = models["DBSCAN"]
        print("DBSCAN sizes (noise first):", [int((lab == l).sum()) for l in sorted(set(lab))])
    profile.round(3).to_csv(f"{OUTPUT_DIR}/kmeans_profile.csv")
    print("\nK-Means profile (mean z-score):")
    print(profile.round(2).to_string())

    # ---- scaling sensitivity ---------------------------------------------------
    sens = scaling_sensitivity(cleaned, scaled, best_k)
    print("\nSCALING SENSITIVITY (K-Means, same k):")
    for key, value in sens.items():
        print(f"  {key}: {value:.4f}")

    # ---- runtime ---------------------------------------------------------------
    timings = measure_runtime(scaled, best_k, best_dbscan)
    print("\nRUNTIME (single fit, seconds):")
    for name, seconds in timings.items():
        print(f"  {name}: {seconds:.4f}")
    pd.Series(timings).round(5).to_csv(f"{OUTPUT_DIR}/runtime.csv", header=["seconds"])

    run_self_tests(features, cleaned, scaled, models, comparison, best_k)
    print("\nDone. Files saved in ./outputs")


if __name__ == "__main__":
    main()