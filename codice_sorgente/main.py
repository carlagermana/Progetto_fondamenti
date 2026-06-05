"""
Progetto di data analysis sul dataset Country-data.csv.

Lo script esegue l'intera pipeline richiesta:
- caricamento e controllo del dataset;
- analisi descrittiva e matrice di correlazione;
- standardizzazione con StandardScaler;
- PCA, scree plot e biplot;
- scelta del numero di cluster con Elbow e Silhouette Score;
- clustering K-Means sul piano delle prime due componenti principali;
- clustering basato su grafo K-Neighbors tramite SpectralClustering;
- esportazione di grafici, tabelle, assegnazioni e relazione PDF.

Esecuzione consigliata dalla root del progetto:
    python3 output_progetto/codice_sorgente/main.py
"""

from __future__ import annotations

import os
import textwrap
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "output_progetto"
SOURCE_DIR = OUTPUT_DIR / "codice_sorgente"
PLOTS_DIR = OUTPUT_DIR / "grafici"
TABLES_DIR = OUTPUT_DIR / "tabelle"
REPORT_DIR = OUTPUT_DIR / "relazione"
DATA_DIR = OUTPUT_DIR / "dati"
ROOT_DATASET_PATH = ROOT_DIR / "Country-data.csv"
OUTPUT_DATASET_PATH = DATA_DIR / "Country-data.csv"
CACHE_DIR = OUTPUT_DIR / "ambiente" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Evita warning rumorosi di joblib sul conteggio dei core in alcuni ambienti macOS.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FULL_FEATURES = [
    "child_mort",
    "exports",
    "health",
    "imports",
    "income",
    "inflation",
    "life_expec",
    "total_fer",
    "gdpp",
]
SELECTED_FEATURES = ["child_mort", "income", "life_expec", "total_fer", "gdpp"]
FEATURE_DESCRIPTIONS = {
    "country": "nome dello stato",
    "child_mort": "mortalità sotto i 5 anni per 1000 nati vivi",
    "exports": "esportazioni di beni e servizi in percentuale del PIL",
    "health": "spesa sanitaria totale in percentuale del PIL",
    "imports": "importazioni di beni e servizi in percentuale del PIL",
    "income": "reddito netto per persona",
    "inflation": "crescita annua del PIL totale",
    "life_expec": "aspettativa di vita media alla nascita",
    "total_fer": "numero medio di figli per donna",
    "gdpp": "PIL pro capite",
}
CLUSTER_NAMES = {
    0: "profilo fragile",
    1: "profilo intermedio",
    2: "profilo avanzato",
    3: "profilo ad alto reddito anomalo",
}


@dataclass
class AnalysisResult:
    """Contiene i principali risultati prodotti dalla pipeline."""

    data: pd.DataFrame
    selected_features: list[str]
    pca_selected: PCA
    final_k: int
    elbow_k: int
    silhouette_k: int
    silhouette_final: float
    silhouette_knn: float
    ari_kmeans_knn: float
    cluster_profile: pd.DataFrame
    pca_variance: pd.DataFrame
    k_selection: pd.DataFrame
    subset_comparison: pd.DataFrame
    images: dict[str, Path]


def ensure_directories() -> None:
    """Crea le cartelle di output senza toccare le directory fornite dal docente."""

    for directory in [SOURCE_DIR, PLOTS_DIR, TABLES_DIR, REPORT_DIR, DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """Carica il CSV e verifica che le colonne attese siano presenti."""

    DATASET_PATH = ROOT_DATASET_PATH if ROOT_DATASET_PATH.exists() else OUTPUT_DATASET_PATH
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset non trovato: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    expected_columns = ["country"] + FULL_FEATURES
    missing_columns = sorted(set(expected_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Colonne mancanti nel dataset: {missing_columns}")

    # Copia di servizio nella cartella output, utile per rendere il progetto autocontenuto.
    data.to_csv(DATA_DIR / "Country-data.csv", index=False)
    return data


def standardize(data: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, StandardScaler]:
    """Applica StandardScaler: ogni variabile viene centrata e scalata."""

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    return scaled_data, scaler


def save_basic_tables(data: pd.DataFrame) -> None:
    """Esporta statistiche descrittive, valori mancanti e matrice di correlazione."""

    overview = pd.DataFrame(
        {
            "metrica": ["osservazioni", "colonne", "valori_mancanti_totali"],
            "valore": [data.shape[0], data.shape[1], int(data.isna().sum().sum())],
        }
    )
    overview.to_csv(TABLES_DIR / "dataset_overview.csv", index=False)
    data.isna().sum().rename("missing_values").to_csv(TABLES_DIR / "missing_values.csv")
    data[FULL_FEATURES].describe().T.round(3).to_csv(TABLES_DIR / "statistiche_descrittive.csv")
    data[FULL_FEATURES].corr().round(3).to_csv(TABLES_DIR / "matrice_correlazione.csv")


def compute_elbow_k(k_selection: pd.DataFrame) -> int:
    """
    Stima il punto di gomito misurando la massima distanza dalla retta
    che congiunge il primo e l'ultimo valore di inertia.
    """

    points = k_selection[["k", "inertia"]].to_numpy(dtype=float)
    first_point, last_point = points[0], points[-1]
    line = last_point - first_point
    shifted_points = points - first_point
    cross_2d = line[0] * shifted_points[:, 1] - line[1] * shifted_points[:, 0]
    distances = np.abs(cross_2d / np.linalg.norm(line))
    return int(points[int(np.argmax(distances)), 0])


def evaluate_kmeans_range(x_values: np.ndarray, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    """Calcola inertia/WCSS e Silhouette Score per diversi valori di k."""

    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(x_values)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(x_values, labels),
            }
        )
    return pd.DataFrame(rows)


def plot_feature_distributions(data: pd.DataFrame) -> Path:
    """Crea istogrammi compatti delle feature numeriche."""

    output = PLOTS_DIR / "distribuzioni_feature.png"
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.ravel()
    for ax, feature in zip(axes, FULL_FEATURES):
        ax.hist(data[feature], bins=22, color="#4c78a8", edgecolor="white")
        ax.set_title(feature)
        ax.set_ylabel("frequenza")
    fig.suptitle("Distribuzione delle feature numeriche", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_correlation_matrix(data: pd.DataFrame) -> Path:
    """Visualizza la matrice di correlazione tra feature."""

    output = PLOTS_DIR / "matrice_correlazione.png"
    corr = data[FULL_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FULL_FEATURES)), FULL_FEATURES, rotation=45, ha="right")
    ax.set_yticks(range(len(FULL_FEATURES)), FULL_FEATURES)
    for i in range(len(FULL_FEATURES)):
        for j in range(len(FULL_FEATURES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Matrice di correlazione", fontsize=15, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_scree(pca: PCA, prefix: str) -> Path:
    """Produce lo scree plot con varianza spiegata e cumulata."""

    output = PLOTS_DIR / f"scree_plot_{prefix}.png"
    ratios = pca.explained_variance_ratio_
    components = np.arange(1, len(ratios) + 1)
    cumulative = np.cumsum(ratios)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.bar(components, ratios * 100, color="#72b7b2", label="varianza spiegata")
    ax1.set_xlabel("Componenti principali")
    ax1.set_ylabel("Varianza spiegata (%)")
    ax1.set_xticks(components)
    ax2 = ax1.twinx()
    ax2.plot(components, cumulative * 100, marker="o", color="#e45756", label="cumulata")
    ax2.set_ylabel("Varianza cumulata (%)")
    ax2.set_ylim(0, 105)
    ax1.set_title("Scree Plot della PCA", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_biplot(
    scores: np.ndarray,
    loadings: np.ndarray,
    explained_ratio: np.ndarray,
    feature_names: list[str],
) -> Path:
    """Crea il biplot: score degli stati e loading delle variabili originali."""

    output = PLOTS_DIR / "biplot_pca_feature_selezionate.png"
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(scores[:, 0], scores[:, 1], s=36, alpha=0.72, color="#4c78a8")
    scale_x = scores[:, 0].max() - scores[:, 0].min()
    scale_y = scores[:, 1].max() - scores[:, 1].min()
    arrow_scale = min(scale_x, scale_y) * 0.38

    for idx, feature in enumerate(feature_names):
        ax.arrow(
            0,
            0,
            loadings[idx, 0] * arrow_scale,
            loadings[idx, 1] * arrow_scale,
            color="#d62728",
            alpha=0.85,
            head_width=0.08,
            length_includes_head=True,
        )
        ax.text(
            loadings[idx, 0] * arrow_scale * 1.12,
            loadings[idx, 1] * arrow_scale * 1.12,
            feature,
            color="#7f1d1d",
            fontsize=10,
        )

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}% varianza)")
    ax.set_title("Biplot: score e loading sulle prime due PC", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_k_selection(k_selection: pd.DataFrame, final_k: int) -> Path:
    """Mostra insieme metodo Elbow e Silhouette Score."""

    output = PLOTS_DIR / "elbow_silhouette.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(k_selection["k"], k_selection["inertia"], marker="o", color="#4c78a8")
    axes[0].axvline(final_k, linestyle="--", color="#e45756", label=f"k scelto = {final_k}")
    axes[0].set_title("Metodo Elbow")
    axes[0].set_xlabel("Numero di cluster k")
    axes[0].set_ylabel("WCSS / inertia")
    axes[0].legend()

    axes[1].plot(k_selection["k"], k_selection["silhouette"], marker="o", color="#59a14f")
    axes[1].axvline(final_k, linestyle="--", color="#e45756", label=f"k scelto = {final_k}")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Numero di cluster k")
    axes[1].set_ylabel("Silhouette media")
    axes[1].legend()
    fig.suptitle("Scelta del numero di cluster", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_pca_clusters(
    scores: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    explained_ratio: np.ndarray,
    title: str,
    filename: str,
) -> Path:
    """Disegna gli stati nel piano PCA e, per K-Means, mostra anche i centroidi."""

    output = PLOTS_DIR / filename
    colors_map = ListedColormap(["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=labels, cmap=colors_map, s=50, alpha=0.83)
    if centers is not None:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="X",
            s=260,
            c="black",
            edgecolor="white",
            linewidth=1.2,
            label="centro di massa",
        )
        ax.legend()
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}% varianza)")
    ax.set_title(title, fontsize=15, fontweight="bold")
    legend = ax.legend(*scatter.legend_elements(), title="cluster", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_cluster_profile(cluster_profile: pd.DataFrame) -> Path:
    """Visualizza i profili medi standardizzati dei cluster K-Means."""

    output = PLOTS_DIR / "profilo_cluster_heatmap.png"
    values = cluster_profile[SELECTED_FEATURES]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-1.8, vmax=1.8)
    ax.set_xticks(range(len(SELECTED_FEATURES)), SELECTED_FEATURES, rotation=35, ha="right")
    ax.set_yticks(range(len(cluster_profile)), cluster_profile.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Profilo medio standardizzato dei cluster", fontsize=15, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_top_countries(assignments: pd.DataFrame) -> Path:
    """Mostra una tabella grafica con esempi di paesi per ogni gruppo."""

    output = PLOTS_DIR / "esempi_paesi_per_cluster.png"
    rows = []
    for cluster_id, group in assignments.groupby("cluster_kmeans"):
        ordered = group.sort_values("distanza_centroide").head(7)
        rows.append([f"Cluster {cluster_id}", ", ".join(ordered["country"].tolist())])

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Cluster", "Paesi più centrali"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    ax.set_title("Esempi di paesi vicini al centroide", fontsize=15, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def compare_feature_subsets(data: pd.DataFrame) -> pd.DataFrame:
    """Valuta dataset completo e sottoinsieme di feature tramite PCA a due componenti."""

    rows = []
    subsets = {
        "tutte_le_feature": FULL_FEATURES,
        "subset_socio_sanitario": SELECTED_FEATURES,
    }
    for subset_name, features in subsets.items():
        x_scaled, _ = standardize(data, features)
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        scores = pca.fit_transform(x_scaled)
        k_values = evaluate_kmeans_range(scores, 2, 8)
        best_row = k_values.loc[k_values["silhouette"].idxmax()]
        rows.append(
            {
                "scenario": subset_name,
                "numero_feature": len(features),
                "varianza_prime_2_PC": pca.explained_variance_ratio_.sum(),
                "k_migliore_silhouette": int(best_row["k"]),
                "silhouette_migliore": best_row["silhouette"],
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(TABLES_DIR / "confronto_subset_feature.csv", index=False)
    return comparison


def assign_readable_cluster_names(assignments: pd.DataFrame) -> dict[int, str]:
    """
    Ordina i cluster K-Means dal più fragile al più avanzato usando un indice semplice:
    mortalità e fertilità alte aumentano fragilità, reddito/PIL/aspettativa di vita la riducono.
    """

    profile = assignments.groupby("cluster_kmeans")[
        ["child_mort", "income", "life_expec", "total_fer", "gdpp"]
    ].mean()
    fragility = (
        profile["child_mort"].rank(ascending=True)
        + profile["total_fer"].rank(ascending=True)
        + profile["income"].rank(ascending=False)
        + profile["gdpp"].rank(ascending=False)
        + profile["life_expec"].rank(ascending=False)
    )
    ordered_clusters = fragility.sort_values(ascending=False).index.tolist()
    labels = {}
    for position, cluster_id in enumerate(ordered_clusters):
        labels[int(cluster_id)] = CLUSTER_NAMES.get(position, f"profilo {position + 1}")
    return labels


def run_analysis() -> AnalysisResult:
    """Esegue tutte le elaborazioni e salva tabelle/grafici intermedi."""

    ensure_directories()
    data = load_dataset()
    save_basic_tables(data)

    images = {
        "distribuzioni": plot_feature_distributions(data),
        "correlazione": plot_correlation_matrix(data),
    }

    # PCA sul dataset completo: utile per scree plot e confronto con il subset.
    x_full_scaled, _ = standardize(data, FULL_FEATURES)
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(x_full_scaled)
    images["scree_full"] = plot_scree(pca_full, "tutte_feature")

    subset_comparison = compare_feature_subsets(data)

    # Pipeline finale: subset interpretabile + StandardScaler + PCA a due componenti.
    x_selected_scaled, _ = standardize(data, SELECTED_FEATURES)
    pca_selected_full = PCA(random_state=RANDOM_STATE)
    pca_selected_full.fit(x_selected_scaled)
    images["scree_selected"] = plot_scree(pca_selected_full, "feature_selezionate")

    pca_selected = PCA(n_components=2, random_state=RANDOM_STATE)
    selected_scores = pca_selected.fit_transform(x_selected_scaled)
    pca_variance = pd.DataFrame(
        {
            "PC": [f"PC{i}" for i in range(1, len(pca_selected_full.explained_variance_ratio_) + 1)],
            "varianza_spiegata": pca_selected_full.explained_variance_ratio_,
            "varianza_cumulata": np.cumsum(pca_selected_full.explained_variance_ratio_),
        }
    )
    pca_variance.to_csv(TABLES_DIR / "pca_varianza_feature_selezionate.csv", index=False)

    images["biplot"] = plot_biplot(
        selected_scores,
        pca_selected.components_.T,
        pca_selected.explained_variance_ratio_,
        SELECTED_FEATURES,
    )

    k_selection = evaluate_kmeans_range(selected_scores, 2, 10)
    k_selection.to_csv(TABLES_DIR / "scelta_k_elbow_silhouette.csv", index=False)
    elbow_k = compute_elbow_k(k_selection)
    silhouette_k = int(k_selection.loc[k_selection["silhouette"].idxmax(), "k"])

    # La scelta finale privilegia la silhouette quando il subset PCA evidenzia cluster più separati.
    final_k = silhouette_k
    images["elbow_silhouette"] = plot_k_selection(k_selection, final_k)

    kmeans = KMeans(n_clusters=final_k, random_state=RANDOM_STATE, n_init=20)
    kmeans_labels = kmeans.fit_predict(selected_scores)
    kmeans_centers = kmeans.cluster_centers_
    silhouette_final = silhouette_score(selected_scores, kmeans_labels)
    images["cluster_kmeans"] = plot_pca_clusters(
        selected_scores,
        kmeans_labels,
        kmeans_centers,
        pca_selected.explained_variance_ratio_,
        "Cluster K-Means sulle prime due componenti principali",
        "cluster_kmeans_pca_centroidi.png",
    )

    # Clustering K-Neighbors: SpectralClustering costruisce una matrice di affinità dai vicini.
    n_neighbors = min(10, len(data) - 1)
    knn_model = SpectralClustering(
        n_clusters=final_k,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=RANDOM_STATE,
    )
    knn_labels = knn_model.fit_predict(selected_scores)
    silhouette_knn = silhouette_score(selected_scores, knn_labels)
    ari_kmeans_knn = adjusted_rand_score(kmeans_labels, knn_labels)
    images["cluster_knn"] = plot_pca_clusters(
        selected_scores,
        knn_labels,
        None,
        pca_selected.explained_variance_ratio_,
        "Cluster basati su grafo K-Neighbors nel piano PCA",
        "cluster_kneighbors_pca.png",
    )

    # Il grafo dei vicini è salvato in forma compatta: numero di collegamenti per osservazione.
    knn_graph = kneighbors_graph(selected_scores, n_neighbors=n_neighbors, include_self=False)
    pd.DataFrame({"country": data["country"], "numero_vicini": np.asarray(knn_graph.sum(axis=1)).ravel()}).to_csv(
        TABLES_DIR / "grafo_kneighbors_sintesi.csv",
        index=False,
    )

    assignments = data.copy()
    assignments["PC1"] = selected_scores[:, 0]
    assignments["PC2"] = selected_scores[:, 1]
    assignments["cluster_kmeans"] = kmeans_labels
    assignments["cluster_kneighbors"] = knn_labels
    assignments["distanza_centroide"] = np.linalg.norm(selected_scores - kmeans_centers[kmeans_labels], axis=1)
    readable_names = assign_readable_cluster_names(assignments)
    assignments["profilo_cluster"] = assignments["cluster_kmeans"].map(readable_names)
    assignments.to_csv(TABLES_DIR / "assegnazione_cluster_stati.csv", index=False)
    images["top_countries"] = plot_top_countries(assignments)

    scaled_selected_frame = pd.DataFrame(x_selected_scaled, columns=SELECTED_FEATURES)
    scaled_selected_frame["cluster_kmeans"] = kmeans_labels
    cluster_profile = scaled_selected_frame.groupby("cluster_kmeans").mean().round(3)
    cluster_profile["dimensione"] = scaled_selected_frame.groupby("cluster_kmeans").size()
    cluster_profile["nome_profilo"] = cluster_profile.index.map(readable_names)
    cluster_profile.to_csv(TABLES_DIR / "profilo_cluster_kmeans_standardizzato.csv")
    images["profilo_cluster"] = plot_cluster_profile(cluster_profile)

    raw_profile = assignments.groupby("profilo_cluster")[SELECTED_FEATURES].mean().round(2)
    raw_profile["numero_stati"] = assignments.groupby("profilo_cluster").size()
    raw_profile.to_csv(TABLES_DIR / "profilo_cluster_valori_originali.csv")

    return AnalysisResult(
        data=data,
        selected_features=SELECTED_FEATURES,
        pca_selected=pca_selected,
        final_k=final_k,
        elbow_k=elbow_k,
        silhouette_k=silhouette_k,
        silhouette_final=silhouette_final,
        silhouette_knn=silhouette_knn,
        ari_kmeans_knn=ari_kmeans_knn,
        cluster_profile=cluster_profile,
        pca_variance=pca_variance,
        k_selection=k_selection,
        subset_comparison=subset_comparison,
        images=images,
    )


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    """Rende più leggibili i paragrafi lunghi nel PDF."""

    cleaned = " ".join(textwrap.dedent(text).strip().split())
    return Paragraph(cleaned, style)


def image_block(path: Path, width_cm: float = 15.5, height_cm: float | None = None) -> Image:
    """Inserisce un'immagine mantenendo dimensioni omogenee nella relazione."""

    img = Image(str(path), width=width_cm * cm, height=height_cm * cm if height_cm else None)
    if height_cm is None:
        img._restrictSize(width_cm * cm, 10.5 * cm)
    return img


def dataframe_table(df: pd.DataFrame, max_rows: int = 8) -> Table:
    """Converte un DataFrame in tabella ReportLab compatta."""

    shown = df.head(max_rows).copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: f"{value:.3f}")
    values = [shown.columns.tolist()] + shown.astype(str).values.tolist()
    table = Table(values, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f3e46")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0b0b0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f7f8")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def footer(canvas, doc) -> None:
    """Aggiunge numero pagina e titolo breve alla relazione."""

    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1.6 * cm, 1.0 * cm, "Progetto Fondamenti di Scienza dei Dati")
    canvas.drawRightString(19.4 * cm, 1.0 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def build_report(result: AnalysisResult) -> Path:
    """Genera la relazione PDF con struttura simile all'esempio fornito."""

    report_path = REPORT_DIR / "relazione_progetto_country_clustering.pdf"
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ProjectTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=20,
            leading=24,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustified",
            parent=styles["BodyText"],
            alignment=TA_JUSTIFY,
            fontSize=10.5,
            leading=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#444444"),
            spaceAfter=6,
        )
    )

    story = []
    body = styles["BodyJustified"]
    heading = styles["Heading1"]
    subheading = styles["Heading2"]

    story.append(Paragraph("Relazione sul progetto: Profilazione degli stati con PCA e clustering", styles["ProjectTitle"]))
    story.append(Paragraph("Dataset: Country-data.csv", styles["Heading3"]))
    story.append(
        paragraph(
            """
            Il progetto analizza 167 stati descritti da indicatori socio-economici, sanitari e
            demografici. L'obiettivo è individuare gruppi di stati simili tramite tecniche di
            apprendimento non supervisionato: standardizzazione, analisi delle componenti
            principali, K-Means e clustering basato su K-Neighbors.
            """,
            body,
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        dataframe_table(
            pd.DataFrame(
                {
                    "voce": ["osservazioni", "feature numeriche", "valori mancanti", "k finale"],
                    "valore": [result.data.shape[0], len(FULL_FEATURES), int(result.data.isna().sum().sum()), result.final_k],
                }
            )
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Introduzione e obiettivo", heading))
    story.append(
        paragraph(
            """
            Il clustering ricerca gruppi di oggetti tali che gli elementi dello stesso gruppo siano
            più simili tra loro e più differenti dagli elementi degli altri gruppi. Nel linguaggio
            usato a lezione, la procedura è non supervisionata: non sono disponibili etichette di
            classe, quindi la validazione è interna e si basa su coesione e separazione dei cluster.
            """,
            body,
        )
    )
    story.append(
        paragraph(
            """
            La pipeline adottata segue le fasi richieste dalla consegna: caricamento dei dati,
            controllo delle feature, StandardScaler, analisi delle correlazioni, PCA, scelta del
            numero di gruppi con Elbow e Silhouette Score, K-Means e confronto con una tecnica
            basata sul grafo dei vicini più prossimi.
            """,
            body,
        )
    )
    story.append(Paragraph("Feature del dataset", subheading))
    feature_rows = pd.DataFrame(
        [{"feature": key, "descrizione": value} for key, value in FEATURE_DESCRIPTIONS.items()]
    )
    story.append(dataframe_table(feature_rows, max_rows=10))
    story.append(PageBreak())

    story.append(Paragraph("Preprocessing dei dati", heading))
    story.append(
        paragraph(
            """
            Le variabili hanno scale molto diverse: ad esempio income e gdpp sono grandezze
            economiche con valori nell'ordine delle migliaia, mentre health, exports e imports
            sono percentuali. Per evitare che le variabili con scala più ampia dominino le distanze
            euclidee, è stato applicato StandardScaler. Ogni feature viene trasformata sottraendo
            la media e dividendo per la deviazione standard, ottenendo media circa zero e
            deviazione standard unitaria.
            """,
            body,
        )
    )
    story.append(Paragraph("Statistiche descrittive principali", subheading))
    stats = result.data[FULL_FEATURES].describe().T.reset_index().rename(columns={"index": "feature"})
    story.append(dataframe_table(stats[["feature", "mean", "std", "min", "50%", "max"]], max_rows=9))
    story.append(PageBreak())

    story.append(Paragraph("Analisi esplorativa", heading))
    story.append(
        paragraph(
            """
            La prima analisi grafica riguarda le distribuzioni delle variabili. Alcune feature
            economiche risultano molto asimmetriche: pochi paesi presentano reddito e PIL pro
            capite molto alti, mentre molti stati si concentrano su valori bassi o intermedi.
            """,
            body,
        )
    )
    story.append(image_block(result.images["distribuzioni"]))
    story.append(PageBreak())

    story.append(Paragraph("Correlazioni nello spazio delle feature", heading))
    story.append(
        paragraph(
            """
            La matrice di correlazione mostra relazioni coerenti con l'interpretazione del
            problema: child_mort e total_fer tendono a muoversi insieme, mentre sono in relazione
            negativa con life_expec. Le variabili income e gdpp sono invece fortemente associate
            e rappresentano una dimensione economica comune.
            """,
            body,
        )
    )
    story.append(image_block(result.images["correlazione"], width_cm=14.5))
    story.append(PageBreak())

    story.append(Paragraph("PCA e selezione delle feature", heading))
    story.append(
        paragraph(
            """
            La PCA trasforma il dataset in un nuovo sistema di assi ortogonali. Le componenti
            principali sono ordinate per varianza spiegata: PC1 è la direzione lungo cui i dati
            variano maggiormente, PC2 la seconda direzione ortogonale. La consegna richiede anche
            di valutare un sottoinsieme di feature; per questo è stato confrontato il dataset
            completo con un subset socio-economico/sanitario.
            """,
            body,
        )
    )
    story.append(dataframe_table(result.subset_comparison, max_rows=5))
    story.append(
        paragraph(
            f"""
            Il subset scelto è formato da {", ".join(result.selected_features)}. Le prime due
            componenti spiegano il {result.pca_selected.explained_variance_ratio_.sum() * 100:.1f}%
            della varianza, quindi il piano PCA conserva molta informazione utile per visualizzare
            e clusterizzare gli stati.
            """,
            body,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Scree Plot e Biplot", heading))
    story.append(image_block(result.images["scree_selected"], width_cm=14.5))
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        paragraph(
            """
            Nel biplot gli score rappresentano gli stati nel nuovo sistema di coordinate, mentre
            le frecce rappresentano i loading delle variabili originali. Vettori vicini indicano
            correlazione positiva, vettori opposti indicano correlazione negativa.
            """,
            body,
        )
    )
    story.append(image_block(result.images["biplot"], width_cm=14.5))
    story.append(PageBreak())

    story.append(Paragraph("Scelta del numero di cluster", heading))
    story.append(
        paragraph(
            f"""
            Il metodo Elbow osserva la riduzione del WCSS/inertia al crescere di k, mentre la
            silhouette media misura quanto ogni osservazione è coerente con il proprio cluster
            rispetto agli altri. Nel nostro caso il gomito automatico indica k={result.elbow_k},
            mentre la silhouette è massima per k={result.silhouette_k}. La scelta finale è
            k={result.final_k}, perché fornisce la separazione media migliore nel piano delle
            prime due componenti principali.
            """,
            body,
        )
    )
    story.append(image_block(result.images["elbow_silhouette"], width_cm=15.5))
    story.append(dataframe_table(result.k_selection, max_rows=9))
    story.append(PageBreak())

    story.append(Paragraph("Clustering K-Means", heading))
    story.append(
        paragraph(
            f"""
            K-Means è un metodo partizionale center-based: ogni cluster è rappresentato da un
            centroide e ogni punto viene assegnato al centroide più vicino. Applicato alle prime
            due componenti principali del subset selezionato, ottiene silhouette media pari a
            {result.silhouette_final:.3f}. Nel grafico sono mostrati anche i centri di massa dei
            cluster, come richiesto dalla consegna.
            """,
            body,
        )
    )
    story.append(image_block(result.images["cluster_kmeans"], width_cm=15.5))
    story.append(PageBreak())

    story.append(Paragraph("Profilazione dei cluster", heading))
    story.append(
        paragraph(
            """
            Il profilo medio standardizzato permette di interpretare i gruppi. Valori positivi
            indicano feature sopra la media del dataset, valori negativi indicano feature sotto la
            media. In generale emergono un gruppo fragile, con mortalità infantile e fertilità
            alte, un gruppo intermedio e un gruppo più avanzato con income, gdpp e life_expec più
            elevati.
            """,
            body,
        )
    )
    story.append(image_block(result.images["profilo_cluster"], width_cm=15.2))
    profile_for_pdf = result.cluster_profile.reset_index().rename(columns={"cluster_kmeans": "cluster"})
    story.append(dataframe_table(profile_for_pdf, max_rows=6))
    story.append(PageBreak())

    story.append(Paragraph("Clustering K-Neighbors", heading))
    story.append(
        paragraph(
            f"""
            Poiché il dataset non contiene una variabile target, K-Neighbors è stato usato in
            forma non supervisionata: si costruisce un grafo di affinità collegando ogni stato ai
            suoi vicini più prossimi nel piano PCA e poi si applica SpectralClustering su tale
            grafo. La silhouette ottenuta è {result.silhouette_knn:.3f}; l'Adjusted Rand Index tra
            K-Means e K-Neighbors è {result.ari_kmeans_knn:.3f}, quindi i due metodi sono
            confrontabili ma non identici.
            """,
            body,
        )
    )
    story.append(image_block(result.images["cluster_knn"], width_cm=15.2))
    story.append(PageBreak())

    story.append(Paragraph("Esempi di stati e conclusioni", heading))
    story.append(image_block(result.images["top_countries"], width_cm=15.5))
    story.append(
        paragraph(
            """
            L'analisi conferma che la struttura principale del dataset è guidata da una
            contrapposizione tra indicatori di fragilità demografica/sanitaria e indicatori di
            sviluppo economico. La standardizzazione è necessaria per rendere confrontabili le
            variabili; la PCA consente di proiettare gli stati in uno spazio a bassa dimensione;
            Elbow e Silhouette supportano la scelta del numero di gruppi; K-Means fornisce una
            partizione compatta e interpretabile, mentre K-Neighbors evidenzia una struttura
            locale basata sulle relazioni di prossimità.
            """,
            body,
        )
    )
    story.append(Paragraph("Librerie utilizzate", subheading))
    story.append(
        paragraph(
            """
            pandas per il caricamento e la gestione dei dati, numpy per il calcolo numerico,
            matplotlib per le visualizzazioni, scikit-learn per StandardScaler, PCA, KMeans,
            SpectralClustering, kneighbors_graph e metriche di validazione, reportlab per la
            generazione della relazione PDF.
            """,
            body,
        )
    )

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return report_path


def create_source_zip() -> Path:
    """Crea lo zip richiesto con i soli codici sorgenti e file di supporto."""

    zip_path = SOURCE_DIR / "codice_sorgente_country_clustering.zip"
    included_files = [
        SOURCE_DIR / "main.py",
        SOURCE_DIR / "README.md",
        SOURCE_DIR / "requirements.txt",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in included_files:
            archive.write(file_path, arcname=file_path.name)
    return zip_path


def write_execution_summary(result: AnalysisResult, report_path: Path, zip_path: Path) -> None:
    """Salva un riepilogo testuale dei risultati principali."""

    summary_path = OUTPUT_DIR / "README_output.md"
    summary = f"""# Output progetto Country clustering

## Risultati principali

- Osservazioni analizzate: {result.data.shape[0]}
- Feature numeriche iniziali: {len(FULL_FEATURES)}
- Feature selezionate per la clusterizzazione finale: {", ".join(result.selected_features)}
- Varianza spiegata dalle prime due PC del subset: {result.pca_selected.explained_variance_ratio_.sum() * 100:.2f}%
- Numero di cluster scelto: {result.final_k}
- Silhouette K-Means: {result.silhouette_final:.3f}
- Silhouette K-Neighbors: {result.silhouette_knn:.3f}
- Adjusted Rand Index K-Means vs K-Neighbors: {result.ari_kmeans_knn:.3f}

## File principali

- Relazione PDF: `{report_path.relative_to(OUTPUT_DIR)}`
- Zip sorgenti: `{zip_path.relative_to(OUTPUT_DIR)}`
- Grafici: `grafici/`
- Tabelle e assegnazioni: `tabelle/`
- Dataset copiato per riproducibilita: `dati/Country-data.csv`

## Riproduzione

```bash
python3 output_progetto/codice_sorgente/main.py
```
"""
    summary_path.write_text(summary, encoding="utf-8")


def main() -> None:
    result = run_analysis()
    report_path = build_report(result)
    zip_path = create_source_zip()
    write_execution_summary(result, report_path, zip_path)
    print(f"Relazione creata: {report_path}")
    print(f"Zip sorgenti creato: {zip_path}")
    print(f"k finale: {result.final_k}")
    print(f"Silhouette K-Means: {result.silhouette_final:.3f}")
    print(f"Silhouette K-Neighbors: {result.silhouette_knn:.3f}")


if __name__ == "__main__":
    main()
