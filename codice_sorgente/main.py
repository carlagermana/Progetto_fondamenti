"""
Progetto di data analysis sul dataset Country-data.csv.

Carla Germana' - Fabio Raineri - 

il codice contiene l'intera pipeline :

- caricamento e controllo del dataset;
- analisi descrittiva e matrice di correlazione;
- standardizzazione con StandardScaler;
- PCA, scree plot e biplot;
- scelta del numero di cluster con Elbow e Silhouette Score;
- clustering K-Means sul piano delle prime due componenti principali;
- clustering basato su grafo K-Neighbors tramite SpectralClustering;
- esportazione di grafici, tabelle e assegnazioni.

Esecuzione dalla root del progetto:
    python3 codice_sorgente/main.py
"""
#---------------------------------------------

from pathlib import Path # lavora con file e cartelle

# Path(__file__) indica il percorso di questo script.
# resolve() lo trasforma in un percorso completo.
# parents[1] risale alla cartella principale del progetto.
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR
#Servirà per salvare i file di output dello script in una cartella fissa, senza dover scrivere ogni volta il percorso a mano

# Qui salviamo in variabili i percorsi delle cartelle usate dal programma,
# cosi nel codice possiamo usare nomi semplici come PLOTS_DIR e TABLES_DIR.
SOURCE_DIR = OUTPUT_DIR / "codice_sorgente"
PLOTS_DIR = OUTPUT_DIR / "grafici"
TABLES_DIR = OUTPUT_DIR / "tabelle"
REPORT_PDF_PATH = OUTPUT_DIR / "relazione" / "relazione_progetto_country_clustering.pdf"
DATA_DIR = OUTPUT_DIR / "dati"
ROOT_DATASET_PATH = ROOT_DIR / "Country-data.csv"
OUTPUT_DATASET_PATH = DATA_DIR / "Country-data.csv"

#---------------------------------------------- IMPORTAZIONE DELLE LIBRERIE
import matplotlib

matplotlib.use("Agg") # significa: "non aprire nessuna finestra, salva il grafico direttamente come immagine"

import matplotlib.pyplot as plt #pyplot è la parte di matplotlib che si usa per creare grafici

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap #colori personalizzati per i grafici
from sklearn.cluster import KMeans, SpectralClustering #raggruppano i dati in gruppi (cluster)
from sklearn.decomposition import PCA #ridurre la dimensionalità dei dati
from sklearn.metrics import adjusted_rand_score, silhouette_score #misurano quanto sono buoni i cluster
from sklearn.neighbors import kneighbors_graph #trova i vicini più simili
from sklearn.preprocessing import StandardScaler #per la normalizzazione dei dati

#---------------------------------------------- 
RANDOM_STATE = 42
# Alcuni algoritmi, come KMeans, usano scelte casuali all'inizio.
# Questo valore fisso serve a ottenere gli stessi risultati a ogni esecuzione.

#---------------------------------------------- FEATURES DEL DATASET
# tutte le caratteristiche del dataset - Sono i dati di ogni paese:
FULL_FEATURES = [ 
    "child_mort", #Mortalità infantile
    "exports", #Esportazioni
    "health", #Spesa sanitaria
    "imports", #Importazioni
    "income", #Reddito medio
    "inflation", #inflazione
    "life_expec", #Aspettativa di vita
    "total_fer", #Tasso di fertilità
    "gdpp", #PIL pro capite
]

SELECTED_FEATURES = ["child_mort", "income", "life_expec", "total_fer", "gdpp"]
#contiene le variabili scelte per fare il clustering 
"""
 Sono state scelte perché descrivono bene il livello socio-economico e sanitario dei Paesi:

  - child_mort: mortalità infantile, alta nei Paesi più fragili
  - income: reddito medio, indica il livello economico
  - life_expec: aspettativa di vita, misura il benessere sanitario
  - total_fer: fertilità totale, spesso più alta nei Paesi meno sviluppati
  - gdpp: PIL pro capite, altra misura economica importante

  In pratica sono variabili molto interpretabili: aiutano a distinguere Paesi più fragili, intermedi e
  avanzati.

"""
#---------------------------------------------- CLUSTER_NAMES
#  CLUSTER_NAMES serve invece a dare un nome leggibile ai cluster:
CLUSTER_NAMES = {
    0: "profilo fragile",
    1: "profilo intermedio",
    2: "profilo avanzato",
    3: "profilo ad alto reddito anomalo",
}
#----------------------------------------------   CREAZIONE DELLE CARTELLE DI OUTPUT 

def ensure_directories() -> None: #-> None indica che la funzione non restituisce nessun valore.
    """Crea le cartelle di output"""

    for directory in [SOURCE_DIR, PLOTS_DIR, TABLES_DIR, DATA_DIR]:  #lista di "variabili" (che rappresentano i percorsi delle cartelle)
        directory.mkdir(parents=True, exist_ok=True)
    """
    Per ognuna delle 4 cartelle:
        Opzione mkdir() crea la cartella
        parents= True crea anche le cartelle intermedie se mancano
        exist_ok=Truenon dà errore se la cartella esiste già
    """

#---------------------------------------------- CARICA IL CSV e VERIFICA CHE LE COLONNE ATTESE SIANO PRESENTI
def load_dataset() -> pd.DataFrame: #restituirà come output (return) un oggetto di tipo pd.DataFrame

    #Cerca il file prima nella cartella principale, poi in quella di output.
    DATASET_PATH = ROOT_DATASET_PATH if ROOT_DATASET_PATH.exists() else OUTPUT_DATASET_PATH
    
    #Controlla che esista
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset non trovato: {DATASET_PATH}")  # blocca tutto e avvisa
        #Se non lo trova → errore chiaro 

    #Carica il CSV -  Legge il file e lo mette in una tabella (DataFrame)
    data = pd.read_csv(DATASET_PATH)

    #Crea la lista delle colonne che ti aspetti
    expected_columns = ["country"] + FULL_FEATURES
    # → ["country", "child_mort", "exports", "health", ...]

    #Trova le colonne mancanti
    missing_columns = sorted(set(expected_columns) - set(data.columns))
    """
    missing_columns = sorted(set(expected_columns) - set(data.columns))
        È una sottrazione tra insiemi:
        colonne attese:  {"country", "child_mort", "income", "gdpp", ...}
        colonne nel CSV: {"country", "child_mort", "income", ...}
                  ─────────────────────────────────────────
        differenza →     {"gdpp"}  ← manca questa!
    """
    
    #Se manca qualcosa → blocca tutto
    if missing_columns:
        raise ValueError(f"Colonne mancanti nel dataset: {missing_columns}")

    # Copia nella cartella output
    data.to_csv(DATA_DIR / "Country-data.csv", index=False)
    """
    Questo comando prende la tabella di dati (il DataFrame data)
        e la salva fisicamente sul computer come un file CSV.

    DATA_DIR / "Country-data.csv": Indica dove salvare il file e come chiamarlo. 
    Sfrutta la libreria pathlib 
        (il simbolo / unisce il percorso della cartella DATA_DIR al nome del file "Country-data.csv").
    index=False: È un parametro di Pandas. Quando carichiamo un file, 
        Pandas assegna automaticamente un numero di riga a ogni riga (0, 1, 2, 3...).
        Dicendo index=False, eviti che questi numeri di servizio vengano salvati nel 
            file CSV come una colonna aggiuntiva innaturale
    
    """
    return data #restituisce la tabella

#---------------------------------------------- NORMALIZZAZIONE DEI DATI - in modo che tutte le variabili siano sulla stessa scala 
def standardize(data: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, StandardScaler]:
    """Applica StandardScaler: ogni variabile viene centrata e scalata."""

    scaler = StandardScaler() # crea lo scaler - Lo scaler è uno strumento che ricorda come abbiamo normalizzato i dati.
    scaled_data = scaler.fit_transform(data[features]) #normalizza (solo le features - colonne scelte)
    return scaled_data, scaler # restituisce dati normalizzati + scaler
        #lo scaler perché servirà dopo per fare il processo al contrario e ritrovare i valori originali

    """
    Il problema senza normalizzazione:
    PIL (gdpp):        45000  ← numeri enormi
    Mortalità infantile:   8  ← numeri piccoli
    L'algoritmo darebbe troppo peso al PIL solo perché ha numeri più grandi, anche se non è più importante!!!
    
    La soluzione — StandardScaler:
        Trasforma ogni colonna così:

             valore originale - media
            ─────────────────────────
                deviazione standard

    (I parametri della funzione)
    La funzione ha bisogno di due strumenti per poter lavorare:

        data: pd.DataFrame
            "Passami una tabella di dati di Pandas e all'interno della funzione la chiamerò data".
        features: list[str]
            "Passami una lista di stringhe (testi) che chiamerò features". Questa lista conterrà i nomi delle colonne della tabella che vuoi effettivamente standardizzare (es. ["income", "inflation", "gdpp"]).

    L'Output (->)
        -> tuple[... ]
    restituisce una tupla, ovvero una coppia (o un gruppo) di oggetti diversi nello stesso momento.
    """

#---------------------------------------------- SALVA 4 TABELLE CSV CON LE STATISTICHE DI BASE DEL DATASET
def save_basic_tables(data: pd.DataFrame) -> None:
    """Esporta statistiche descrittive, valori mancanti e matrice di correlazione."""

    overview = pd.DataFrame(
        {   #nome col 1
            "metrica": ["osservazioni", "colonne", "valori_mancanti_totali"], 
            #nome col 2
            "valore": [data.shape[0], data.shape[1], int(data.isna().sum().sum())],
                                                    #conta tutti i valori mancanti nel dataset.

        }
    )

    overview.to_csv(TABLES_DIR / "dataset_overview.csv", index=False)
  #Questo comando prende la tabella di dati (il DataFrame data)
        #e la salva fisicamente sul computer come un file CSV.

    #Conta quanti valori mancano per ogni colonna:
    data.isna().sum().rename("missing_values").to_csv(TABLES_DIR / "missing_values.csv")

    """
    Statistiche descrittive → statistiche_descrittive.csv
        Per ogni colonna mostra media, min, max, ecc.:
    """
    data[FULL_FEATURES].describe().T.round(3).to_csv(TABLES_DIR / "statistiche_descrittive.csv")
    

    """
    Matrice di correlazione → matrice_correlazione.csv
        data[FULL_FEATURES].corr()
    Mostra quanto due variabili sono collegate ("la loro relazione lineare")

                child_mort income  gdpp
    child_mort       1.0    -0.8   -0.7  ← alta mortalità = basso reddito
    income          -0.8     1.0    0.9


    """
    data[FULL_FEATURES].corr().round(3).to_csv(TABLES_DIR / "matrice_correlazione.csv")

#----------------------------------------------


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

#*************************$$******************************************


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


def run_analysis():
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

    compare_feature_subsets(data)

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

    return {
        "data": data,
        "selected_features": SELECTED_FEATURES,
        "pca_selected": pca_selected,
        "final_k": final_k,
        "elbow_k": elbow_k,
        "silhouette_k": silhouette_k,
        "silhouette_final": silhouette_final,
        "silhouette_knn": silhouette_knn,
        "ari_kmeans_knn": ari_kmeans_knn,
    }


def main() -> None:
    result = run_analysis()

    print(f"k finale: {result['final_k']}")
    print(f"Silhouette K-Means: {result['silhouette_final']:.3f}")
    print(f"Silhouette K-Neighbors: {result['silhouette_knn']:.3f}")


if __name__ == "__main__":
    main()
