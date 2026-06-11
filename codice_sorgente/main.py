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

"""
Profilo fragile
  Paesi con condizioni socio-economiche e sanitarie più difficili. In genere hanno:

  - mortalità infantile più alta
  - reddito più basso
  - PIL pro capite più basso
  - aspettativa di vita più bassa
  - fertilità più alta

  Profilo intermedio
  Paesi in una situazione di mezzo. Non sono tra i più fragili, ma nemmeno tra i più avanzati. Hanno valori medi
  o misti.

  Profilo avanzato
  Paesi con condizioni migliori. In genere hanno:

  - mortalità infantile bassa
  - reddito più alto
  - PIL pro capite più alto
  - aspettativa di vita alta
  - fertilità più bassa

  Profilo ad alto reddito anomalo
  Gruppo di Paesi con reddito o PIL molto alto, ma con caratteristiche che possono renderli un po’ diversi dagli
  altri Paesi avanzati. Per esempio possono avere valori economici molto elevati rispetto al resto del dataset.

"""

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

#---------------------------------------------- METODO DEL GOMITO PER TROVARE NUM. OTTIMALE DI CLUSTER
def compute_elbow_k(k_selection: pd.DataFrame) -> int: #Trova il numero ottimale di cluster (gruppi in cui dividere i paesi) usando il metodo del gomito.

    """
    Quando aumentiamo i cluster, l'errore (inertia) scende. 
    Ma ad un certo punto scende poco, lì c'è il gomito, il numero ottimale di cluster.

    """
    points = k_selection[["k", "inertia"]].to_numpy(dtype=float) # Prende i punti del grafico

    #Traccia una retta dal primo all'ultimo punto
    first_point, last_point = points[0], points[-1]
    line = last_point - first_point

    shifted_points = points - first_point
    """
    *                    ← primo punto
    \
      \
        \
          \           ← retta immaginaria
            \
             *       ← ultimo punto
    """

    #Misura la distanza di ogni punto dalla retta
    cross_2d = line[0] * shifted_points[:, 1] - line[1] * shifted_points[:, 0]
    distances = np.abs(cross_2d / np.linalg.norm(line))

    """
    *                    
  \    *  ← distanza massima = GOMITO !!!
    \
      \        *
        \           
          \
            *
    
    """
    return int(points[int(np.argmax(distances)), 0])    # Restituisce il k (numero ottimale di cluster) con distanza massima

#---------------------------------------------- SILHOUETTE SCORE PER DIVERSI VALORI K
                        #dati normalizzati (array numpy) k minimo e massimo da provare (default 2 -10)
def evaluate_kmeans_range(x_values: np.ndarray, k_min: int = 2, k_max: int = 10) -> pd.DataFrame: # restituisce una tabella
    """Calcola inertia/WCSS(somma degli errori quadratici)  e Silhouette Score per diversi valori di k."""
    #proviamo tutti i k da 2 a 10, così possiamo scegliere il migliore
    rows = []
    for k in range(k_min, k_max + 1): #Prova ogni k da 2 a 10
        #Per ogni k crea e addestra KMeans

        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(x_values) # labels → [0, 2, 1, 3, 0, ...]  ← gruppo assegnato ad ogni paese

        #Salviamo i risultati in una tabella
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(x_values, labels),
            }
        )
        # Esempio di tabella finale:
        # k | inertia | silhouette
        # 2 | 9000    | 0.45
        # 3 | 6000    | 0.61
        # 4 | 4000    | 0.52
    return pd.DataFrame(rows) #ritorna tab - è il tipo

#---------------------------------------------- CREAZIONE DI ISTOGRAMMI DELLE FEATURE NUMERICHE 
def plot_feature_distributions(data: pd.DataFrame) -> Path: #restituisce il percorso
    #9 istogrammi, uno per ogni variabile del dataset
    output = PLOTS_DIR / "distribuzioni_feature.png" # → dove salvare l'immagine cioè nella cartella PLOTS_DIR

    fig, axes = plt.subplots(3, 3, figsize=(13, 10)) # → crea una griglia 3x3 di grafici
    axes = axes.ravel() # → trasforma la griglia in una lista semplice [ax1, ax2, ax3, ...]

    # → per ogni variabile disegna il suo istogramma
    for ax, feature in zip(axes, FULL_FEATURES):
        ax.hist(data[feature], bins=22, color="#4c78a8", edgecolor="white")
        ax.set_title(feature)
        ax.set_ylabel("frequenza")

    fig.suptitle("Distribuzione delle feature numeriche", fontsize=16, fontweight="bold") # → titolo generale sopra tutti i grafici
    fig.tight_layout(rect=(0, 0, 1, 0.96)) # per sistemare gli spazi tra i grafici

    # salva l'immagine, libera memoria, restituisce il percorso
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output

    """
    Gli Assi (X e Y)
        Asse Orizzontale (X): Rappresenta il valore della feature. Nel caso di child_mort, va da 0 a oltre 200.
        Asse Verticale (Y - "frequenza"): Rappresenta il numero di volte (il conteggio) in cui quel valore compare nel tuo set di dati. Più la barra è alta, più sono i Paesi o i casi che hanno quel valore.

    Le Barre (I "Bin")
        I dati vengono raggruppati in "colonne" (chiamate bin).
        per il primo grafico la prima barra blu altissima a sinistra in child_mort: si trova tra lo 0 e il 20 circa, ed è alta quasi 60.
        Significa che nel  dataset ci sono quasi 60 Paesi che hanno un tasso di mortalità infantile molto basso (compreso tra 0 e 20).
    """

#---------------------------------------------- CREAZIONE MATRICE DI CORRELAZIONE TRA LE VARIABILI(FEATURES)(9 X 9 caratteristiche = 18 riquadri )
def plot_correlation_matrix(data: pd.DataFrame) -> Path:

    output = PLOTS_DIR / "matrice_correlazione.png" # → dove salvare l'immagine cioè nella cartella PLOTS_DIR
   
    #Calcola le correlazioni
    corr = data[FULL_FEATURES].corr() 

    fig, ax = plt.subplots(figsize=(10, 8)) #prepara un foglio 10x8 con un grafico vuoto pronto da riempire

    # Disegna la matrice a colori
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    """
     rosso  → correlazione positiva  (+1)
     bianco → nessuna correlazione   (0)
     blu    → correlazione negativa  (-1)
    """
    #Aggiunge i nomi delle variabili sugli assi
    ax.set_xticks(range(len(FULL_FEATURES)), FULL_FEATURES, rotation=45, ha="right") # nomi sull'asse X (ruotati 45°)
    ax.set_yticks(range(len(FULL_FEATURES)), FULL_FEATURES) # nomi sull'asse Y

    #per scrivere il numero dentro ogni cella della matrice
    for i in range(len(FULL_FEATURES)):
        for j in range(len(FULL_FEATURES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    
    ax.set_title("Matrice di correlazione", fontsize=15, fontweight="bold") # → aggiunge il titolo in cima al grafico
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # → aggiunge la barra colori a destra
        # fraction = quanto è larga, pad = quanto è distante dal grafico
    fig.tight_layout() # sistema automaticamente spazi e margini
    fig.savefig(output, dpi=180) # salva l'immagine PNG (dpi=180 = buona qualità)    dots per inch (punti per pollice) 
    plt.close(fig) # libera la memoria, elimina il grafico dalla RAM

    return output       #   restituisce il percorso del file salvato    
                        #   esempio "mio_progetto/grafici/matrice_correlazione.png"

#---------------------------------------------- SCREE PLOT CON VARAINZA SPIEGATA (PCA)
# Riceve un oggetto PCA (già addestrato) e una stringa prefix per distinguere i file.
# Restituisce il Path del file salvato.
def plot_scree(pca: PCA, prefix: str) -> Path:  
    #viene chiamata due volte (genera un grafico diverso per due volte)
    #chimata per pca su tutte le features
    #chiamata per pca su features selezionate scree plot usando solo: child_mort, income, life_expec, total_fer, gdpp

    output = PLOTS_DIR / f"scree_plot_{prefix}.png" # → dove salvare l'immagine cioè nella cartella PLOTS_DIR
    ratios = pca.explained_variance_ratio_     # Frazione di varianza spiegata da ogni componente (array che somma a 1)
    components = np.arange(1, len(ratios) + 1)     # Asse X: numeri interi 1, 2, 3, ... per ogni componente principale(autovalori)
    cumulative = np.cumsum(ratios)     # Somma cumulata: quanta varianza(informazione) totale si cattura aggiungendo componenti

    fig, ax1 = plt.subplots(figsize=(9, 5.5)) #preparazione del foglio per il grafico
    
    # Barre: quanta informazione aggiunge ogni singola componente
    ax1.bar(components, ratios * 100, color="#72b7b2", label="varianza spiegata")
    ax1.set_xlabel("Componenti principali")
    ax1.set_ylabel("Varianza spiegata (%)")
    
    ax1.set_xticks(components) #Imposta manualmente i segni sull'asse X.

    # Crea un secondo asse a destra per la linea cumulata
    ax2 = ax1.twinx() 

    # Linea rossa: informazione totale man mano che si aggiungono componenti
    ax2.plot(components, cumulative * 100, marker="o", color="#e45756", label="cumulata")
    ax2.set_ylabel("Varianza cumulata (%)")
    ax2.set_ylim(0, 105)

    ax1.set_title("Scree Plot della PCA", fontsize=15, fontweight="bold")
    
    fig.tight_layout()     # Aggiusta i margini
    # Salva il file e libera la memoria
    fig.savefig(output, dpi=180)
    plt.close(fig) 
    return output #ritorna il path

#---------------------------------------------- CREA IL BIPLOT!: score degli stati e loading delle variabili originali.

def plot_biplot(
    scores: np.ndarray, #cosa prende come parametro la funwione
    loadings: np.ndarray,
    explained_ratio: np.ndarray,
    feature_names: list[str],
) -> Path:  #il ritorno che ci aspettiamo dalla funzione
    """Crea il biplot: score degli stati e loading delle variabili originali."""

    """
    A cosa serve il biplot
        Il biplot sovrappone due informazioni nello stesso grafico:

        Punti (scores) — dove si posiziona ogni osservazione (es. ogni stato) nello spazio delle componenti principali
        Frecce (loadings) — quanto e in che direzione ogni variabile originale contribuisce alle componenti

    ci permette di vedere insieme quali stati si assomigliano e quali variabili li influenzano.

    """
    output = PLOTS_DIR / "biplot_pca_feature_selezionate.png" # → dove salvare l'immagine cioè nella cartella PLOTS_DIR
    fig, ax = plt.subplots(figsize=(9, 7)) #prepara un foglio con un grafico vuoto pronto da riempire
   
    # Disegna i punti: ogni osservazione posizionata su PC1 (asse X) e PC2 (asse Y)
    ax.scatter(scores[:, 0], scores[:, 1], s=36, alpha=0.72, color="#4c78a8")
    
    scale_x = scores[:, 0].max() - scores[:, 0].min()
    scale_y = scores[:, 1].max() - scores[:, 1].min()
    arrow_scale = min(scale_x, scale_y) * 0.38
    """
    Calcola quanto devono essere lunghe le frecce per stare bene nel grafico.
    Passo 1 — misura quanto spazio occupano i punti su X e su Y:

    scale_x = differenza tra il punto più a destra e quello più a sinistra
    scale_y = differenza tra il punto più in alto e quello più in basso

    Passo 2 — prende il lato più stretto tra i due e usa il 38% di quello come lunghezza massima delle frecce.
    Il risultato è che le frecce si adattano automaticamente alla nuvola di punti — né troppo piccole da non vedersi, né così grandi da uscire dal grafico.
    """

    # Disegna una freccia per ogni variabile originale
    for idx, feature in enumerate(feature_names):
        ax.arrow(
            0,  # Parte dall'origine
            0,
            loadings[idx, 0] * arrow_scale, # Direzione su PC1
            loadings[idx, 1] * arrow_scale, # Direzione su PC2
            color="#d62728",
            alpha=0.85,
            head_width=0.08,
            length_includes_head=True,
        )
        ax.text(       # Per scrivere l'etichetta della variabile, posizionata leggermente oltre la punta della freccia
            loadings[idx, 0] * arrow_scale * 1.12,
            loadings[idx, 1] * arrow_scale * 1.12,
            feature,
            color="#7f1d1d",
            fontsize=10,
        )

    ax.axhline(0, color="grey", linewidth=0.8) # Linee di riferimento orizzontale e verticale sull'origine
    ax.axvline(0, color="grey", linewidth=0.8)

    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}% varianza)")  # Etichette degli assi con la percentuale di varianza spiegata
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}% varianza)")

    ax.set_title("Biplot: score e loading sulle prime due PC", fontsize=15, fontweight="bold")
    fig.tight_layout() # Aggiusta i margini

    fig.savefig(output, dpi=180)     # Salva il file e libera la memoria
    plt.close(fig)

    return output #ritorna il path del file salvato nella cartella grafici

#---------------------------------------------- METODO ELBOW E SILHOUETTE SCORE insieme per decidere il numero di cluster

def plot_k_selection(k_selection: pd.DataFrame, final_k: int) -> Path:

    output = PLOTS_DIR / "elbow_silhouette.png" # //
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))     # Due grafici affiancati

    """
    axes è l'array che contiene i due grafici creati da plt.axes è l'array che contiene i due grafici creati da plt.subplots(1, 2).
        axes[0] — grafico sinistro (Elbow)
        axes[1] — grafico destro (Silhouette)
    Se avessimo fatto plt.subplots(1, 1) avresti ricevuto un singolo oggetto ax

    """

    # --- Grafico sinistro: metodo Elbow ---
    # Linea che mostra come cala l'inerzia all'aumentare di K
    """inerzia 
    è l'errore totale del modello di clustering.
    Misura quanto i punti sono "sbagliati" rispetto al loro centroide. 
    Più i punti sono lontani dal centro del loro cluster, più l'errore è alto. 
    """
    axes[0].plot(k_selection["k"], k_selection["inertia"], marker="o", color="#4c78a8")
    # Linea verticale tratteggiata sul K scelto
    axes[0].axvline(final_k, linestyle="--", color="#e45756", label=f"k scelto = {final_k}")
    axes[0].set_title("Metodo Elbow")
    axes[0].set_xlabel("Numero di cluster k")
    axes[0].set_ylabel("WCSS / inertia")
    axes[0].legend()

    # --- Grafico destro: Silhouette Score ---
    # Linea che mostra la qualità dei cluster al variare di K
    axes[1].plot(k_selection["k"], k_selection["silhouette"], marker="o", color="#59a14f")
    # Linea verticale tratteggiata sul K scelto
    axes[1].axvline(final_k, linestyle="--", color="#e45756", label=f"k scelto = {final_k}")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Numero di cluster k")
    axes[1].set_ylabel("Silhouette media")
    axes[1].legend()

    # Titolo comune a entrambi i grafici
    fig.suptitle("Scelta del numero di cluster", fontsize=15, fontweight="bold")
    fig.tight_layout() # sistema i margini 

    fig.savefig(output, dpi=180)# Salva il file e libera la memoria
    plt.close(fig)
    return output #ritorna il percoso del file 

#---------------------------------------------- GLI STATI i punti NEL PIANO CON PC (DOPO PCA)
def plot_pca_clusters(
    scores: np.ndarray, # Coordinate delle osservazioni nello spazio PCA
    labels: np.ndarray, # Numero del cluster assegnato a ogni osservazione
    centers: np.ndarray,  # Coordinate dei centroidi (solo K-Means)
    explained_ratio: np.ndarray, # Varianza spiegata da PC1 e PC2
    title: str, # Titolo del grafico
    filename: str, # Nome del file di output
) -> Path:
    """
    Disegna i punti (stati) nel piano delle prime due componenti principali, colorati per cluster.
    per K-Means mostra anche i centroidi con una X nera.
    """
    output = PLOTS_DIR / filename #dove salvare il file
    # Palette di 5 colori, uno per ogni cluster possibile
    colors_map = ListedColormap(["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"])
    fig, ax = plt.subplots(figsize=(9, 6.5)) #foglio per il greafico

    # Disegna i punti colorandoli in base al cluster di appartenenza
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=labels, cmap=colors_map, s=50, alpha=0.83)
    
    # Mostra i centroidi solo se forniti (K-Means), con una X nera ben visibile
    if centers is not None:
        ax.scatter(   #scatter è il grafico a punti — ogni punto rappresenta un'osservazione. In matplotlib si crea con ax.scatter(x, y) e posiziona ogni punto in base alle sue coordinate
            centers[:, 0], 
            centers[:, 1],
            marker="X", # Forma a X per distinguerli dai punti normali
            s=260, # Più grandi dei punti normali per risaltare
            c="black",
            edgecolor="white",  # Bordo bianco per staccarsi dallo sfondo
            linewidth=1.2,
            label="centro di massa",
        )
        ax.legend()

    # Linee di riferimento sull'origine
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)

    # Etichette degli assi con la percentuale di varianza spiegata
    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}% varianza)")

    ax.set_title(title, fontsize=15, fontweight="bold") #imposta il titolo del grafico

    legend = ax.legend(*scatter.legend_elements(), title="cluster", loc="best") #creazione della legenda
    ax.add_artist(legend)
    #a legenda è un riquadro che spiega il significato visivo degli elementi rappresentati — cioè associa a ogni colore, una variabile.
    
    fig.tight_layout() #aggiusta i margini automaticamente.
    fig.savefig(output, dpi=180) #salva il file ad alta risoluzione.
    plt.close(fig) #libera la memoria
    return output # restituisce il percorso del file salvato.

#---------------------------------------------- PNG di una tabella con esempi di paesi per ogni gruppo (cluter)

def plot_top_countries(assignments: pd.DataFrame) -> Path:
    """
    Prende assignments che è un DataFrame con almeno tre colonne:Prende assignments che è un DataFrame con almeno tre colonne:
        cluster_kmeans — il numero del cluster assegnato ad ogni paese
        distanza_centroide — quanto è lontano ogni paese dal centro del suo cluster
        country — il nome del paese
      è la tabella con i risultati del K-Means, una riga per paese.
    """

    output = PLOTS_DIR / "esempi_paesi_per_cluster.png"
    rows = []
    # Itera su ogni cluster
    for cluster_id, group in assignments.groupby("cluster_kmeans"):
        # Ordina i paesi per distanza dal centroide e prende i 7 più vicini
        ordered = group.sort_values("distanza_centroide").head(7)
        # Crea una riga: [numero cluster, paesi separati da virgola]
        rows.append([f"Cluster {cluster_id}", ", ".join(ordered["country"].tolist())])

    fig, ax = plt.subplots(figsize=(10, 3.8)) # //
    # Nasconde gli assi — serve solo come contenitore per la tabella
    ax.axis("off")

    # Disegna la tabella con i dati e le intestazioni delle colonne
    table = ax.table(cellText=rows, colLabels=["Cluster", "Paesi più centrali"], loc="center")
    # Disabilita il font automatico e imposta una dimensione fissa
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    # Scala la tabella: 1 = larghezza invariata, 1.7 = righe più alte per leggibilità
    table.scale(1, 1.7)
    # //
    ax.set_title("Esempi di paesi vicini al centroide", fontsize=15, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output

    """
    Nel grafico/tabella esempi_paesi_per_cluster.png, i cluster non sono fatti direttamente “a mano”.

  Sono i cluster creati da K-Means usando queste variabili selezionate:

  child_mort, income, life_expec, total_fer, gdpp

  Il procedimento è:

  1. Il programma prende queste 5 variabili per ogni Paese.
  2. Le standardizza, cioè le mette sulla stessa scala.
  3. Applica la PCA e tiene le prime 2 componenti principali.
  4. Su queste 2 componenti applica K-Means.
  5. K-Means divide i Paesi in gruppi simili tra loro.

  Quindi i cluster sono basati su somiglianze tra Paesi rispetto a:

  - mortalità infantile
  - reddito medio
  - aspettativa di vita
  - fertilità
  - PIL pro capite

  Nel grafico esempi_paesi_per_cluster.png, però, non vengono mostrati tutti i Paesi. Vengono mostrati solo
  alcuni esempi per ogni cluster:

  ordered = group.sort_values("distanza_centroide").head(7)

  Questo significa:

  > per ogni cluster vengono presi i 7 Paesi più vicini al centro del cluster.

  Quindi sono Paesi “rappresentativi” di quel gruppo.
    """

#----------------------------------------------









# $ $ $ $ $. $ $ $ $ $$ $ $ $ $ $ $ $ $$ $ $ $ $



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
