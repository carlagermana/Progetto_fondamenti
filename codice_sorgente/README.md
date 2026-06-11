# Progetto Country-data clustering

Questo codice realizza una pipeline completa di data analysis sul dataset `Country-data.csv`.

## Contenuto

- `main.py`: script principale commentato.
- `requirements.txt`: librerie necessarie.

## Esecuzione

Dalla root del progetto:

```bash
pip install -r codice_sorgente/requirements.txt
```

```bash
python3 codice_sorgente/main.py
```

Lo script legge il dataset originale dalla root del progetto oppure, se non presente, la copia in `dati/Country-data.csv`. 
Salva tutti gli output dentro le cartelle del repository, senza modificare directory esterne.

## Output generati

- `grafici/`: distribuzioni, matrice di correlazione, scree plot, biplot, cluster PCA.
- `tabelle/`: statistiche, matrice di correlazione, scelta di `k`, assegnazione cluster.
- `dati/`: copia del dataset.

La relazione PDF si trova in `relazione/`.
