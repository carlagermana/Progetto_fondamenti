# Progetto Country-data clustering

Questo codice realizza una pipeline completa di data analysis sul dataset `Country-data.csv`.

## Contenuto

- `main.py`: script principale commentato.
- `requirements.txt`: librerie necessarie.

## Esecuzione

Dalla root del progetto:

```bash
python3 output_progetto/codice_sorgente/main.py
```

Lo script legge il dataset originale dalla root del progetto oppure, se non presente, la copia
in `output_progetto/dati/Country-data.csv`. Salva tutti gli output dentro `output_progetto`,
senza modificare le directory della consegna o del materiale del corso.

## Output generati

- `grafici/`: distribuzioni, matrice di correlazione, scree plot, biplot, cluster PCA.
- `tabelle/`: statistiche, matrice di correlazione, scelta di `k`, assegnazione cluster.
- `relazione/`: relazione PDF finale.
- `dati/`: copia del dataset usata per riproducibilita.
