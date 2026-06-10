# Output progetto Country clustering

## Risultati 

- Osservazioni analizzate: 167  (paesi)
- Feature numeriche iniziali: 9 (caratteristica per paese)
- Feature selezionate per la clusterizzazione finale: child_mort, income, life_expec, total_fer, gdpp
- Varianza spiegata dalle prime due PC(principal components) del subset: 91.60%
- Numero di cluster scelto: 3
- Silhouette K-Means: 0.574
- Silhouette K-Neighbors: 0.570
- Adjusted Rand Index K-Means vs K-Neighbors: 0.909

## File principali

- Relazione: `relazione/relazione_progetto_country_clustering.pdf`
- Zip sorgenti: `codice_sorgente/codice_sorgente_country_clustering.zip`
- Grafici: `grafici/`
- Tabelle: `tabelle/`
- Dataset: `dati/Country-data.csv`

## Come visualizzare il progetto

Installazione delle librerie necessarie(requirements.txt):

```bash
pip install -r codice_sorgente/requirements.txt
```

Per eseguire lo script:

```bash
python3 codice_sorgente/main.py
```
