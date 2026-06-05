# Output progetto Country clustering

## Risultati principali

- Osservazioni analizzate: 167
- Feature numeriche iniziali: 9
- Feature selezionate per la clusterizzazione finale: child_mort, income, life_expec, total_fer, gdpp
- Varianza spiegata dalle prime due PC del subset: 91.60%
- Numero di cluster scelto: 3
- Silhouette K-Means: 0.574
- Silhouette K-Neighbors: 0.570
- Adjusted Rand Index K-Means vs K-Neighbors: 0.909

## File principali

- Relazione PDF: `relazione/relazione_progetto_country_clustering.pdf`
- Zip sorgenti: `codice_sorgente/codice_sorgente_country_clustering.zip`
- Grafici: `grafici/`
- Tabelle e assegnazioni: `tabelle/`
- Dataset copiato per riproducibilita: `dati/Country-data.csv`

## Riproduzione

```bash
python3 output_progetto/codice_sorgente/main.py
```
