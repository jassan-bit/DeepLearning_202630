BDS se calcula sobre los residuos de horizonte h=1 de cada bloque y semilla, con dimensiones 2 y 3 y radio de 1,5 DE muestral. Los bloques mensuales son cortos y los p-valores asintóticos se interpretan como diagnósticos exploratorios. El promedio y la DE de p-valores son descriptivos; no constituyen un contraste combinado ni una probabilidad de independencia.

| D | Modelo | Dimensión | p definidos / previstos | Media p (descriptiva) | DE p (descriptiva) | Rechazos Holm, α=0,05 |
|---|---|---|---|---|---|---|
| 7 | CNN1D | 2 | 120/120 | 0,189190 | 0,289672 | 54 |
| 7 | CNN1D | 3 | 120/120 | 0,216587 | 0,305149 | 54 |
| 7 | LSTM | 2 | 120/120 | 0,104282 | 0,216751 | 55 |
| 7 | LSTM | 3 | 120/120 | 0,115311 | 0,220061 | 48 |
| 7 | MLP | 2 | 120/120 | 0,187543 | 0,298378 | 54 |
| 7 | MLP | 3 | 120/120 | 0,187511 | 0,299019 | 50 |
| 14 | CNN1D | 2 | 120/120 | 0,040308 | 0,151474 | 95 |
| 14 | CNN1D | 3 | 120/120 | 0,036687 | 0,144666 | 98 |
| 14 | LSTM | 2 | 120/120 | 0,086290 | 0,195357 | 73 |
| 14 | LSTM | 3 | 120/120 | 0,114850 | 0,233369 | 69 |
| 14 | MLP | 2 | 120/120 | 0,057291 | 0,193246 | 76 |
| 14 | MLP | 3 | 120/120 | 0,064782 | 0,182154 | 66 |
| 21 | CNN1D | 2 | 120/120 | 0,000347 | 0,003615 | 116 |
| 21 | CNN1D | 3 | 120/120 | 0,004778 | 0,037973 | 114 |
| 21 | LSTM | 2 | 120/120 | 0,093804 | 0,200501 | 80 |
| 21 | LSTM | 3 | 120/120 | 0,123288 | 0,262537 | 76 |
| 21 | MLP | 2 | 120/120 | 0,031146 | 0,128393 | 93 |
| 21 | MLP | 3 | 120/120 | 0,046430 | 0,150166 | 92 |
| 28 | CNN1D | 2 | 120/120 | 0,000531 | 0,004030 | 113 |
| 28 | CNN1D | 3 | 120/120 | 0,002424 | 0,023429 | 109 |
| 28 | LSTM | 2 | 120/120 | 0,063567 | 0,181633 | 84 |
| 28 | LSTM | 3 | 120/120 | 0,065453 | 0,182395 | 83 |
| 28 | MLP | 2 | 120/120 | 0,044668 | 0,153850 | 88 |
| 28 | MLP | 3 | 120/120 | 0,053219 | 0,168621 | 90 |

- [bds_by_fold.csv](../_static/filipe_multihistory_20261104/analysis_cv/bds_by_fold.csv).
- [bds_by_fold_summary.csv](../_static/filipe_multihistory_20261104/analysis_cv/bds_by_fold_summary.csv).
- [bds_mean_across_folds_descriptive.csv](../_static/filipe_multihistory_20261104/analysis_cv/bds_mean_across_folds_descriptive.csv).
- [bds_pvalues_descriptive_summary.csv](../_static/filipe_multihistory_20261104/analysis_cv/bds_pvalues_descriptive_summary.csv).
