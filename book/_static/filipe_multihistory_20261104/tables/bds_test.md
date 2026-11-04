BDS se calcula sobre los residuos de horizonte h=1 de cada bloque y semilla, con dimensiones 2 y 3 y radio de 1,5 DE muestral. Los bloques mensuales son cortos y los p-valores asintóticos se interpretan como diagnósticos exploratorios. El promedio y la DE de p-valores son descriptivos; no constituyen un contraste combinado ni una probabilidad de independencia.

| D | Modelo | Dimensión | p definidos / previstos | Media p (descriptiva) | DE p (descriptiva) | Rechazos Holm, α=0,05 |
|---|---|---|---|---|---|---|
| 7 | CNN1D | 2 | 200/200 | 0,105102 | 0,221707 | 84 |
| 7 | CNN1D | 3 | 200/200 | 0,118528 | 0,232420 | 76 |
| 7 | LSTM | 2 | 200/200 | 0,050000 | 0,184918 | 135 |
| 7 | LSTM | 3 | 200/200 | 0,051976 | 0,158720 | 124 |
| 7 | MLP | 2 | 200/200 | 0,076018 | 0,189438 | 97 |
| 7 | MLP | 3 | 200/200 | 0,094188 | 0,207507 | 100 |
| 14 | CNN1D | 2 | 200/200 | 0,004959 | 0,055291 | 184 |
| 14 | CNN1D | 3 | 200/200 | 0,015195 | 0,095313 | 179 |
| 14 | LSTM | 2 | 200/200 | 0,017436 | 0,095400 | 147 |
| 14 | LSTM | 3 | 200/200 | 0,024050 | 0,104575 | 133 |
| 14 | MLP | 2 | 200/200 | 0,041902 | 0,134103 | 121 |
| 14 | MLP | 3 | 200/200 | 0,043533 | 0,130718 | 115 |
| 21 | CNN1D | 2 | 200/200 | 0,017463 | 0,099971 | 191 |
| 21 | CNN1D | 3 | 200/200 | 0,015312 | 0,096886 | 191 |
| 21 | LSTM | 2 | 200/200 | 0,012820 | 0,064395 | 153 |
| 21 | LSTM | 3 | 200/200 | 0,023574 | 0,104464 | 142 |
| 21 | MLP | 2 | 200/200 | 0,034867 | 0,126915 | 143 |
| 21 | MLP | 3 | 200/200 | 0,028493 | 0,113573 | 141 |
| 28 | CNN1D | 2 | 200/200 | 0,008418 | 0,061554 | 191 |
| 28 | CNN1D | 3 | 200/200 | 0,007675 | 0,048608 | 191 |
| 28 | LSTM | 2 | 200/200 | 0,013675 | 0,078342 | 179 |
| 28 | LSTM | 3 | 200/200 | 0,020768 | 0,108517 | 176 |
| 28 | MLP | 2 | 200/200 | 0,010747 | 0,086350 | 186 |
| 28 | MLP | 3 | 200/200 | 0,008909 | 0,061053 | 181 |

- [bds_by_fold.csv](../_static/filipe_multihistory_20261104/test/analysis/bds_by_fold.csv).
- [bds_by_fold_summary.csv](../_static/filipe_multihistory_20261104/test/analysis/bds_by_fold_summary.csv).
- [bds_mean_across_folds_descriptive.csv](../_static/filipe_multihistory_20261104/test/analysis/bds_mean_across_folds_descriptive.csv).
- [bds_pvalues_descriptive_summary.csv](../_static/filipe_multihistory_20261104/test/analysis/bds_pvalues_descriptive_summary.csv).
