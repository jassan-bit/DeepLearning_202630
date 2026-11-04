Validación forward chaining de 2024. Media ± DE muestral entre diez semillas; las fechas y el objetivo n=7/h=1 se mantienen. Persistencia es una referencia determinista, sin réplicas de entrenamiento. Los intervalos, cobertura y recuentos definidos se conservan en el CSV.

| Entrada D (días) | Modelo | RMSE | MAE | MSE | MAPE (%) | R² | MASE |
|---|---|---|---|---|---|---|---|
| 7 | MLP | 0,0060491 ± 0,0002780 | 0,0043804 ± 0,0002110 | 0,000036661 ± 0,000003390 | 19,33 ± 0,74 | 0,61567 ± 0,03554 | 1,66168 ± 0,08422 |
| 7 | LSTM | 0,0063354 ± 0,0001870 | 0,0047263 ± 0,0001245 | 0,000040169 ± 0,000002366 | 21,79 ± 0,80 | 0,57889 ± 0,02481 | 1,78982 ± 0,04645 |
| 7 | CNN1D | 0,0062296 ± 0,0003373 | 0,0046780 ± 0,0002212 | 0,000038910 ± 0,000004072 | 21,98 ± 1,00 | 0,59209 ± 0,04269 | 1,77956 ± 0,08783 |
| 7 | Persistencia | 0,0047890; referencia determinista | 0,0029435; referencia determinista | 0,000022934; referencia determinista | 13,55; referencia determinista | 0,75957; referencia determinista | 1,11835; referencia determinista |
| 14 | MLP | 0,0069510 ± 0,0005341 | 0,0051882 ± 0,0003986 | 0,000048573 ± 0,000007632 | 22,36 ± 1,42 | 0,49079 ± 0,08001 | 1,96807 ± 0,15406 |
| 14 | LSTM | 0,0077018 ± 0,0001999 | 0,0056700 ± 0,0001658 | 0,000059354 ± 0,000003095 | 24,44 ± 1,44 | 0,37777 ± 0,03244 | 2,14989 ± 0,06420 |
| 14 | CNN1D | 0,0092696 ± 0,0003187 | 0,0067505 ± 0,0002064 | 0,000086017 ± 0,000005865 | 28,90 ± 0,94 | 0,09825 ± 0,06148 | 2,56498 ± 0,07841 |
| 14 | Persistencia | 0,0047890; referencia determinista | 0,0029435; referencia determinista | 0,000022934; referencia determinista | 13,55; referencia determinista | 0,75957; referencia determinista | 1,11335; referencia determinista |
| 21 | MLP | 0,0083825 ± 0,0004626 | 0,0061462 ± 0,0002981 | 0,000070459 ± 0,000007788 | 25,35 ± 1,09 | 0,26135 ± 0,08165 | 2,33300 ± 0,11435 |
| 21 | LSTM | 0,0087796 ± 0,0003524 | 0,0062492 ± 0,0002399 | 0,000077193 ± 0,000006080 | 26,22 ± 1,02 | 0,19075 ± 0,06374 | 2,36819 ± 0,09224 |
| 21 | CNN1D | 0,0098231 ± 0,0002269 | 0,0072237 ± 0,0001826 | 0,000096539 ± 0,000004484 | 30,59 ± 0,84 | -0,01206 ± 0,04701 | 2,74184 ± 0,07091 |
| 21 | Persistencia | 0,0047890; referencia determinista | 0,0029435; referencia determinista | 0,000022934; referencia determinista | 13,55; referencia determinista | 0,75957; referencia determinista | 1,11245; referencia determinista |
| 28 | MLP | 0,0093860 ± 0,0008195 | 0,0067748 ± 0,0004255 | 0,000088702 ± 0,000015418 | 27,19 ± 1,22 | 0,07010 ± 0,16163 | 2,58069 ± 0,16686 |
| 28 | LSTM | 0,0093129 ± 0,0002589 | 0,0064968 ± 0,0001772 | 0,000086790 ± 0,000004804 | 26,11 ± 0,86 | 0,09015 ± 0,05036 | 2,47789 ± 0,06705 |
| 28 | CNN1D | 0,0099066 ± 0,0002304 | 0,0073046 ± 0,0001872 | 0,000098188 ± 0,000004577 | 31,18 ± 0,92 | -0,02935 ± 0,04798 | 2,79384 ± 0,07414 |
| 28 | Persistencia | 0,0047890; referencia determinista | 0,0029435; referencia determinista | 0,000022934; referencia determinista | 13,55; referencia determinista | 0,75957; referencia determinista | 1,12126; referencia determinista |

[Descargar resumen completo](../_static/filipe_multihistory_20261104/analysis_cv/metrics_summary.csv).
