# Resultado posterior de la validación controlada nueva

Este documento es posterior a los resultados; no reemplaza el plan ni sus fuentes congeladas.
Se completaron **640** réplicas y **639360** remuestreos cruzados; el control antiguo añadió **0** muestras. Tiempo del controlador: **187.684 s**.
Plan y código públicos antes de ejecutar: `7d304faa0dd583c6109a688ff179ec7e937f6561`. Fuentes intactas: `True`.

| Caso nulo | Nuevo bootstrap-t | IC95 % individual CP | Cribado de desarrollo |
|---|---:|---|---|
| EPA_fresh_nominal | 21/256 = 8.20 % | 5.15–12.27 % | exploratory_tolerance_screen_passed |
| EPA_fresh_strong_dependence | 4/128 = 3.12 % | 0.86–7.81 % | exploratory_tolerance_screen_passed |
| EPA_fresh_seed_only | 2/128 = 1.56 % | 0.19–5.53 % | exploratory_tolerance_screen_passed |

**Decisión del cribado controlado: `exploratory_tolerance_screen_passed_declared_nulls_only`.** El umbral era límite superior CP ≤0,13 frente a nominal0,10. Es una tolerancia exploratoria individual; no certifica nivel exacto, garantía simultánea o validez universal.

**Resultado adverso que permanece:** el subconjunto primario de dos métodos no queda aprobado. Sus tasas de falso positivo familiar corregidas por Holm se conservan a continuación; no se elimina después el método o rival que provoca el problema.

| Caso nulo | FWER de P1/P2 con Holm38 | IC95 % individual CP | Decisión exploratoria |
|---|---:|---|---|
| EPA_fresh_nominal | 4/256 = 1.56 % | 0.43–3.95 % | exploratory_tolerance_screen_passed |
| EPA_fresh_strong_dependence | 9/128 = 7.03 % | 3.27–12.93 % | inconclusive |
| EPA_fresh_seed_only | 14/128 = 10.94 % | 6.11–17.67 % | inflation_detected |

En la alternativa predeclarada, el MCS nuevo rechaza globalmente en **112/128** (87.50 %), pero retiene sólo al candidato en **39/128** (30.47 %). La potencia conjunta del subconjunto de dos métodos es **18/128** (14.06 %; IC95 % 8.55–21.31 %). La cifra conjunta cuenta sólo la intersección de los p-valores ajustados P1/P2; no incorpora toda la regla de márgenes, efectos y adecuación. No es potencia de una batería completa ni satisface por sí sola el diseño prospectivo de mercado.

Las tres variantes de MCS son comparaciones de una extensión metodológica, no tres pruebas primarias distintas. El MCS original no se altera ni se declara erróneo por usar studentizadores fijos. Las diferencias entre v1 y v2 incluyen cambio del estimador de covarianza y re-studentización; el control nuevo de studentizador fijo distingue esta última intervención.

Los CSV de `tables/` conservan todos los casos, variantes, intervalos por rival, FWER del subconjunto de dos métodos y seis comparaciones pareadas corregidas por Holm. El escenario alternativo informa potencia sintética de ese subconjunto, sin certificar ≥0,90 por contraste o potencia conjunta de una batería primaria completa.

**No se habilita confirmación ni se acredita R3.** Faltan admisibilidad y calibración suficiente de tres procedimientos primarios pertinentes, candidata viable, paneles admitidos y dimensionamiento prospectivo real. V3 conserva NO-GO y V4 sigue abandonada.

No se retocaron métodos, parámetros, bloques, semillas o criterios después de esta validación. Asistencia de IA declarada.
