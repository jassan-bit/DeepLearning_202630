# Adecuación retrospectiva V4: ejecución terminada

**Estado:** análisis auxiliar ejecutado, sin confirmación, acreditación R3 o
certificado de adecuación. Asistencia de IA declarada.

El plan y las ocho fuentes auxiliares se publicaron antes del cálculo en
`e8485812241e492f683f0e6ea77459ca3423f30c`, verificados por el proceso principal
mediante API Git y auditoría Merkle. El anclaje es de las **nuevas operaciones**;
los calendarios y el resultado adverso del piloto ya eran conocidos. No se
presenta como preregistro ni devuelve independencia a los datos de desarrollo.

- Plan: `analysis_plan.json`, SHA256 `864f321f69783810ee306a29fe47588a6aadb7bb4dd7dc0e2cb2c1791796a80e`.
- Congelamiento previo: `IMPLEMENTATION_FREEZE.json`, SHA256 `5744f7385640c9636bf8ed8f37141074b5d9e66434cd2dd289f48c3ee1e957eb`.
- **46 entradas** del commit `d6cf7f1af6800ec1ac85a26347bdde52380e362f`, verificadas antes y después.
- Ejecución real: `2026-11-21T03:15:20.579411Z` a `03:15:20.633731Z` UTC.
  El pipeline midió **0,05434 s**; excluye arranque del intérprete, importaciones,
  elaboración del plan y pruebas de software.
- **0 entrenamientos de pronóstico, 0 forward passes, 0 predicciones nuevas y
  0 remuestreos**. Se hicieron **44 regresiones OLS auxiliares** —22 ARCH y22 MZ—,
  contadas por separado de los veinte ajustes neuronales anteriores.

## Resultados y lectura

Las **1.210 filas** corresponden a 22 trayectorias sobre las mismas **55 fechas**:
cuatro recetas neuronales por cinco semillas y dos controles analíticos una vez.
El calendario de evaluación expuesto es 7 de septiembre–31 de octubre de 2024.

| Diagnóstico | Posiciones computadas | Rechazos tras Holm88 |
|---|---:|---:|
| Ljung–Box rezagos1–7 | 22 | 0 |
| Ljung–Box rezagos1–14 | 22 | 0 |
| ARCH-LM rezago7 | 22 | 0 |
| Mincer–Zarnowitz conjunto, HAC14 | 22 | 13 |

Las 88 posiciones se conservaron; no hubo estadísticas no soportadas en esta
ejecución. Los **13 rechazos son señales de desviación de la calibración lineal
condicionadas a supuestos asintóticos**, no una prueba general de inutilidad.
Corresponden a las cinco trayectorias aditivas, dos de cinco sigma-CRPS V4,
dos de cinco U-CRPS, dos de cinco MSE y ambos controles analíticos.

Con **55 fechas**, la referencia marginal clásica y la calidad del HAC siguen
sin calibración finita demostrada. Ljung–Box no es robusto a heterocedasticidad,
y Holm no corrige un p marginal inválido. Los ceros de rechazo de Ljung–Box/ARCH
no certifican ausencia de dependencia, homocedasticidad, potencia o adecuación.
El solapamiento de objetivos motiva comprobar dependencia; no la impone en el
error óptimo a un paso, que sería una diferencia de martingala.

Los **44 intervalos de sesgo medio** con HAC7/HAC14 son puntuales, asintóticos y
condicionados a cada predictor guardado. No tienen cobertura simultánea
certificada ni incluyen variabilidad poblacional de inicializaciones. **No son
intervalos de ventaja primaria de RMSE**.

## Sensibilidad descriptiva y resultado adverso conservado

| Receta | RMSE media individual, sept7–30 (24 fechas) | RMSE media individual, oct1–31 (31 fechas) |
|---|---:|---:|
| V4 sigma-CRPS | 0,001982974 | 0,002164303 |
| U-CRPS control | 0,001952526 | 0,002158059 |
| MSE-only control | 0,001938335 | 0,002145686 |
| Aditiva positiva | 0,002117654 | 0,002245723 |
| Piso | 0,002683806 | 0,002870104 |
| Piso+exceso medio | 0,002033058 | 0,002174995 |

No se elige un mes favorable ni se llama completo a septiembre. Las escalas y
condiciones de los dos periodos difieren. Ambos cortes se mantienen expuestos.

Sobre las cinco semillas pareadas, la diferencia V4 menos MSE-only de RMSE es
**+0,00002969978 sigma7**, con DE **0,00002340272** y Cohen dz descriptivo **1,2691**.
Las cinco diferencias son positivas. Esto conserva el deterioro registrado de
V4 y su abandono; no promueve otra receta. Ese dz sólo resume dispersión de
cinco repeticiones sobre un calendario fijo, sin cubrir incertidumbre temporal.
No satisface el requisito de tamaño de efecto confirmatorio por sí solo.

La comparación con los otros controles y el estimando secundario **RMSE de la
predicción promediada** se conservan en CSV, separados de la media de RMSE
individuales. Los dos controles analíticos no se duplican para inventar semillas.

## Sobreajuste y calibración que no pueden reconstruirse

Las veinte curvas y recibos se verificaron, incluida la época seleccionada y su
MSE de parada. La RMSE de evaluación/parada tiene razones entre aproximadamente
0,463 y0,578, según receta/semilla. Son calendarios diferentes y la parada se
usó para elegir la época; razones menores que uno no demuestran ausencia de
sobreajuste. No hay predicciones guardadas de ajuste/parada para un contraste
completo. El coste compuesto CRPS de las dos recetas correspondientes no es la
MSE puntual de validación. Los dos controles neuronales de MSE conservan su
objetivo propio; tampoco se interpreta la media de pérdidas durante una época
como la pérdida de ajuste del checkpoint final.

No existe un calibrador o distribución predictiva V4 guardados que permita
evaluar PIT o cobertura de intervalos sin nuevas inferencias. No se hereda uno
de E2 o de la aditiva. Calibración lineal MZ y cobertura probabilística son objetos
distintos. Continúa intacta la guarda de **diez semillas** de la inferencia
cruzada primaria: las cinco disponibles no la cumplen.

## Archivos y reproducción

`results/OUTPUT_MANIFEST.json` enumera los diez archivos derivados: ocho CSV,
el resumen y el recibo. Contiene errores individuales, diagnóstico/multiplicidad,
intervalos, meses, estimandos, efectos descriptivos y curvas. No contiene pesos.

`METHODS_AND_SCOPE.md` conserva las instrucciones previas y llama al anclaje
con un nombre genérico. El archivo efectivamente publicado es `PUBLIC_ANCHOR.json`.
Desde este directorio y con las entradas fijadas disponibles:

```bash
python audit_v4_saved.py \
  --input-root /ruta/al/repositorio/experiments/e3_next_evidence_20261121/development/cpu_pilot_results \
  --anchor PUBLIC_ANCHOR.json --out mi_resultado_nuevo
python audit_v4_saved.py \
  --input-root /ruta/al/repositorio/experiments/e3_next_evidence_20261121/development/cpu_pilot_results \
  --out results --verify-only
```

No se sobreescribe `results`; la reproducción numérica usa otro directorio.
`--verify-only` verificó
46 entradas y los diez resultados, sin repetir análisis. Las **15 guardas de
software** incluyen igualdad numérica con statsmodels, invariancia de unidades,
degeneración, calendarios, semillas y posiciones Holm. El historial conserva
una expectativa manual incorrecta en una fixture ACF, corregida antes del
anclaje y del cálculo real. Esos tests prueban software, no calibración estadística.

`CLOSURE_MATRIX.csv` separa lo ejecutado de la evidencia prospectiva pendiente.
No se alteraron los 796 trabajos, los treinta ajustes V3, la transferencia de
20.400 filas, el piloto V4 o sus decisiones originales.
