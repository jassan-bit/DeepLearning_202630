# MCS secundario de la cohorte guardada de input-only

Se ejecutó un complemento retrospectivo del conjunto de confianza de modelos
(MCS) **Tmax** de Hansen, Lunde y Nason (2011), con el código ya existente del
proyecto conservado **byte a byte**. No se reutilizó la extensión cruzada de MCS
cuya validación anterior falló. No se instaló `arch` ni se modificó el entorno
CPU fijado o el motor A.

## Fuente, plan y ejecución

- Archivo fuente de predicciones: `public_results.zip`, SHA256
  `beabedeb63ff10324b310ba8b86105505c3be783216e38770a092a7be438a17c`.
- El parser compartido de `../retrospective/audit_saved_adequacy.py` comprueba
  archivos, recibos, identidades, calendario común y objetivos idénticos antes
  de construir las pérdidas. Su SHA256 se fija en el ejecutor y el recibo.
- Se conservaron **las ocho recetas**, seis neuronales con cinco semillas
  guardadas cada una y dos controles analíticos sin réplicas artificiales:
  **32 trayectorias, 1.760 predicciones y 55 fechas**, del 7 de septiembre al
  31 de octubre de 2024. No se contaron semillas como diseños nuevos.
- `analysis_plan.json` se escribió el **22-11-2026 a las 03:00:11 UTC**,
  `IMPLEMENTATION_FREEZE.json` a las **03:02:00 UTC**, antes de ejecutar este
  cálculo a las **03:02:01 UTC**. Todo ocurre después del piloto y de la
  nominación de input-only: **no es un preregistro ni devuelve independencia al
  calendario**.
- El plan fijó nivel MCS **90 %** (`alpha=0.1`), **1.999 remuestreos** por
  sensibilidad y bloques circulares comunes de **7 y 14 días**, con PCG64 y
  semilla `2026112203`. El nivel se utiliza en el sentido de la construcción
  MCS bajo sus supuestos, no como cobertura finita demostrada en estos datos.

## Resultado ejecutado

| Bloque | Recetas iniciales | Recetas retenidas | Exclusión | p secuencial de exclusión | p EPA del conjunto remanente |
|---|---:|---:|---|---:|---:|
| 7 días | 8 | 7 | `floor_only` | 0,0005 | 0,2005 |
| 14 días | 8 | 7 | `floor_only` | 0,0005 | 0,1530 |

En ambos casos el conjunto remanente contiene `additive_mse`,
`floor_plus_mean_excess`, `global_crps`, `global_mse`, `input_only_mse`,
`output_only_mse` y `seoi_mse`. **Input-only permanece junto con seis rivales;
no aparece un ganador único.** La inclusión no demuestra equivalencia,
adecuación, superioridad práctica ni R3.

El valor 0,0005 corresponde a la regla real `(1 + excedencias)/(B + 1)` y su
resolución mínima `1/2000`; no se declara un p de cero. El algoritmo se detiene
al no rechazar EPA: **no calcula p individuales completos de los
supervivientes**. `membership.csv` deja esas celdas vacías y registra por
separado el límite inferior secuencial obtenido del último test global.
No se rellenaron p=1 por conveniencia.

## Alcance y límites

La pérdida secundaria es, para cada fecha, la **media del error cuadrático de
las semillas ajustadas fijas** de cada receta. No se promedian los pronósticos
antes de elevar al cuadrado. El resultado no contrasta la métrica primaria
histórica **media de RMSE individuales**, y no se invoca monotonía de la raíz
para fingir esa equivalencia. Tampoco contrasta un margen práctico de RMSE ni
se cuenta como tercera prueba de la familia primaria de pares.

La familia contiene las **ocho recetas del desarrollo V5**, no los 21 sistemas
del motor confirmatorio. Por ambas razones —familia y estimando— este
complemento no completa por sí solo la casilla de MCS/CD de la comparación
confirmatoria exigida por la guía.

Sólo se remuestrea el calendario; no se incluye incertidumbre de otras
inicializaciones, selección de arquitectura o nominación posterior. Los
supuestos de dependencia débil y estabilidad temporal del bootstrap no quedan
verificados por este cálculo. Con 55 fechas, el bloque de 14 días abarca menos
de cuatro longitudes de bloque: el soporte es reducido. Se reportan ambas
sensibilidades, sin escoger la favorable ni atribuir cobertura conjunta a sus
dos resultados. Es otra familia exploratoria, separada de Holm128 y de la
confirmación.

Se ejecutaron **3.998 sorteos de índices de calendario**, reutilizados dentro
de cada secuencia de eliminación. El pipeline consumió **0,01794 s CPU** y
0,01793 s de pared, excluyendo importaciones y preparación del entorno.
Hubo **cero entrenamientos, inferencias de modelos, predicciones nuevas,
descargas de mercado y aperturas de una prueba reservada nueva**. Las fuentes
conservaron sus hashes.

## Reproducción y verificación

Desde esta carpeta, con las dependencias del entorno CPU ya fijado:

```bash
python -m unittest -v test_mcs_saved.py
python run_mcs_saved.py --output results-reproduced
```

El ejecutor rechaza sobrescribir resultados existentes. Su entrada por defecto
es el ZIP del experimento `e3_input_only_candidate_20261121`; puede indicarse
otra ruta con `--input`, siempre que coincida con el hash fijado. El parser
hermano y la referencia MCS también se verifican por hash. Se incluyen
**nueve pruebas de software** de empate exacto, control evidentemente peor,
permutación, monotonía, confianza, invariancia de contrastes, falta de
equivalencia entre métricas y varianza degenerada. Estos fixtures no son
validación Monte Carlo ni evidencia de mercado.

Los resultados completos están en `results/mcs_full_results.json`, la tabla en
`results/membership.csv` y los hashes/contabilidad en
`results/execution_receipt.json` y `results/OUTPUT_MANIFEST.json`.

## Procedencia y atribución

La copia sin cambios `mcs_tmax_reference.py` procede de
[`experiments/e3_estadistica_posterior_20261117/mcs.py`](https://github.com/jassan-bit/DeepLearning_202630/blob/fdf697aa870571b4e333a32198aa97fec69c9bad/experiments/e3_estadistica_posterior_20261117/mcs.py),
blob Git `da5999673bf57ee424eebfdb8d7d161ec4b894be`, SHA256
`bb14d47277a06e4c59ac1145633dd1f5d6c23f065cac5885b5433324aeca6af4`.
Se preserva su procedencia y alcance de derechos existentes; no se concede una
licencia global nueva ni permisos sobre datos ajenos.

Referencia del método: Hansen, P. R., Lunde, A. y Nason, J. M. (2011),
*The Model Confidence Set*, Econometrica 79(2), 453–497,
[doi:10.3982/ECTA5771](https://doi.org/10.3982/ECTA5771).
La implementación y revisión de este ejecutor y sus pruebas usan asistencia de
IA declarada; no se atribuyen a una persona externa.
