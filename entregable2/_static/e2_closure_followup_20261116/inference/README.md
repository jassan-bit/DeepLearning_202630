# Comparación OOF–prueba y descomposición condicional de pérdidas

Este suplemento posterior utiliza únicamente pronósticos guardados. No entrena
modelos, genera nuevos forwards, modifica A2/RQ1 ni transforma el calendario
conocido en confirmación prospectiva.

## Comparación con la validación utilizada realmente en HPO

Se recuperan las **249 predicciones OOF de cada ganador registrado**: tres
predictores de fold con 83 fechas cada uno. Se compara su RMSE concatenada con
la media de las seis RMSE individuales de prueba608 de A2. La diferencia es
**prueba menos OOF**. No se sustituyen esas predicciones por calibración53.

Los calendarios se remuestrean por separado: bloques móviles dentro de cada
fold y dentro de prueba. En prueba se remuestrean seis inicializaciones dentro
de cada modelo; no existe una segunda inicialización OOF que se pueda inventar.
Los modelos comparten los índices temporales de cada réplica. Se utilizan
5.000 réplicas, bloque principal14 y sensibilidades7/28. La familia bilateral
principal tiene trece comparaciones con Holm13. Los IC son marginales.

**Sólo SAE rechaza igualdad con Holm13:** RMSE OOF=0,0056136885,
RMSE prueba=0,0038034296, diferencia=−0,0018102588, IC95 %
[−0,0030418907; −0,0009580752], p ajustado=0,01039792.
Las otras doce comparaciones no rechazan. Su falta de rechazo no demuestra
consistencia, igualdad o equivalencia.

Los pesos, tamaños de entrenamiento y periodos de OOF y prueba son distintos.
Además, OOF seleccionó la configuración y la parada. Este cálculo condiciona
en esa selección: no vuelve a ejecutar HPO ni identifica si una diferencia
proviene del tamaño de entrenamiento, deriva, dificultad o sobreajuste.
**La comparación empírica está ejecutada; una validación externa nueva del
procedimiento de selección no queda establecida.**

## Reparto finito de la variación de la pérdida

La cantidad descompuesta es el **error cuadrático diario**, en el tensor
equilibrado de trece modelos × seis inicializaciones ×608 fechas. No es la
RMSE de A2. Proyecciones ortogonales separan cinco componentes, con ponderación
uniforme de modelos, semillas y fechas. Su suma reproduce la varianza empírica
total con error relativo máximo de 1,10×10⁻¹⁵.

| Componente | Fracción observada (%) | IC95 % marginal, bloque14 (%) |
|---|---:|---:|
| Diferencias entre los trece modelos | 5,51 | [3,92;14,59] |
| Inicialización anidada en modelo | 0,25 | [0,12;0,76] |
| Calendario compartido | 35,99 | [24,63;43,12] |
| Interacción modelo–calendario | 49,51 | [38,98;56,58] |
| Resto inicialización–calendario | 8,75 | [5,34;15,23] |

El bootstrap remuestrea conjuntamente fechas por bloques y, dentro de cada
modelo, sus inicializaciones; cada réplica recompone todas las proyecciones.
La varianza total de la pérdida es 8,84528027×10⁻⁹, en unidades de volatilidad
elevadas a la cuarta potencia. Los IC de las fracciones no son simultáneos.

El reparto describe **estas trayectorias y este catálogo**. No es un ajuste
REML de efectos aleatorios ni estima variación entre mercados, entre nuevas
particiones o de volver a seleccionar hiperparámetros. El componente principal
de inicialización no recoge por sí solo toda su influencia: el resto también
incluye su interacción con el calendario. La dependencia temporal, la deriva
observada y la adecuación aproximada de los bloques siguen siendo limitaciones.

## Reanálisis independiente

Con Python y NumPy, guardar en una misma carpeta:

- `replay_inference.py`
- `inference_inputs.npz`
- `inference_inputs_manifest.json`

No se necesitan Git, Internet, checkpoints, otros paquetes del entregable ni
un ledger del proyecto para ejecutar el reanálisis del profesor.

```bash
python replay_inference.py --output resultados_nuevos
```

La salida debe ser una carpeta que aún no exista. El modo predeterminado
recalcula las dos tablas completas con sus IC, p-valores y sensibilidades.
Para revisar sólo métricas puntuales y componentes, sin remuestreo:

```bash
python replay_inference.py --output puntos_nuevos --point-only
```

El NPZ incluye pronósticos, objetivos, fechas UTC, máscara e identificadores,
con hashes de los resultados OOF y checkpoints originales de prueba. No contiene
los pesos ni permite reconstruir el entrenamiento o la selección HPO.

## Registro, recuperación y verificación

- [Plan científico ejecutado](recovery_v3/analysis_plan.json), fijado antes del
  cálculo; el código exacto se conserva en
  [registered_analysis_script.py](recovery_v3/registered_analysis_script.py).
- [39 admisiones OOF](recovery_v3/result_v1/winning_OOF_admission_39.csv).
- [Comparaciones e IC](recovery_v3/result_v1/conditional_OOF_test_comparisons_39.csv).
- [Componentes e IC](recovery_v3/result_v1/finite_loss_variance_components_15.csv).
- [Resumen](recovery_v3/result_v1/summary.json) y
  [réplicas numéricas](recovery_v3/result_v1/bootstrap_draws.npz).
- [Plan de recuperación de metadatos](export_recovery_plan.json),
  [recibo de recuperación](metadata_recovery_receipt.json) y
  [QA del replay portable](portable_replay_QA.json).

Dos intentos previos se detuvieron en la admisión por los dos formatos de fecha
de las fuentes: timestamps UTC y fechas de calendario. Se preservan ambos
planes y sus códigos. El tercer intento completó todas las estadísticas y
archivos numéricos; falló después al serializar una ruta relativa del manifiesto.
La recuperación conserva esos resultados con sus hashes y sólo completa
metadatos y el paquete de reanálisis. El código de uso futuro corrige esa
serialización sin cambiar el algoritmo estadístico. Las tres salidas de error
están en `run_logs/`; todos sus cargos permanecen en el ledger compartido.

El replay completo se verifica desde una carpeta temporal externa al repositorio,
sin `PYTHONPATH`. Se exige igualdad byte a byte de ambas tablas frente al cálculo
original. Esta repetición de QA se contabiliza aparte y no constituye un nuevo
experimento ni una reproducción del entrenamiento.
