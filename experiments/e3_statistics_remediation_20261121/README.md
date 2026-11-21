# Pruebas de hipótesis y adecuación: trabajo ejecutado

Este suplemento desarrolla exclusivamente el componente del **15 %** del Entregable 3. Declara asistencia de IA. Parte de `d6cf7f1af6800ec1ac85a26347bdde52380e362f` y conserva los resultados anteriores. Los identificadores técnicos son UTC del 21 de noviembre de 2026; el trabajo corresponde al 20 de noviembre en Bogotá.

## Decisión después de ejecutar

**El MCS propuesto mejora sus resultados de desarrollo controlado; la batería primaria sigue sin aprobarse.** Se completaron **640 réplicas nuevas**, cuatro escenarios y **639.360 remuestreos** conjuntos de semillas y fechas, en **187,684 s** del controlador, dentro del máximo de 300 s. Las tres variantes usan los mismos datos y remuestreos; los controles no agregan nuevas extracciones.

| Escenario nulo | MCS anterior sobre los datos nuevos | Control de covarianza nueva fija | Propuesta bootstrap-t | IC95 % Monte Carlo de la propuesta |
|---|---:|---:|---:|---|
| Dependencia moderada | 62/256 = 24,22 % | 56/256 = 21,88 % | **21/256 = 8,20 %** | 5,15–12,27 % |
| Dependencia fuerte | 36/128 = 28,13 % | 33/128 = 25,78 % | **4/128 = 3,13 %** | 0,86–7,81 % |
| Variación sólo entre semillas | 90/128 = 70,31 % | 92/128 = 71,88 % | **2/128 = 1,56 %** | 0,19–5,53 % |

Los tres límites superiores satisfacen el **filtro exploratorio previo de 13 %**, para un nominal MCS de 10 %. Son intervalos individuales; no certifican nivel exacto, validez simultánea o aplicación a cualquier mercado. El MCS estándar puede usar escalas bootstrap fijas: el resultado anterior no se atribuye a un error obvio de signo ni a una infracción automática de aquel algoritmo. Se investiga una extensión no lineal y cruzada, con dos modificaciones y un control que permite distinguirlas. Los **27/128** del estudio anterior permanecen en su archivo original; la tabla usa un experimento nuevo y controles pareados.

**El resultado adverso de la batería es importante:** el subconjunto de dos procedimientos con Holm38 presenta **14/128 = 10,94 %** de error familiar en el escenario de semillas, con IC95 % **6,11–17,67 %**, frente al nominal 5 %. El escenario temporal fuerte es inconcluso. No se retocó el método después de estos resultados.

En la alternativa sintética, rechazo global del MCS, selección única e intersección pareada son eventos diferentes: **112/128 = 87,50 %**, **39/128 = 30,47 %** y **18/128 = 14,06 %**, respectivamente. Este último evento cubre sólo dos procedimientos y sus p-valores, sin la regla completa de margen, efectos y adecuación. Ninguno acredita potencia de la batería de tres procedimientos o del diseño de mercado.

- [Decisión y límites de calibración](calibration/RESULTS_DECISION.md).
- [Tablas de las tres variantes](calibration/tables/mcs_calibration_comparison.csv).
- [Error familiar y potencia parcial](calibration/tables/family_fwer_and_subset_power.csv).
- [Comparaciones pareadas predeclaradas](calibration/tables/paired_mechanism_comparisons.csv).

## Qué se añadió sobre predicciones guardadas

El análisis nuevo usa exclusivamente las **22 trayectorias, seis recetas y 1.210 predicciones CPU** ya publicadas: cinco semillas por receta neuronal y un trayecto por control analítico. Declara una familia **Holm88** antes de ejecutar: Ljung–Box 7/14, ARCH-LM 7 y Mincer–Zarnowitz HAC14 por trayectoria.

Se computaron los 88 diagnósticos, sin posiciones inválidas. Hay **13 rechazos, todos Mincer–Zarnowitz HAC14**. La ausencia de rechazos ajustados en los otros diagnósticos no demuestra adecuación; sus referencias son asintóticas y sólo hay 55 fechas. Son calendarios ya expuestos y estos resultados no constituyen confirmación.

Se incluyen **44 IC puntuales condicionales de sesgo**, sensibilidad de los 24 días disponibles de septiembre y 31 de octubre, cinco diferencias pareadas descriptivas y un inventario de veinte curvas. Los intervalos de sesgo no son IC primarios de RMSE ni intervalos predictivos. No existen predicciones guardadas de entrenamiento/parada ni un calibrador V4 que permita completar las otras comprobaciones; no se fabricaron esos insumos. Las **44 regresiones auxiliares** son diagnósticas, sin entrenar modelos de pronóstico o producir nuevos forecasts.

[Resultados retrospectivos](retrospective/README.md) · [Brechas de adecuación](retrospective/CLOSURE_MATRIX.csv).

## Estimando, guía y potencia

La media de RMSE individuales por semilla difiere de la RMSE de un ensemble y de una comparación MSE condicionada a una cohorte fija. Cambiar entre ellas puede invertir un ranking. No se cambia el objetivo para conseguir una tercera prueba.

El Cuadro 4 permite MCS, DM-HLN, GW o Clark–West según sus supuestos, Friedman sobre unidades pertinentes y bootstrap estacionario. MCS no queda universalmente excluido como procedimiento global: hacen falta cobertura y función contrastiva justificadas, junto con efectos y margen práctico. Esta extensión experimental no acredita automáticamente ese requisito. Las variantes de bootstrap no son tres procedimientos primarios distintos.

Se reverificaron **71 fuentes distintas y los 486 escenarios anteriores**. Siguen disponibles sólo **3/19 varianzas de desarrollo V3**, con tres pares; faltan dieciséis. Las **76 celdas futuras V3 y 84 V4** permanecen vacías. Las cinco referencias y 25 covarianzas V4 son de desarrollo expuesto y la receta está abandonada. No se ejecutan sus 340 ajustes ilustrativos.

El contrato versionado verifica población, estimando, unidades, covarianza, familia, calibración, presupuesto y persistencia. Conserva **R/T finales nulos**. La referencia normal implementada muestra el piso de varianza temporal, pero no selecciona un diseño ni certifica potencia. Se exige **0,90 por contraste** y reporte conjunto con precisión, sin inventar un mínimo conjunto universal de 0,90.

[Guía y auditoría de potencia](guide_power/README.md) · [Matriz vigente de este componente](requirements_matrix_current.csv).

## Procedencia y reproducción

- [Plan y fuentes retrospectivos publicados antes de ejecutarlos](https://github.com/jassan-bit/DeepLearning_202630/commit/e8485812241e492f683f0e6ea77459ca3423f30c).
- [Plan y fuentes estadísticos publicados antes de la validación nueva](https://github.com/jassan-bit/DeepLearning_202630/commit/7d304faa0dd583c6109a688ff179ec7e937f6561).
- Los planes son anteriores a sus cálculos nuevos; no convierten calendarios expuestos en conjuntos independientes ni son un protocolo confirmatorio de mercado.

Desde el repositorio completo:

```bash
export PYTHONDONTWRITEBYTECODE=1
PKG=experiments/e3_statistics_remediation_20261121
python "$PKG/bundle_tools/verify_package.py" --packet "$PKG"
python "$PKG/retrospective/audit_v4_saved.py" \
  --input-root experiments/e3_next_evidence_20261121/development/cpu_pilot_results \
  --out "$PKG/retrospective/results" --verify-only
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python "$PKG/calibration/run_validation.py" --root "$PKG/calibration"
```

Estos comandos verifican artefactos y preflight, sin repetir la validación Monte Carlo. Los generadores, requisitos y comandos de reproducción están en cada componente; una repetición deliberada debe usar un directorio nuevo y conservar su identidad de reproducción.

## Conservación y pendientes

Se conservan **796 trabajos aditivos, 30 ajustes V3, 51 trayectorias/20.400 predicciones retrospectivas y 20 ajustes/1.210 predicciones V4**. Esta operación añade **cero entrenamientos de pronóstico, forwards o predicciones de mercado**. No promueve una ablación, no inicia V5 y no cambia el abandono V4 o el NO-GO V3.

Render, los ONNX, el DOI y los recibos humanos automáticos anteriores conservan sus versiones. La reproducción humana auténtica y R2 siguen requiriendo sus actuaciones. La sustentación **4,5** se registra como dato proporcionado por el autor, sin afirmar una defensa observada.

[Acciones concretas](NEXT_ACTIONS.md) · [Valoración orientativa del componente del 15 %](COMPONENT_ASSESSMENT.md) · [Acta posterior de publicación](https://github.com/jassan-bit/DeepLearning_202630/tree/e3-statistical-remediation-publication-d6cf7f1/proof).

**R3 no se acredita.**
