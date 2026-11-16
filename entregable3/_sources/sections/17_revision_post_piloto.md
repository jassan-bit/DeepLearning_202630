# Revisión posterior al primer piloto

Este corte del **16 de noviembre de 2026** incorpora el primer entrenamiento
real del candidato. Continúa siendo desarrollo exploratorio. Los documentos
anteriores conservan su cronología; la selección final y la verificación de
novedad permanecen abiertas.

## Resultado real y trazabilidad

La corrida
[`9e40ce379a7b4ee1a3e2fb306071cdcb`](https://github.com/jassan-bit/DeepLearning_202630/tree/f7826d120f3a6eb8e9a0952dd9842cd77f152f34/runs/desarrollo_v1)
se ejecutó en una NVIDIA A100-SXM4 de 40 GB, con Python 3.12.12 y
PyTorch 2.8.0+cu126. Comparó `full` con `M00`, que elimina conjuntamente la
penalización de discordancia y la saturación radial. Usó 532 ejemplos de ajuste,
siete fechas purgadas y 116 de validación interna; una sola semilla, 20261116.

Ambos modelos alcanzaron el máximo de 40 épocas. El mejor RMSE de `full`
corresponde a la época 35; el de `M00`, a la 39. El tiempo conjunto registrado
fue **369.48 segundos** del límite de 1200. No se detuvieron por agotamiento
del presupuesto ni por la paciencia de ocho épocas.

`full` tuvo **12.43 % más RMSE** que `M00`. La validación sirvió para seleccionar
época y luego examinar el resultado; esta diferencia no es una prueba estadística
ni una comparación independiente contra el Top 10 de E2.

## Una limitación medible: predicciones incompatibles con la historia

La entrada contiene siete cierres y, por tanto, seis retornos conocidos. Para
su media $m_6$, su suma de desviaciones cuadradas $S_6$ y el retorno futuro $x$,
la definición de la variable objetivo implica

$$
\sigma_7(t)^2=\frac{S_6}{7}+\frac{6}{49}(x-m_6)^2,
\qquad
\sigma_7(t)\geq \ell_t=\sqrt{S_6/7}.
$$

El piso $\ell_t$ usa únicamente datos disponibles al pronosticar. Es una
especialización de la actualización clásica de la dispersión de
[Chan, Golub y LeVeque (1983)](https://doi.org/10.1080/00031305.1983.10483115).
La [derivación y sus supuestos](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_revision_post_piloto_20261116/theory.md)
se conservan en el repositorio.

En estas 116 fechas, ninguna etiqueta viola el piso. Sin embargo, **ambas redes
lo incumplen en 32 predicciones (27.59 %)**. Esos casos concentran 50.06 % del
error cuadrático total de `full` y 54.12 % del de `M00`. Esto identifica un fallo
de coherencia de sus salidas; no permite atribuirlo a un componente concreto.

## Diagnóstico posterior sin nuevos entrenamientos

Si la etiqueta satisface $y\geq\ell_t$, reemplazar una predicción por
$\max(\widehat y,\ell_t)$ no aumenta su error absoluto ni cuadrático.
Se aplicó la misma regla a los dos modelos, sobre sus checkpoints ya seleccionados.
El piso solo se incluye como referencia analítica.

```{csv-table} Métricas reproducidas sobre las mismas 116 fechas de desarrollo
:file: ../../experiments/e3_revision_post_piloto_20261116/audit/table.csv
:header-rows: 1
```

Las proyecciones y la referencia analítica se examinaron **después de observar
el piloto**. Son diagnósticos exploratorios: no sustituyen las métricas originales,
no incorporan nuevas redes entrenadas y no acreditan superioridad confirmatoria.
`M00` mantiene menor error que `full` también tras aplicarles la misma proyección.

La mediana de la razón realizada entre la parte conocida $S_6/7$ y
$\sigma_7(t)^2$ es 0.91333 en esta cohorte. Esa razón contiene el objetivo futuro
en el denominador: **no es un $R^2$, varianza explicada ni predictibilidad general**.

## Qué aporta la revisión de fuentes

La [matriz de nueve fuentes](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_revision_post_piloto_20261116/literature/review_matrix.csv)
explicita los objetivos, rezagos y alcances de lectura. Los antecedentes impiden
presentar esta corrección sencilla como una novedad ya demostrada:

- [ConstraintNet (2021)](https://cdn.aaai.org/ojs/16841/16841-13-20335-1-2-20210518.pdf)
  ya incluye restricciones por muestra y una salida escalar sobre un umbral.
- [Spulbar y Ene (2026)](https://doi.org/10.3390/math14203661) separan la parte
  observada y la futura de una ventana móvil financiera, con un objetivo distinto.
- [HARNet](https://arxiv.org/html/2205.07719),
  [GARCH-NN/GARCH-LSTM](https://arxiv.org/html/2402.06642) y
  [DeepVol](https://arxiv.org/html/2210.04797) ofrecen mecanismos documentados,
  pero sus rezagos, datos intradía o definiciones de volatilidad difieren de D7×5→σ7.

Por ello, el siguiente diseño debe justificar una modificación algorítmica
concreta frente a los antecedentes, además de respetar la parte conocida del
objetivo y aprender la innovación futura. Una cabeza positiva, un piso o trasladar
un método a BTC no bastan por sí solos para sustentar R1. Tampoco se supone que
una corrección residual mejorará el pronóstico por el hecho de simplificar el objetivo.

La revisión fue focal, apoyada en fuentes originales y consultas web registradas.
Los intentos de acceso a tres API bibliográficas no produjeron búsquedas utilizables.
La búsqueda sistemática completa y la verificación humana externa siguen pendientes;
no se declara ausencia de antecedentes.

## Decisión y reproducción

La [decisión de continuación](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_revision_post_piloto_20261116/decision.md)
conserva el resultado adverso y prioriza el estudio de coherencia e innovación.
El modelo final permanece pendiente de selección. La
[bitácora acumulada](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_desarrollo_v1/design_log.md)
registra cinco estructuras examinadas como software, dos variantes evaluadas
con datos de mercado y tres diagnósticos analíticos posteriores sin ajuste.

La auditoría se reproduce desde la raíz del repositorio con:

```bash
python scripts/e3_audit_development.py --check
python -m pytest tests/test_development_audit.py -q
```

Los CSV originales y su resumen se verifican contra el manifiesto inmutable de
GitHub. El NPZ se verifica por SHA-256 y sólo se abren sus tres arreglos originales
de entrenamiento. No se accede a calibración ni evaluación. Los cuadros del
capítulo se generan desde la misma tabla del diagnóstico.

El trabajo se elaboró con asistencia de IA. Las revisiones internas no se
presentan como reproducción humana independiente ni verificación externa de novedad.
