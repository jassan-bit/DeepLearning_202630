# SE-OI V5: antecedentes, modificación y contribución acotada

Fecha de preparación: **21 de noviembre de2026, America/Bogota**. Objeto: `seoi_mse`, una propuesta nueva de desarrollo denominada **Scale-Equivariant Overlap-Innovation (SE-OI V5)**. La asistencia de IA intervino en formulación, búsqueda bibliográfica focal, código, pruebas y redacción. Este documento no es un dictamen docente ni una acreditación de R1, R2 o R3.

## 1. Problema e hipótesis que sí se pueden sostener

El piloto V3 `atom_crps` fue2,7043% peor en RMSE que su control `atom_mse_only`; la receta V4 sigma-CRPS fue1,4433% peor que su mejor control CPU. Se conservan ambas decisiones adversas y las cohortes por separado. Las mediciones posteriores de gradientes mostraron interacción heterogénea, con predominio de alineación; **no identificaron CRPS, conflicto de gradientes o selección temporal como causa única del deterioro**.

La propuesta cambia una limitación arquitectónica comprobable: una representación y una escala de innovación fijadas por estadísticas globales no imponen por construcción que multiplicar los retornos por una constante positiva multiplique exactamente la predicción de volatilidad. SE-OI impone esa coherencia dimensional y acopla la normalización causal a la parte de la etiqueta que todavía no es observable. La plausibilidad es que el encoder pueda reutilizar formas de ventanas a distintas escalas. Es una **hipótesis inductiva**, no una explicación causal ya encontrada ni una garantía de utilidad.

La candidata se entrena con MSE porque el objetivo primario del piloto es RMSE. Su novedad propuesta no consiste en retirar CRPS, cambiar lambda o promover la ablación ganadora de V3: reside en la representación y en la restitución de unidades dentro de la lectura estructural. Los pesos anteriores no se transfieren ni se modifican.

## 2. Construcción específica

Para los seis retornos de cierre conocidos al emitir el pronóstico, sean

\[
\bar r_{6,t}=\frac16\sum_{i=1}^{6}r_{t,i},\quad
A_t=\frac17\sum_{i=1}^{6}(r_{t,i}-\bar r_{6,t})^2,\quad
U_t=|r_{t+1}-\bar r_{6,t}|,\quad c=6/49.
\]

`U_t` sólo se usa como etiqueta cuando el retorno siguiente ya pertenece a la partición de ajuste. La identidad de la volatilidad poblacional de siete retornos es

\[
Y_t=T_{A_t}(U_t)=\sqrt{A_t+cU_t^2}.
\]

La escala causal es

\[
s_t=\max\left\{\sqrt{\frac16\sum_{i=1}^{6}r_{t,i}^2},\;10^{-6}r_{\mathrm{fit}}\right\},
\]

donde `r_fit>0` es el RMS de las seis entradas de retorno de todas las ventanas de ajuste, agrupadas sin utilizar validación o evaluación. `10^-6` es adimensional. No se usa el retorno futuro para normalizar la ventana.

Para cada OHLC en la ventana se calcula el log-precio relativo al último cierre conocido, dividido por `s_t` en la candidata. El uso de log-OHLC relativo al último cierre **ya existía en el cargador V3**; no se presenta como cambio nuevo de coordenadas. Es verificable en `load_innovation_data`, líneas132–143 del `src/e3_innovation_model.py` histórico conservado en `experiments/e3_next_evidence_20261121/development/pilot_sources/`. El volumen conserva el tratamiento común `log1p(V)` y estandarización estimada sólo en ajuste; no se trata como una magnitud de retorno. El encoder aprende masas `p_k` y átomos adimensionales `a_k>=0`, con tres átomos. La restitución de unidades es

\[
\gamma_{\mathrm{fit}}=\frac{\operatorname{mean}_{\mathrm{fit}}(U_t)}{r_{\mathrm{fit}}},\qquad
u_{t,k}=s_t\gamma_{\mathrm{fit}}a_{t,k},\qquad
\hat Y_t=\sum_{k=1}^{3}p_{t,k}\sqrt{A_t+c u_{t,k}^{2}}.
\]

Los factores `mean_fit(U)` y `r_fit` se estiman una sola vez en ajuste y son comunes a todas las recetas. En el código se utilizan `u_scale=mean_fit(U)` y `scale=s_t/r_fit`, de modo que `u=u_scale*a*scale` si hay restitución local, y `u=u_scale*a` en el control global. No se introduce un promedio `mean(U/s)` específico por receta: confundiría el factorial. La entrada global divide el log-precio relativo por `r_fit` antes de la estandarización ajustada, lo cual cancela esa constante y conserva la representación global anterior salvo aritmética finita. El MSE se divide por `mean_fit(Y)^2`, común a todas las recetas. No se afirma que la distribución real sea de tres átomos. La geometría de cabeza es FP64, el encoder FP32 y las recetas atómicas tienen1262parámetros; la aditiva conserva1210. Son diferencias explícitas, no una igualación ficticia de capacidad.

## 3. Antecedentes próximos y consulta efectivamente realizada

La búsqueda fue focal y no exhaustiva. No demuestra prioridad mundial ni ausencia de trabajos equivalentes. No se copió código de los repositorios de estos autores; cualquier copia futura requiere su propia comprobación de licencia y versión.

| Fuente primaria | Qué se consultó | Qué ya existía | Diferencia propuesta y límite |
|---|---|---|---|
| Kim, T.; Kim, J.; Tae, Y.; Park, C.; Choi, J.-H.; Choo, J. (2022). *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*. ICLR2022. [Registro](https://openreview.net/forum?id=cGDAkQo1C0p). | PDF primario de25páginas recuperado mediante [espejo de referencia OpenReview](https://openreview.net/references/pdf?id=SRgrKlAhbc); método§3.1 y contexto. Los enlaces canónicos pidieron verificación de navegador y no se cuentan como lectura adicional. | Normalización por instancia, transformación afín e inversión de la escala a la salida. | SE-OI restaura unidades de la innovación dentro de `sqrt(A+cU²)`, no invierte una predicción de precio. Normalizar y desnormalizar, por sí solos, no constituyen la contribución. |
| Passalis, N.; Tefas, A.; Kanniainen, J.; Gabbouj, M.; Iosifidis, A. (2020). *Deep Adaptive Input Normalization for Time Series Forecasting*. IEEE Transactions on Neural Networks and Learning Systems31(9),3760–3765. [DOI10.1109/TNNLS.2019.2944933](https://doi.org/10.1109/TNNLS.2019.2944933). | [Preprint primario completo](https://arxiv.org/pdf/1902.07892), versión2,7páginas; introducción y§II. Datos bibliográficos contrastados con el registro universitario del artículo. | Desplazamiento, escala y gating aprendidos a partir de cada entrada; aplicación financiera a libros de órdenes. | SE-OI emplea una escala causal analítica compartida con la lectura de la etiqueta solapada. No se atribuye novedad a aplicar normalización en finanzas. |
| Liu, Y.; Wu, H.; Wang, J.; Long, M. (2022). *Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting*. Advances in Neural Information Processing Systems35. [Registro](https://papers.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html). | [PDF primario completo](https://proceedings.neurips.cc/paper_files/paper/2022/file/4054556fcaa934b0bf76da52cf4f92cb-Paper-Conference.pdf),13páginas;§§2.2,3.1 y arquitectura. | Stationarization, restitución de estadísticas y atención que reincorpora información eliminada. | SE-OI no propone de-stationary attention. Debe reconocer que su restricción de escala puede eliminar información útil de régimen; la equivariancia no prueba generalización. |

El mecanismo atómico y la transformación exacta de la etiqueta proceden del desarrollo V3 documentado; V5 conserva esa atribución y cambia su parametrización. Aprender masas/soportes y puntuar una distribución con CRPS también tienen antecedentes previos: Yang et al.(2019), *Fully Parameterized Quantile Function for Distributional Reinforcement Learning*, NeurIPS32, [fuente primaria](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f471223d1a1614b58a7dc45c9d01df19-Abstract.html), cuyo resumen primario se consultó; y Gneiting y Raftery(2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA102(477),359–378, [DOI10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437), cuya identidad y propiedad CRPS se consultaron en el PDF universitario. La candidata primaria no utiliza el auxiliar CRPS; se mantiene como control de pérdida.

## 4. Propiedad formal pertinente

La propiedad distintiva se formula para **pesos y referencias de entrenamiento fijos**: transformar sólo la historia disponible de una ventana como `l'=b+k*l`, `k>0`, implica `r'=k*r` y `A'=k²*A`. Si el piso no se activa ni antes ni después, `s_t'=k*s_t`; por tanto, la entrada local de precios no cambia. El volumen y su tratamiento tampoco cambian. Masas y átomos adimensionales quedan iguales y `u'=k*u`, pues `gamma_fit` se conserva. En consecuencia,

\[
\hat Y_t'=
\sum_kp_{t,k}\sqrt{k^2A_t+c(k s_t\gamma_{\mathrm{fit}}a_{t,k})^2}
=k\hat Y_t.
\]

La lectura también cumple `Yhat>=sqrt(A)` y es la esperanza exacta bajo la ley finita programada. Esta última propiedad ya estaba en V3 y no se reclama como nueva. Un piso físico fijado no puede tratarse como coescalado sin hacerlo realmente.

Como comprobación dimensional adicional, si se transforman coherentemente **todas** las ventanas de historia/ajuste y también `r_fit`, `mean_fit(U)` y `mean_fit(Y)`, la homogeneidad del máximo da `s_t'=k*s_t` incluso con piso activo y conserva `gamma_fit`. Esta variante conjunta también puede cumplirse en los controles globales correctamente reescalados; **no es una propiedad exclusiva de V5 ni evidencia de mejora sobre ellos**. La restricción útil que se ensaya es la respuesta local con el estado entrenado fijado.

Supuestos: datos positivos para log-precios; `r_fit>0`, escalas ajustadas positivas y finitas; `k>0`; misma convención OHLC y las mismas transformaciones dimensionales. Se deben rechazar entradas degeneradas en lugar de introducir una constante física que invalide la prueba. La verificación numérica comprueba tolerancias de aritmética finita, no exactitud simbólica del computador.

No se demuestra mejora de mercado, calibración, consistencia del estimador, invariancia a cambios arbitrarios de régimen ni corrección del fracaso de V3/V4. Las entradas con idéntica forma normalizada pueden corresponder a leyes futuras diferentes; éste es un riesgo empírico de la propuesta.

## 5. Comparación factorial y decisión falsable

Las seis recetas corresponden a nuevas ejecuciones de una **cohorte CPU nueva y común** cuando el autor corra el piloto. Los controles conservan su identidad funcional anterior cuando sus fórmulas coinciden; no se reutilizan los trabajos, pesos o predicciones anteriores como resultados emparejados nuevos. La equivalencia algebraica de una fórmula de entrada no implica heredar la cohorte GPU V3 ni la cohorte CPU V4.

| Receta | Precios normalizados por escala local | Restitución local de átomos | Objetivo | Papel |
|---|---:|---:|---|---|
| `seoi_mse` | Sí | Sí | MSE | Candidata primaria V5. |
| `input_only_mse` | Sí | No | MSE | Aísla normalización de entrada. |
| `output_only_mse` | No | Sí | MSE | Aísla restitución local. |
| `global_mse` | No | No | MSE | Base global del factorial; identidad de control previa, sin promoción. |
| `global_crps` | No | No | MSE+0,1CRPS_U | Control de pérdida global. |
| `additive_mse` | No | No | MSE | Referencia aditiva de1210parámetros. |

Cinco semillas por receta son **30trabajos planificados**, no30diseños ni una potencia demostrada. La regla previa exige todos los trabajos completos, **mejora relativa de RMSE media de al menos2% frente al mejor control**, victoria en al menos4/5semillas emparejadas frente al control neuronal de menor RMSE media y MAE no mayor de1,05veces el MAE **del control neuronal seleccionado por su mejor RMSE medio**, no el menor MAE entre controles. Debe aplicarse la traducción numérica exacta del protocolo; si faltan resultados o hay fallos, no se declara continuación favorable. El2% es una decisión interna de desarrollo, no el umbral universal de R3. No se modifica después de conocer los resultados.

Las purgas de desarrollo suman **14fechas**: siete entre ajuste terminado el30de junio y selección iniciada el8de julio; otras siete entre selección terminada el30de agosto y evaluación de desarrollo iniciada el7de septiembre. Un contador `selection_assessment_purge_count=7` sólo describe el segundo intervalo, no las14fechas totales. El calendario de evaluación sigue expuesto y no se convierte en prueba confirmatoria por esta nueva partición operativa.

Un desenlace favorable sólo apoyaría estudiar continuidad y dimensionar una confirmación con la familia completa. Uno adverso conserva íntegramente la receta, controles y decisión; no promueve otra receta por ranking. Los tests sintéticos/de software no cuentan como ajustes reales ni como confirmación.

## 6. Declaración precisa de contribución

**Se propone y programa una adaptación de normalización causal y restitución de unidades para una ley atómica del componente desconocido de una volatilidad móvil solapada. La escala se inserta dentro de la transformación estructural dependiente de la historia, imponiendo equivariancia positiva en el mapa de predicción bajo las condiciones enunciadas.** La comparación factorial puede comprobar si esta restricción aporta utilidad sobre sus controles. La búsqueda y la propiedad sostienen una tesis técnica acotada que debe revisarse como V5; no acreditan automáticamente R1 o superioridad R3.

Las alternativas Faithful y agregación temporal se consideraron el21de noviembre y no se implementaron en esta propuesta. Faithful tiene antecedentes explícitos en Stirn et al.(2023), [AISTATS/PMLR206](https://proceedings.mlr.press/v206/stirn23a.html); la igualdad de su trayectoria media con su propio control MSE no constituye ganancia RMSE. La agregación causal de expertos tiene antecedentes financieros y no se presenta como una invención no explorada. Estas decisiones de diseño se conservan como exploración de hoy, no como preregistros anteriores a los pilotos históricos.
