# Propuesta: cabeza de momentos para CNN-LSTM con atención

**16 de noviembre de 2026. Decisión: conservar para revisión matemática y de admisibilidad; no promover a modelo original ni iniciar otro piloto.** La base sigue siendo CNN causal → LSTM unidireccional → atención temporal por contenido. Se propone cambiar la cabeza de salida y su criterio de aprendizaje. No se han implementado ni entrenado redes nuevas en este corte.

## Problema concreto y procedencia

El [cierre v2](../e3_cierre_v2_20261116/README.md) dejó como mejor referencia de desarrollo a `additive_positive_head`, RMSE **0,0031446148**. La receta half-t5 con MSE y NLL obtuvo **0,0031982970** y no superó la heurística de continuación. Ese resultado no identifica una causa ni demuestra que Student describa mal el mercado.

La limitación que se estudia es más precisa: una escala half-t5 impone simultáneamente `m=E[U²]=5s²/3`, `q=E[U⁴]=25s⁴=9m²` y una media transformada. Bajo mala especificación, el óptimo de su contraste NLL puede diferir del óptimo de MSE de σ₇. La escala libre sí puede representar las medias puntuales del problema; no se afirma una falta universal de capacidad para la media. El conflicto entre NLL y media ya tiene antecedentes, especialmente **Faithful**.

El [contraejemplo demostrado](theory.md) usa `P(U=0)=3/4`, `P(U=2)=1/4`, `A=0`: el óptimo MSE tiene `s²=9π²/320≈0,277583`; el óptimo NLL, `s²=0,4`. La derivada NLL en el óptimo de media es distinta de cero. La verificación incluye además una ley exponencial continua. Estos ejemplos prueban una posibilidad poblacional; **no diagnostican la causa del resultado de BTC**. La MSE poblacional mínima conserva su varianza irreducible y no es cero.

El [paper guía sigue siendo TimeMixer](paper_guide.md). El defecto de calendario medido en ETTh1 no se transfiere al adaptador BTC sin calendario. La guía permite justificar otra limitación con evidencia equivalente: aquí el origen es el acoplamiento de la cabeza v2 y sus antecedentes, no un defecto nuevo atribuido a TimeMixer.

## Mecanismo propuesto

Se conserva D7×5 OHLCV y el pronóstico de un día de la desviación poblacional de siete retornos logarítmicos, sin anualizar. Con los seis retornos conocidos al origen,

\[
A=\frac17\sum_{i=t-6}^{t-1}(r_i-\bar r_6)^2,\quad
U=|r_t-\bar r_6|,\quad V=U^2,\quad
Y=\sigma_7(t)=\sqrt{A+\tfrac6{49}V}.
\]

`A` es observable; `U`, `V` y `V²` son **etiquetas**, nunca entradas. Sean `c=6/49`, `m≈E[V|X]` y `q≈E[V²|X]`. Tres lecturas del mismo contexto producen

\[
m=s_u^2\operatorname{softplus}(a),\quad q=m^2(1+b^2),\quad
\rho=\frac{d^2}{1+d^2}.
\]

La salida se construye como

\[
L=\sqrt A+\frac{cm}{\sqrt{A+cq/m}+\sqrt A},\quad
H=\sqrt{A+cm},\quad \widehat\mu=L+\rho(H-L).
\]

Con los **momentos verdaderos**, la media condicional está en `[L,H)` cuando `q>m²`; `L` se alcanza y `H` es un supremo no alcanzado. Si `q=m²`, ambas cotas coinciden. El [teorema, la demostración y los bordes](theory.md) están documentados. Con momentos aprendidos sólo se garantiza pertenencia a la región calculada: puede no contener la media real. La envolvente **no es un intervalo de predicción**, de confianza ni de cobertura, y no impone un máximo a cada observación futura.

La propuesta inicializa `b=√8` y `d=1` (`q/m²=9`, `rho=1/2`), como ancla matemática respecto de t5, sin búsqueda por desempeño. Los pesos de esas lecturas deberían empezar en cero para producir esos valores constantes. `b=0` y `d=0` tienen derivadas nulas; el signo es no identificable y hay límites de redondeo/overflow. La inicialización no garantiza convergencia. La lectura de `a` partiría de la media de `V` de ajuste, sin usar validación. Estas son **decisiones propuestas**, aún sin ejecución.

## Pérdida y propiedad comprobable

Las escalas fijas se obtendrían únicamente de ajuste: `s_u=media(U)`, `s_y=media(Y)`, positivas. Con `Ṽ=V/s_u²`, `m̃=m/s_u²`, `q̃=q/s_u⁴`,

\[
\mathcal L=((Y-\widehat\mu)/s_y)^2
+\lambda_m[\log\widetilde m+\widetilde V/\widetilde m]
+\lambda_q[\log\widetilde q+\widetilde V^2/\widetilde q].
\]

Se proponen `lambda_m=lambda_q=0,1` para una eventual especificación inicial; no se probaron pesos alternativos ni se congeló un protocolo. Los scores quasi-Gamma/QLIKE elicitan medias y son conocidos; no suponen una distribución Gamma ni constituyen una densidad conjunta.

La hipótesis matemática es que, si `m0>0`, `E[U⁴|X]<∞` y la clase representa conjuntamente los momentos y la media verdaderos, éstos minimizan a la vez los tres componentes del riesgo condicional. La demostración expresa el exceso como suma de términos no negativos. No exige `E[U⁸]` para definir la esperanza del score, pero esa octava potencia puede ser necesaria para varianza finita de gradientes. No se prueban robustez, convergencia de SGD, capacidad universal del backbone ni fidelidad de trayectoria de entrenamiento de Faithful. El caso verdadero `m0=q0=0` queda **excluido del teorema de pérdida**: requeriría una rama separada; agregar un epsilon cambiaría la afirmación exacta.

La fórmula admite otros problemas con `Y=√(A+cV)`, `A≥0`, `c>0`, `V≥0`; para una ventana de `n` retornos, `c=(n−1)/n²`. Esa transferibilidad algebraica no acredita por sí sola originalidad.

## Aporte potencial y dictamen de novedad

La [matriz de nueve antecedentes](literature/review.md) distingue ocho lecturas focales de cuerpo y una de metadatos/resumen. La cota inferior se especializa directamente de **Merhav (2023), ec. (8)**; los scores tienen antecedente en **Gneiting (2011)**. Cabezas de momentos, propagación no lineal y guardas de salida también son conocidas. **No se reclama una desigualdad nueva.**

El aporte a examinar sería la adaptación algorítmica conjunta de momentos de la innovación y media transformada admisible dentro de esta red. No encontrar su combinación exacta en una búsqueda focal no prueba ausencia de antecedentes ni sustantividad. Las doce consultas web y tres intentos de APIs [registrados](native_search/summary.md) no completan la revisión nativa exigida. Los tres intentos fallaron en el túnel antes de respuesta bibliográfica; los resultados son desconocidos, no cero. Faltan búsqueda por citas, repetición antes de entrega y verificación por una persona externa al diseño. Los agentes de IA no sustituyen esa persona.

**R1 pendiente; R2 no acreditado para esta cabeza aún no implementada; R3 no evaluado.** La procedencia y el uso de IA se declaran en [provenance.md](provenance.md). Si el análisis cercano no permite defender una adaptación sustantiva, se descarta su pretensión de originalidad, aunque el método pudiera funcionar.

## Controles y falsación para una evaluación futura

No basta vencer a la receta v2 fallida. La hipótesis predictiva propuesta es que el acoplamiento de momentos reduzca RMSE de σ₇ respecto de una cabeza directa con **los mismos tres readouts activos y los mismos scores auxiliares**, sin deterioro relevante en extremos. Sólo un futuro diseño justo puede contrastarla.

| Control propuesto | Qué aislaría | Capacidad prevista |
|---|---|---:|
| Cabeza aditiva v2 congelada | Referencia de desarrollo más fuerte observada. | 1210 |
| Media directa `√A+s_y softplus(r)` + las mismas cabezas `m,q` y scores | Aporte de la envolvente; las tres lecturas son activas. | **1236**, igual a propuesta |
| Control inspirado en Faithful | Separar la supervisión de la media del gradiente auxiliar al tronco; aux cabezas reciben contexto con `stop-gradient`. | 1236 si usa las mismas tres lecturas |
| Plug-in Jensen `H(A,m)` | Necesidad de corregir la transformación no lineal mediante `q,rho`. | 1210 con una lectura activa |
| Cabeza de momentos con `q=9m²` fijo y `rho` aprendido | Necesidad de un segundo momento libre. | 1223 con dos lecturas activas |
| Half-t5 media sola / media+NLL; M00 proyectado | Continuidad con resultados anteriores. | Capacidad y recetas originales documentadas |

El control inspirado en Faithful requiere especificar su adaptación, sus condiciones de optimizador y la comparación exacta antes de invocar su teorema; un detach arbitrario no hereda la garantía. No se añaden parámetros inactivos para simular igualdad de capacidad. Las ablaciones con distinta capacidad la declaran. Las ejecuciones nuevas deberían tener presupuesto total comparable, mismos datos, inicialización del tronco y selección de época, sin reajustar los rivales congelados de E2.

La propuesta sumaría **26 parámetros**, de 1210 a **1236**: tronco/atención 1197 más tres lecturas de 13. Las lecturas y álgebra se proponen FP64; el codificador y la atención permanecen FP32. El conteo es analítico, no la inspección de una red ya implementada.

La hipótesis se considera sin respaldo si la cabeza directa equiparada o el control inspirado en Faithful igualan o mejoran su RMSE, o si quitar `q`/envolvente no perjudica el desempeño bajo el protocolo fijado. Un resultado impreciso se reportaría como no concluyente. Una mejora global acompañada de deterioro en extremos obliga a revisar la explicación. Los grupos por objetivo realizado son diagnósticos retrospectivos, nunca entradas ni selectores operativos.

No se eligen hoy nuevas particiones, semillas finales o umbrales a partir de resultados nuevos. El **margen confirmatorio heredado `δ=0,00025772842664981106` no cambia**. No se reaplica automáticamente el 2 % del piloto v2 como prueba de R3. Siguen pendientes potencia, mínimo de semillas según la guía, familia completa Top 10 y baselines fuertes, tres conjuntos adicionales elegidos previamente y revisión humana.

## Evidencia producida y reproducción

El [verificador escalar](../../scripts/e3_verify_moment_head.py) integra cinco leyes artificiales: uniforme, exponencial, dos puntos, lognormal y Pareto `α=5`. Comprueba 20 combinaciones de ley y `A`, extremales, homogeneidad, derivadas y exceso de riesgo. Optimiza riesgos poblacionales **sintéticos escalares**; no entrena redes. El [JSON](verification/metrics.json) y la [figura](verification/moment_envelope.svg) distinguen estas operaciones de datos de mercado. Las [pruebas](../../tests/test_moment_head.py) cubren bordes, integrales, conflicto de escala y contratos de reproducción.

```bash
python -m pip install -r requirements-e3-close-cpu.lock.txt
python -m pytest tests/test_moment_head.py -q
python scripts/e3_verify_moment_head.py --check
```

La copia descargable del libro se verifica contra los archivos maestros en CI. La versión exacta de Python que generó el JSON se conserva como procedencia; `--check` admite otro parche de Python 3.12 y mantiene tolerancias numéricas explícitas.

La [bitácora](design_log.md) conserva las correcciones de formulación. El total de recetas neuronales entrenadas en mercado sigue siendo **cinco** entre v1 y v2; esta propuesta no es una sexta. No se abrieron NPZ ni checkpoints ni se realizaron nuevas inferencias de mercado. El consumo de desarrollo se mantiene en **556,212659362 de 1200 segundos**, con **643,787340638 segundos** aritméticamente disponibles; el saldo no inicia una ejecución ni habilita otra campaña por sí solo.

El siguiente paso es resolver la admisibilidad y completar la revisión bibliográfica externa, luego cerrar controles, presupuesto y protocolo antes de implementar una eventual prueba. La cabeza aditiva v2 conserva su papel de referencia mientras tanto.
