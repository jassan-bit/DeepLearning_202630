# 6 Bloque C El modelo original

El trabajo actual es un **candidato neuronal de desarrollo**. Dispone de formulación, implementación asistida por IA y una propiedad demostrada de la cabeza latente. No se presenta como modelo con novedad o superioridad ya acreditadas.

## 6.1 Relación con el paper guía de E2

La continuidad elegida es CNN causal → LSTM unidireccional → atención temporal para BTCUSDT, D7×5 y horizonte de un día. El mecanismo candidato penaliza discordancia respecto a un promedio causal y acota radialmente los valores agregados.

La limitación de calendario de TimeMixer medida en ETTh1 no se traslada automáticamente al adaptador BTC sin esos marcadores. El estrés artificial de E2 motiva investigar sensibilidad, pero puede romper coherencia OHLCV y no prueba que la regla nueva resuelva un defecto económico real. La [sección 7.1](07_diseno_riguroso.md) relaciona esa evidencia con una hipótesis falsable y sus límites.

## 6.2 Condiciones de originalidad y admisibilidad

| Condición | Evidencia actual | Estado que debe conservarse |
|---|---|---|
| R1 Novedad para el fenómeno y la tarea | Regla exacta y matriz de siete trabajos cercanos. | Pendiente de búsqueda efectiva, equivalencias, originales adicionales y revisión humana externa. |
| R2 Núcleo algorítmico propio y defendible | `src/original_model.py`, escrito desde la formulación con primitivas PyTorch; pruebas y asistencia de IA declarada. | La existencia del archivo no acredita por sí sola comprensión, autoría responsable y defensa del estudiante. |
| R3 Superioridad estadísticamente verificada | No hay evaluación confirmatoria del candidato sobre mercado. | Pendiente de campaña justa, potencia, familia completa, pruebas y conjuntos adicionales. |

CNN, LSTM, atención por contenido, consultas aprendidas y saturación no se reclaman como invenciones. La contribución candidata debe diferir sustantivamente en mecanismo, pérdida o entrenamiento; cambiar datos o hiperparámetros no basta. Las tres ablaciones funcionales no se convierten en tres innovaciones.

## 6.3 Revisión de literatura y evidencia de novedad


Se examinaron focalmente los originales de [Kim, Papamakarios y Mnih (2021)](https://proceedings.mlr.press/v139/kim21i.html) y [Dasoulas, Scaman y Virmaux (2021)](https://proceedings.mlr.press/v139/dasoulas21a.html). Ambos son antecedentes directos de atención y propiedades Lipschitz. Una consulta aprendida y una cota, por separado, ya tienen antecedentes.

Las seis cadenas bilingües generaron 18 solicitudes a OpenAlex, Crossref y arXiv. **Las 18 fueron denegadas por la política de red antes de ejecutar las búsquedas.** El número de resultados es desconocido, no cero. Las ocho consultas GitHub y el examen de tres exportaciones heredadas no sustituyen la búsqueda exigida en tres bases. Tampoco se presenta un subagente de IA como la persona externa requerida.

| Trabajo cercano | Relación que debe distinguirse |
|---|---|
| Kim et al., *The Lipschitz Constant of Self-Attention* | Atención L2 entre pares y estabilidad: no toda atención se vuelve una contribución nueva por tener cota. |
| Dasoulas et al., *Lipschitz Normalization* | Consulta aprendida y normalización de scores; el candidato acota valores y añade discordancia, pero su novedad exacta está por verificar. |
| TimeMixer | El procesamiento multiescala ya tiene antecedentes. La diferencia debe ser de mecanismo. |
| Luong et al., atención en traducción | La atención por contenido es un componente convencional. |
| Peng et al., CNN-LSTM con atención para tendencia cripto | La familia híbrida y su uso en criptomonedas no se pueden presentar como inéditos. Su objetivo no se equipara a σ₇ sin revisar el original. |
| CEEMDAN-RF-LSTM para volatilidad realizada | Separar mecanismo y definición del objetivo; no comparar métricas de tareas distintas como si fueran homogéneas. |
| HARNet | La causalidad y los inductive biases para volatilidad ya tienen precedentes. |

La saturación $T_B$ es también el gradiente de $\phi_B(h)=B^2(\sqrt{1+\|h\|^2/B^2}-1)$, de forma pseudo-Huber radial. Esta identidad impide presentarla aislada como una función inventada aquí. No se atribuye una fuente histórica primaria no verificada. El [registro completo](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/e3_bloque_c_20261115/literature) distingue lecturas nuevas, heredadas y sólo metadatos. **No se emite aún una declaración de novedad.**



## 6.4 Ciclo de desarrollo

| Etapa | Evidencia disponible | Condición siguiente |
|---|---|---|
| Revisión y hueco | Antecedentes y limitación exploratoria delimitados. | Completar R1 antes de una declaración de novedad. |
| Formulación | Regla, derivadas, pseudocódigo y cota latente. | Comprobar utilidad y límites en desarrollo real. |
| Implementación | Encoder y cabeza desde primitivas; configuración guardable. | Integrar el ejecutor de entrenamiento común sin cambiar los rivales congelados. |
| Pruebas | 35 casos de software sobre datos artificiales. | Estas pruebas no sustituyen HPO, ablación de desempeño o comparación de mercado. |
| Diseño y confirmación | Bitácora y estados abiertos documentados. | Fijar datos, presupuesto, piloto, semillas y protocolo antes de los experimentos definitivos. |

La [sección 7](07_diseno_riguroso.md) desarrolla las decisiones previas a esa confirmación. No se han elegido recetas mediante resultados de los conjuntos adicionales ni se ha generado un tag de preregistro.

## 6.5 Implementación y formulación del candidato

El encoder usa convolución con relleno sólo histórico, activación tanh y LSTM unidireccional. Recibe ventanas ya estandarizadas y produce estados $H=(h_1,\ldots,h_7)$. Una matriz fija $P$, independiente de $H$, calcula promedios causales. La cabeza implementa

$$
d_j=\sqrt{\|h_j-(PH)_j\|_2^2+\varepsilon^2},\qquad
a=\operatorname{softmax}\bigl((Hq-\lambda d)/\tau\bigr),
$$

$$
v_j=\frac{Bh_j}{\sqrt{B^2+\|h_j\|_2^2}},\qquad
f(H)=\operatorname{softplus}\left(w^\top\sum_j a_jv_j+b\right).
$$

Los parámetros efectivos satisfacen cotas mediante normalizaciones suaves; no se presume que cualquier peso libre esté acotado sin esa parametrización. La salida es positiva en unidades $\sigma_7/a_{\rm tr}$ y el pipeline invierte la escala del objetivo fuera del módulo. Los detalles, derivadas y algoritmo completos están en la demostración descargable y en la [sección 7.3](07_diseno_riguroso.md).

La configuración inicial tiene **1.211 parámetros**, ocho canales CNN y doce estados LSTM. B=1, Q=1, W=4, Λ=1 y τmín=0,25 son decisiones iniciales de software, no ganadores de HPO. Las tres variantes funcionales eliminan penalización, saturación o contenido; la base sin ambas modificaciones todavía está pendiente, como explica la sección 7.4.

### Pruebas y límites de exportación

Se aprobaron **35 casos pytest**: formas y entradas inválidas, semilla reproducible sin alterar el RNG global, causalidad del encoder, independencia entre muestras, gradientes activos, `gradcheck`, restricciones suaves, ablaciones aisladas y serialización. El sobreajuste diminuto usa ocho ejemplos artificiales y 350 pasos de Adam; no es entrenamiento en Bitcoin.

La ida y vuelta TorchScript se probó con lotes 1, 3 y 4 y tolerancia declarada. El trazado LSTM produce advertencias de comprobaciones de forma tratadas como constantes: hay que aplicar `validate_input` antes de usar una traza. No se entrega ese ensayo como artefacto final entrenado en mercado.

La cabeza usa una multiplicación densa por $P$: su término dominante es $O(n^2d)$, además del encoder. No se declara coste lineal de una implementación de promedio móvil que todavía no se usa. La complejidad asintótica no sustituye FLOPs, memoria y latencias medidos.

### Uso y reproducción

```python
from src.original_model import CandidateConfig, OriginalModel

model = OriginalModel(CandidateConfig(), seed=0)
# x contiene ventanas ya transformadas y estandarizadas de forma lote×7×5.
scaled_prediction = model(x)
```

```bash
python -m pip install -r requirements-e3-original-cpu.lock.txt
python -m pytest tests/test_original_model.py -q
python scripts/e3_verify_original.py --check
```

El workflow CPU conserva esas comprobaciones. Compartir forma y escala con E2 no acredita la integración en el ejecutor común, que sigue pendiente. El autor deberá comprender y modificar el código y la demostración; la asistencia de IA está declarada en las fuentes y la procedencia.

- {download}`Código del candidato <../_static/e3_bloque_c_20261115/original_model.py>`.
- {download}`Pruebas del módulo <../_static/e3_bloque_c_20261115/test_original_model.py>`.
- {download}`Derivación y pseudocódigo completos <../_static/e3_bloque_c_20261115/theory.md>`.
- {download}`Informe de verificación <../_static/e3_bloque_c_20261115/software_verification.json>`.
- {download}`Matriz de antecedentes <../_static/e3_bloque_c_20261115/review_matrix.csv>`.
- {download}`Bitácora de consultas <../_static/e3_bloque_c_20261115/search_attempts.csv>`.
- {download}`Referencias BibTeX <../_static/e3_bloque_c_20261115/references.bib>`.

## 6.6 Validez del modelo y regla de evaluación

La guía distingue: si R1 o R2 fallan, tope 3,0; si R1/R2 y la propiedad se acreditan pero no R3, tope 4,0; la ruta al 5,0 exige cumplir la familia completa y los demás requisitos de evaluación. Es una regla condicionada a evidencia y evaluación académica, no una calificación calculada para este avance.

Aquí las condiciones siguen abiertas. Una demostración de la cabeza y pruebas de software no convierten R1 o R3 en aprobadas ni certifican la defensa académica del autor. Las limitaciones y resultados adversos deberán conservarse.
