# 21. Comparadores temporales y decisión NAS por receta

**Vigencia documental (12/11/2026):** el [catálogo obligatorio es de dieciséis sistemas (§43)](43_catalogo_obligatorio.md). La [adjudicación vigente (§44)](44_arquitecturas_publicadas.md) conserva o restaura nueve cuerpos definidos del Top 10 y mantiene NAS únicamente para la variante TiDE. Existe selección HPO/NAS ejecutada del plan anterior de catorce redes; no acredita resultados de las cuatro recetas restauradas. El benchmark final, k, R y el cierre inferencial siguen pendientes. [Enmienda registrada (§45)](45_enmienda_protocolo.md) · [Corte histórico de resultados (§41)](41_avance_optimizacion_11_origenes.md).

Las seis familias NAS de [§27](27_nas_hpo_presupuesto.md) pertenecen al plan anterior. Las especificaciones y tablas siguientes conservan el corte de preparación del 08–09/11/2026; sus pendientes y decisiones propuestas se interpretan a esa fecha. No sustituyen las identidades, el calendario ni las decisiones de §43–§45.

**Once propuestas del corte histórico:** [fichas de cuerpos y adaptaciones](22_fichas_recetas.md). El híbrido adicional queda como opcional en el alcance actual.


**Especificación propuesta, sin ejecución.** Este corte desarrolla los dos siguientes pasos de §20: comparadores F2a y examen de arquitectura por modelo. No añade candidatos al Top 10, no declara un ganador y no inicia HPO, NAS, piloto o entrenamiento. Los prototipos actuales permanecen sin modificar.

La guía exige conservar el contrato de E1 y resolver explícitamente las diferencias de información y de presupuesto. Una receta bibliográfica definida no es sinónimo de una implementación fiel; cambiar el cuerpo de una red y mantener su nombre tampoco demuestra una exención NAS. Las decisiones se separan por fuente, cuerpo implementado, adaptación de tarea y condición pendiente.

## 21.1 Comparadores exigidos por F2a

Se conserva **naive estacional y ARIMA no estacional**, además de persistencia. La elección ARIMA se apoya en su separación documentada entre estimación de parámetros y filtrado causal de nuevas observaciones; no demuestra superioridad frente a ETS. Con los diez modelos y las tres referencias de E1, el alcance obligatorio es de **16 sistemas**. Híbrido y Ridge son opcionales y no se necesitan para completar ese catálogo. La especificación descargable siguiente conserva la propuesta histórica.

{download}`Especificación completa de controles, pruebas pendientes y fuentes <../_static/controls_nas_20261108/controls.md>`.

| Control propuesto | Receta y selección | Alcance |
|---|---|---|
| Naive estacional | ŷ[t]=y[t−m], m∈{7,14,28}; elegir sólo en desarrollo | Hipótesis semanal/quincenal/cuatro semanas, no estacionalidad demostrada; 28 no es mes calendario. Cero ajustes numéricos, pero selección no gratuita. |
| ARIMA | p,q∈{0,1,2}, d∈{0,1}: 18 órdenes; constante si d=0, sin drift si d=1; no estacional ni exógenas | Máxima verosimilitud en espacio de estados, parámetros fijos dentro del bloque, actualización causal del filtro. Versión y tolerancias por fijar y probar. |

**Tres relojes distintos:** ajustar parámetros y escala con etiquetas≤o−8; condicionar estado sólo con etiquetas ya observadas≤t−1; puntuar y[t] después de emitir y guardar ŷ[t]. Incorporar días del embargo al filtro no los incorpora a una optimización, pero sí supone historia distinta de D7, que debe declararse. Ningún estado suavizado con observaciones posteriores puede reconstruir pronósticos anteriores.

La etiqueta y[t] necesita ocho cierres completos t−7…t. Persistencia necesita t−8…t−1; naive estacional requiere t−m−7…t−m. ARIMA utiliza historia univariada de desarrollo, no las mismas 35 entradas de la red. La comparación es entre sistemas, no aislamiento de arquitectura con información idéntica.

Conservar calendario y el día incompleto 24/03/2023. Naive no sustituye un rezago inválido por otro. ARIMA mantiene faltantes sobre frecuencia diaria y propaga estado sin actualización de medición; no usar `dropna` comprimiendo tiempo. Esto requiere pruebas del adaptador, todavía no implementado. La escala y/a usa media positiva de su propio entrenamiento y queda fija; los pronósticos finitos negativos se truncan a cero por regla previa, conservando raw y tasa de truncamiento. Una salida fallida nunca se sustituye por cero.

La máscara de evaluación debe ser común a redes, verdad, persistencia y los tres periodos candidatos; publicar exclusiones por soporte, no por error. **Las 305/53/608 fechas son soportes de referencia, no cobertura nueva certificada para estos controles.** No se eliminan filas válidas del entrenamiento neuronal porque un comparador requiera más historia. MASE conserva un denominador común train-only sobre diferencias de fechas consecutivas, no uno elegido por periodo ganador.

### Dos escenarios de selección, no acumulativos

A, propuesto como continuidad del borrador: 18 órdenes×10 folds=180 ajustes; una vez seleccionado y congelado, un ajuste piloto y uno final si se autorizan, total 182. La selección global por 305 fechas concatenadas no es OOF anidado. Naive compara tres periodos sin ajustes numéricos y no se repite por semilla. El piloto de 53 fechas desde 08/11 usa parámetros hasta 31/10; evaluación histórica desde 01/01/2025 usaría parámetros hasta 24/12, sin refit posterior.

B, alternativa si se decide evaluar selección anidada por origen:18 ajustes internos hasta o−98 y un refit ganador hasta o−8; antes de validación interna o−90…o−8, filtrar el puente o−97…o−91 con parámetros fijos. Diez folds=190 ajustes, piloto 19 y final 19, total 228. No sumar A y B ni elegir entre ellos después de ver resultados. El coste de 18 órdenes no acredita por sí solo equidad frente al espacio neural aún pendiente.

Convergencia, inicialización, límites y fallos deben quedar registrados. Propuesta inicial: una inicialización determinista, L-BFGS máximo 200 iteraciones, sin reinicios; al menos 90 etiquetas válidas como umbral de ingeniería, no garantía estadística. Si un candidato falla, conserva recibo y consumo; no borrar fechas favorables, ampliar rangos ni rescatar un ganador usando el test. Toda esta política debe validarse antes de datos reales.

## 21.2 Matriz histórica de fuentes y decisiones NAS

La revisión de este corte comprobó diseños publicados y código real, separando fuente y adaptación; todavía no fijaba m. Las diferencias de la tabla describen aquellas implementaciones, incluidas variantes luego sustituidas. La adjudicación posterior de [§44](44_arquitecturas_publicadas.md) fija **m=1, TiDE**, para el diseño vigente y documenta las cuatro restauraciones con identidades nuevas. No se renombran los resultados anteriores como resultados de esos cuerpos.

| Modelo | Diseño en fuente primaria | Diferencia relevante del código D7 actual | Decisión defendible antes de entrenar |
|---|---|---|---|
| FCN-1D | Sí: Wang et al., §II.B, pp. 2–3 y Fig. 1b | Tres convoluciones de ancho 16 y kernels 3/3/1, sin BatchNorm; no el cuerpo 128/256/128, 8/5/3 de la fuente | Restaurar cuerpo publicado y justificar cabeza escalar, o tratar el compacto como diseño propio sujeto a §14; no eximir el compacto por su nombre |
| ResNet-1D | Sí: Wang et al., §II.C, p. 3 y Fig. 1c | Dos bloques de dos convoluciones, ancho 16, sin BatchNorm; fuente: tres bloques y nueve convoluciones principales | Misma bifurcación: cuerpo publicado documentado frente a diseño compacto propio; adjudicación pendiente |
| Inception-1D | Sí: Fawaz et al., §3.1–3.2, pp. 5–6 y código de autor | Un módulo, kernels 1/3/5 y ancho 16; sin las seis unidades, normalización y conexiones residuales del cuerpo fuente | Resolver red individual frente a ensemble y restaurar diseño o aplicar §14 al propio; la red actual no es InceptionTime publicado literal |
| LSTM-autoencoder con predictor | Código primario del autor sí contiene diseños explícitos; texto/tablas del paper no revalidados en esta revisión | Un encoder y un decoder de ancho 16 con proyección, frente a preentrenamiento apilado y preset de dos encoders 25/26 | Verificar el diseño del paper y elegir receta explícita, o reconocer diseño propio y aplicar §14; ni ausencia de diseño ni exención por paper demostradas todavía |
| TimesNet | Sí: §3 y Table 7, p. 14 del PDF | D7 se extiende a 8; se usan las cuatro frecuencias no DC en lugar del top-k original; cuerpo y dimensiones mezclan decisiones del paper/código | No basta decir «cabeza cambiada». Justificar/adjudicar modificación estructural; restaurar configuración compatible o incluir diseño adaptado en §14 si se considera nuevo |
| TSMixer | Sí: Fig. 1, p. 2; Appendix E, Table 8, p. 21; implementación de autor | Conserva seis bloques y feature MLP 512; cambia longitud, salida escalar, momentos y dropout | Candidato fuerte a cuerpo publicado más puente de tarea, sujeto a registro explícito; no es réplica exacta ni exención automática de toda la receta |
| TiDE | Sí: Table 8, p. 21; código primario de encoder/decoder | Covariables propias, sentinel futuro cero y decoder temporal de salida 8 en vez de 1, además de cabeza | Cambio estructural real que exige adjudicación; no presentarlo como mero reemplazo de readout |
| TimeMixer | Sí: §3; Table 7, p. 14; script ETTh1 de autor | Reduce tres downsamplings publicados a dos, dando longitudes 7→3→1; elimina proyección de variables y calendario | Justificar/adjudicar el cambio de escalas; fuente explícita no prueba que esta receta esté exenta |
| PatchTST | Sí: §4.1, p. 6; Appendix A.1.4, p. 14; es ejemplo explícito de la guía | Mantiene cuerpo ETTh pequeño, pero rellena D7 hasta 16 y crea dos patches; cabeza conjunta y momentos | Documentar geometría/padding y puente; exención condicional si se admite configuración publicada adaptada, no por nombre |
| iTransformer | Sí: Appendix A.2, Algorithm 1, p. 13; script ETTh1 de autor | Conserva encoder L=2, D=256, FF=256, H=8; entrada 7 y cabeza conjunta sustituyen forecast por variable | Candidato fuerte a cuerpo publicado más puente; registrar adaptaciones y adjudicar, sin llamarlo réplica literal |
| CNN-1D + LSTM + atención temporal | No se verificó una fuente primaria exacta para esta receta completa; esto no afirma que no exista literatura semejante | Composición propia CNN32→LSTM32→atención16, kernel3, una LSTM, cabeza escalar | Si se evalúa como arquitectura propia de E2, planificar §14 salvo justificación académica expresa. La guía E2 no resuelve por sí sola el tratamiento de una propuesta separada E3 |


{download}`Evidencia completa por modelo, fuentes primarias, localizadores y limitaciones <../_static/controls_nas_20261108/nas.md>`.

Los cuatro prototipos compactos cambian más que la cabeza. Para FCN/ResNet/Inception hay diseños primarios explícitos; restaurarlos exige especificar normalización, kernels, bloques y miembro individual frente a ensemble. Para LSTM-SAE se verificó código del autor con variantes definidas, pero no se recuperó el cuerpo del artículo de forma verificable en esta revisión: no se concede exención a partir de una tabla no releída.

Entre los seis adaptadores, TSMixer e iTransformer preservan mejor el cuerpo de una configuración fuente, mientras TimesNet, TiDE y TimeMixer presentan cambios estructurales de D7 y PatchTST cambia geometría de parches. Los seis incorporan diez momentos de la misma ventana a la cabeza. No son días adicionales ni pesos preentrenados, pero sí diferencias que deben adjudicarse y controlarse.

El candidato CNN-LSTM-atención carece de una fuente exacta verificada para su receta completa. Proponer tres bloques CNN/LSTM/atención con anchos independientes es una posible correspondencia para §14, aún no aprobada ni implementada. No se presume originalidad ni ausencia de trabajos semejantes.

**Resolución posterior:** [§44](44_arquitecturas_publicadas.md) documenta la receta fuente o adaptada de cada fila y el único dominio NAS obligatorio, TiDE. Las recomendaciones de recuperación de cuerpos y adjudicación del corte anterior conservan su trazabilidad; no autorizan NAS para los once por defecto.


## 21.3 Coste por escenarios, no tiempos inventados

La calculadora `scripts/e2_nas_cost_scenarios.py` sólo realiza aritmética: no importa frameworks de entrenamiento ni lee observaciones. Para m modelos sometidos a NAS, F orígenes que repiten toda la búsqueda, k finalistas y R₂ semillas de fase 2:

- Fase 1: m × F × 1.000 trayectorias cortas, una semilla por arquitectura.
- Fase 2: m × F × (k + c) × R₂ trayectorias completas, con k ≥ 10 y R₂ ≥ 5.
- c=1 si la referencia debe añadirse; c=0 si ya está entre los finalistas. No duplicar la misma arquitectura para inflar coste o evidencia.
- Tiempo simbólico: m × F × [1.000 × t_corta + (k + c) × R₂ × t_completa]. No se han medido t_corta ni t_completa y no se calculan horas. Si los costes difieren por modelo, sumar sus tiempos; no asumir uno común. No multiplicar otra vez por folds ya incluidos dentro de la trayectoria cronometrada.

Una trayectoria significa el pipeline completo correspondiente. Preentrenamiento AE, ajustes internos y refits que incluya deben entrar en su tiempo, no tratarse como gratuitos ni confundirse con un único paso de optimizador. La fórmula no incluye HPO previo, ajuste final, comparadores, robustez, explicaciones o ablaciones; esos componentes se presupuestan aparte. F=1 no equivale a ejecutar toda la búsqueda dentro de diez folds anidados.

| Escenario aritmético | F | Cortas | Completas máximas (k=10, R₂=5, c=1) |
|---|---:|---:|---:|
| m=0, sólo si ninguna receta queda sujeta a NAS por decisión expresa | 1 | 0 | 0 |
| m=1 | 1 | 1.000 | 55 |
| m=4, ejemplo de guía | 1 | 4.000 | 220 |
| m=5 | 1 | 5.000 | 275 |
| m=11 | 1 | 11.000 | 605 |
| m=5, búsqueda completa repetida en diez orígenes | 10 | 50.000 | 2.750 |

**La tabla es aritmética histórica:** la adjudicación vigente fija m=1 para TiDE, y [§45](45_enmienda_protocolo.md) registra una sola NAS inicial. k se fija mediante el piloto de ruido; el ejemplo k=10 de esta tabla no sustituye esa regla. Los escenarios no autorizan once búsquedas. [Cálculos reproducibles](../_static/controls_nas_20261108/cost_scenarios.json).

La guía ofrece m=4, k=10, R₂=5 como ejemplo: 4.000 trayectorias cortas y hasta 220 completas (p55). Reproducir ese cálculo no autoriza el cómputo ni afirma que cuatro sea el número de modelos aplicable a este catálogo. R₂ tampoco es el R final de la comparación.

## 21.4 Qué queda antes de ejecutar

La condición arquitectónica y el calendario ya están registrados en §44–§45. Para ejecutar la ruta vigente faltan conciliar el presupuesto y los resultados compatibles, completar la transición del controlador y acreditar comparadores, pilotos y ajustes con sus artefactos. La reserva histórica conocida no recupera condición confirmatoria. Un presupuesto insuficiente se documenta; no se recortan silenciosamente arquitecturas, semillas o fallos.
