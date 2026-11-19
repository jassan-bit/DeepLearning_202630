# Protocolo de E3: borrador consolidado posterior

**Fecha real: 17 de noviembre de 2026, America/Bogota. Estado: BORRADOR NO CONGELADO NI PREREGISTRADO. Tag confirmatorio: pendiente.**

Este documento consolida los requisitos de la sección 8 de la guía después de
conocer los pilotos, las réplicas y los resultados históricos. Su creación en la
ruta canónica no registra retrospectivamente aquellos experimentos ni autoriza
nuevos ajustes, calibraciones o consumo GPU. La versión del libro de partida es
`8fccb216f36a3e4fc5cb38ecdec769b7f5379397`.

El borrador histórico de `experiments/e3_bloque_d/protocolo_borrador.md` y el plan
del análisis posterior se conservan. El plan posterior se fijó en `4a21b0329b99e864ec64fc2df49260aa874141bf`
y su código en `0213266404a0aa3af985d7306dce6b13892c71ec`, antes de calcular sus
p-valores y después de conocer las predicciones. Ese orden no restaura el
cegamiento de las 608 fechas de prueba.

## 1. Estándar y mecanismo

Rige `experiments/estandar.md`: propiedad formal pertinente y verificación;
familia completa; al menos tres pruebas distintas admisibles para la ruta;
potencia por contraste de al menos 0,90 y potencia conjunta reportada; mínimo
diez semillas o más según dimensionamiento; al menos tres conjuntos adicionales.
El registro con tag y la ratificación humana del estándar siguen pendientes.

La referencia evaluada es CNN causal → LSTM unidireccional → atención temporal
por contenido, con salida aditiva positiva, 1210 parámetros y pérdida MSE
normalizada. No es el primer mecanismo de penalización/saturación ni la cabeza
de momentos. Sus componentes convencionales y la propiedad condicional del
capítulo 7 no se declaran nuevos. Falta fijar qué aporte se defenderá, su
limitación cuantificada, hipótesis de mecanismo, predicciones secundarias,
procedencia y adaptación sustantiva cuando corresponda. No se traslada una
hipótesis de otro mecanismo a esta receta por compartir el encoder.

La bitácora `experiments/design_log.md` y su inventario conservan ocho recetas
de desarrollo y diez réplicas adicionales de una receta; **Nd global sigue sin
conciliar**. Una semilla nueva no equivale a un diseño nuevo. La defensa del
autor, la revisión de novedad y la verificación humana independiente no quedan
certificadas por disponer de estos archivos.

## 2. Métricas, hipótesis y márgenes

La tarea principal sigue siendo pronóstico a un día de sigma7 poblacional de
siete retornos Close, sin anualización. La métrica primaria es la media de las
RMSE individuales, no la RMSE de un ensamble. Para el rival j:

$$
\Delta_j=\operatorname{media}_r(\mathrm{RMSE}_{j,r})-
\operatorname{media}_r(\mathrm{RMSE}_{E3,r}),\qquad
H_{0,j}:\Delta_j\le\delta,\quad H_{1,j}:\Delta_j>\delta.
$$

El margen heredado BTC es **δ=0,00025772842664981106**. No se convierte en δ²
para contrastar MSE ni se recalcula para favorecer el resultado. Para los
conjuntos adicionales falta fijar y justificar antes de sus resultados cómo
se conserva o adapta el margen a su escala. La orientación positiva favorece E3.

El margen de adecuación δg tiene el mismo valor numérico heredado, con otra
hipótesis: para una brecha comparable validación−ajuste, contrastar que sea
menor que δg. Falta fijar estimando y remuestreo de esa brecha; probar igualdad
a cero o no rechazar sobreajuste no demuestra su tolerancia.

MAE, sMAPE y R² son métricas secundarias descriptivas del suplemento posterior.
La reconstrucción de los intervalos de E2 usa sólo predicciones y cuantiles
guardados; no recalibra. E3 no tiene cobertura o anchura calibradas. En una
campaña futura deberán fijarse las reglas de puntuación apropiadas, las familias
secundarias y el nivel de intervalo. Un intervalo único del 90 % no determina
por sí solo una distribución para calcular CRPS o PIT. No se interpretará la
DE entre semillas como intervalo predictivo.

## 3. Familia y control de multiplicidad

Se conservan los **16 rivales**: FCN-1D, ResNet-1D, Inception-1D,
LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST,
iTransformer, MLP16, LSTM32, GRU32, Persistence, SeasonalNaive y ARIMA. El
ejemplo m≈12 de la guía no reduce este inventario. Las recetas conservan sus
fuentes; cerrar A y los artefactos confirmatorios sigue pendiente.

El análisis posterior ya calculado controla una familia global de **48
hipótesis elementales**, tres procedimientos por rival, con Holm y α=0,05.
Los tres p ajustados deben rechazar para cada rival; se reporta su máximo.
La afirmación frente a todos exige todos los rivales, sin seleccionar el menor
p o excluir un fallo. Este registro no altera aquella familia ni sus p-valores.

Para la confirmación falta cerrar las pruebas admisibles de la Tabla 4 para
F2 temporal, sus nulas, estimandos, supuestos, ajuste y tratamiento de fallos.
Se declarará la integración del procedimiento global o de control múltiple.
El bootstrap circular ejecutado no se renombra como bootstrap estacionario;
HAC de influencia RMSE no se renombra DM–HLN; la t entre semillas no se convierte
en una prueba temporal. Sus alcances distintos no son una prohibición de la
guía, pero requieren justificación y no acreditan automáticamente la batería.

DM, Giacomini–White o Clark–West requieren sus diseños y supuestos; Clark–West
no se aplica sin anidamiento. El MCS auxiliar usa MSE diario medio entre
semillas y nivel 90 %: sus diez supervivientes no son diez ganadores ni prueban
el margen de RMSE. Las familias de ablación, adicionales, regímenes,
sensibilidades y adecuación se enumerarán y controlarán por separado antes de
su ejecución, sin fraccionarlas después de ver sus resultados.

## 4. Datos, particiones y semillas

La cronología histórica es conocida. E3 usó 532 ejemplos de ajuste, siete
purgados y 116 de validación de desarrollo; E2 usó 655 de ajuste y 53 de
calibración. Las 608 fechas de prueba comunes ya fueron evaluadas. Tampoco se
restaura el cegamiento de reservas históricas publicadas por cambiar su nombre.

Antes de confirmar deben fijarse fuentes, versiones/hashes y derechos, cortes
y soporte temporal de entradas/etiquetas, purgas, tratamiento de faltantes,
transformaciones estimadas sólo en ajuste y una política común de selección,
ajuste final y calibración. La auditoría posterior de 1316 ventanas tiene
12 PASS, 3 PARTIAL y 1 NA; sus controles abiertos no se cierran por repetir una
comprobación de hashes.

Las semillas históricas neuronales 271801–271810 están documentadas. La lista
de una campaña futura y su R definitivo se fijarán después del dimensionamiento,
antes de ejecutar. Cada clásico determinista conserva una trayectoria; copiarla
diez veces no crea independencia. Se documentarán Python, NumPy y framework,
generadores, flags de determinismo, precisión, versiones y límites entre
plataformas, sin prometer determinismo no verificado.

La unidad de remuestreo debe conservar la dependencia y el emparejamiento. Los
días solapados y las semillas sobre el mismo calendario no son mercados
independientes. Se fijarán bloques, réplicas, generadores y sensibilidades antes
de analizar. Un modelo mixto no elimina por sí solo la autocorrelación; sus
efectos de modelo/unidad/semilla y su estructura residual requieren justificación.

## 5. Equidad, presupuesto y potencia

Faltan la política común de ajuste y el presupuesto operativo por modelo:
configuraciones, espacio relativo, épocas máximas, parada, intentos/fallos,
tiempo y hardware. Se declarará si la igualdad se mide en configuraciones,
tiempo o ambos. Si cambia el coste por configuración, se iguala el tiempo total
o se reportan ambos análisis. El esfuerzo de diseño se separa de HPO. Se
reportarán parámetros, FLOPs, entrenamiento, inferencia, RAM y energía con su
método, y controles de capacidad cuando procedan.

Los consumos cerrados se conservan separados: ampliación E2
5130,970719594005 s y E3 acumulado 948,439625042 s, que incluye etapas previas.
No son presupuestos equivalentes ni una asignación nueva. Los cálculos sobre
predicciones guardadas no añaden entrenamiento o inferencia.

La potencia y MDE prospectivas **no están dimensionadas**. Se requiere un
piloto adecuado de al menos cinco pares y una alternativa prevista, sin usar
el efecto del test conocido como potencia observada. Para la nula de margen,
la distancia relevante es Δalternativa−δ>0. La fórmula normal de R es sólo una
aproximación bajo sus supuestos, no una garantía para Holm, remuestreo temporal
y una regla de tres pruebas. Se dimensionarán tanto R como el calendario.

Se reportará potencia por rival bajo la regla completa y potencia conjunta
de superar a todos, conservando dependencias. El producto de potencias sólo
es ilustrativo bajo independencia; no estima la potencia real. Si diez
semillas no bastan, se aumentan antes de congelar. Los límites discretos de
Wilcoxon no demuestran potencia suficiente. Equivalencia y no inferioridad
requieren hipótesis/márgenes propios; ausencia de rechazo no basta.

## 6. Tres conjuntos adicionales y robustez

**Ningún conjunto adicional está seleccionado.** Se registrarán tres fuentes
públicas de la misma tarea y primaria, cada una distinta del principal en al
menos dos ejes: tamaño, dominio, ruido, desbalance o dimensionalidad. Fuente,
licencia, versión/hash y razón de selección se fijan antes de conocer desempeño.
Todos los rivales y E3 se evalúan, sin excepciones, con auditoría y presupuesto
comparables. Si un conjunto guía rediseño, pasa a desarrollo.

Se reportará cada conjunto por separado y se consolidarán rangos mediante
Friedman–Iman–Davenport y pares con Holm, o una alternativa bayesiana/ROPE fijada
antes del resultado. No se promedian RMSE de escalas distintas; tres conjuntos
tienen potencia limitada. Se definirán unidad, efecto/IC y probabilidad de
superioridad sin confundir proporciones empíricas con probabilidad posterior.

La matriz de robustez incluye hiperparámetros, semillas y particiones,
perturbaciones de al menos tres severidades, curvas de tamaño de muestra y
escalabilidad, cambio de distribución fuera del diseño, calibración/reglas
de puntuación, cobertura/ancho por subgrupos, eficiencia/Pareto y errores
compartidos. Las perturbaciones de OHLCV conservarán plausibilidad y
disponibilidad temporal. Los diagnósticos F2 aplicables, sus criterios y sus
límites se fijarán antes de su evaluación. No se fabrican scores probabilísticos
sin la salida necesaria ni cobertura de E3 desde dispersión de semillas.

## 7. Decisión R3 e informes

R3 requiere simultáneamente: mejor media frente a cada rival; tres pruebas
admisibles con control familiar y ganancia práctica; efecto no despreciable
con estimador apropiado e IC de ganancia positivo; consistencia en los tres
adicionales conforme a la guía; adecuación del propio modelo; y robustez a
la elección metodológica. Los umbrales de Cliff y Cohen requieren la unidad
y el estimador pertinentes, no se intercambian automáticamente con Cohen dz
condicionado al calendario. La adecuación incluye baseline trivial, brecha
tolerable, calibración, residuos y coherencia entre validación y prueba.

Se entregarán métricas media/DE/IC con alcance y recursos; tabla de hipótesis,
supuestos, estadísticos, p crudos/Holm, efectos/IC y conclusión; MCS; tablas por
conjunto y consolidación; Tablas A/B de E3; potencia/MDE/conjunta; figuras de
robustez y costes; discusión de resultados adversos, no concluyentes y límites.
Ninguno de estos documentos asigna una nota o certifica R1/R2/R3.

## 8. Cierre, desviaciones y versiones

Para registrar este borrador faltan mecanismo/hipótesis defendibles, revisión
humana, cierre A, datasets y reserva confirmatoria real, equidad, pruebas y
supuestos, piloto/potencia, R, presupuesto, artefactos y plan completo de
reportes. Entonces se registrarán fecha y tag reales previos a las ejecuciones.
No se escribe un tag inexistente ni se retrofecha esta revisión.

Cualquier desviación conservará fecha, motivo y efecto sobre la interpretación;
los análisis no previstos serán exploratorios. Cambiar arquitectura, pérdida o
espacio de búsqueda después del cierre exige nueva versión y conserva el
historial. La asistencia de IA está declarada; el autor debe revisar y defender
el contenido y ninguna comprobación automática sustituye su responsabilidad.
