# V3: decisión científica y protocolo preparatorio

**Estado al 20 de noviembre de 2026: NO-GO para confirmación.** Este documento
es un borrador posterior al piloto, sin etiqueta de congelamiento, sin datos
confirmatorios admitidos y sin autorización para entrenar. No reconstruye un
prerregistro anterior. La asistencia de IA está declarada.

El piloto `20261120T031424176339Z` conserva sus treinta ajustes. La candidata
completa `atom_crps` no superó la regla original de continuación: fue
**2,7043119888 % peor que `atom_mse_only`** en la media de RMSE individuales de
tres semillas sobre 55 fechas de desarrollo. No se reduce el umbral ni se
promociona automáticamente la ablación λ=0. El dictamen documental favorable
de R1 para V3 no acredita utilidad predictiva, R2 o R3; los 796 trabajos de la
aditiva conservan su identidad y alcance.

La pregunta adicional limitada que sí queda preparada por separado es el
[diagnóstico de transferencia con pesos fijos](decision_zero_shot.md): cinco
recetas y dos controles en D01–D03, con escalas BTC congeladas y sin ajuste.
Su decisión se debe registrar antes de producir esas predicciones; los
calendarios ya expuestos lo mantienen exploratorio. No habilita la confirmación.

## Qué tendría que decidir el autor

Antes de otra campaña, el autor debe formular y fechar por qué merece
comprobar una pregunta nueva pese al resultado negativo. Podría cerrar la
contribución como mecanismo válido sin superioridad demostrada, o justificar
una investigación acotada sobre transferencia/representación del solapamiento.
Una nueva candidata o nueva regla de continuación sería otro diseño y otra
decisión, posterior a estos resultados. La evidencia actual no obliga a gastar
otra campaña grande.

Si decide buscar R3, debe aportar un calendario aún no observado, una
justificación operativa de márgenes por dominio, una batería temporal revisada,
potencia suficiente y un presupuesto concreto. Estos requisitos están en
`protocol.DRAFT.json`; los campos no resueltos son `null` o `false`, nunca una
fecha, aprobación o tag ficticios.

## Tarea, estimando y comparación completa

La tarea sigue siendo predecir a un paso la desviación poblacional σ7,
`ddof=0`, en unidades originales sin anualización, con las mismas variables
causales OHLCV. El solapamiento de seis rendimientos conocidos induce
`Y²=A+(6/49)U²`; no genera siete observaciones independientes.

La métrica primaria es la **media de las RMSE individuales** de las semillas.
La ganancia frente a j es `Δj=E[RMSEj]−E[RMSEV3]`. Se distingue de la RMSE
de un ensamble y de la raíz del MSE agregado. La hipótesis prevista es
`H0: Δj≤δ`, frente a `Δj>δ`. MAE, CRPS_U cuando haya una distribución
comparable, cobertura, anchura y coste se conservan como secundarios.

La familia principal propuesta incluye **19 rivales, 20 sistemas y 57
componentes de prueba**: Top 10 completo, MLP16/LSTM32/GRU32,
Persistence/SeasonalNaive/ARIMA, aditiva y los dos controles físicos.
`atom_mse_only`, `fixed_probs` y `atom_plugin` forman una familia de ablación
separada; su desempeño no permite retirar rivales. Entre los veinte sistemas
hay quince redes, tres clásicos y dos controles analíticos.

El margen BTC histórico `δ=0.00025772842664981106` es una referencia heredada,
no una elección nueva a partir del resultado V3 ni el umbral heurístico de 2 %.
Falta documentar su uso operativo para V3. Los márgenes de otros dominios
requieren justificación en sus escalas; no se copia ese número automáticamente.

## Datos y tratamiento común antes de abrir prueba

Las 55 fechas del piloto y los `clean_test` originales de D00–D03 ya son
conocidos. Tampoco las semillas representan mercados nuevos. Para confirmar
se requiere una admisión nueva con registro de exposición: adquisición
prospectiva de objetivos después de la decisión y la purga completa, o un
holdout externo genuinamente no abierto. Su fecha final es desconocida; **365
fechas en una sensibilidad no significan que existan 365 días independientes**.
WIKI/AAPL D01 termina en 2013: un futuro AAPL exige admitir otra fuente, no
rebautizar una franja histórica conocida.

Los tres paneles adicionales conservarán la tarea y la métrica, diferirán en
al menos dos ejes y tendrán fuente/licencia/hash/reloj/unidad de volumen y
dependencias documentados antes del resultado. ETH y derivados BTC comparten
exposiciones cripto; tres paneles no equivalen a tres mercados independientes.

Se propone una geometría común de ajuste, selección, calibración y prueba,
con purgas de siete ventanas; las proporciones 55/15/10/resto son una plantilla
que debe fijarse en fechas absolutas y tamaños viables. Los escaladores sólo
ven ajuste. Todos los sistemas usan los mismos objetivos/fechas/covariables.
Para las redes, tres tasas relativas, máximo cuarenta épocas, paciencia ocho
y límite por ajuste son una plantilla común. Los clásicos conservan recetas
congeladas; no se inventan HPO ni semillas duplicando predicciones deterministas.
Se fijarán configuraciones **y tiempo**: un mismo número de intentos no iguala
el cómputo de modelos de coste distinto.

## Tres procedimientos F2: revisión necesaria

La guía, Cuadro 4 y §§8.4–8.7, exige tres pruebas pertinentes, supuestos y
control familiar. Este borrador propone:

1. Bootstrap estacionario pareado del funcional real de ganancia RMSE,
   respetando fechas y semillas completas; longitud media de bloque 14 como
   plantilla, sensibilidad 7/28 y 9.999 remuestreos. Se requiere estacionariedad
   débil o segmentos justificados, mezcla y suficientes bloques.
2. Una adaptación por función de influencia de Diebold–Mariano con HAC para
   esa media de RMSE y corrección HLN pertinente. El horizonte real es uno;
   la ventana siete y su dependencia no se sustituyen por un horizonte siete.
   Esta extensión no se presenta como DM canónico de una pérdida aditiva:
   necesita justificar expansión, dirección, margen y aproximación finita.
3. Una extensión de capacidad predictiva condicional Giacomini–White para
   predictores no anidados, con instrumentos causales fijados en ajuste y
   dirección favorable; Clark–West sólo si se demuestra el anidamiento.
   Rechazar una igualdad condicional ómnibus no demuestra superioridad. La
   adaptación y su alineación con la métrica requieren revisión específica.

**Esta batería todavía no está validada ni congelada.** Un MCS auxiliar al
90 % sobre MSE puede informar selección, pero sobrevivir al MCS no prueba el
margen de RMSE ni sustituye automáticamente una de las tres pruebas. Tampoco
t/Wilcoxon/sign sobre semillas se convierten en tres pruebas temporales F2.

Se propone Holm global sobre los **57 p-valores**, sin dividirlo en tres
familias más pequeñas. Para cada rival manda el máximo de sus tres p
ajustados: todos deben rechazar y cumplir dirección/margen; para «supera a
todos» deben pasar los diecinueve. Se informarán IC temporales, su carácter
marginal/simultáneo y efectos con unidad apropiada: Cliff ≥0,147 o Cohen
≥0,2 cuando corresponda. No se confunde un efecto condicionado a semillas
con un efecto entre mercados.

## Potencia que puede calcularse ahora

La guía §8.7.2 exige un piloto de al menos cinco semillas para estimar las
diferencias por rival; el piloto V3 sólo tiene tres. Además, de sus seis
controles disponibles, tres son ablaciones: sólo la aditiva y los dos
analíticos pertenecen a la familia principal. **Faltan los dieciséis rivales
de E2.** Por tanto no se selecciona un R final ni se certifica la potencia de
la batería.

Se ejecutó exclusivamente el cálculo CPU reproducible de `planning.py`, con
un registro previo de escenarios y los pronósticos ya guardados. No hubo
fits, inferencia, HPO, calibración ni potencia observada posterior. Las
486 filas de escenarios son evaluaciones analíticas: **cero Monte Carlo**.
Las alternativas hipotéticas son `g1=1,25δ`, `1,5δ`, `2δ`; la distancia de potencia
es **g1−δ**, nunca la ganancia observada. Se usa `α=.05/57` como referencia
conservadora de Holm.

Las tablas separan variabilidad de inicialización en el mismo calendario y
variabilidad temporal por influencias/HAC con rezagos 7/14/28. No se suman
ambas sin identificar un modelo jerárquico. La cota normal del SD simultánea
para los seis controles es 10,93 veces el SD de tres semillas: ilustra una
incertidumbre grande, con normalidad no verificada. En esa sensibilidad,
para una alternativa `1,25δ`, las referencias de pares van de 70 a 551; para
`2δ`, de 10 a 40. **Ninguno es el R final recomendado.** Parte de esas cifras
corresponde a ablaciones y no resuelve las varianzas ausentes.

La referencia temporal a potencia marginal 0,90 y SD HAC sin multiplicar
varía entre 76–5.159 fechas para `1,25δ` según control/rezago, y 5–323 para
`2δ`. Los extremos cortos no satisfacen por sí mismos las condiciones de una
aproximación asintótica; los largos extrapolan mucho más allá de 55 fechas.
Duplicar el SD temporal cuadruplica esa referencia. No se interpretan como
calendarios ya adquiridos, independientes o recomendados.

La potencia conjunta tampoco es la marginal. Si todos los componentes
tuvieran marginal 0,90, la hipótesis de independencia daría `0,9^57≈0,00247`;
la dependencia común perfecta puede dar 0,90; sin conocer la dependencia, la
cota de unión inferior es cero. Un escenario conservador suficiente para
conjunta 0,90 requiere marginal `1−0,1/57≈0,998246` por componente. La guía
exige **reportar** la conjunta; no exige que sea 0,90. Estas son sensibilidades,
no una potencia conjunta demostrada para tres procedimientos F2 distintos.

## Adecuación, robustez, coste y puerta de ejecución

Antes de R3 faltan la superioridad frente al trivial, brechas admisibles,
residuos/estabilidad y calibración adecuados bajo el mismo tratamiento.
Se prevén Ljung–Box, ARCH, BDS si el tamaño permite, forma residual,
Mincer–Zarnowitz con HAC, razones fuera/dentro de muestra y estabilidad
Giacomini–Rossi. Para la salida discreta se debe tratar correctamente la masa
en U=0 al construir PIT; cobertura y anchura requieren calibración separada
ligada a pesos/versiones. El calibrador de la aditiva no se hereda a V3.

La robustez incluirá bloques alternativos, MAE secundaria, regímenes causales,
perturbaciones previamente fijadas y estabilidad de semillas sin seleccionar
subconjuntos favorables. Nadeau–Bengio sólo procede con particiones repetidas
que justifiquen su corrección; no se fabrica usando días o semillas.
Los dominios se presentan en sus escalas: mejor media en dos de tres y sin
inferioridad significativa frente a ningún rival; no rechazar inferioridad
no demuestra equivalencia. La comparación global con tres dominios tendrá
potencia limitada.

Un ejemplo contable, **sin autorización**, es `4×15×(3+R)×120 s`; con R=10
son 26 horas máximas de ajustes neuronales, además de costes clásicos,
preparación y análisis. No fija R ni autoriza ese gasto. Se exige hardware y
software comunes, capacidad/FLOPs, tiempos activos y de pared, inferencia
CPU batch1 con calentamiento y memoria claramente identificada. RSS del
proceso no se llama memoria incremental.

`preflight.py` prepara un recibo CPU reanudable y valida documentos/hashes.
Rechaza cambios de identidad de una preparación; conserva ambos masters.
El controlador antiguo aporta políticas de fuente inmutable, admisión,
cohorte y checkpoint, pero su trabajador aditivo **no implementa V3**. No se
lo llama como si lo hiciera. El adaptador entregado no importa modelos ni
tiene una ruta que entrene: actualmente devuelve `BLOCKED_PREPARATION_ONLY`.
La celda Colab exige fijar primero el commit real y SHA del manifiesto de
integración. Un tag final, autorización de presupuesto y motor V3 revisado
siguen siendo intervenciones concretas posteriores.
