# Batería temporal V4 y estudios conservados

La implementación vigente es `temporal-fixed-seeds-mse-v4`. El plan, código, runner y pruebas se publicaron en el commit `70f0838cdf8809a469ea1ca837e5011faff551e2` antes de observar la primera réplica V4. Su decisión operacional usa **α=0,025**, más exigente que el 0,05 de la guía. El dictamen ejecutado es **INCONCLUSIVE_BOUNDED_VALIDATION**: el nulo AR(0,5) dio16/512 falsos rechazos y CP95 superior0,05025287955875999, ligeramente mayor que0,05. No se redondea para aprobar. Las otras cotas fijadas pasan. El informe completo está en `qa/statistical_validation/v4/report.json`; no hay admisión final.

## Estimando, semillas y tiempo

Para sistema j, semilla s y fecha t, q[j,s,t]=(prediction[j,s,t]−y[t])². Se promedian pérdidas sobre el conjunto **fijo** de modelos entrenados: Q[j,t]=mean_s(q[j,s,t]). El estimando temporal es E_t[Q[rival,t]−Q[candidate,t]], condicionado a esos modelos. Se usa la secuencia cronológica, **no R×T observaciones iid**.

Se distinguen la media histórica de RMSE individuales, mean_s(sqrt(mean_t(q))), y la raíz del MSE agregado, sqrt(mean_t(Q)), funcional prospectivo nuevo. El RMSE de promediar predicciones como ensamble no sustituye ninguno. La enmienda no reescribe la regla V5 o E1; también se exige que la media histórica de RMSE sea favorable. La monotonicidad de la raíz permite expresar la nueva pérdida agregada en RMSE; no iguala los funcionales ni implica aceptación académica automática.

El margen m=0,02 se traduce en W[t]=(1−m)²Q[rival,t]−Q[candidate,t]. Su media positiva equivale a ganancia relativa de RMSE agregado mayor que m, para MSE positivos. Igualdad y margen son hipótesis distintas y se fijan antes de datos nuevos.

## Tres procedimientos pertinentes

| Procedimiento | Hipótesis y unidad | Dependencia y límites |
|---|---|---|
| DM-HLN/HAC | Media del diferencial d o W igual a cero, sobre fechas | Covarianza Bartlett/Newey–West y factor HLN, con referencia fixed-b escalar; horizonte1 no significa independencia |
| Bootstrap estacionario-t | Media del diferencial igual a cero, centrado y studentizado | Bloques compartidos por sistemas, HAC recalculada en cada réplica; semillas no se convierten en fechas |
| Giacomini–White condicional | E[z_t·d_t]=0 o E[z_t·W_t]=0 | Momentos vectoriales predictibles y HAC/fixed-b Wald; rechazo no direccional, no demuestra por sí solo inferioridad de la media |

Los instrumentos reales son constante y sqrt(known_floor_A)/rfit, con rfit estimado sólo en ajuste. No se usa la etiqueta futura como instrumento. Una matriz ausente deja GW no disponible (p=1); una constante duplicada es singular y no cuenta como copia de DM. En los DGP sintéticos se usa un instrumento exógeno predecible, independiente del diferencial.

No se fuerza Clark–West sin anidamiento ni se convierten inclusiones de un MCS global en p direccionales por rival. Se requieren estacionariedad/dependencia débil, momentos adecuados, FCLT vectorial y covarianza no singular. Autocorrelación, mitades temporales y número de condición se reportan; esos diagnósticos no demuestran los supuestos.

## Referencias fixed-b, HAC y bootstrap-t

V1/V2 omitían parte de la variación finita de HAC al usar t/chi-cuadrado. V3 introdujo referencias gaussianas fixed-b escalar para DM y vectorial para Wald GW, preservando tamaño, bandwidth y factor HLN. Kiefer–Vogelsang (2005), teorema 3 y sección 3.3, fundamentan la simulación del pivote bajo FCLT. La malla gaussiana finita aproxima ese pivote: no es la ley exacta de cualquier mercado.

La fórmula HAC de sumas móviles de longitud L+1, incluidos extremos acolchados, reproduce exactamente Bartlett/Newey–West con normalización T. Las pruebas comparan ambas expresiones. El núcleo C++ opcional sólo acelera aritmética: no tiene RNG o decisiones propias y no usa fast-math. Se comprueba equivalencia con NumPy en escalas 1e−12, 1 y 1e12, intervalos y conteos de cola; fuente/flags se identifican por hash.

El bootstrap-t recalcula HAC por remuestra. La diferencia de MSE usa intervalo pivotal y la ganancia studentización delta del log-ratio MSE. Se conservan intervalos marginales **95 % y99 %**, con sus niveles reales. La regla operacional añade CI99 y su estudio exige cobertura≥0,95 de ese intervalo más fuerte; no se le llama CI95 ni se ocultan posibles fallos de CI95. No son intervalos simultáneos para todos los rivales.

## Multiplicidad y direcciones

Cada panel tiene 120 casillas: 20 rivales × 3 métodos × igualdad/margen. Se guardan estadísticos, p crudos, ajustados Holm, dirección, estado, conteo de cola, draws y regla (count+1)/(B+1). La referencia gaussiana tiene su propio número de draws 49999; no usa por error el B=8191 del bootstrap.

La conjunción positiva requiere seis p ajustados<0,025, medias EPA/margen favorables, media histórica RMSE favorable, efecto temporal mean(d)/sd(d)≥0,2, CI95 de MSE con límite inferior >0 y CI99 de MSE/ganancia con límite inferior >0/margen. El efecto es descriptivo por fechas, no una desviación de medias dividida por raíz de T ni una muestra iid ficticia de semillas.

La familia adicional conserva Holm180 sobre EPA: 3 paneles × 20 rivales × 3 métodos. El veto de **inferioridad media** usa media negativa y DM/bootstrap ajustado significativo. GW permanece en Holm y se reporta separadamente si rechaza momentos con media muestral negativa; un Wald no direccional no basta para declarar inferioridad media.

La API por panel siempre informa r3_accredited=false: aún necesita el compuesto de potencia, admisión, reserva auténtica, consistencia, adecuación y robustez. Nadeau–Bengio no se aplica a una única reserva temporal sin remuestreo de particiones entrenamiento/prueba.

## Alcance por panel; aplicación a cuatro paneles

El Monte Carlo estudia **un panel temporal completo con 20 rivales** por réplica. No se copian resultados cuatro veces como experimentos independientes. Su aplicación a cuatro paneles reales queda condicionada a los supuestos en cada uno. Holm es un ajuste fuerte para EPA adicionales bajo p marginales superuniformes y no necesita independencia entre paneles. Estas cinco simulaciones no prueban superuniformidad para todos los mercados ni una ley conjunta universal de cuatro paneles. La potencia de 80 contrastes y su conjunta bajo DGP empírico se calcula separadamente con predicciones reales de desarrollo.

La regla ya fijada también permite una cota entre grupos: principal120
a0,025 y adicionalEPA180 a0,025 suman como máximo0,05 por Bonferroni para las
afirmaciones primarias y el veto de inferioridad, **si** sus p marginales son
superuniformes bajo cada nulo verdadero. Holm no necesita independencia:
con m₀ nulos verdaderos, el primer rechazo de uno verdadero en el orden
exige algún p verdadero ≤α/m₀; la unión de m₀ eventos tiene probabilidad ≤α.
Esta cota teórica no valida por sí sola p aproximados bootstrap/fixed-b para
un mercado. Los restantes análisis adicionales y sensibilidades conservan
su alcance descriptivo o la familia prefijada que les corresponda; no se
agregan como nuevos contrastes confirmatorios sin control de multiplicidad.

## Versiones previas y nueva validación

| Estudio | Plan congelado | Evidencia conservada |
|---|---|---|
| V1 | T192/R10/B4999/HAC12/bloque12; 80×5 réplicas | Fallos FWER/cobertura;16/80 falsos rechazos nuloAR.5 y 21/80 solapado |
| V2 | T1536/R10/B4999/HAC48/bloque64; 80×5 nuevas | Fallos de cobertura parcial/solapada y FWER 11/80 en frontera; CPU acumulada V1/V2 = 168,46 s |
| V3 | T768/R10/B=8191/HAC48/bloque32/ref49999; 192×5 | Inconclusa: FWER 5, 4, 10, 9/192 no demuestra límite superior≤.05; CI99 parcial 188/192 tampoco demuestra inferior≥.95; alternativa fuerte 192/192 no repara esos límites |
| V4 | Mismos estadísticos/T/R/B/HAC/bloque/referencia; α.025; 512×5 nuevas | Dictamen en v4/report.json; no se acumula con V3 ni se escogen réplicas por resultado |

V3 dejó 960 filas completas y todos los hashes congelados intactos; el entorno se sustituyó antes de que existiera report.json. El resumen se reconstruyó mecánicamente de ese inventario exacto sin generar o repetir réplicas. v3/RECOVERY_RECEIPT.json conserva esa recuperación. CPU/reloj originales no son reconstruibles y se informan desconocidos. Se conserva la contradicción textual 300 s/numérico 900 s del plan V3: su runner usaba el número 900. El nuevo presupuesto V4 es inequívoco: 2000 CPU s.

V4 endurece la decisión y aumenta precisión con semillas independientes. Los criterios CP **no se relajan**: FWER sólo se demuestra si CP95 superior ≤0,05; cobertura operacional CI99 sólo si CP95 inferior ≥0,95; potencia fuerte sólo si CP95 inferior ≥0,90 para cada rival. Las cotas CP son puntuales, no simultáneas sobre diagnósticos; la admisión exige que todos pasen. Cualquier cota no concluyente se informa inconclusa, sin redondear a aprobación.

La cobertura se evalúa para el **rival0 prefijado** en cada escenario, como
declaran los planes. El evento FWER cubre todos los nulos verdaderos entre
los veinte rivales y tres métodos; la potencia fuerte conserva veinte
contadores y la conjunta. La cobertura de rival0 no se presenta como una
verificación de cobertura simultánea de veinte intervalos heterogéneos.

Los escenarios son nulos completos/parciales, dependencia AR(0,5), solapamiento de cinco barras, frontera práctica y alternativa fuerte. El caso fuerte no demuestra potencia ante 3 % de ganancia o una campaña de mercado. Cuatro threads comparten un único reloj CPU y referencias fijas. Se reservan 32 CPU s para tareas en curso antes de dejar de iniciar réplicas. Un agotamiento conserva todo e informa escenarios incompletos. Los JSONL completos se comprimen determinísticamente con SHA256 comprimido/original.

**21 pruebas deterministas V4 pasan**, incluidas invariancia de estadísticos/intervalos al cambiar sólo α y GW significativo con media negativa sin veto falso de inferioridad. No sustituyen calibración estocástica.

Reproducción vigente, directorio nuevo:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_statistics.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python qa/statistical_validation/v4/run_validation.py --output-dir /tmp/temporal-v4-reproduction
```

Para V3 se usa una **copia temporal**, restaurando v3/statistics_frozen.py y tests_frozen.py a engine/statistics.py y tests/test_statistics.py. El runner V3 verifica los hashes originales. No se sobrescribe el código vigente ni se borran adversos. Las fuentes/documentos V1/V2 también permanecen archivados.

## Fuentes primarias

- Diebold–Mariano1995:doi:10.1080/07350015.1995.10524599.
- Harvey–Leybourne–Newbold1997:doi:10.1016/S0169-2070(96)00719-4.
- Newey–West1987:doi:10.2307/1913610.
- Politis–Romano1994:doi:10.1080/01621459.1994.10476870.
- Giacomini–White2006:doi:10.1111/j.1468-0262.2006.00718.x.
- Kiefer–Vogelsang (2005):CAE05-08,teorema3/sección3.3, https://cae.economics.cornell.edu/05-08.pdf; lectura primaria documentada en v3/primary_source.json, sin inventar hash de un PDF no descargado.
- Holm1979:Scandinavian Journal of Statistics6,65–70.

Estas fuentes motivan construcción y supuestos; no garantizan error finito para cualquier proceso, muestra, bandwidth o reserva de mercado.
