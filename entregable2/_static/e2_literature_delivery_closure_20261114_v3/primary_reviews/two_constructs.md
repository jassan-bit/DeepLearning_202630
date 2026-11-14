# Revisión primaria focal de dos dudas de constructo

Fecha: 14/11/2026 UTC. Propuesta de uso y texto para revisión; no modifica el paquete publicado ni sus decisiones.

**Resultado:** ambas fuentes son admisibles en la narrativa de métodos aplicados a BTC/cripto. La reconstrucción de sus etiquetas y sus efectos cuantitativos sigue sin poder verificarse. Mantener esa limitación se justifica por contradicciones o vacíos del propio texto, no por diferir de σ7. Ningún PDF adicional del usuario es necesario para llegar a esta conclusión.

## Alcance efectivamente leído

Los recibos locales previos declaran `original_file_materialized=false` y SHA-256 nulo: acreditan lecturas históricas, no originales locales disponibles para releer. Esta revisión accedió de nuevo a dos fuentes primarias abiertas mediante la herramienta web y leyó focalmente los pasajes indicados abajo. No afirma lectura continua nueva de ambos artículos, verificación visual de sus PDF ni lectura de la versión editorial final de CryptoMamba.

## E2-0059 / PR_GARCH

Zahid, Iqbal y Koutmos (2022), *Forecasting Bitcoin Volatility Using Hybrid GARCH Models with Machine Learning*. DOI: **10.3390/risks10120237**.

Fuente primaria consultada: [HTML editorial de MDPI](https://www.mdpi.com/2227-9091/10/12/237), identificado como Risks 10(12), 237, publicado el 13/12/2022.

- **Población y rol confirmados:** BTC; §3.1 declara cierres diarios de Bitcoincharts. §3.3 introduce estimaciones GARCH como entradas de híbridos con LSTM/GRU/BiLSTM. No equivale a comparar redes recurrentes autónomas.
- **Duda confirmada en el original:** §3.1, Ecs.12–14, parte de cierres diarios pero define la agregación de retornos cuadrados sobre observaciones dentro de un día. Falta establecer qué observaciones o agregación produjeron la etiqueta evaluada; no corresponde suponer n=1 ni una ventana de siete días. La diferencia no se explica por un simple cambio de horizonte.
- **Horizonte separado del target:** §3.3 declara pronósticos a 7/14/21 días; §4.3 describe un procedimiento rolling de un paso y vuelve a referirse a esos horizontes. El texto consultado no reconstruye inequívocamente su alineación.

**Decisión propuesta:** conservar como estudio empírico de híbridos GARCH–DL para BTC, con operacionalización de volatilidad no verificada. Permite discutir el uso de pronósticos econométricos como características y las dificultades de reporte. Mantener pendiente la elegibilidad aplicada verificable según el protocolo vigente; no extraer un efecto comparable ni inferir superioridad universal.

Localizadores: §3.1/Ecs.12–14 (HTML líneas 293–311); §3.3 (325–328); §4.3 (392–403). Referencias web de la sesión: `turn951140view0`, `turn187519view2`. Localizador de la ficha histórica: PDF pp.6–8, §§3.1–3.3; pp.12–14, §4.3.

## E2-0289 / E1N075

Zhao et al. (2026), *CryptoMamba-SSM: Linear Complexity State Space Models for Cryptocurrency Volatility Prediction*. DOI: **10.1109/OJCS.2026.3651226**.

Fuente primaria consultada: [manuscrito aceptado expuesto íntegramente en ResearchGate](https://www.researchgate.net/publication/399710354_CryptoMamba-SSM_Linear_Complexity_State_Space_Models_for_Cryptocurrency_Volatility_Prediction). El pie identifica la versión de autor sin edición final y el DOI correcto; la cabecera conserva metadatos de plantilla. No se afirma haber consultado la versión editorial final.

- **Población y rol confirmados:** experimentos BTC/cripto; modelo SSM compuesto, con extracción multiescala y atención, y filas de comparadores recurrentes. Es admisible para narrativa de métodos.
- **Duda confirmada:** apéndice H.1 identifica Garman–Klass, Parkinson y Rogers–Satchell como estimadores del objetivo empírico; no asigna uno, su escala o transformación a cada tabla. B.1/Ec.21 define teóricamente varianza condicional de logretornos. Esa definición no completa la receta de las etiquetas empíricas.
- **Distinción adicional comprobada:** en III.E/Ecs.18–20, la media pronostica volatilidad y σ² representa incertidumbre predictiva. No debe confundirse esa salida de incertidumbre con la varianza de los retornos de B.1/Ec.21.
- **Horizontes:** III.E declara cabezas 1/5/30 minutos; Tabla 2 añade 15 minutos sin explicar allí esa cabeza.

**Decisión propuesta:** conservar como estudio empírico contextual de SSM multiescala/atención para volatilidad cripto. Mantener pendientes la asignación de target/escala por resultado y la elegibilidad de sus efectos. Diferir de σ7 no es motivo de exclusión narrativa. El código de otro CryptoMamba sobre precios no resuelve estos puntos.

Localizadores: III.E/Ecs.18–20, p.5 (HTML 998–1078); Tabla 2, p.7 (1274–1289); B.1/Ec.21, p.13 (2657–2687); H.1, p.16 (3383–3398). Referencias web de la sesión: `turn187519view1`, `turn619218view3`, `turn187519view0`, `turn951140view2`.

## Texto concreto propuesto para §62

La lectura primaria focal confirma el uso narrativo de dos estudios cuyos efectos permanecen sin adjudicar. Zahid et al. documentan la combinación de pronósticos GARCH con redes recurrentes; su definición intradía no queda conciliada con los cierres diarios declarados. CryptoMamba-SSM permite discutir SSM multiescala y atención, pero no asigna inequívocamente sus estimadores de volatilidad y escalas a cada tabla. En este último, la varianza de incertidumbre de la cabeza probabilística tampoco debe confundirse con la varianza de los retornos. Estas limitaciones motivan en E2 una etiqueta completamente especificada, la separación entre horizonte y ventana del objetivo y comparadores evaluados bajo un mismo contrato. No se usa ninguna de estas fuentes para atribuir una ventaja cuantitativa a las implementaciones E2.

## Qué queda realmente pendiente

No falta acceso para decidir su uso contextual. Falta información verificable que el texto focal no proporciona: construcción y alineación de la etiqueta de Zahid; estimador/escala por tabla y especificación del horizonte adicional de CryptoMamba. Una aclaración de autor, corrección o implementación identificada del estudio podría resolverlo. No se requieren esos elementos para conservar la discusión crítica anterior. No cambiar los 13 incluidos aplicados ni los dos contextos del paquete vigente con la evidencia actual.
