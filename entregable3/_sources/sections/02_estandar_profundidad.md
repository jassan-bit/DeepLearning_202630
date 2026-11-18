# 2 Estándar de profundidad exigido

La guía establece un único estándar: **profundidad teórica con evaluación rigurosa**. Los cinco requisitos siguientes se mantienen simultáneamente. La evidencia favorable ya obtenida no los sustituye ni permite reducirlos después de ver los resultados.

| Requisito | Exigencia de la guía | Evidencia actual y trabajo restante |
|---|---|---|
| Sustento teórico | Al menos una propiedad demostrada formalmente, con supuestos explícitos y verificación numérica | Se distinguen la cota del primer módulo de 1211 parámetros, la teoría de momentos y una cota condicional propia de la cabeza aditiva de 1210 parámetros en [7.3](07_diseno_riguroso.md). Esta última tiene comprobación sintética, con piso y parámetros fijos y estados acotados. No demuestra novedad, estabilidad desde OHLCV ni beneficio predictivo; falta conectar la propiedad con la contribución y la evidencia real exigidas. |
| Evaluación amplia | Al menos tres conjuntos adicionales de naturaleza distinta, elegidos antes de conocer sus resultados | No se han seleccionado ni evaluado para E3. Las 608 fechas BTC compartidas y las diez semillas no son varios conjuntos independientes. |
| Familia completa | Comparación frente a todo el Top 10 y todos los baselines fuertes, con corrección de Holm | Se conservaron los **16 rivales**, y el análisis posterior calculó los **48 contrastes** de tres procedimientos con Holm global. Esto aporta evidencia favorable bajo sus supuestos; no convierte el calendario conocido en confirmación independiente. |
| Potencia y repeticiones | Potencia de al menos **0,90 por contraste**, potencia conjunta reportada y al menos **10 semillas**, o más si el dimensionamiento lo exige | E3 y las trece redes E2 ya tienen diez semillas; los clásicos tienen una trayectoria. **La potencia y el número suficiente de repeticiones no se han dimensionado prospectivamente** para la campaña exigida. |
| Pruebas por contraste | Mínimo **tres pruebas distintas**, apropiadas para el problema, con hipótesis, supuestos y control familiar justificados | Se ejecutaron bootstrap temporal, HAC de influencia de RMSE y t pareada entre semillas. Sus ámbitos temporal y de inicialización están documentados en [8](08_verificacion_estadistica.md). La adecuación y combinación deben justificarse para la afirmación y la ruta F2a; contar tres p-valores no cierra por sí solo el requisito. |

La guía exige pruebas **distintas y apropiadas**; no establece como condición literal que todas tengan idéntico ámbito de incertidumbre. La diferencia de alcance de los procedimientos debe explicarse y justificarse, sin inventar un requisito adicional de «homogeneidad» ni ignorar los supuestos de cada prueba.

## 2.1 Declaración del estándar y cronología

Se incorpora el documento [experiments/estandar.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/estandar.md), con **fecha real del 17 de noviembre de 2026**, para documentar el estándar y los pendientes de una futura campaña confirmatoria. Es posterior a los pilotos, entrenamientos y resultados conocidos. **No es un preregistro de esos experimentos, no acredita una firma humana y su registro con tag previo al futuro protocolo sigue pendiente.**

La guía exige una declaración fechada y un **tag Git real anterior al congelamiento del protocolo**. Publicar ahora el documento no retrofecha el compromiso ni completa automáticamente esa condición. La plantilla A.1 del [apéndice](apendice_a_plantillas.md) se conserva como plantilla histórica y no se transforma en una declaración suscrita.

El estándar no se relaja por falta de tiempo o por un resultado desfavorable. Si un análisis de potencia válido requiere más semillas, se aumentan **antes de congelar la campaña confirmatoria**, sin reducir la familia de rivales o el número de pruebas para obtener rechazo. La extensión posterior ya realizada de seis a diez semillas queda identificada como tal y no se presenta retrospectivamente como dimensionamiento previo.

## 2.2 Métrica, márgenes y afirmaciones

La primaria es la **media de RMSE individuales por ejecución**, en unidades originales de sigma7; no la RMSE de un ensemble. Para cada rival se orienta la ganancia como RMSE del rival menos RMSE de E3, definida en la ecuación {eq}`eq-e3-primary-mean-rmse-gain`. Se conserva el margen heredado

$$
\delta=0{,}00025772842664981106.
$$ (eq-e3-practical-margin-standard)

El borrador heredó **el mismo valor numérico para δg**, pero su hipótesis es diferente: δ se refiere a una ganancia entre modelos y δg a una brecha entre ajuste y evaluación comparable. Compartir valor numérico no hace intercambiables sus significados. Falta justificar y ejecutar la evaluación de brecha del candidato bajo un diseño apropiado; los diagnósticos CPU de E2 no la sustituyen.

La superioridad exigida por la guía incluye relevancia práctica, tamaño de efecto, adecuación, estabilidad de la conclusión y consistencia en los conjuntos adicionales. El resultado posterior favorable frente a los 16 rivales es evidencia con alcance concreto; ni una media menor, ni una cota formal, ni la pertenencia al MCS certifican por sí solas el conjunto de condiciones de R3. No rechazar una diferencia tampoco demuestra equivalencia.

## 2.3 Estado del compromiso y viabilidad

| Registro necesario | Estado real |
|---|---|
| Estándar escrito y fecha actual | Documento incorporado; declara expresamente su carácter posterior a los resultados históricos |
| Tag anterior al futuro protocolo | Pendiente; no se inventa un tag ni se usa uno de otro experimento |
| Viabilidad de cómputo | Hay tiempos reales de las ejecuciones terminadas; falta dimensionar la campaña completa, incluidos tres conjuntos y la potencia requerida |
| Número de semillas suficiente | Diez observadas por red; suficiencia por potencia pendiente |
| Conjuntos y propiedad a comprometer | Selección previa de los tres conjuntos y propiedad pertinente de la receta/contribución final, pendientes |
| Revisión y responsabilidad humana | La asistencia de IA está declarada; revisión, comprensión y defensa de la persona autora pendientes de acreditación |

La [matriz inicial de requisitos](01_alcance_requisitos.md) se conserva como anexo de trazabilidad. La [sección 8](08_verificacion_estadistica.md) distingue el análisis posterior ejecutado del diseño confirmatorio pendiente. Las [secciones 9](09_dashboard_predictivo.md) y [10](10_despliegue_render.md) especifican el producto; la [14](14_evaluacion_calificacion.md), las condiciones de evaluación; y la [15](15_integridad_academica.md), las reglas de integridad. Ninguna de estas matrices asigna una nota o convierte un pendiente en evidencia cumplida.
