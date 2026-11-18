# Revisión de IA del candidato E3 congelado

**Estado: AIREVIEWONLY.** Fecha: 2026-11-18. Revisor: agente de IA con contexto nuevo respecto de los dictámenes anteriores del proyecto. No se leyeron esos dictámenes para elaborar esta revisión. Esta separación de contexto es limitada: no constituye una persona independiente, no elimina posibles sesgos compartidos entre modelos y no equivale a evaluación humana externa. `human_external_review_completed=false`; no se acredita R1 ni se asigna nota académica.

La evidencia leída permite defender una adaptación concreta: incorporar un piso algebraico de σ7, calculable en el origen del pronóstico, y aprender un exceso positivo. No basta para sostener una innovación algorítmica general frente a las parametrizaciones conocidas de salidas restringidas. Tampoco demuestra que la aplicación específica carezca de originalidad: los textos inspeccionados no exhiben exactamente el mismo piso de seis retornos conocidos. Esa originalidad limitada permanece sin establecer.

## Objeto y trazabilidad

Se inspeccionó el código público de `jassan-bit/DeepLearning_202630` en el commit **314f6e7e888b0748c8790cf3c0d9f4fc71bd6a0b**, especialmente [e3_innovation_model.py](https://github.com/jassan-bit/DeepLearning_202630/blob/314f6e7e888b0748c8790cf3c0d9f4fc71bd6a0b/src/e3_innovation_model.py#L254), `DevelopmentModel("M00")` y el codificador/atención de `original_model.py`. El objeto de este dictamen es **InnovationModel("additive_positive_head")**, no las dos variantes de media por cuadratura.

La búsqueda nativa entregada está en **edf6fe312987fd7299f6ca850a20a67e4f1c3cbc**, prefijo `runs/r1_search/20261118T202323797581Z`. Se leyó su recibo: Crossref, arXiv y OpenAlex devolvieron 10 registros por cada una de seis consultas: 180 filas de resultados, no necesariamente 180 trabajos únicos. DBLP y Semantic Scholar constan bloqueados/no disponibles. El recibo declara primera página, cribado incompleto y ningún texto completo leído por la recolección. Esa cobertura no prueba exhaustividad ni novedad. Esta revisión seleccionó las fuentes siguientes y añadió búsquedas focalizadas; no presenta las 180 filas como lecturas de cuerpo.

## Qué calcula realmente el candidato

El código usa siete cierres conocidos para obtener seis retornos logarítmicos, su media m y

```text
A(x) = (1/7) sum_{i=1}^6 (r_i - m)^2
σ7² = A(x) + (6/49)(r_futuro - m)^2
b(x) = sqrt(A(x))
σ_estimado(x,A) = b(x) + y_scale [10^-4 + Softplus(score_θ(x))]
```

La segunda igualdad se deriva expandiendo la varianza poblacional de los siete retornos alrededor de su media conjunta. Es válida para ese objetivo que comparte seis retornos observados y añade uno desconocido. No debe trasladarse sin recalcularla a una desviación con denominador 6, a siete retornos enteramente futuros, ni a una varianza condicional de la innovación.

El piso es agudo: se alcanza cuando el retorno nuevo coincide con m. La cabeza implementada queda estrictamente por encima del piso por el margen positivo fijo. La condición exige que el llamador suministre el A correcto, coherente con la ventana y las unidades; `forward` comprueba forma y no negatividad de A, no reconstruye esa coherencia.

El codificador es una convolución causal 5→8, núcleo 3, seguida de LSTM unidireccional de 12 estados y atención temporal por contenido. La revisión del código da 128 parámetros convolucionales + 1056 del LSTM + 12 de consulta + 12 de lectura + sesgo + temperatura = **1210**. El modelo M00 desactiva penalización por discordancia y saturación radial. La cabeza aditiva usa una Softplus sobre el escalar lineal; cambia la precisión de lectura a FP64 y no añade parámetros. La variable U derivada de las etiquetas es auxiliar de otras variantes: esta cabeza devuelve `scale=None`, tiene peso NLL cero y su pérdida es MSE escalado de σ7. No produce por sí misma una distribución predictiva ni intervalos calibrados.

## Comparaciones primarias efectivamente leídas

**ConstraintNet, antecedente más cercano por mecanismo.** Se leyó el cuerpo de la [prepublicación primaria arXiv:2003.10258v1](https://arxiv.org/html/2003.10258), §§3.1–3.4, ecuaciones 1–3 y 5. Formula un guardia diferenciable sin parámetros que transforma una salida latente y una restricción específica de la muestra en una salida factible para todos los pesos. La cabeza E3 comparte ese principio. Su restricción es una semirrecta analítica determinada por x; la prepublicación desarrolla principalmente politopos/sectores y entrada de restricciones ajustables. La versión editorial [AAAI 2021, DOI 10.1609/aaai.v35i8.16841](https://doi.org/10.1609/aaai.v35i8.16841) quedó bloqueada por el portal; no se presume identidad de sus detalles con la prepublicación de 2020.

**Precedente adicional de la forma algebraica.** Opschoor y van der Wel, [A Smooth Shadow-Rate Dynamic Nelson-Siegel Model for Yields at the Zero Lower Bound](https://pure.eur.nl/ws/portalfiles/portal/158886821/A_Smooth_Shadow-Rate_Dynamic_Nelson-Siegel_Model_for_Yields_at_the_Zero_Lower_Bound.pdf), DOI 10.1080/07350015.2024.2365779, publicado en línea el 15-07-2024; registro institucional: JBES 43(2), 298–311 (2025). Se leyó §2.2 y se verificó visualmente la ecuación 6 en la página impresa 4 (página 6 del PDF): límite inferior + Softplus(valor latente − límite inferior). Es evidencia previa de esa transformación escalar, con otro problema: rendimientos de bonos, límite prefijado y factores DNS, sin CNN–LSTM ni piso de σ7 dependiente de la ventana.

**OC-BNN.** Se leyó [Yang et al., arXiv:2010.10969v2](https://arxiv.org/html/2010.10969), §4, definiciones 4.1–4.4 y §§4.1–4.3. Incorpora restricciones de salida dependientes de la entrada mediante priors bayesianos condicionados o amortizados. Es antecedente del objetivo de respetar conocimiento funcional, pero su inferencia probabilística y satisfacción aproximada no son la parametrización determinista aditiva de E3. No es evidencia de implementación idéntica.

**Pronóstico probabilístico de volatilidad cripto.** Se leyó [Dudek, Orzeszko y Fiszeder, arXiv:2508.15922v1](https://arxiv.org/html/2508.15922), §§II–III, ecuaciones 1–8. Modela cuantiles de varianza realizada diaria formada con retornos intradiarios y transforma pronósticos puntuales mediante simulación de residuos, regresión cuantílica o bosques. Difiere tanto en estimando como en mecanismo. Impide presentar el pronóstico probabilístico de volatilidad cripto como terreno vacío; sus componentes no deben atribuirse a la cabeza aditiva.

**CNN–LSTM en cripto.** Se leyó [Livieris et al., Electronics 10(3), 287 (2021)](https://www.mdpi.com/2079-9292/10/3/287), §§3–5. Usa ramas CNN–LSTM para series de BTC/ETH/XRP, retardos 7 y 14 y salida lineal para precio/movimiento. Establece antecedente de arquitectura y aplicación general, no del piso examinado.

**Lectura limitada.** De [D’Amato, Levantesi y Piscopo, Physica A 596, 127158 (2022)](https://iris.uniroma1.it/handle/11573/1628787) solo se leyó el resumen institucional: red recurrente Jordan y comparación con SETAR/NAR. El cuerpo editorial respondió 403 y el repositorio restringe el PDF. No se infieren de ese resumen su activación, pérdida o fórmula exacta de objetivo.

## Alcance de las afirmaciones

- **Método conocido aplicado:** codificador CNN–LSTM/atención, activación positiva y parametrización mediante un piso pertenecen a familias establecidas. La composición concreta puede ser un diseño de ingeniería útil.
- **Propiedad teórica defendible:** la identidad de varianza demuestra el piso del objetivo especificado; la fórmula de la cabeza lo respeta para A coherente y parámetros finitos. No prueba calibración, exactitud de una esperanza condicional, superioridad ni un certificado global de sensibilidad desde OHLCV.
- **Utilidad de entrenamiento:** aprender el exceso puede evitar salidas imposibles y cambiar el sesgo inductivo. La MSE busca aproximar la media condicional de σ7 bajo las limitaciones del modelo/optimización. Esto no es evidencia empírica de mejora ni de originalidad.
- **Originalidad:** no se identificó en los cuerpos seleccionados el mismo cálculo del piso de seis retornos seguido de esta cabeza. No encontrarlo en esta selección no demuestra primera publicación. Hace falta justificar qué aporta científicamente esa adaptación además de instanciar una salida restringida conocida.

La identidad también explica por qué un objetivo rodante solapado puede resultar predecible a partir de información ya conocida. Una evaluación útil tendría que separar ese componente de la ganancia aprendida sobre controles como piso solo, piso más exceso medio, M00 y M00 proyectado. Esta revisión no ejecutó ajustes, predicciones de mercado, contrastes ni modificaciones del modelo. Ningún resultado de entrenamiento previo se utilizó para resolver la cuestión de originalidad.

**Dictamen limitado:** hay una propiedad correcta y una aplicación específica defendible. La diferencia frente a los antecedentes leídos es principalmente el cálculo del piso para este estimando y su integración en el modelo congelado. La evidencia disponible no establece una contribución algorítmica sustantiva nueva; la originalidad de la aplicación específica queda abierta. Esto es una revisión de IA trazable, sin aprobación humana externa ni certificación R1.
