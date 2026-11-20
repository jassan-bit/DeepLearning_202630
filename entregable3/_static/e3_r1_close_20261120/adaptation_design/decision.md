# Adaptación candidata delimitada: V3 existente, sin crear otra arquitectura

Preparado con asistencia de IA. Este dictamen revisa el diseño, no acredita R1,
R2 o R3 y no ejecuta ajustes, inferencia neuronal, lectura de mercado ni
remuestreos. Los 796 trabajos terminados corresponden a la referencia aditiva;
no pertenecen a V3.

## Decisión

**Conservar el candidato V3 ya implementado como la única adaptación de desarrollo
que conviene contrastar ahora.** No añadir una V4 de atención, normalización o
proyección de gradientes sólo para buscar una etiqueta de novedad. V3 modifica la
estructura de salida y el entrenamiento, de manera verificable, frente a la
guarda escalar aditiva. Esa diferencia permite presentarlo y defenderlo como
**adaptación algorítmica candidata**; no demuestra todavía que sea inédita para
el fenómeno y tarea de aplicación requeridos por R1.

La contribución que podría investigarse es estrecha: aprender la ley del único
componente futuro no observado de una volatilidad móvil solapada, transformarla
por su identidad exacta y puntuar simultáneamente la ley de ese componente y la
media de la volatilidad resultante. Una descripción como «red probabilística para
volatilidad» sería demasiado amplia y contradice antecedentes conocidos.

## Objeto y modificación concreta

Sean seis rentabilidades conocidas, su media m, y la futura r. Defínanse

```text
A = (1/7) Σ_{i=1}^6 (r_i-m)²,  U = |r-m|,  c=6/49,
Y=σ₇ = g_A(U) = sqrt(A+c U²).
```

La referencia aditiva de 1210 parámetros aprende directamente una predicción
puntual `sqrt(A)+sY·(10^-4+softplus(z))`. V3 reutiliza su codificador y atención,
pero sustituye esa lectura por tres posiciones positivas ordenadas `a_k(x)` y
masas `p_k(x)` aprendidas:

```text
a_k = sU Σ_{j≤k} (10^-4+softplus(z_j)),
p = softmax(l₁,l₂,0),
G_x = Σ_k p_k δ_{a_k},
μ_G(x)=Σ_k p_k sqrt(A(x)+c a_k²),
L = ((μ_G-Y)/sY)² + λ CRPS(G_x/sU,U/sU),  λ=0,1.
```

`A` es observable al pronosticar. `U` y `Y` realizados son etiquetas únicamente
de entrenamiento. `sU` y `sY` se estiman sólo con ajuste. La suma finita es exacta
bajo **la ley del modelo** y no bajo la verdadera ley de mercado desconocida.
Los valores K=3, λ=0,1 y la parametrización corresponden al prototipo preparado
el 18 de noviembre; esta revisión no los retunea con los resultados de 8.18.

Tres diferencias programadas pueden explicarse línea por línea: (i) el estado de
salida tiene átomos y probabilidades, no un exceso escalar; (ii) la transformación
depende del componente observado de cada muestra y se aplica **antes** de
promediar; (iii) el entrenamiento incluye una puntuación propia de la etiqueta U.
Cambiar de CNN, ancho, activación o datos no es el argumento de originalidad.

## Qué es conocido y qué queda como adaptación específica

| Elemento | Antecedente y alcance de lectura | Conclusión delimitada |
|---|---|---|
| Red que estima una distribución condicional | Bishop (1994), resumen institucional primario. El PDF fue abierto, pero su extracción es ilegible; no se declara nueva lectura completa. | Conocido. No reclamar que V3 inventa la predicción neuronal de leyes o mezclas. |
| MDN aplicada a volatilidad financiera | Schittenkopf, Dorffner y Dockner (1998), §2, pp. impresas 2–3, ecuaciones (1)–(9); §3, p.5, usa ATX y retornos cuadrados. Lectura focal del cuerpo primario. | Conocido desde antes de las redes actuales; volatilidad financiera no es un dominio virgen para salidas distribucionales. Su objetivo no es por ello σ₇ poblacional centrada con seis retornos ya observados. |
| Mezcla profunda y GARCH | Feng, Li y Zhang (2023), resumen editorial primario: red autorregresiva profunda y mezcla gaussiana para probabilidad de volatilidad bajo GARCH. Cuerpo no recuperado en esta revisión. | Antecedente de aplicación probabilística, no evidencia de la composición exacta de V3 ni de su ausencia. |
| Distribución categórica y softmax | Bellemare, Dabney y Munos (2017), §4.1 p.5 y §4.2 pp.5–6 del PDF. Átomos de soporte fijo y actualización Bellman proyectada con KL. | Conocido. V3 aprende posiciones y masas y usa CRPS supervisado; no implementa Bellman, C51 o su garantía de contracción. C51 es trabajo cercano, no la fuente de una implementación copiada. |
| CRPS y propiedad estricta | Gneiting y Raftery (2007), §4.2, p.367, ecuación (21); lectura focal ya registrada por el prototipo. | Conocido. La orientación de pérdida y la normalización se conservan; no se inventa una regla de puntuación. |
| Restricciones de salida por muestra | ConstraintNet, antecedente focal de la referencia aditiva. | La garantía de soporte no es nueva. V3 añade una ley y entrenamiento sobre U, no sólo otro enlace positivo. |
| Transformación específica del objetivo solapado | Identidad elemental de varianza y suma de esperanza bajo una ley finita. | Adaptación de estructura y entrenamiento a contrastar; ni la identidad ni una transformación de una ley son teoremas nuevos. La prioridad de esta composición exacta sigue sin establecerse. |

## La limitación que puede justificarse

La limitación matemática concreta corresponde al **control de sustitución por
segundo momento**: conocer `E[U²|x]` no basta para conocer
`E[sqrt(A+c U²)|x]`. Dos leyes con el mismo segundo momento pueden producir
medias de σ₇ diferentes. Es una consecuencia de la transformación no lineal.

**Esto no es una incapacidad de la referencia aditiva.** Una cabeza escalar
flexible entrenada directamente con MSE puede representar la media condicional
óptima de Y. La implementación aditiva incluso puede representar la misma media
puntual que una ley V3, siempre que el exceso normalizado supere su incremento
mínimo. V3 no establece una clase de funciones puntuales universalmente más rica
ni una mejora garantizada de RMSE. La justificación investigable es una
**restricción inductiva y señal auxiliar de entrenamiento**, con posibles
beneficios o perjuicios de muestra finita y especificación.

La mala cobertura y el deterioro con ruido observados en 8.18 son diagnósticos
reales del modelo aditivo; no prueban que una ley de tres átomos los corrija.
Tampoco se deriva calibración de mercado del mero uso de un score propio.

## Propiedades formales que sí se pueden defender

1. **Soporte e identidad:** con A≥0 y masas válidas, cada átomo transformado está
   en `[sqrt(A),∞)`, y su media también. La salida calcula exactamente
   `E_G[g_A(U)]` bajo la ley finita programada.
2. **Jensen y límite de aproximación:** para V=U² en `[vmin,vmax]`, A>0,

   ```text
   (c²/8) Var(V)/(A+c vmax)^(3/2)
      ≤ sqrt(A+c E[V]) − E[sqrt(A+c V)]
      ≤ (c²/8) Var(V)/(A+c vmin)^(3/2).
   ```

   Es el límite estándar de segundo orden: `g''(v)=-c²/[4(A+cv)^(3/2)]` y el
   término lineal de Taylor alrededor de E[V] tiene esperanza cero. No es un
   teorema nuevo del proyecto. El caso A=0 se trata aparte; no se promete una
   cota uniforme finita si el soporte incluye cero.
3. **Utilidad bajo especificación correcta, exclusivamente contra plug-in:** si
   la ley G es la verdadera ley condicional de U, la media exacta `μ_G` minimiza
   el riesgo cuadrático. Para `b=sqrt(A+c E_G[U²])`,

   ```text
   E[(Y-b)²|x] − E[(Y-μ_G)²|x] = (b-μ_G)² ≥ 0.
   ```

   Esta comparación es con el control plug-in, no con la referencia aditiva,
   que también puede aprender μ_G. Bajo mala especificación no hay signo
   garantizado para la ventaja frente al plug-in o cualquier otro predictor.
4. **Riesgo conjunto correctamente especificado:** si la ley real pertenece a
   la clase representable, es alcanzable en el mínimo poblacional global y se
   cumplen los momentos necesarios, la estricta propiedad de CRPS y la
   descomposición cuadrática hacen que esa ley minimice ambos términos. Tres
   átomos, capacidad finita y entrenamiento práctico no satisfacen en general
   esas hipótesis. No se heredan convergencia del optimizador o consistencia
   universal.

La verificación `analytic_checks.py` evalúa 100 leyes finitas artificiales, las
cotas de curvatura y la identidad de diferencia de riesgo. Incluye dos controles
de honestidad: una cabeza aditiva que reconstruye el mismo punto, y un par de
aproximaciones con tres átomos donde añadir CRPS con λ=0,1 mejora el riesgo
conjunto pero empeora el riesgo puntual. No hay entrenamiento ni uso de mercado.

## Hipótesis refutable y controles mínimos

**Hipótesis candidata, no preregistrada:** bajo un presupuesto y particiones de
desarrollo comunes, aprender la ley del componente U y usar su esperanza
transformada proporciona una mejora práctica predefinida frente a la referencia
aditiva y frente a las ablaciones de la propia ley. La magnitud δ, semillas,
familia y regla de decisión deben quedar fijadas antes de la nueva ejecución;
los resultados de 8.18 no se usan para rellenar una predicción anterior.

Predicciones secundarias que pueden refutar la explicación:

- Con **la misma ley ya aprendida**, la diferencia exacto/plug-in coincide con
  el Jensen-gap y está dentro de las cotas. Si falla, hay un error de formulación
  o implementación, no una contribución.
- Una mejora atribuida al aprendizaje de la ley debe acompañarse de mejor score
  de U fuera del ajuste y debe examinarse por bins de Jensen-gap **predicho al
  origen**, fijados sin objetivos futuros. Que falle esa relación debilita la
  explicación; no obliga a que la aditiva sea peor por identidad matemática.
- Retirar CRPS, fijar masas y sustituir la agregación por el plug-in deben
  examinarse por separado. Si no se detecta un aporte, no se atribuye causalmente
  la ganancia a ese componente. La variación de capacidad y el ruido deben
  conservarse como limitaciones.

| Control | Aislamiento y condición |
|---|---|
| Aditiva histórica | Comparador principal conservado; 1210 parámetros frente a 1262 de V3, diferencia declarada. |
| `atom_mse_only` existente | Misma cabeza, 1262 parámetros e inicio; λ=0 aísla la pérdida auxiliar. |
| `fixed_probs` existente | 1236 parámetros; masas uniformes aísla parcialmente su aprendizaje. No afirmar igualdad de capacidad ni añadir parámetros inactivos para aparentarla. |
| Plug-in del segundo momento | Misma ley y `sqrt(A+c Σp a²)` en vez de `Σp sqrt(A+c a²)`; primero control funcional pareado, y luego una variante entrenada bajo política común si se decide contrastar su efecto de aprendizaje. Este control aún requiere integración explícita. |
| Modelo de ley de Y directa, opcional | Igual número de salidas y score correctamente normalizado, para investigar si la ventaja procede de descomponer el solapamiento. No es requisito ya ejecutado ni motivo para ampliar sin límite el piloto. |

La guía exige al menos tres componentes propios aislados en la ablación. No
declarar satisfecho ese requisito por nombrar controles sin ejecutarlos ni
confundir las tres longitudes de bloque con tres pruebas distintas.

Un piloto mecánico puede estudiar estas ablaciones, pero la comparación final
debe conservar los rivales fuertes ya identificados. En particular, no retirar
`floor_plus_mean_excess` —el rival más cercano en los cuatro conjuntos de 8.18—
ni la aditiva histórica al añadir V3. Conservar los 19 sistemas actuales y añadir
V3 da 20 sistemas en esa comparación; el ejecutor/planner histórico de 18 no
queda automáticamente actualizado ni dimensiona esa familia nueva.

## Pseudocódigo y frontera de ejecución

```text
Entradas de diseño: fuente del modelo, datos de desarrollo y particiones,
  escalas de ajuste, K=3, λ=0,1, variantes y semillas, presupuesto HPO/ajuste,
  δ_dev, regla de continuación y fallos, fechas reales/tag del nuevo plan.
Antes de ajustar: verificar fuente, hashes, separación de diseño/confirmación.
Para cada receta y semilla predefinida:
  estimar escalas sólo en ajuste;
  por lote: derivar A de historia observable; formar etiquetas U,Y sólo en ajuste;
  codificar X → contexto; producir átomos ordenados y masas;
  transformar cada átomo con sqrt(A+c a²); sumar con sus masas;
  calcular MSE normalizado y CRPS_U normalizado; actualizar parámetros;
  detener por parada/presupuesto comunes, con fallos y consumo conservados;
  elegir receta sólo por validación según el criterio fijado;
  guardar pesos, configuración, escalas y selección antes de scoring.
Comparar todas las variantes, incluidas las desfavorables, en desarrollo;
  conservar o descartar V3 según la regla anterior, sin certificar R1/R3.
Sólo después: fijar una versión y protocolo confirmatorio nuevos, dimensionar
  potencia por contraste/conjunta y ≥10 semillas o más si lo exige el análisis.
```

Los cuatro conjuntos D00–D03 ya tienen resultados expuestos. Si sus resultados
guían esta adaptación, deben considerarse desarrollo o evaluación exploratoria
para la versión nueva; no heredan cegamiento ni validez confirmatoria. La
selección y admisión de evidencia realmente reservada necesita su propio plan.
Una versión nueva no puede reutilizar los 796 trabajos como entrenamientos V3.

## Fuentes primarias y alcance

- Bishop (1994), [página institucional](https://www.microsoft.com/en-us/research/publication/mixture-density-networks/), resumen; cuerpo PDF ilegible en extracción.
- Schittenkopf, Dorffner y Dockner (1998), [PDF institucional](https://research.wu.ac.at/ws/portalfiles/portal/18954278/document.pdf), §2 ecuaciones (1)–(9), §3 caso ATX; DOI 10.57938/a6057c38-dbad-42dc-bebc-bd2c71fe537f. No confundirlo con el reporte OFAI de 1999 cuyo acceso agotó el tiempo.
- Feng, Li y Zhang (2023), [editorial primaria](https://aimspress.com/article/doi/10.3934/era.2023194), resumen y metadatos; no se atribuye lectura del cuerpo.
- Bellemare, Dabney y Munos (2017), [PDF PMLR](https://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf), §4.1–4.2; soporte fijo y Bellman/KL, distintos del entrenamiento supervisado V3.
- Gneiting y Raftery (2007), [PDF primario](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), §4.2, ecuación (21), lectura focal ya registrada por el prototipo.

**Dictamen:** V3 tiene una diferencia estructural y de entrenamiento explícita
frente a la guarda escalar, suficiente para investigarlo como adaptación
candidata. La prioridad de su composición específica y su contribución útil
siguen sin establecerse. No debe presentarse como R1 acreditada ni como una
garantía de mejor RMSE, cobertura o robustez.
