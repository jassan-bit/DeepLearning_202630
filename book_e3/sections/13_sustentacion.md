# 13 Sustentación

La guía exige **20 minutos de exposición sobre el dashboard desplegado**, seguidos de preguntas. Este capítulo es un guion de preparación, no una defensa realizada. El servicio preparado tiene Contexto, EDA y Modelos, catálogo de 17 sistemas y cinco predictores ONNX CPU, incluido TSMixer; doce sistemas muestran resultados precalculados. El autor comunica Live en la [URL Render](https://deeplearning-e3-dashboard.onrender.com); falta la aceptación pública de la versión con su commit y humo de cinco modelos. Las pruebas de software no acreditan comprensión individual del autor.

## 13.1 Distribución del tiempo

Los cinco bloques que enumera la guía suman 18 minutos. La agenda siguiente conserva esos tiempos y propone los dos minutos restantes para transiciones y margen operativo. Las preguntas van después de la exposición.

| Ventana propuesta | Duración | Contenido obligatorio | Evidencia que deberá acompañarlo |
|---|---:|---|---|
| 00:00–02:00 | 2 min | Problema, datos y hallazgos principales del EDA. | Objetivo σ7 de siete retornos, h=1 sin anualización, fuente/periodo, figura temporal interpretada y límites de datos. |
| 02:00–05:00 | 3 min | Benchmark, mejores configuraciones y predicción en vivo con explicación local. | Tabla de 17 sistemas; configuración elegida en validación; entrada admitida, pronóstico, intervalo y explicación del mismo artefacto. E3 debe identificar su perfil posterior individual, límite temporal y cobertura histórica; no trasladar la media de diez semillas. |
| 05:00–10:00 | 5 min | Modelo original, limitación, relación con paper guía, fuente, mecanismo, demostración y novedad. | Ecuaciones, pseudocódigo, código, supuestos, verificación numérica y revisión de antecedentes con su nivel de evidencia. |
| 10:00–16:00 | 6 min | Protocolo y evaluación estadística/ampliada. | Familia completa, pruebas/supuestos, Holm, efectos/IC, potencia por contraste/conjunta, calibración, robustez, coste y veredicto respaldado. |
| 16:00–18:00 | 2 min | Conclusiones, limitaciones y trabajo futuro. | Qué está demostrado, qué no, escenarios adversos y tareas restantes, sin declarar superioridad no acreditada. |
| Margen hasta 20:00 | 2 min | Transiciones, cambio de vista y contingencia operativa. | Ensayo cronometrado; no sustituye contenido ni extiende la exposición a la sesión de preguntas. |

La demo deberá usar la URL pública y una entrada precargada verificada por cada capacidad que se exhiba. Se registrarán versión, estado del servicio y procedencia del ejemplo. Las capturas o el vídeo de contingencia se identificarán como material grabado y no como una predicción ejecutada en vivo.

## 13.2 Preparación de la demostración

El ensayo local puede recorrer Contexto, EDA y Modelos y usar MLP16, LSTM32, GRU32, CNN-LSTM E3 y TSMixer mediante ONNX Runtime CPU. Los otros doce sistemas conservan resultados precalculados: nueve redes esperan exportación/paridad y los tres clásicos requieren historia o estado adicional al contrato de siete barras. No se justifica su ausencia por memoria sin evidencia. La tabla resume diez semillas por red; cada curva e inferencia usa la semilla representante 271801, fijada por su ID. Se distinguirán esas unidades durante la exposición.

Se prepararán los ejemplos históricos verificados y una entrada CSV válida de siete barras diarias consecutivas anteriores a la fecha objetivo. Los tres ejemplos conservados se fijaron por posiciones cronológicas, no por aciertos del modelo. La predicción es volatilidad poblacional σ7 a un día, sin anualización; no es una probabilidad de subida ni una predicción del precio. Los cuatro predictores E2 habilitados, incluido TSMixer, reutilizan sus cuantiles congelados para el nominal del 90 %. E3 adopta un perfil posterior separado de la semilla 271801, con `q = 0.005402463770651868` y `[max(0, predicción−q), predicción+q]`. Sus 53 objetivos de calibración van del 8 de noviembre al 30 de diciembre de 2024 y el runtime sólo lo aplica a objetivos posteriores al último día. Para fechas anteriores o iguales, muestra el punto sin intervalo.

La cobertura histórica individual E3 es **94,08 %** sobre 608 fechas y **85,56 %** en 2026Q1, diferente de la media de diez semillas y sin garantía futura o por grupo. La verificación ORT de 661 ventanas conserva q y la cobertura de evaluación; el redondeo de un caso situado en el cuantil cambia la cobertura de calibración de 49/53 a 48/53, sin reajustarlo. La demostración debe exponer estos límites y la adopción posterior del intervalo; no presentar el calibrador como parte del entrenamiento original o del nuevo prototipo V3.

La sensibilidad local por intervenciones del mismo artefacto no se presentará como una atribución causal o como fidelidad y estabilidad globales ya demostradas. El aviso fuera de dominio sólo contrasta rangos observados en ajuste: su ausencia no certifica fiabilidad. El historial E3 de la semilla 271801 conserva 40 épocas y mejor época 38; la pérdida normalizada durante entrenamiento y la RMSE de validación tienen ejes y unidades distintos.

La [prueba de humo pública](10_despliegue_render.md) recorrerá las páginas y cada modelo habilitado, con sus estados, errores, intervalo/explicación disponibles y tiempos medidos. Distinguirá arranque frío y respuesta caliente y conservará la URL, versión y registros. Las comprobaciones locales del capítulo 9 preparan ese recorrido; no lo sustituyen. Las capturas o vídeos de contingencia identificarán su versión y carácter grabado.

La explicación debe distinguir el líder por validación disponible de E2 del líder descriptivo de prueba, una métrica de cohorte de una predicción de semilla representante y un nominal de cobertura de su verificación empírica. Se mantendrán visibles la exposición histórica del test y los límites de la evaluación del candidato.

| Evidencia disponible antes del ensayo | Alcance que se puede defender |
|---|---|
| [Aplicación y capturas verificadas](09_dashboard_predictivo.md) | El corte histórico de cuatro ONNX tiene 25 casos unitarios y seis recorridos de navegador; el contenedor local también pasó los seis. El corte actualizado tiene recibos separados de TSMixer y del perfil posterior E3 en [11.6](11_informe_tecnico.md). Son pruebas de software, no sustentación o reproducción humana. |
| [Exportaciones e ida y vuelta](05_configuraciones_exportacion.md) | Cinco ONNX contrastados sobre 608 ventanas por modelo, con tolerancia absoluta $10^{-6}$ en σ7. La extensión TSMixer y el perfil E3 comprueban también 53 ventanas de calibración, sin reajuste. La paridad no es evaluación confirmatoria; el perfil posterior E3 conserva su propio alcance. |
| [Pruebas del módulo científico](06_modelo_original.md) | Se conservan 61 casos aprobados en el entorno local. El CI del dashboard registra 60 aprobados y una omisión opcional de la referencia SciPy para cuantiles de otra cabeza; no se cuentan como 61 casos ejecutados en ese CI ni como evidencia de dominio personal. |
| Servicio Render, ensayo y defensa | La URL pública y su prueba de humo deben acreditarse en [10](10_despliegue_render.md). No se registran aquí como realizados el ensayo cronometrado, la exposición o las respuestas del autor. |

## 13.3 Preguntas y comprobaciones técnicas

| Pregunta prevista por la guía | Preparación y estado actual |
|---|---|
| ¿Qué limitación concreta del paper guía o métodos existentes resuelve el mecanismo y qué evidencia la sustenta? | Presentar la limitación y su evidencia de [6](06_modelo_original.md) y [7](07_diseno_riguroso.md), separadas de la ganancia descriptiva. La receta evaluada tiene 1210 parámetros y salida aditiva positiva; no lleva la penalización/saturación del primer diseño ni la cabeza de momentos. Sus diez semillas dan RMSE media 0,0029394324 en 608 fechas conocidas. Ese resultado no demuestra por sí solo que haya corregido causalmente la limitación propuesta. |
| Si proviene de otra aplicación, ¿qué adaptación algorítmica hizo y por qué era necesaria? | Mostrar fuente original, formulación anterior, cambio programado y comparación con los trabajos cercanos. Si se defiende un mecanismo propio, sustentar igualmente su diferencia con la literatura. No basta trasladar datos o cambiar hiperparámetros; la revisión científica de novedad sigue abierta. La verificación externa se aporta por la modalidad de IA declarada de 6.13, conforme a la indicación docente comunicada. |
| ¿Qué propiedad demuestra y qué supuestos no cumplen exactamente los datos reales? | Derivar la cota de la cabeza aditiva de [7.3](07_diseno_riguroso.md), con piso, escala y pesos fijos y filas latentes acotadas. Allí $K/a_{\rm tr}=77{,}32121111929344$; se comprobaron 1728 pares sintéticos y 96 normas de gradiente, sin violaciones observadas con la tolerancia declarada. La prueba formal, no el número finito de ensayos, sostiene la desigualdad. No hay una cota del encoder respecto de OHLCV crudo ni una demostración de que esa propiedad cause la menor RMSE. |
| ¿Cuántas variantes $N_d$ probó y cómo controló selección sobre validación? | Mostrar el [inventario y la bitácora](07_diseno_riguroso.md): el corte inicial de cuatro variantes artificiales, ocho recetas de desarrollo completadas y diez réplicas de una receta son conteos distintos. $N_d$ global sigue sin conciliar. La consolidación posterior conserva las fechas originales y no convierte decisiones conocidas en preregistro ni corrige por sí sola el sesgo del ganador. |
| Al retirar un componente, ¿qué predijo la hipótesis y qué observó? | Escribir la predicción secundaria antes de mostrar el resultado y distinguir qué receta se interviene. Las pruebas funcionales del primer diseño y los controles de los pilotos no completan la ablación estadística del aporte finalmente defendido. El factorial, las interacciones con IC y el control de capacidad correspondientes siguen pendientes. |
| ¿Quién integra la familia y cómo corrigió comparaciones múltiples? | Enumerar los diez del Top 10, MLP16, LSTM32, GRU32, Persistence, SeasonalNaive y ARIMA: 16 rivales frente a E3. Mostrar en [8](08_verificacion_estadistica.md) los 48 contrastes posteriores, Holm global a 0,05 y el máximo de los tres p ajustados por rival. La nula de ganancia usa $\delta=0{,}00025772842664981106$. Los procedimientos temporal e inicialización tienen alcances distintos; los 48 rechazos y la permanencia de E3 entre diez sistemas del MCS90 no acreditan automáticamente R3 o un ganador único. |
| ¿Qué MDE y potencia conjunta alcanzó? | No se han dimensionado prospectivamente para este candidato. La guía pide un piloto con al menos cinco semillas, potencia de al menos 0,90 por contraste y potencia conjunta reportada. Diez semillas observadas y p pequeños no sustituyen ese cálculo. No trasladar la potencia histórica de E2 ni multiplicar marginales bajo independencia no justificada. |
| ¿Dónde no supera a los benchmarks y por qué? | Conservar el primer piloto desfavorable y el diagnóstico posterior de las 18 fechas de objetivo alto: la aditiva tuvo RMSE 22,50 % mayor que M00 proyectado. Ese grupo usa volatilidad realizada y no define un régimen observable al pronosticar. Examinar los errores compartidos en Modelos y presentar las causas como hipótesis por comprobar, sin ocultar el resultado adverso. |
| ¿Qué cambia al retirar el conjunto adicional más desfavorable? | D01 WIKI/AAPL está admitido y D02 ETHUSDT Spot/D03 BTCUSD_PERP seleccionados, pero no hay conjuntos adicionales evaluados por modelos ni un resultado que recalcular. Una futura sensibilidad debe mostrar la conclusión con y sin cada conjunto y mantener visible el análisis completo. No permite eliminar el conjunto desfavorable para mejorar la afirmación principal; los usados para rediseñar pasan a desarrollo. |

Durante las preguntas el docente podrá pedir derivar un gradiente, modificar una pérdida o parte del mecanismo y ejecutar una prueba unitaria o un experimento reducido. La persona autora debe poder explicar qué cambia en la formulación, los supuestos, la cota y los tests. Que el código pase CI no demuestra ese dominio técnico. La imposibilidad de explicarlo es evidencia en contra de las condiciones R1 y R2, según la guía.

## 13.4 Práctica técnica, ensayo y trazabilidad

Para preparar una derivación breve, la persona autora deberá explicar la salida $\widehat y=\sqrt A+a_{\rm tr}[10^{-4}+\operatorname{softplus}(z)]$, con $A$ observable al origen y $a_{\rm tr}$ calculado sólo en ajuste. Con ambos fijos, $\partial\widehat y/\partial z=a_{\rm tr}\operatorname{sigmoid}(z)$. Para la pérdida de un caso $\ell=(\widehat y-y)^2/a_{\rm tr}^2$, resulta $\partial\ell/\partial z=2(\widehat y-y)\operatorname{sigmoid}(z)/a_{\rm tr}$; al promediar un lote aparece además su factor $1/B$. Debe poder conectar esa cuenta con el `forward`, el escalado, autograd y los supuestos de la cota. Repetir esta fórmula no acredita que haya comprendido o defendido el código.

Los comandos se preparan **antes** de la exposición, desde la raíz de un checkout completo y en los entornos separados fijados en el repositorio. El entorno científico se instala con `requirements-e3-original-cpu.lock.txt`; una comprobación breve de los parámetros compartidos, pérdida/gradientes y guardado/carga es:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest \
  tests/test_innovation_model.py \
  -k 'three_variants_share_seeded_parameters or normalized_loss_is_exactly_specified or mixed_precision_state_roundtrip' -q
```

Estos casos usan ventanas y etiquetas sintéticas y abarcan las tres cabezas v2; no se atribuye todo el conjunto exclusivamente a la aditiva. Si el docente pide un cambio, se trabajará sobre una copia de práctica identificada por revisión, indicando qué comportamiento o prueba debería cambiar antes de ejecutarla. El resultado no reemplaza la versión congelada ni una ablación de mercado.

En el entorno de inferencia fijado por `requirements.txt`, el siguiente recorrido comprueba admisión de CSV, transformación/piso, intervalos disponibles con su límite temporal, caché, integridad y sensibilidad de los cinco artefactos:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover \
  -s tests -p 'test_e3_dashboard_inference.py' -v
```

Esta segunda comprobación sí realiza inferencias CPU de ingeniería sobre pesos conservados; no entrena ni recalibra. No demuestra dominio individual, fidelidad causal o disponibilidad de Render. La prueba de humo sobre la URL pública seguirá el procedimiento del capítulo 10.

El ensayo deberá registrar fecha, participantes, versión, duración por bloque, ejemplo usado y fallos/acciones pendientes. Se prepararán comandos concretos en el entorno fijado y con insumos verificados; la prueba de software no se describirá como una campaña científica completa.

No se afirma que el ensayo o la sustentación hayan ocurrido. El [anexo final](cierre_reproducibilidad.md) mantiene separados el uso de IA, la revisión personal pendiente y las firmas; la [sección 14](14_evaluacion_calificacion.md) conserva los criterios de evaluación sin asignar calificación actual.
