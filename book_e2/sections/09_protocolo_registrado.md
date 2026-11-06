# 9. Preregistro, potencia y piloto

**Estado al 06/11/2026:** protocolo versionado antes de iniciar el piloto Bitcoin de esta etapa; P0 y su evaluación fuera de muestra en desarrollo están **ejecutados y analizados**, con 220 trayectorias y 440 ajustes cerrados. El HPO, el piloto P1, los ajustes definitivos y el benchmark final siguen pendientes. La guía completa de **65 páginas** ya fue recibida y revisada: los pendientes actuales corresponden a implementación y evidencia experimental.

El registro público es el commit [`599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0`](https://github.com/jassan-bit/DeepLearning_202630/commit/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0), con recibo de registro del **06/11/2026 a las 14:52:32 UTC**. El inicio de P0/OOF se condicionó a ese registro y a sus verificaciones previas. Registrar decisiones ahora mejora el procedimiento prospectivo; **no devuelve el cegamiento al test histórico conocido ni convierte E2 en una confirmación independiente**.

## 9.1 Diecinueve decisiones documentadas

La siguiente matriz organiza diecinueve decisiones operativas. Las reglas quedan especificadas; su registro no afirma que los experimentos ni todas las verificaciones finales estén terminados. El R definitivo y las recetas finales se obtendrán mediante las reglas registradas, sin completar esos campos con resultados imaginados.

| N.º | Decisión | Regla registrada y alcance |
|---|---|---|
| 1 | Tarea y unidad | Ruta F2a de series de tiempo, horizonte diario h=1 y una única serie BTCUSDT. Semillas, fechas y meses no se cuentan como nuevos mercados independientes. |
| 2 | Entradas y objetivo | Se conserva D7×5 OHLCV y σ₇ como DE poblacional, `ddof=0`, de siete retornos logarítmicos diarios. No se añaden rezagos de σ₇ a las redes. |
| 3 | Identidad de datos | Mantener la partición y el contrato E1; exigir índices, versiones, hashes y auditoría del pipeline real antes de ejecutar cada fase correspondiente. Un hash identifica datos, no demuestra ausencia de fuga. |
| 4 | Métrica | Media entre semillas de la RMSE por trayectoria en escala original y sobre fechas elegibles comunes. Las métricas secundarias de §4/§6 conservan su cobertura y papel complementario. |
| 5 | Mejora mínima | δ=**0,00025772842664981106** unidades de volatilidad diaria. No se recalcula con el resultado del nuevo benchmark. |
| 6 | Sobreajuste | δg=δ como tolerancia operativa previa para la brecha RMSE validación−train, calculada con la misma regla de inferencia. Es una elección prospectiva, no una aprobación retrospectiva de E1. |
| 7 | Catálogo | **20 configuraciones:** Top 10, GRU-E2/LSTM-E2/TCN-E2, tres anclas E1, Ridge, ARIMA, persistencia y referencia ingenua semanal. |
| 8 | Equidad del ajuste | **15 recetas ajustables**, doce candidatos por receta: **180 candidatos**. Asignación común de **172.800 CPU-s por receta**, con registro del consumo y de los intentos incompletos. No es HPO ya realizado. |
| 9 | Arquitectura final | Conservar estructuras respaldadas por originales o presets identificados y revisar explícitamente su adaptación D7. Las recetas pequeñas P0 no sustituyen las configuraciones finales publicadas. |
| 10 | NAS condicional | Aplicar §14 solamente cuando corresponda: espacio completo de 1.000 arquitecturas, control, piloto de ruido, fases y selección previamente especificados. **No se ha ejecutado NAS.** |
| 11 | Partición temporal | Desarrollo forward enero–octubre de 2024, piloto noviembre–diciembre y reserva histórica heredada. Cortes de etiquetas y embargo de siete días se comprueban en cada índice; no se barajan ni comprimen fechas. |
| 12 | Aleatoriedad | Semillas P0 nuevas **660101–660105**, emparejadas entre modelos. Los conjuntos de semillas P0, HPO, P1 y finales se mantienen separados. |
| 13 | Familias | Treinta hipótesis frente a GRU/D7-c03 con α=0,02; treinta frente a GRU-E2 con α=0,02; 199 diagnósticos con α=0,01. Holm dentro de cada familia y asignación global total de 0,05. |
| 14 | Contrastes e intervalos | Comparaciones temporales pareadas, nula centrada en δ, bloques de siete días y sensibilidades de catorce/veintiocho; justificación de la longitud exigida en desarrollo. RQ3 usa la intersección-unión de mejora y menor latencia. |
| 15 | Función de P0 | Factibilidad, gap entre regímenes registrado y variabilidad de recetas fijas. La familia de tres contrastes del deterioro por régimen usa exclusivamente OOF; noviembre–diciembre aporta descripción y variabilidad. |
| 16 | Dimensionamiento | P1 posterior al HPO y al congelamiento de recetas; potencia objetivo 0,80, alternativas de ganancia/erosión 2δ y límite **R=200**. R se elegirá con simulación registrada y su incertidumbre Monte Carlo. |
| 17 | Coste | CPU común, un thread y latencia batch 1; calentamiento, bloques de temporización e intercalado de modelos fijados. El cronómetro tiene una unidad distinta de la evaluación predictiva. |
| 18 | Exclusiones y parada | Conservar fallos, NaN, divergencias, límites de recursos e interrupciones documentadas. Máximo de cien épocas y paciencia doce; sin reponer selectivamente candidatos o semillas por su resultado. |
| 19 | Orden y cambios | Registro → verificaciones → P0 → HPO/NAS aplicable → recetas congeladas → P1 → dimensionamiento → pesos cerrados → fase única de scoring E2 → análisis registrado. Los cambios requieren enmienda fechada. |

Persistencia, referencia semanal y ARIMA usan información endógena causal con un soporte diferente del D7×5 de las redes. Esa excepción se identifica al interpretar la comparación. Las tres anclas E1 conservan su receta; las variantes E2 permiten estudiar el ajuste comparable sin reescribirlas.

## 9.2 Ejecución completada: P0 y evaluación OOF

P0 comprende **TimeMixer, iTransformer, PatchTST y GRU/D7-c03**, con las cinco semillas nuevas indicadas. Las recetas se fijaron antes de sus ajustes y no se seleccionarán arquitecturas a partir de este piloto.

| Componente registrado y ejecutado | Calendario | Trayectorias | Ajustes de optimizador |
|---|---|---:|---:|
| Piloto con pesos congelados | 01/11–30/12/2024; sesenta fechas potenciales | 4×5=20 | 40 |
| OOF con las mismas recetas | Diez orígenes mensuales enero–octubre de 2024; **305 fechas potenciales** | 4×5×10=200 | 400 |
| Total de esta etapa | Desarrollo histórico identificado | **220** | **440** |

Cada trayectoria requiere selección de épocas y un reajuste desde inicialización nueva, de ahí los dos ajustes de optimizador. El [recibo de ejecución](../_static/remediation_preflight/execution_summary.json) verifica que se completaron las cifras registradas y que fueron elegibles las 305 fechas OOF y sesenta del piloto. Los meses comparten historia de entrenamiento y no son diez datasets independientes.

La inferencia principal de limitación en P0 usa la cohorte **OOF de enero–octubre**, con los tres modelos del Top 10 y Holm a **α=0,01**. Contrasta si RMSE en alta volatilidad menos RMSE en régimen normal supera δ. El régimen se define con σ₇(t−1) y el cuantil 0,75 de entrenamiento 2023, sin añadirlo como entrada de la red. Esta pregunta es distinta de la brecha train–validación respecto a δg.

Los sesenta días de noviembre–diciembre quedan como análisis **descriptivo** de factibilidad y variabilidad de esas recetas. No generan otra familia de rechazo por régimen ni se elegirá después el calendario con el resultado más favorable. La familia P0 es un diagnóstico de desarrollo separado del presupuesto de error del benchmark final; no se afirma un FWER conjunto entre ambas etapas.

Los resultados siguientes proceden de los ajustes cerrados y del análisis registrado; se mantienen separados de los del benchmark final.

### 9.2.1 Resultados observados de Bitcoin

Se cerraron **las 220 trayectorias y 440 ajustes previstos**, sin fallos registrados. El registro de las 14:52:32 UTC precedió al primer ajuste; todos los pesos cerraron antes del primer scoring. El [recibo de ejecución](../_static/remediation_preflight/execution_summary.json) conserva los tiempos reales, recuentos y hashes. Se puntuaron **1.200 filas semilla×fecha de piloto** y **6.100 de OOF**: son sesenta y 305 fechas únicas, respectivamente, para cada receta; no 7.300 observaciones de mercado independientes. No se accedió a datos de prueba de 2025–2026.

| Receta fija P0 | RMSE OOF, media ± DE entre semillas | IC temporal nominal 95 % | RMSE noviembre–diciembre, media ± DE |
|---|---:|---:|---:|
| TimeMixer | 0,00760339 ± 0,00032045 | [0,00579635; 0,00949957] | 0,00572273 ± 0,00054997 |
| iTransformer | 0,00751167 ± 0,00047332 | [0,00586802; 0,00923716] | 0,00564681 ± 0,00033395 |
| PatchTST | 0,00625683 ± 0,00027579 | [0,00479305; 0,00789627] | 0,00532612 ± 0,00029717 |
| GRU | 0,00501423 ± 0,00015372 | [0,00419800; 0,00582205] | 0,00464975 ± 0,00023966 |

Estas RMSE son medias de cinco RMSE por trayectoria, no la RMSE de un ensamble. Los IC se obtuvieron sobre el calendario con bloques de siete días y 4.999 réplicas; son nominales y condicionados a las redes ajustadas. La DE entre semillas no sustituye la incertidumbre temporal. GRU tiene menor RMSE descriptiva en este piloto; **no hay evidencia aquí de superioridad global de los tres prototipos sobre esa referencia**. Tampoco constituye el ranking de veinte configuraciones con HPO.

### 9.2.2 Limitación por régimen: resultado registrado

Los **115 orígenes de alta volatilidad conocida** y 190 restantes se definieron con el umbral de entrenamiento fijado antes de puntuar. Los tres contrastes principales OOF/bloque7 rechazan H₀: gap≤δ con Holm a α=0,01:

| Modelo | RMSE alta−resto | IC95 nominal | p unilateral | p Holm, tres hipótesis | Decisión registrada |
|---|---:|---:|---:|---:|---|
| TimeMixer | 0,00584116 | [0,00264550; 0,00898261] | 0,0004 | 0,0008 | Gap > δ respaldado en P0/OOF |
| iTransformer | 0,00591431 | [0,00310376; 0,00863974] | 0,0002 | 0,0006 | Gap > δ respaldado en P0/OOF |
| PatchTST | 0,00432492 | [0,00158748; 0,00713583] | 0,0020 | 0,0020 | Gap > δ respaldado en P0/OOF |

Las sensibilidades registradas de catorce y veintiocho días conservan la dirección positiva y p Holm inferiores a 0,01; se publican como sensibilidades, no como nuevas oportunidades de decisión. La brecha descriptiva de GRU en OOF es **0,00318119**, sin añadirlo posteriormente a la familia de tres contrastes.

**Brecha acotada por la evidencia:** bajo el contrato D7×5, estos tres prototipos fijos presentan deterioro de RMSE en episodios de alta volatilidad de desarrollo de Bitcoin superior a δ. La limitación observada ya tiene evidencia en Bitcoin además de ETTh1. Su presencia también en GRU apunta a dificultad compartida del régimen; la diferencia absoluta puede reflejar escala/heterocedasticidad y no identifica un defecto causal exclusivo del mecanismo. No demuestra que sobreviva al HPO, que sea mayor que la de todas las referencias, ni que una modificación transferida la corrija. El **gap corregible del modelo original E3 sigue por demostrar** mediante adaptación, ablaciones y comparación justa.

La cohorte de noviembre–diciembre permanece descriptiva y **la dirección del gap se invierte en las cuatro recetas**: TimeMixer −0,00199168, iTransformer −0,00075023, PatchTST −0,00100891 y GRU −0,00160510. La fragilidad no es universal en todos los calendarios. No se eligió después por producir un resultado más favorable. El desarrollo de 2024 y la selección histórica de GRU ya eran conocidos en E1: los cortes OOF verifican los ajustes de pesos de esta etapa, sin recuperar cegamiento ni eliminar toda selección histórica.

### 9.2.3 Qué dicen estos datos sobre las semillas

La siguiente proyección usa únicamente noviembre–diciembre y la alternativa previamente fijada 2δ, α=0,02/30 y cuatro grados de libertad para la DE. **No utiliza el efecto observado como alternativa de planificación.**

| Modelo frente a GRU | DE de diferencias entre cinco semillas | Potencia condicional proyectada con R=5 | R con DE puntual | R usando cota superior de DE | Potencia condicional con R=200 y esa cota |
|---|---:|---:|---:|---:|---:|
| TimeMixer | 0,00041370 | 0,0075 | 48 | >200 | 0,4277 |
| iTransformer | 0,00025710 | 0,0223 | 22 | 140 | 0,9516 |
| PatchTST | 0,00050627 | 0,0052 | 69 | >200 | 0,2310 |

La incertidumbre de la DE cambia mucho el R proyectado. Con su cota superior, TimeMixer y PatchTST no alcanzan 0,80 ni con R=200 bajo esta aproximación condicional. Por tanto, no se acredita suficiencia por usar cinco o diez semillas. Estos números **no fijan R final**: falta HPO, congelamiento de recetas, P1 y simulación que incluya la dependencia temporal y las familias previstas.

### 9.2.4 Evidencia publicada y análisis adicional

- [Métricas agregadas](../_static/remediation_pilot_results/aggregate_metrics.csv) y [métricas por trayectoria/semilla](../_static/remediation_pilot_results/seed_metrics.csv).
- [Gap, contrastes y sensibilidades](../_static/remediation_pilot_results/gap.csv), [ganancias descriptivas frente a GRU](../_static/remediation_pilot_results/comparative_gains.csv) y [proyección de potencia](../_static/remediation_pilot_results/power.json).
- [Resumen del experimento](../_static/remediation_pilot_results/summary.json) y [procedencia por hashes](../_static/remediation_pilot_results/provenance.json).
- [Análisis de errores](12_incertidumbre_errores.md): 305 fechas clasificadas por receta y cien casos únicos por modelo en una muestra privada estratificada. Las predicciones individuales y los checkpoints no se publican.

![Resultados del diagnóstico P0: error global y brecha de régimen](../_static/remediation_pilot_results/pilot_summary.png)


## 9.3 Lo que P0 no decide y lo que sigue bloqueado

P0 no estima la potencia de recetas todavía no seleccionadas por HPO, no determina qué arquitectura es superior en el benchmark final y no concede una exención general de NAS. Las diez adaptaciones finales del Top 10 conservan una revisión pendiente: la identificación del mecanismo publicado se distingue de la verificación de su configuración y de la cabeza de σ₇.

Por ello, el control de fases **bloquea HPO, P1, ajustes definitivos y scoring** mientras esas revisiones finales no estén verificadas. Los valores no reportados por un artículo deben resolverse con evidencia de un preset identificable o mantenerse como pendientes; no se reemplazan por los valores pequeños de P0. Si una arquitectura queda realmente sin definir, se activa el NAS condicional y se exigen sus recibos de fase 2 y ganador antes de continuar.

El [protocolo de ajuste equitativo](11_hpo_equidad.md) y la [revisión de arquitecturas](14_arquitecturas.md) desarrollan esos controles para las fases siguientes.

Cuando corresponda NAS, el registro conserva tres bloques, anchos {16,32,64,128,256}, los ocho patrones de dropout 0,2 y las 1.000 combinaciones. Un piloto de ruido fija k=20 o 40 con una regla previa, se añade el control 64–64–64/dropout010 y se confirman los finalistas con al menos cinco semillas distintas de la fase 1. Esta especificación no acredita que la búsqueda se haya ejecutado.

P1 se realizará **después del HPO y del congelamiento de sus recetas**. El dimensionamiento distinguirá variabilidad entre semillas y dependencia del calendario, conservará las familias completas y reproducirá la decisión conjunta de RQ3. La alternativa de planificación es 2δ: detectar exactamente el valor de la frontera nula δ no permite prometer potencia del 80 %. La alternativa de coste es una reducción relativa del 20 % de latencia, medida por separado.

Se elegirá el menor R de la cuadrícula registrada cuyo límite inferior unilateral de Wilson al 95 % para la potencia simulada alcance 0,80 en las celdas previstas. Si ninguna opción lo consigue, se registrará R=200 con **objetivo de potencia no alcanzado**, efecto mínimo detectable y limitaciones. Los modelos sin piloto propio usarán supuestos proxy identificados para planificar; no recibirán una certificación empírica de potencia. Aumentar semillas no crea nuevas fechas independientes.

Los contrastes finales usarán el α de su familia, incluidos los límites simultáneos, en vez de conservar automáticamente 0,05/30. Se mantienen los límites del test histórico conocido y la necesidad de verificar supuestos, auditoría de fuga, calibración y pipeline final.

## 9.4 Documentos y descarga

- [Espacios HPO, asignación y catálogo](../_static/remediation_protocol/hpo.json).
- [Preregistro: hipótesis, familias, pilotos, potencia y reglas](../_static/remediation_protocol/preregistration.json).
- [Decisiones arquitectónicas y NAS condicional](../_static/remediation_protocol/architecture_decisions.json).
- [Validación documental del protocolo](../_static/remediation_protocol/validation.json) y [registro de 180 candidatos sin evaluar](../_static/remediation_protocol/candidate_registry.json).
- [Descargar protocolo, código y evidencias de esta etapa](../_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip).

Los JSON conservan el estado que tenían al registrarse, anterior a los ajustes de esta etapa. Sus contadores iniciales de ejecución no se modifican para aparentar un registro posterior. La validación comprueba contratos, identidades y reglas; los resultados y recibos de ejecución demuestran por separado qué se completó y con qué alcance.
