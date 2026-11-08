# 19. Plan experimental propuesto y auditoría sin entrenamiento

**Cotejo posterior con la guía original de 65 páginas recibida el 08/11/2026:** este borrador queda condicionado por las correcciones de [§20](20_contraste_guia.md). La cuadrícula reducida y el total de16 sistemas no cubren todos los requisitos; no son el plan definitivo autorizado.

**Corte del 08/11/2026.** Este documento registra decisiones y comprobaciones posteriores a resultados históricos conocidos. **No es un preregistro prospectivo, no acredita HPO ni piloto del catálogo vigente y no demuestra superioridad.** Los protocolos y resultados de E1 y de versiones anteriores permanecen sin cambios. La ejecución requiere cerrar las condiciones pendientes y registrar una enmienda con su exposición previa antes de nuevos ajustes.

## 19.1 Alcance y condición de entrada

El catálogo comprende los diez modelos de §15 y el candidato adicional CNN-1D + LSTM + atención temporal. Las referencias separadas son MLP16, LSTM32 y GRU32 vigentes de E1, Ridge como control no profundo propuesto y persistencia causal. El borrador inicial contaba quince configuraciones de benchmark más el candidato once: **dieciséis sistemas**, no dieciséis arquitecturas nuevas ni una lista que reintroduzca modelos retirados. La ruta F2a vigente añade naive estacional y ARIMA o ETS (PDF p30): manteniendo Ridge, la propuesta debe contemplar al menos 18 sistemas, con recetas y coste de esos controles pendientes. Las ablaciones son controles adicionales y no cuentan como nuevos modelos del Top 10.

La entrada y el objetivo permanecen D7×5 OHLCV relativo, σ₇ poblacional sin anualización y horizonte uno conforme a §17. La receta de software es una adaptación y no una reproducción literal de los artículos. La guía recibida se conserva como registro histórico: sus selecciones antiguas, TCN, R y estados de ejecución no sustituyen este catálogo ni el E1 vigente. El PDF original ya se cotejó posteriormente; las páginas y correcciones actuales están en §20. El JSON histórico de recepción no se reescribe retroactivamente.

## 19.2 Auditoría del snapshot

Se auditó el snapshot diario de desarrollo2023–2024 mediante `scripts/audit_e2_real_data.py`, contrastando sus SHA256 con el registro histórico y reconstruyendo etiquetas y escaladores. [Recibo agregado](../_static/plan_audit_20261108/audit.json).

| Comprobación ejecutada | Resultado |
|---|---|
| Calendario | 731 filas diarias UTC, sin duplicados ni compresión de fechas |
| Calidad | 24/03/2023 incompleto conservado; no imputación |
| Elegibilidad | 716 ventanas D7;15 no elegibles por soporte/etiquetas |
| Objetivo | σ₇ poblacional recomputada; diferencia máxima9,72×10⁻¹⁷ |
| Entrada | Siete días t−7…t−1, sin barra del objetivo |
| CV histórica | 305 fechas de enero–octubre2024 |
| Piloto histórico | 60 fechas,01/11–30/12/2024; no se reemplaza |
| Soporte de ajuste final | 709 refit,619 entrenamiento interno,83 validación interna |
| Transformaciones | Reconstrucción independiente y perturbación futura sin cambios en train/escaladores |
| Ejecución de modelos | Cero ajustes, cero pasos de optimizador, cero HPO y cero descargas |

La auditoría usa los derivados diarios existentes, **no reagrega los archivos originales de minutos**. Verificar hashes acredita identidad del snapshot registrado, no certificación independiente de Binance. No se leyó ni recuperó la evaluación2025–2026, por lo que no se certifican aquí sus608 filas. Tampoco se prueba fuga a nivel de un runner real inexistente.

El flujo actual rechaza correctamente este snapshot por su guarda `synthetic_only`; no se retiró la guarda ni se cambió su procedencia para pasar pruebas. Sólo la función matemática de transformación es compartida. **Datos de desarrollo auditados no equivale a flujo de entrenamiento real integrado.**

## 19.3 Recetas y presupuesto propuestos

Se propone usar `provisional_default` de §18 como estructura fija, sin ensemble InceptionTime, sin variación de profundidad ni NAS declarada. Los cuatro prototipos usan ancho16 y dropout0; el híbrido CNN32/LSTM32/atención16, kernel3, una capa LSTM y dropout0,1. Las seis adaptaciones mantienen sus mecanismos y heads documentados. Los valores son puntos de partida del diseño, **no óptimos validados**.

La cuadrícula reducida inicialmente propuesta, **insuficiente para el espacio mínimo de §11.2 de la guía (p47)**, es LR∈{0,001; 0,003} × weight decay∈{0; 0,0001}, cuatro configuraciones por red. Es una desviación explícita respecto de la cuadrícula LR/dropout histórica de E1: los cuatro prototipos no implementan dropout y variar ese parámetro crearía intentos inexistentes o duplicados. Se fija el dropout estructural de cada receta. AdamW, MSE en y escalada, batch32 con último batch conservado (sin recomendar batch1; registrar tamaño efectivo), máximo100 épocas supervisadas, paciencia12, mejora mínima10⁻⁵ y clipping1. No se altera una receta después de mirar la evaluación.

Para LSTM-AE: hasta20 épocas de reconstrucción train-only, sin selección por la validación externa, seguidas de fine-tuning supervisado de encoder/predictor; decoder no participa en esa pérdida. Se crea un optimizador nuevo al pasar a supervisión. Cada reajuste reinicia también el preentrenamiento, no hereda pesos posteriores. La política propuesta usa20 épocas fijas de reconstrucción y no una búsqueda oculta; cualquier parada por fallo se registra. El costo incluye ambas fases.

| Partida propuesta | Cálculo | Etapas máximas programadas |
|---|---|---:|
| Once candidatos, selección supervisada | 11×4 configuraciones×2 semillas×10 folds×(ajuste interno+refit) | 1.760 |
| Preentrenamientos AE independientes | 4×2×10×2 | 160 |
| Tres referencias E1 retunadas para E2 | 3×4×2×10×2 | 480 |
| Subtotal del escenario reducido neuronal | Supervisado2.240 + reconstrucción160 | 2.400 |
| Ridge propuesto | α∈{0,01; 0,1; 1; 10}×10 folds, ajuste cerrado una vez | 40 |
| Persistencia | Sin parámetros aprendidos | 0 |

Estos subtotales conservan la contabilidad del borrador inicial, no el presupuesto completo conforme a la guía: faltan expansión de espacios, NAS donde aplique y controles temporales. Estas cantidades **no son horas, FLOPs, equivalencia de cómputo ni ejecuciones realizadas**. Los refits dependen de la época elegida internamente; todo preentrenamiento, fallo, reintento y control entra en el consumo. Hardware homogéneo, memoria y techo de tiempo por receta deben fijarse tras una prueba técnica acotada autorizada; no se deducen tiempos reales del smoke. Un presupuesto que no permita terminar se declara insuficiente, sin favorecer al híbrido ni regalar intentos adicionales.

Las referencias retunadas serán nuevas ejecuciones etiquetadas E2, no modificaciones de pesos/resultados E1. Sus anclas históricas se presentan por separado. Ridge se ajustará sobre D7×5 transformada y escalada con el mismo entrenamiento; enlace no negativo por truncamiento explícito al evaluar σ, sin confundirlo con Softplus. La persistencia usa σ₇ del último día observado y requiere ocho cierres t−8…t−1, frente a las siete barras de la entrada neuronal. En desarrollo,716 ventanas neuronales no equivalen a716 casos de comparación común:714 tienen también persistencia; faltan08/01/2023 y01/04/2023. La revisión independiente verificó que CV 305 y piloto propuesto53 sí conservan soporte de persistencia completo. Exigir una máscara común al evaluar todos los sistemas finalmente incluidos, sin quitar por ello dos filas elegibles del entrenamiento neuronal. Ninguno agrega información futura.

### Calendario, selección y semillas propuestas

El desempate por ID descrito a continuación pertenece al borrador reducido de HPO, no a la selección arquitectónica: para NAS rige la parsimonia en empate técnico de la guía p56.

Diez folds de enero–octubre2024 con los mismos soportes/embargos de E1. La selección concatena305 fechas elegibles por trayectoria, calcula RMSE original y promedia entre semillas; no promedia RMSE mensuales sin ponderar. Dos semillas propuestas de búsqueda:27101 y27102; refit=semilla+10.000.000 y orden de lotes=semilla+17, con generadores separados. Desempate: menor RMSE sin redondeo y luego ID lexicográfico prefijado; no elegir por test. La parada interna de épocas **no convierte la selección global de hiperparámetros en validación anidada**: es desarrollo/selección, no OOF insesgado del procedimiento completo.

Tras congelar receta, se propone un piloto auxiliar 08/11–30/12/2024 (53 fechas), semillas28101–28110. El ajuste y sus escaladores usan sólo etiquetas hasta31/10, origen08/11 y embargo01–07/11. El inicio evita llamar causal a una evaluación desde01/11 cuando la receta utiliza octubre completo y se exige embargo de siete días. No se cambia el piloto histórico E1. Este bloque también es conocido: dimensionamiento exploratorio, nunca reserva nueva. Se propone aplicar el piloto a las14 redes y10 semillas:280 ajustes supervisados más20 reconstrucciones AE,300 etapas adicionales a la tabla, no ejecutadas, más un ajuste Ridge y cero de persistencia. Las fechas exactas y muestras deben validarse antes de ejecutar. R definitivo queda pendiente del dimensionamiento de variación entre inicializaciones; incertidumbre y potencia temporal se evalúan por separado, y transferir estimaciones de53 a608 fechas es un supuesto fuerte; las semillas definitivas no se asignan todavía.

### Controles y análisis por registrar

- Tres ablaciones del híbrido ya existen como código aislado: CNN-only, LSTM-only y CNN-LSTM sin atención. Mantener datos, transformaciones y protocolo; sus diferencias de capacidad/readout impiden interpretar todo cambio como efecto puro de un único bloque. Sin selección por el test. Para aislar tres componentes se proponen además −CNN=LSTM+atención, −LSTM=CNN+atención y −atención=CNN+LSTM+pooling uniforme sobre los mismos siete estados, conservando la cabeza. Estas tres intervenciones **no están implementadas**; no se da por cumplido el requisito de aislar tres componentes con los controles actuales. Tras fijar recetas, sus ajustes internos+refits sumarían6R etapas, fuera de la tabla de selección y sin reutilizar pesos del híbrido.
- Los seis adaptadores incorporan cinco medias y cinco DE de D7 escalada. Proponer controles sin esos momentos y un control sólo-momentos. Las seis variantes sin momentos añadirían12R etapas internas+refit a igual R, y el control sólo-momentos necesita receta/presupuesto propio. No están implementados ni comprometidos: la comparación inicial es del **adaptador completo**, no del backbone puro.
- Métrica primaria: media de RMSE por semilla en unidades originales; MAE, MSE, R², MAPE con cobertura de y>0 y MASE con denominador train-only sobre diferencias consecutivas. No confundir promedio de semillas con ensemble.
- Margen histórico δ=0,00025772842664981106 preservado como referencia práctica explícita, no reestimado a favor del candidato. Contrastes pareados por fecha y dependencia temporal; bootstrap por bloques con sensibilidad de longitud fijada antes de resultados. Longitud de bloque pendiente de justificar con desarrollo, no del test. La guía fija al menos2.000 remuestreos para IC (p22) y al menos10.000 para la prueba de remuestreo/permutación pareada (p27); se propone10.000 para ambos, todavía sin ejecutar. Se remuestrean bloques comunes de fechas, recalculando RMSE por trayectoria antes de promediar semillas; dispersión entre semillas y variación temporal se informan separadas. No se usan semillas o meses como datasets ficticios de Friedman.
- Familia primaria provisional, aún por conciliar con la batería de tres pruebas admisibles por contraste (p23) y los controles F2a añadidos: doce comparaciones del candidato frente a cada uno de los diez modelos, GRU32-E2 y persistencia. H0: ganancia de RMSE del candidato≤δ, H1: ganancia>δ; alfa familiar0,05 con corrección Holm. Otras parejas se etiquetan secundarias. El cálculo de potencia debe usar una alternativa por encima deδ, no la frontera nulaδ; su magnitud requiere justificación previa. El mejor competidor no se elige a posteriori en test. No se atribuyen intervalos válidos a comparaciones todavía no ejecutadas.
- Robustez con tres severidades, calibración predictiva y explicación de tres mejores requieren especificar perturbaciones físicamente válidas, un bloque de calibración separado y criterio de selección basado en desarrollo. No se calibran intervalos con residuos de etiquetas incluidas en el ajuste final hasta24/12. Una eventual rama secundaria ajustada sólo hasta31/10 y calibrada08/11–30/12 debe mantener ese mismo predictor al evaluar y quedar separada del primario; su cobertura bajo dependencia/deriva no está garantizada. No se simulan resultados ni se considera cerrado este requisito. La condición NAS por modelo y el piloto de limitación del paper guía siguen pendientes, no satisfechos por fijar defaults.

**Decisiones descritas, no preregistradas ni cerradas como cumplimiento de la guía:** objetivo/soportes heredados, catálogo, estructura provisional, cuadrícula propuesta y contabilidad, separación de fases AE, no test para selección. **Puertas aún abiertas:** ejecutor real integrado y auditado, controles de momentos, NAS/guía, hardware/tiempos, política de calibración/robustez, dimensionamiento R e inferencia. Cerrar un borrador no equivale a autorizar las2.400 etapas ni a cumplir la guía completa.

## 19.4 Límites de inferencia y próximos pasos

La reserva de 608 fechas (01/01/2025–31/08/2026) ya es conocida y sólo puede informar comparación exploratoria. La reserva anterior de 60 fechas (01/09–30/10/2026) también fue observada. Ni volver a sellar archivos ni cambiar semillas recupera independencia respecto de decisiones previas. No se utiliza ninguna de ellas para elegir recetas, épocas, presupuesto o márgenes. Una afirmación confirmatoria futura exige un soporte temporal realmente no observado, congelado antes de examinar sus resultados; aquí no se inventa ese soporte.

El R=309 de E1 y sus semillas pertenecen a su propio dimensionamiento. No se transfieren potencia ni suficiencia a once recetas diferentes o a series con dependencia temporal. Semillas de optimización y fechas son fuentes distintas de variación: repetir semillas no multiplica el tamaño de la muestra de mercado.

**Siguiente puerta de ejecución:** completar el ejecutor real separado del smoke sintético, revisar el plan y sus recursos, resolver controles y registrar las decisiones con fecha real. Sólo después correspondería solicitar/ejecutar selección y piloto autorizados. Este corte no inicia entrenamiento de mercado, búsqueda ni dimensionamiento.
