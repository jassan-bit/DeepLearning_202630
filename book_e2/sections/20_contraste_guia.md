# 20. Cotejo del plan con la guía original

**Vigencia al 11/11/2026:** El PDF aportado nuevamente el 11/11/2026 tiene SHA256 93499dcfda6a36d9621524a81821470017ba114ca6b08241a35de5a54f74a654. La revisión vigente de exigencias/evidencias está en §38; este capítulo conserva el cotejo y estado histórico del 08/11/2026. [Matriz de exigencias y evidencia](38_cumplimiento_guia.md) · [Enmienda de calendario](37_ajuste_guia.md).


**Siguiente especificación:** [comparadores temporales, matriz NAS y costes por escenarios](21_comparadores_nas.md). Propuesta sin entrenamiento, búsqueda ni cierre automático del protocolo.


**Fuente:** Proyecto Final de Deep Learning: Entregable 2, PDF original de 65 páginas aportado por el usuario el 08/11/2026, nombre `Entregable 2DeepLearningProject(3).pdf`. Las referencias siguientes son páginas impresas, coincidentes con el número de página del archivo en los pasajes cotejados. Se leyó su texto íntegro; no se distribuye el PDF original ni se modifica su contenido.

[Registro estructurado del cotejo](../_static/e2_guide_alignment_20261108.json).

Este cotejo corrige el alcance del borrador de §19. No ejecuta el plan, no convierte retrospectivamente decisiones en preregistro y no certifica cumplimiento integral. La auditoría acotada de desarrollo y sus resultados siguen vigentes; los recibos previos permanecen intactos.

## 20.1 Correcciones que afectan el próximo paso

| Disposición de la guía | Estado comprobado y consecuencia para E2 |
|---|---|
| Heredar métrica primaria, δ, unidad y partición de E1; prevalece disposición más restrictiva (p2) | Mantener RMSE, δ y contrato de §17. MASE recomendada en ruta temporal (p31) no se usa para cambiar por silencio la primaria heredada. Test de 608 fechas ya conocido sigue exploratorio. |
| Diez modelos, al menos cuatro familias; tres bases E1 y referencias adicionales (p11) | No cambiar los diez elegidos ni el híbrido candidato. Separar MLP16/LSTM32/GRU32 como bases. Falta cerrar puntajes reproducibles, calidad y justificación de selección con la revisión bibliográfica. |
| Ruta F2a: naive, naive estacional y ARIMA o ETS (p30) | §7 ya fijó F2a por diagnóstico residual. Persistencia y Ridge no bastan. Añadir controles estacional y ARIMA o ETS a la planificación, sin entrenarlos. Manteniendo Ridge: al menos 18 sistemas frente a los 16 del borrador, sin ampliar el Top 10. |
| Espacio mínimo HPO: LR logarítmica, WD, dropout, batch, profundidad y ancho; ejes específicos por familia (p47) | LR×WD con cuatro puntos es sólo un escenario preliminar. Batch/anchos/dropout fijos no satisfacen por sí solos el mínimo. Hay que implementar/justificar espacios y restricciones por modelo; no repetir intentos idénticos para aparentar cumplimiento. |
| Arquitectura indefinida: búsqueda condicional de 1.000 diseños, misma oportunidad (pp13,46,52–55) | Fijar `provisional_default` no exime. Resolver por modelo si la fuente define exactamente el diseño empleado; nuestras adaptaciones compactas necesitan justificación específica. No afirmar que todos requieren NAS ni que todos están exentos. |
| Tres pruebas admisibles por contraste confirmatorio (p23) | Holm sobre una sola prueba no basta. Completar batería temporal y supuestos antes de registro, sin usar semillas/meses como datasets independientes ficticios. |
| IC con ≥2.000 remuestreos (p22); contraste pareado con ≥ 10.000 (p27) | Propuesta de 10.000 por bloques temporales comunes, longitud aún por justificar con desarrollo. No hay IC ni p-valores nuevos calculados. |
| Potencia objetivo ≥ 0,80; piloto≥ 5 semillas en 2–3 modelos; declarar MDE si recursos insuficientes (p43) | Diez semillas del piloto propuesto no demuestran potencia. Catorce redes propuestas amplían ese mínimo, pero su presupuesto sigue sin autorizarse. R=309 de E1 no se hereda. |
| Auditoría formal completa (pp44–46) | Nuestro auditoría de 731 días y 716 ventanas verifica snapshot, calendario, etiquetas, soporte y escalado. No certifica casi duplicados, atributos sospechosos, validación adversarial o sanidad de etiquetas aleatorias del nuevo runner; tampoco reabre test. |
| Limitación del paper guía con métrica, magnitud e IC; piloto en≥ 3 del Top 10 (pp16–17) | Se conserva la réplica y el piloto externo de tres modelos de §5: TimeMixer, iTransformer y PatchTST. Acreditan L1 operativa bajo ETTh1, no una limitación predictiva Bitcoin ni motivación validada del híbrido. La transferencia al contrato D7 sigue pendiente. |
| Robustez: 3 severidades; explicar Top 3; aislar 3 componentes (p16) | La ablación recae sobre el mejor modelo, que aún no se ha identificado, no necesariamente sobre el candidato 11. La interpretabilidad corresponde a los tres mejores y requiere fidelidad y estabilidad entre semillas. Los tres controles compactos existentes no aíslan tres componentes. Intervenciones de un bloque y controles de momentos siguen pendientes; no contar su propuesta como ejecución. |

## 20.2 Búsqueda arquitectónica: puerta previa, no ejecución implícita

La guía distingue diseño definido por el artículo de una familia sin arquitectura especificada (p52). La configuración publicada de PatchTST aparece como ejemplo de exención (p53), pero no permite declarar exenta cualquier variante propia. Por cada receta se debe registrar fuente, capas/dimensiones originales, diferencias reales y decisión fundamentada. Las seis implementaciones reutilizadas y cuatro prototipos compactos no tienen el mismo grado de fidelidad.

Cuando aplique: tres bloques, anchos{16,32,64,128,256} y ocho patrones binarios de dropout 0,2 producen 125×8=1.000 diseños. Si la familia no admite naturalmente tres bloques, declarar correspondencia aproximada conservando cardinalidad (p53). Primero ajustar otros hiperparámetros sobre referencia(64,64,64) y patrón(0,1,0), usando validación (p54). Fase 1: todos 1.000 con una semilla y presupuesto reducido;20–30% de épocas es ejemplo, no regla universal. Fase 2: k≥ 10 finalistas más referencia, con≥ 5 semillas distintas de fase1 (p54). Elegir k según ruido, estimar coste antes y declarar desviaciones de recursos; no recortar silenciosamente (p55).

La selección arquitectónica usa validación y parsimonia cuando hay empate técnico según δ o IC solapados (p56), adaptando correctamente la dirección a minimizar RMSE. No aplicar el ID lexicográfico del borrador como sustituto de esa regla. El conjunto indistinguible, sesgo del ganador y análisis completo de espacio deben reportarse. La prueba de 608 fechas conocida impide presentarla como una nueva confirmación sellada.

Por tanto,2.400 etapas neuronales+40 ajustes Ridge y 300 etapas piloto de §19 son **subtotales de un escenario reducido**, no una estimación completa de guía. No incluyen NAS, ampliación HPO, controles temporales, robustez ni ablaciones. No se ha solicitado ni iniciado ese cómputo.

## 20.3 Ruta temporal y batería estadística pendiente

La ruta F2a continúa vigente; no se elimina la dependencia temporal cambiando la etiqueta a regresión tabular. La selección del periodo naive estacional y del clásico ARIMA o ETS debe basarse en desarrollo y quedar congelada; no se inventa que periodo 7 sea estacionalidad demostrada. Declarar historia disponible, política de actualización/refit y máscara de evaluación común. No se agregan rezagos del objetivo como canales de las redes sin una enmienda explícita.

La guía incluye DM con corrección HLN, Ljung–Box, ARCH-LM, BDS y calibración Mincer–Zarnowitz (pp19,30–32). Antes del registro, especificar pérdida del diferencial DM, estimación HAC y rezagos, ventanas y familias de corrección. DM sobre pérdida cuadrática contrasta precisión en MSE, no por sí solo ganancia>δ en la primaria RMSE. Complementar con remuestreo pareado de RMSE por bloques y una prueba no paramétrica/bayesiana admisible; establecer además procedimiento global/multiplicidad y la regla conjunta de decisión. La lista de nombres no acredita una batería implementada ni tres pruebas válidas.

Los contrastes de brecha train-validación necesitan margen δg fijado previamente (pp25,42), aún pendiente. No aceptar un modelo sólo por superar un promedio: revisar ajuste, sobreajuste, estabilidad, calibración y referencias (p22). La ausencia de significancia no prueba equivalencia (p19). No se trasladan p-valores o diagnósticos de E1 a estas redes.

## 20.4 Orden de trabajo sin entrenar todavía

1. Completar matriz fuente–diseño–NAS por receta y revisar cobertura de las cuatro familias y puntajes del Top 10.
2. Especificar espacios HPO mínimos y controles F2a; distinguir presupuesto de tiempo, intentos y fases AE/NAS. Estimar recursos antes de autorización de ejecución.
3. Completar ejecutor real separado del smoke, sus pruebas causales y la auditoría restante sin reabrir test para seleccionar.
4. Cerrar batería inferencial,δg, bloques, fallos, calibración, robustez y política de selección; registrar con fecha real y exposición previa declarada.
5. Sólo tras autorización correspondiente, realizar las fases de ajuste y piloto. No hay entrenamiento de mercado nuevo en este cotejo.

También siguen pendientes la incertidumbre predictiva de los tres mejores, calibración disjunta y cobertura condicional;≥ 100 casos estratificados por modelo principal (p49); al menos una cota de referencia de desempeño y su margen γ (pp48–49), sin llamar a persistencia techo de Bayes.

La revisión sistemática (pp9–12), búsquedas transversales mínimas y tabla de transferencia (p10), trazabilidad EDA (p9), reproducción en entorno limpio (p41) y reporte completo siguen siendo entregables, no reemplazables por esta auditoría. El PDF de E2 menciona contenedor obligatorio para E3 (p41), pero no aporta por sí solo toda la guía específica de E3 ni sus resultados.
