# Evidencia vigente de reproducibilidad y preregistro

Este expediente acompaña los puntos 8 y 9 de la guía. Consolida registros ya ejecutados y mejora su presentación; conserva los datos, modelos, resultados, presupuestos y documentos históricos. Los recuentos de requisitos son subcondiciones auditables, no pesos ni calificaciones.

## Reproducción y entorno

La guía acepta otra persona **o un entorno limpio**. La reproducción registrada utilizó checkout y venv nuevos, con el mismo motor científico, recetas, épocas, semillas y tipo de GPU: 78 ajustes neuronales y tres comparadores, 81 predictores. La discrepancia máxima de pronóstico fue **0** en 53 fechas de calibración y 608 de evaluación. Esa alternativa de entorno limpio está acreditada; no se exige también un segundo equipo para reconocerla.

El auditor CPU reconstruye las 81 RMSE y las dieciséis medias A2 desde predicciones ya guardadas, con discrepancia máxima **2,6020852139652106×10⁻¹⁸**. Esta comprobación aritmética es distinta de repetir entrenamiento o comparar tensores. Desde el checkout:

```bash
python scripts/audit_e2_delivery.py --require-supplements --output work/e2_delivery_audit.json
```

No necesita red, GPU ni paquetes externos. No regenera todas las tablas, figuras y experimentos de todos los capítulos, ni repite los sorteos bootstrap. La automatización integral conserva alcance parcial; el comando de construcción del libro tampoco se presenta como replay científico completo.

`environment_observed.lock.txt` recupera las **44 distribuciones Python instaladas** enumeradas por `importlib.metadata.distributions()`. Sus bytes reproducen exactamente el SHA256 **6d02a90c8551905126417f6c92c5248b66d249c8b5930babb96c7f60e6f93716**, registrado por el productor para el lock instalado en la reproducción limpia. `environment_observed.json` enlaza la fuente y el algoritmo de recuperación. No hubo una nueva instalación.

El entorno observado es Python 3.12.12/Torch 2.8.0+cu126/CUDA12.6/A100, con FP32 y la política determinista documentada. La versión de imagen/SO y el driver no quedan íntegramente fijados por ese recibo: no se sustituyen por los del equipo editorial. CPU de diagnósticos, CPU del paper ETTh1 y construcción del libro tienen dependencias propias. El contenedor es **opcional en E2**; el requisito de E3 conserva su alcance futuro.

## Licencias y uso responsable

`license_register.csv` y `license_sources.json` distinguen datos, código y pesos:

- Binance Vision: texto de términos conservado en E1, CC BY-NC-SA4.0 salvo designación específica y condiciones adicionales; uso académico no comercial y atribución. No es una licencia MIT de los datos.
- ETTh1: copia exacta de `LICENSE` en el mismo commit de datos `zhouhaoyi/ETDataset@1d16c8f4f943005d613b5bc962e9eeb06058cf07`, **CC BY-ND4.0**. `LICENSE.ETDataset.txt` tiene el Git blob **bbf5362553802d33a9ab499ae77754cdc3b86195**. Se conserva atribución y la condición de no compartir material adaptado; el CSV original se adquiere por separado y no se redistribuye en el libro.
- TiDE: licencia Apache2.0 conservada junto a su fuente. No se extiende automáticamente al resto de puertos y bibliotecas.
- Otros componentes: la adjudicación exhaustiva de permisos y avisos sigue parcial. Una URL pública o una cita no acreditan por sí mismas derechos uniformes de redistribución.
- El benchmark no importa pesos preentrenados ajenos. El preentrenamiento SAE usa su propio train y no elimina las obligaciones de datos/código.

El reporte ético conserva los límites de una sola fuente/activo y de los regímenes temporales. No hay identificadores personales en las velas utilizadas; los grupos de volatilidad no demuestran equidad demográfica. El capítulo distingue consecuencias potenciales del error de beneficios o daños financieros realmente medidos, y cobertura global de cobertura por subgrupo.

## Preregistro y potencia

El punto 9 muestra los doce campos de la plantilla, su valor actual, evidencia y cronología. Las decisiones heredadas de E1, la continuación reducida y los suplementos posteriores conservan ámbitos diferentes. La métrica, δ, unidad y partición se heredan conforme a la guía; no se afirma recuperar un test ciego. El δg previo está documentado y su prueba comparable para los predictores actuales permanece como requisito separado.

Los tres procedimientos originales de los veinte pares se registraron concretamente antes del scoring actual: bootstrap pareado, t y cambios de signo. Holm controla cada familia original de veinte; no se cuenta como otro método ni se exige un global extra por ese solo motivo. Las correcciones no verifican los supuestos de cada prueba. El refinamiento de 4.999 a 20.000 conserva la misma familia y método; los veinte p ajustados siguen en 1.

`power_design_readout.json` recoge el piloto de cinco pares iTransformer/GRU32 y 53 fechas. La planificación real fue **t no central unilateral**, H0 en gananciaδ y alternativa **2δ**, con α=0,0025 y R=6. La potencia publicada es **0,1229533091842126**, frente al objetivo 0,80; la ganancia total mínima al 80 % es **0,0008786736025621022**, con exceso sobreδ de **0,0006209451759122912**.

El cociente documental d=δ/DE≈**1,020680829** se obtiene de dos cantidades ya publicadas. No es un nuevo cálculo de potencia, un R mínimo ni el tamaño Cohen/Cliff con IC de los contrastes finales. La fórmula bilateral ilustrativa de la guía no se atribuye a la planificación unilateral ejecutada; esta última tampoco es potencia exacta de Holm o del bootstrap temporal. Se conserva la limitación por recursos sin ampliar R tras ver la prueba.

Los inventarios exploratorios se informan por programa. No se suman los 1.484 registros de un corte histórico a las 231 oportunidades válidas de la selección reducida como ensayos nuevos del mismo diseño. El universo histórico completo de configuraciones y métricas examinadas mantiene cobertura parcial. Los suplementos no seleccionan otro modelo final con test y no se transforman en preregistros anteriores a sus resultados.

## Comunicación y figura de calibración

`top3_coverage_width.svg` y `.png` separan dos tipos de evidencia: nueve coberturas con sus **IC95% temporales marginales publicados**, B=10.000 y bloque14, y nueve anchuras medias como **puntos descriptivos sin IC publicado**. `calibration_plot_readout.json` fija las dos fuentes, los valores copiados, el script y los hashes. No se promedian IC individuales ni se calculan intervalos de anchura inexistentes. La figura anterior permanece accesible sin alterar sus bytes.

El script reproduce únicamente esta presentación:

```bash
python scripts/plot_e2_calibration_communication.py --output /tmp/e2-calibration-communication
```

La bibliografía BibTeX conserva 29 entradas: 26 con DOI, ocho de ellos arXiv, y 29 con URL. El formato se analizó sin claves duplicadas ni entradas sin DOI/URL; eso no es una nueva resolución en vivo de todos los identificadores. Los tres registros sin DOI tienen fuentes verificables OpenReview, JSTOR y JMLR. No se inventan DOI editoriales para preprints o publicaciones que usan otro identificador.

## Contenido y verificación

- `guide_checklist.csv` y `.json`: 48 subcondiciones de los puntos 8/9, con evidencia, límites y tareas concretas.
- `decision_traceability.csv`: doce vínculos entre hallazgos, decisiones permitidas y fuentes.
- `current_evidence_index.json`: fuentes con SHA256 y ámbitos explícitos.
- `environment_observed.*`, `power_design_readout.json`, `license_register.csv`, `license_sources.json` y `LICENSE.ETDataset.txt`: las recuperaciones documentales descritas arriba.
- `top3_coverage_width.*` y `calibration_plot_readout.json`: nueva presentación de resúmenes e intervalos existentes.
- `08_reproducibilidad_before_update.md` y `09_protocolo_registrado_before_update.md`: copias exactas de los capítulos anteriores.
- `manifest.json`: tamaños y SHA256 de este expediente.

Esta actualización no entrenó modelos, ejecutó pronósticos, calculó nuevos scores o pruebas, repitió remuestreos o cambió el presupuesto. El balance final de 84 h autorizadas, 80,2561 h contabilizadas y ninguna reserva abierta conserva su fecha y alcance. Los resultados y enlaces históricos se mantienen.
