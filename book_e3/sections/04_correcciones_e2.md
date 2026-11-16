# 4 Bloque A Correcciones del Entregable 2

No se han recibido observaciones del docente, según lo informado por el autor. Las filas siguientes corresponden a **revisión interna**, auditorías ejecutadas y límites documentados de E2. No se atribuyen al profesor ni se interpreta su ausencia como aprobación.

## 4.1 Trazabilidad de correcciones

| Observación del docente | Hallazgo interno | Acción realizada y estado | Ubicación en el Jupyter Book | Evidencia | Impacto en resultados |
|---|---|---|---|---|---|
| No recibida | A01 Organización y estado del informe. | Actualizadas portada y navegación; las secciones 1 a 7 siguen la guía. | [Sección 1](01_introduccion.md) y [concordancia](concordancia_secciones.md). | Índice y fuentes del libro. | Sin modificación de métricas o ranking. |
| No recibida | A02 Auditoría de fuga parcial. | Inventario E3 y auditoría real train/cal de E2 ejecutados; faltan el contenido de prueba y controles completos. | Sección 4.2. | [Auditoría E2 vigente](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html). | Sin fallo de datos demostrado en los controles ejecutados; no certifica ausencia integral de fuga. |
| No recibida | A03 Aplicación de δg. | E2 añadió un diagnóstico CPU sobre trece redes; dos establecieron tolerancia condicional. La puerta CPU–CUDA estricta no se dio por aprobada. | Sección 4.3. | [Trece contrastes CPU](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/e2_corrections_20261115/cpu_diagnostic/statistics/results/CPU_delta_g_Holm13.csv). | No cambia la conclusión primaria ni acredita adecuación conjunta. |
| No recibida | A04 Seis semillas y baja potencia histórica. | Conservadas la limitación y la MDE; dimensionamiento nuevo de E3 pendiente. | [Sección 2](02_estandar_profundidad.md). | [Piloto E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/e2_benchmark_final_20261113/pilot_power_MDE.json). | No se amplían semillas después de observar la prueba. |
| No recibida | A05 Calendario conocido. | Documentada exposición previa y herencia permitida; confirmación E3 pendiente. | [Sección 7.2](07_diseno_riguroso.md). | [Contrato E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/17_contrato_sincronizado.html). | Conserva el alcance retrospectivo de las 608 fechas. |
| No recibida | A06 Limitación y originalidad del híbrido. | Formulación y antecedentes disponibles; R1 sigue abierto. | [Secciones 6 y 7](06_modelo_original.md). | [Matriz de antecedentes](../_static/e3_bloque_c_20261115/review_matrix.csv). | Sin novedad ni mejora predictiva acreditadas del candidato. |
| No recibida | A07 Equidad de presupuesto. | E2 recuperó oportunidades HPO y costes con límites; presupuesto E3 pendiente. | [Sección 7.2](07_diseno_riguroso.md). | [Ampliación E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html). | Sin nuevo HPO o campaña de mercado en esta reorganización. |
| No recibida | A08 Familia y pruebas de E3. | Estándar conservado; protocolo confirmatorio del candidato pendiente. | [Secciones 2 y 7](02_estandar_profundidad.md). | Registro de diseño y requisitos. | No se cambian las familias primarias ni los p-valores de E2. |
| No recibida | A09 CPU y exportación. | Tres referencias TorchScript verificadas. E2 tiene además un suplemento eager CPU con alcance propio. | [Sección 5](05_configuraciones_exportacion.md). | Reportes de ida y vuelta y diagnóstico E2. | Paridad CPU–TorchScript en calibración; no se equipara a la puerta CUDA ni a prueba608. |
| No recibida | A10 Reproducibilidad completa. | Auditores, código y pruebas disponibles; contenedor y reproducción final E3 pendientes. | [Sección 6.5](06_modelo_original.md). | Workflows e informes guardados. | Pruebas de software no sustituyen reproducción de la campaña definitiva. |
| No recibida | A11 Cobertura bibliográfica. | Corpus recibido de E2 conservado y revisión focal del candidato añadida; búsqueda completa R1 pendiente. | [Sección 6.3](06_modelo_original.md). | Matriz, manifiesto de fuentes y bitácora de consultas. | No se cambia el PRISMA ni se infiere ausencia de antecedentes de una consulta bloqueada. |

## 4.2 Auditoría de artefactos y datos reales

El inventario de E3 concilia **16 sistemas y 81 predictores**, 78 neuronales y tres deterministas. Se verificaron los bytes de **167 metadatos**, incluidos 81 escaladores y 81 calibradores, y sus vínculos con los checkpoints. Los **245 hashes** del freeze histórico concuerdan con el manifiesto de respaldo. Listar 79 assets de pesos en Releases no equivale a haber verificado todos sus contenidos; cada descarga efectiva conserva su propia comprobación.

La ampliación actual de E2 también reconstruyó los **655 ejemplos train y 53 de calibración**, comprobó ventanas, vecinos, soportes y transformaciones comunes, y recuperó escaladores físicos. Informó ocho controles verificados, siete parciales y uno no aplicable, como clasificación de esa auditoría, no como porcentaje de cumplimiento de la guía.

Por tanto, no se mantiene el diagnóstico anterior de que sólo existían guardas sintéticas. **La auditoría integral sigue parcial:** no están disponibles en ese entorno las ventanas OHLCV de las 608 fechas de prueba; faltan controles train/test con ajustes adversariales y etiquetas permutadas y la certificación completa de cada port. La recuperación, las identidades de la reproducción y el freeze original se distinguen en sus manifiestos.

## 4.3 Brecha, determinismo y límites numéricos

El suplemento CPU posterior completó 78 trayectorias y contrastó la brecha calibración–train frente a δg con seis pares y Holm13. **Dos de trece redes establecieron tolerancia condicional; once no la establecieron.** Una brecha negativa puede reflejar una calibración más sencilla. Este resultado no acredita adecuación conjunta ni constituye confirmación prospectiva.

La puerta anterior de coincidencia CPU frente a CUDA, con tolerancia 10⁻⁷, conservó su fallo y sus ejecuciones parciales. El suplemento CPU no la convirtió retrospectivamente en aprobada. Tampoco se mezcla con la paridad de exportación CPU–TorchScript de la sección 5, que compara otras salidas bajo otro alcance.

E2 documenta semillas y política de Python/PyTorch/CUDA, FP32 y flags de determinismo. El núcleo no fija una semilla global NumPy; los análisis usan generadores registrados. No se atribuye una llamada global inexistente ni se promete igualdad entre plataformas.

## 4.4 Resultados actualizados y congelamiento

La A2 original conserva SHA-256 `879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1`. Las comprobaciones posteriores y esta reorganización **no cambian su ranking o su conclusión primaria**. Si una corrección demostrada modifica resultados, se recalcularán todos los productos dependientes y se publicará la comparación antes/después.

El hash histórico es un ancla de lectura, no un nuevo cierre del Bloque A. El congelamiento definitivo de los benchmarks para E3 requiere resolver y documentar los controles aplicables; después no se reajustan rivales para favorecer al modelo original.

### Evidencias de referencia

- {download}`Inventario E3 de 81 predictores <../_static/e3_bloques_ab_20261115/audit_predictors.csv>`.
- {download}`Resumen del corte de auditoría de artefactos <../_static/e3_bloques_ab_20261115/audit_summary.json>`.
- [Auditorías y diagnósticos posteriores de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html).
- [Tabla anterior de trazabilidad](06_trazabilidad_correcciones.md), conservada como corte de trabajo.
