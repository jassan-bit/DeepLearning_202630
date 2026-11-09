# 25. Matriz de cierre antes del benchmark

**Corte documental: 09/11/2026.** Fuente de requisitos: *Entregable2DeepLearningProject.pdf*, aportado por el usuario, especialmente §§3–4 y §§9–14 y Apéndices A–B. Esta matriz organiza el trabajo autorizado de preparación; no constituye un preregistro experimental ni acredita ejecuciones nuevas.

## Cómo interpretar el estado

**Parcial** significa que hay evidencia reutilizable, pero no satisface todo el requisito del catálogo vigente. **Pendiente** exige decisiones o mediciones nuevas. Un requisito se cierra con un artefacto verificable y revisión de su correspondencia con el experimento; una descripción del método o una prueba sintética no bastan.

El catálogo mantiene diez modelos y el candidato CNN-1D + LSTM + atención temporal. Los tres modelos de E1 y los comparadores clásicos permanecen identificados como referencias separadas. Los recibos históricos conservan fecha, receta y alcance.

## Requisitos, decisiones y evidencia de cierre

| ID | Guía y ubicación | Estado observado | Decisión o trabajo pendiente | Evidencia para cerrar |
|---|---|---|---|---|
| R01 | §3, [literatura](03_literatura.md) | Parcial: búsquedas, cribados y lecturas documentados; elegibilidad sin cierre global | Conciliar duplicados, resolver textos y justificar selección Top 10, calidad de fuentes y adaptaciones | PRISMA reconciliado, razones de exclusión y matriz final de fuentes por modelo, con pendientes de acceso explícitos |
| R02 | §§1 y 3.4, [problema](01_punto_partida.md) | Hipótesis documentadas; falta conciliación con catálogo y diseño definitivo | Cerrar gap, al menos tres RQ, H0/H1, clasificación confirmatoria/exploratoria, α, δ y criterio de decisión; conservar lo heredado de E1 | Tabla RQ–hipótesis–estimando–prueba–familia–decisión, con trazabilidad a E1 |
| R03 | §§4, 11 y 14, [recetas](22_fichas_recetas.md) y [implementación](23_implementacion_recetas.md) | Cuerpos implementados y probados sintéticamente | Congelar por receta arquitectura, entrada, objetivo, pérdida, inicialización, SAE, ensemble y controles del híbrido | Registro versionado por modelo, hashes de código/fuentes y correspondencia receta–implementación revisada |
| R04 | §14, [arquitecturas](14_arquitecturas.md) y [matriz NAS](21_comparadores_nas.md) | Diseño resuelto en §27: seis variantes NAS y cinco cuerpos fijos; fábricas nuevas pendientes | Implementar y validar el espacio y los mapeos resueltos en [§27](27_nas_hpo_presupuesto.md) | Decisión motivada por receta; para NAS aplicable, espacio, restricciones, fases, selección y coste registrados antes de buscar |
| R05 | §§4.3 y 11, [HPO](11_hpo_equidad.md) | Dominios y T=18 resueltos en §27; sin HPO ejecutado | Dominios enlazados y perfiles cortos publicados en [§29](29_entrenamiento_temporal.md); falta perfilar cohortes y presupuesto reales | Configuraciones ejecutables y presupuesto comparable, incluidos preentrenamiento y selección; revisión frente al mínimo de la guía |
| R06 | §§4, 7 y 10, [contrato](17_contrato_sincronizado.md) y [ejecutor](24_ejecutor_comun.md) | Motor de18 sistemas integrado y probado artificialmente en §29; mercado pendiente | Cerrar alternativa A/B, selección causal, puentes, faltantes, estado ARIMA y máscara común sin comprimir calendario | Plan de particiones y pruebas de integración de todas las fábricas/adaptadores, escalado train-only y persistencia previa a observación |
| R07 | §§9 y 13, [preregistro](09_protocolo_registrado.md) | Nueva enmienda pendiente | Dimensionar repeticiones/potencia, fijar semillas, estimandos, familias, multiplicidad, bloques e inferencia | Protocolo versionado con fecha real de cierre, supuestos de potencia, límites de recursos y registro de exposición previa |
| R08 | §10, [auditoría](10_auditoria_datos.md) | Auditoría histórica parcial; controles finales sin ejecución | Completar controles aplicables, incluidos etiquetas permutadas, desplazamiento y disponibilidad causal | Recibos por pipeline; fallos conservados, causa corregida y repetición de experimentos afectados |
| R09 | §§4, 10 y 17 del libro, [disponibilidad](24_ejecutor_comun.md) | Desarrollo comprobado; 608 fechas finales sin certificar | Verificar derivado final mediante acceso autorizado, calidad, hashes, rezagos y exclusiones prefijadas | Manifiesto de disponibilidad y máscara común congelada; declarar el carácter exploratorio de la reserva conocida |
| R10 | §12.1, [techo y margen](12_incertidumbre_errores.md) | Método documental añadido; estimación pendiente | Elegir y justificar cota de predictibilidad, escala y comparación con baseline | Estimación e incertidumbre, distancia normalizada cuando sea identificable y límites de interpretación |
| R11 | §§4.5 y 12.2–12.3, [incertidumbre y errores](12_incertidumbre_errores.md) | Análisis histórico separado; diseño actual pendiente | Definir calibración disjunta, cobertura, robustez, interpretabilidad y muestreo de errores | Plan previo con soportes y métricas; posteriormente resultados de cobertura, segmentos y casos auditables del catálogo vigente |
| R12 | §8 y Apéndice B, [reproducibilidad](08_reproducibilidad.md) | Versiones directas y pruebas; entorno completo sin reproducción independiente | Fijar dependencias transitivas, fuentes, semillas y comandos; revisar licencias | Entorno fijado y recibo de reproducción independiente con alcance declarado |
| R13 | §§4–6 y 13, [benchmark](06_benchmark_final.md) | Sin benchmark vigente de mercado | Tras cerrar preparación, ejecutar búsqueda aplicable, ajustes, evaluación e inferencia según protocolo | Tabla principal con dispersión e IC, calibración, robustez, coste y ranking; tabla de contrastes con efectos y ajuste familiar |
| R14 | Apéndice A y §8, [trazabilidad EDA](02_eda.md) | Material distribuido; informe final pendiente | Reunir plantillas A.1–A.3 y checklist B; preparar resumen, discusión, respuestas a RQ y conclusiones cuando existan resultados | Enlaces estables a tablas/evidencias y checklist justificado; estructura sugerida A.4 sin resultados inventados |

## Orden de ejecución y dependencias

1. **Justificación científica:** resolver R01–R02. Las incertidumbres bibliográficas deben quedar visibles y no convertirse en inclusiones definitivas por omisión.
2. **Diseño y software:** cerrar R03–R06 y preparar R10–R11. Justificar NAS por receta antes de definir la búsqueda; integrar todos los sistemas antes de presentar el runner como completo.
3. **Registro previo:** cerrar el diseño de R07, con R05 y el plan de R08 ya especificados. Registrar decisiones antes de sus ajustes o contrastes correspondientes; no atribuir retrospectivamente carácter confirmatorio a datos conocidos.
4. **Verificación:** completar R08–R09 y R12 conforme al orden registrado. Las pruebas con ajustes de mercado necesitan su propio protocolo previo; este documento no las ejecuta. Acceso final y evaluación mantienen las guardas existentes.
5. **Experimento y reporte:** ejecutar R13 y las mediciones de R10–R11, después completar R14. Conservar fallos, resultados nulos y cambios como enmiendas fechadas.

## Producto disponible y siguiente cierre

La [tabla por receta de §26](26_registro_recetas.md) reúne fuentes, adaptaciones, dudas NAS, factores HPO, parámetros y estado de integración para los once candidatos y siete referencias separadas. Respalda el inventario R03–R06 y la conciliación arquitectónica local R01; no cierra el PRISMA global ni los dominios HPO o las decisiones NAS.

La [decisión posterior de §27](27_nas_hpo_presupuesto.md) resuelve NAS por receta, dominios HPO y presupuesto contable para Codex. La [integración de §29](29_entrenamiento_temporal.md) enlaza las variantes, etapas, selección y reajuste de18 sistemas y publica perfiles CPU cortos. El siguiente cierre es el preregistro de pilotos de desarrollo y su presupuesto real antes de la campaña; la muestra artificial no fija k/R ni horas totales. No se fija aquí un número de semillas, ganador, exención NAS ni techo numérico sin evidencia.

La [sección 20](20_contraste_guia.md) conserva el cotejo detallado con la guía. La [sección 24](24_ejecutor_comun.md) distingue el smoke sintético del núcleo temporal posterior. Esta matriz complementa esas evidencias sin modificar sus recibos.
