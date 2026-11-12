# 46. Preparación técnica opcional: artefactos y pilotos

**Estado: preparado, sin ejecución de pilotos nuevos acreditada.** El cierre del informe con la evidencia disponible se presenta en [§47](47_entrega_consolidada.md); esta página conserva una ruta técnica para trabajo posterior. No se solicita ampliar el presupuesto ni iniciar otra campaña para redactar los resultados y límites actuales.

La conexión de una GPU no demuestra que exista entrenamiento. Los **1.705 resultados del plan anterior** pertenecen a su campaña histórica; no son corridas de los pilotos descritos aquí. **R y k no se han determinado con esta preparación.** Los ejemplos y verificaciones CPU no sustituyen los pilotos de mercado ni justifican copiar sus decisiones.

## Qué queda preparado

El [controlador de pilotos](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/run_e2_amended_pilots.py) separa la comprobación de admisión de la ejecución. Sin `--execute` comprueba protocolo, fuentes, GPU real y saldo conciliado; no abre los CSV de desarrollo ni ajusta modelos. El [lanzador de Colab](https://github.com/jassan-bit/DeepLearning_202630/blob/main/notebooks/run_e2_pilots_colab.py) conserva ese modo predeterminado y guarda sus artefactos en Google Drive.

El [notebook técnico](https://github.com/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable2_pilotos_artefactos.ipynb) tiene una sola celda de lanzamiento, `EXECUTE_PILOTS=False` y fuente/digest de publicación explícitos. Su uso posterior requiere esos pines válidos y el cierre normal de cualquier controlador anterior. No modifica la sesión privada actual, no evita las desconexiones de Colab y no tiene fallback de entrenamiento CPU.

| Etapa preparada | Trabajo previsto | Qué permite decidir al completarse |
|---|---|---|
| Ruido TiDE | 12 arquitecturas × 5 semillas = **60 ajustes**, en desarrollo anterior a 2024 | k20/k40 mediante la regla de estabilidad de rangos registrada |
| HPO previa al piloto de potencia | TimeMixer, PatchTST y GRU32 × 18 configuraciones × 2 semillas = **108 ajustes**, ancla octubre de 2024 | Recetas y épocas de esos tres representantes; no cierra el HPO de las trece redes |
| Potencia | Tres representantes × cinco semillas = **15 ajustes**; bloque 08/11–30/12/2024, 53 días calendario | Dimensionamiento común de R, con incertidumbre y condiciones de potencia declaradas |

La cota es de **183 ajustes optimizadores** antes de reutilización compatible. Las fases separadas de predicción, replay y análisis añaden trabajo y tiempo, aunque no añadan ajustes de pesos. El piloto de ruido no equivale al NAS completo de 1.000 arquitecturas de TiDE y los quince ajustes no son el benchmark final.

Se conservan los límites registrados: hasta 200 épocas en estos ajustes completos, paciencia y horizonte de scheduler declarados. El piloto de potencia utiliza épocas fijadas por la mediana de las dos semillas HPO ganadoras; no usa sus etiquetas para parada temprana. Las predicciones y el cálculo estadístico se realizan después de congelar todos los pesos necesarios del piloto.

## Qué se guarda y dónde

La carpeta científica predeterminada es:

```text
/content/drive/MyDrive/DeepLearning_202630/E2/amended-pilots-v1
```

La preparación reutilizable queda en la carpeta hermana `amended-pilots-assets`: fuentes autorales fijadas, derivados admitidos, información del entorno y registros del lanzador. El código se reconstruye por su commit; la VM de Colab es transitoria y no es el destino de conservación de los resultados.

| Artefacto | Contenido y límite |
|---|---|
| `execution_spec.json` | Configuración, etapas, semillas y límites de esta ruta |
| `preflights/<fecha>.json` y `preflight.json` | Comprobaciones fechadas; el segundo es la vista actual de admisión |
| `budget_evidence/` y `ledger/` | Evidencia conciliada de los diarios, reservas y tiempo cobrado |
| `attempts/<identidad>/` | Especificación del trabajo, log, arquitectura cuando corresponda, predicciones, replay y resultado terminal |
| Artefactos por identidad | Pesos iniciales y checkpoints, estados de optimizador/RNG y curvas/eventos por época con sus hashes |
| `october_winners.json` | Ganadores HPO de los representantes cuando está completa esa etapa |
| `k_decision.json` / `R_decision.json` | Sólo se escriben como decisiones científicas válidas tras completar y verificar el piloto correspondiente |

Las curvas se conservan en eventos JSONL y recibos por etapa; no se atribuye un CSV de curvas que todavía no se haya exportado. Los fallos, pausas e intentos sin desenlace se preservan. Un proceso interrumpido no se repite automáticamente al reconectar ni se convierte en una corrida completada por encontrar un archivo parcial.

## Presupuesto: admisión antes de datos y ajustes

El techo sigue en **64 horas acumuladas**, incluyendo el gasto anterior. El replay de los diarios consultados arroja aproximadamente **7.472 segundos** disponibles, frente a una reserva original de **7.500 segundos** para admitir el siguiente trabajador completo. La ejecución debe volver a comprobar ese saldo y cualquier sesión abierta; esta página no lo convierte en un saldo en vivo.

Si siguen vigentes esos valores, el controlador devuelve **código 3, `paused_budget_insufficient`**, incluso con la GPU correctamente conectada. Conserva `ready=false`, no inicia el nuevo ajuste y no abre los CSV de desarrollo mediante el lanzador. No reduce la reserva para encajar por segundos, no borra el gasto anterior y no concede tiempo adicional.

La activación del ejecutor de pilotos está deshabilitada en esta entrega, incluso con `--execute`: requiere aceptación CUDA real y revisión de la integración de overhead/caché con el ledger. No se recomienda activarlo durante este cierre. El código 3 también puede identificar ese bloqueo.

El código 0 de una preparación sólo significa que la admisión pasó; con ejecución desactivada no hubo entrenamiento. El código 2 requiere revisar el motivo y conservar los recibos. Ningún código de salida, por sí solo, demuestra que E2 esté completo o que R/k estén cerrados.

## Uso en la continuación del trabajo

La secuencia técnica disponible para una futura continuación es **pilotos → selección pendiente → evaluación común → tablas de resultados**. El trabajo inmediato de cierre utiliza los resultados ya existentes y declara lo que éstos permiten concluir. Cuando se retome la ejecución, deberán conservarse la identidad de las recetas, el saldo conciliado y la separación entre datos conocidos y prueba prospectiva de [§45](45_enmienda_protocolo.md).

Esta preparación no acredita corridas nuevas sobre GPU, no fija R o k y no añade resultados de mercado al informe. Las decisiones que falten permanecen identificadas como limitaciones o pendientes, en vez de rellenarse con datos sintéticos o métricas de otra receta.
