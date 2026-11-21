# Avance ejecutado de desarrollo, estadística y reproducción

Corte realizado el **20 de noviembre de 2026 en Bogotá**; identificadores técnicos UTC del 21 de noviembre. Fuente inicial: `f5c960eefedb5fc49b313dd795b1ec84e784bd78`. Este suplemento declara asistencia de IA y conserva resultados negativos. No acredita R3 ni reproducción humana.

## Resultados obtenidos

| Trabajo | Ejecución y decisión | Evidencia |
|---|---|---|
| Nueva hipótesis de desarrollo V4 | Cambió el score auxiliar del espacio U al objetivo sigma, con su normalización. Diagnóstico de 144 casos y **20 ajustes CPU**, cinco semillas pareadas, cuatro recetas neuronales y dos controles analíticos; **22 trayectorias y 1.210 predicciones** en 55 fechas previamente expuestas. | [Desarrollo](development/README.md), [decisión](development/CURRENT_DEVELOPMENT_DECISION.json), [resultados](development/cpu_pilot_results/recipe_summary.csv). |
| Regla de avance previa | Exigía mejorar al menos 2 % frente al mejor control neuronal y al mejor de todos los controles. V4 obtuvo **1,443261 % más RMSE** que MSE-only: **ABANDON_CURRENT_V4_RECIPE**. No se promueve MSE, no se inicia V5 y no se confirma V4. | [Plan y fuentes anteriores a los ajustes](CPU_PILOT_PREFIT_PUBLICATION.json), [QA posterior](qa/QA_CPU_PILOT_POSTRUN.json). |
| Batería temporal propuesta | Dos extensiones pareadas y MCS global implementados; 19 guardas de software y **1.024 réplicas controladas**, 11 casos, 2.096.128 remuestreos bootstrap. MCS rechazó igualdad en 27/128 casos nulos (**21,09 %**, nominal 10 %; IC95 % 14,38–29,19 %). Configuración **no aprobada para confirmar**. | [Métodos y límites](statistics/METHODS_IMPLEMENTED.md), [decisión](statistics/VALIDATION_DECISION.json), [plan anterior al Monte Carlo](CONTROLLED_PLAN_PUBLICATION.json). |
| Admisión futura | Cuatro propuestas concretas: BTCUSDT, SPY, USO y TLT. **0/4 admitidas**, con fuentes, condiciones y bloqueos explícitos. No se observaron resultados futuros para seleccionarlas. | [Admisión](data_admission/README.md), [planificación condicionada](planning/README.md). |
| Libro y tabla principal | Contenedor real sin red: libro f5c completo, 39 HTML lógicos y 5.191 enlaces/activos; regeneración de 76 filas principales aditiva 19×4 y siete recetas V3. Comparaciones por campo/unidad aprobadas. | [Recibo real](container_profile/EXECUTION_RECEIPT.json), [perfil y comandos](container_profile/README.md), [kit externo](human_reproduction/addon_v1_1/README.md). |

Se conservan los **796 trabajos aditivos**, **30 ajustes V3** y **51 trayectorias/20.400 predicciones retrospectivas**. El resultado original V3 de **2,704312 % peor** que su control MSE y su NO-GO no se sustituyen por los nuevos controles CPU. Una repetición por semilla no es un diseño nuevo. El [inventario canónico](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/design_log.md) incorpora una propuesta V4 y distingue los intentos de los diseños.

Los cinco pares de referencia nuevos de D00 son descriptivos de desarrollo. No llenan las 84 celdas futuras de la familia V4 propuesta, ni justifican 340 ajustes de planificación tras abandonar la receta. No se transfieren varianzas, calibradores ni dictámenes R1 entre versiones.

## Alcance de la guía

La guía pide al menos tres pruebas distintas permitidas que rechacen igualdad de rendimiento, junto con dirección, margen práctico y condiciones adicionales; no impone literalmente tres nulos de margen. MCS está permitido en principio: esta extensión concreta carece de la correspondencia contrastiva suficiente y falla calibración. Potencia mínima de 0,90 por contraste y potencia conjunta reportada siguen pendientes; no se inventa un umbral conjunto universal.

La guía permite resultados costosos precalculados y versionados, con código generador documentado. La persona ajena al código debe ejecutar el libro completo en contenedor, regenerar la tabla principal y emitir su concepto con discrepancias. No exige universalmente repetir 796/30 entrenamientos o el controlador C parcial. El recibo automático usa `software-validation`: **no es una ejecución humana**. Se conserva el f5c exacto utilizado; no se atribuye esa ejecución a la nueva edición del libro.

## Uso y verificación

Este directorio se publica también como archivos descargables del libro. Tras obtenerlo completo desde GitHub:

```bash
python bundle_tools/verify_packet.py --packet .
```

Los manifiestos enumeran archivos, tamaños y SHA-256. No se necesitan entrenamientos para verificar el paquete. Para el caso externo completo, seguir primero [las instrucciones ampliadas](human_reproduction/addon_v1_1/README.md) y el [perfil offline](container_profile/README.md), completar antes de ejecutar la solicitud y tolerancias reales de la persona y conservar sus propios recibos. Los wheels, el checkout completo y las entradas se obtienen bajo sus condiciones; no están embebidos en este suplemento.

Código y resultados nuevos están separados de las comprobaciones históricas de Render H `feb83dbc08197bde7076857bd8ff475de197efd2`, los 14 ONNX, el DOI `10.5281/zenodo.24356217` y la cápsula 0.3.1-REVIEW. Sus alcances permanecen vigentes; este suplemento no es otro depósito Zenodo ni publica pesos nuevos. Ver [procedencia y exclusiones](RIGHTS_AND_EXCLUSIONS.md).

La publicación del libro se comprueba en un acta posterior con SHA efectivo, workflows y HTTP, separada de los planes científicos: [acta de publicación](https://github.com/jassan-bit/DeepLearning_202630/tree/e3-next-evidence-book-publication-f5c960ee/proof).

Consulte la [matriz vigente](requirements_matrix_current.csv) y las [acciones pendientes con responsables](NEXT_ACTIONS.md). Los documentos previos conservan sus fechas y no se convierten retrospectivamente en preregistros.
