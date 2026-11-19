# Adopción de intervalos y extensión de inferencia

Esta versión sirve cinco predictores existentes en ONNX CPU: MLP16, LSTM32, GRU32, CNN–LSTM–atención aditiva E3 y TSMixer. Conserva intactos el manifiesto, los pesos, los metadatos y los datos de `e3_dashboard_v1`. No entrena modelos ni ajusta de nuevo escaladores o calibradores. La preparación, revisión y pruebas de estos archivos se realizaron con asistencia de IA (Codex).

## Intervalo posterior del predictor E3

`artifacts/e3_serving_calibration_v2/manifest.json` declara un perfil separado vinculado mediante SHA-256 al ONNX, checkpoint, metadatos originales, semilla 271801 y escala de salida del predictor E3 de 1210 parámetros. El runtime rechaza un perfil alterado o asociado a otro predictor.

El valor existente **q = 0.005402463770651868 σ₇** procede de 53 residuos de calibración, rango 49 de 53 para nivel nominal 90 %. Se verificaron los hashes y la cronología de sus fuentes. Sus fechas objetivo son 2024-11-08 a 2024-12-30; quedan después del ajuste y la validación. El perfil sólo se aplica si la fecha objetivo solicitada es posterior a 2024-12-30. El intervalo es `[max(0, predicción − q), predicción + q]`. Los pronósticos puntuales, pesos y escalador no cambian.

Las 608 fechas históricas del **predictor individual 271801** tienen cobertura **94.078947 %**; el grupo temporal 2026Q1 tiene **85.555556 %**. La interfaz muestra 94,08 % y 85,56 % y advierte que no hay garantía bajo cambio temporal. No se atribuyen al predictor individual las medias de diez semillas del análisis original. Tampoco se presenta un intervalo marginal como garantía por grupo o bajo dependencia serial.

`calibration_runtime_verification.json` registra 661 ventanas históricas verificadas con ORT CPU: 53 de calibración y 608 de evaluación. El error máximo frente a las predicciones guardadas es **2.584e-8 σ₇**, inferior a la tolerancia declarada 1e-6. El q conserva la precisión original: una observación de calibración situada en el límite cambia su inclusión por una diferencia de aproximadamente 3e-10 tras la conversión; la cobertura empírica de calibración pasa de 49/53 en el cálculo original a 48/53 en ORT. No se retocó el cuantil para ocultar ese efecto. La cobertura histórica de evaluación permanece igual.

El antiguo análisis con `activate_in_dashboard: false` permanece como evidencia histórica. Este perfil separado documenta la decisión posterior de adopción en el servicio. No sustituye la evaluación prospectiva, la comparación bajo tratamiento común, R1 o R3; tampoco se aplica a un nuevo candidato V3.

## Extensión TSMixer

`artifacts/e3_serving_extensions_v2/manifest.json` usa el esquema original `schema_version: 1`, con `models` que contienen `id`, `slug` y `metadata_sha256`. La entrada `TSMixer` resuelve `tsmixer/metadata.json` y `tsmixer/model.onnx`. Los metadatos usan el mismo contrato que los cuatro originales. El runtime combina ambos catálogos y rechaza identificadores duplicados, rutas inválidas y hashes no coincidentes. Mantiene como máximo una sesión residente.

El checkpoint original, la receta seleccionada, el escalador y el calibrador se identifican en `export_capabilities/tsmixer/provenance.json`. Se exportó la semilla fija menor, 271801, sin seleccionar por el resultado de prueba. El ONNX ocupa 165263 bytes y conserva 35044 parámetros. Su paridad se verificó en 608 ventanas de evaluación y 53 de calibración, con error máximo frente al predictor guardado **2.285e-8 σ₇**. Mantiene el q original **0.007345522909209999**, con cobertura histórica individual de 89.309211 %.

`export_capabilities/tsmixer/cpu_examples.json` y `verify_cpu_examples.py` permiten repetir tres ejemplos reales sin Torch. Las pruebas del runtime pasan estas mismas ventanas por el contrato público CSV, con fecha y explicación. No se incorporan pesos arbitrarios ni modelos de sustitución.

La nueva latencia TSMixer se midió por separado y se conserva en su propio recibo. La frontera de Pareto mantiene los cuatro modelos de su perfil histórico común; no mezcla mediciones procedentes de ejecuciones diferentes. La tabla de capacidades distingue cinco inferencias activas de los 17 sistemas históricos. `export_capabilities/capability_matrix.csv` justifica tres clásicos como históricos bajo el contrato de siete barras y deja nueve redes pendientes de exportar y verificar; no declara impedimentos de recursos sin evidencia.

## Verificación y alcance

`unit_tests.log` contiene las comprobaciones de admisión CSV, integridad de datos y artefactos, calibración vinculada al predictor, paridad TSMixer, caché de una sesión e importación sin Torch ni código de entrenamiento. `browser/acceptance.json` conserva el recorrido real del navegador y sus capturas; `browser_attempt_01/` mantiene el intento previo que detectó la incompatibilidad de perfiles del gráfico Pareto y motivó su corrección.

Las comprobaciones locales y del contenedor no constituyen por sí solas aceptación del despliegue público en Render. La aceptación remota debe registrar la URL pública y el commit que efectivamente responde en `/healthz`.
