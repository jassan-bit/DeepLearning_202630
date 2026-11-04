# CV de Filipe con cuatro historias: publicación de resúmenes

D=7/14/21/28 días de OHLCV; objetivo sigma7 poblacional y horizonte de un día. D7 reutiliza la representación relativa de su experimento firmado; las demás historias se ajustan en una ampliación independiente. Los datos originales, predicciones y artefactos registrados no son modificados por este exportador. El control de precios absolutos pertenece exclusivamente al ciclo inicial D7.

La evaluación de test, cuando phase=complete, usa pesos y escaladores fijados antes de abrir su bruto; sus meses son bloques de reporte, no reajustes ni nuevos folds de selección. Los resultados no acreditan un piloto prospectivo ni cierran el respaldo bibliográfico de MLP/CNN1D.

## Resultados y auditoría

`analysis_cv/` conserva métricas y BDS mensuales de validación; `test/analysis/` conserva la evaluación final. `freeze.json`, `final_models_manifest.json`, `test/raw_test_audit.json` y `test/cohort_manifest.json` identifican selección previa, firmas, primera lectura y elegibilidad. Los meses de test no implican nuevos entrenamientos.

## Alcance de la publicación

Esta carpeta pública contiene exclusivamente resúmenes agregados por corrida o mes, pruebas BDS, reglas, fechas, calidad, firmas SHA-256 y código. Las predicciones individuales, vectores de objetivos o residuos, derivados diarios de mercado, escaladores, checkpoints y archivos TAR completos permanecen en el espacio local del experimento; no se ofrecen como descargas públicas. `final_models_manifest.json` publica únicamente identidades y hashes; `run_artifacts_manifest.json` identifica los materiales conservados localmente. `public_allowlist_manifest.json` enumera y verifica los archivos permitidos en esta vista.

## Código y entornos

La carpeta `code/` conserva los nuevos scripts y el repositorio incluye los helpers firmados y locks de dependencias. `prepare_filipe_multihistory.py` usa el entorno NumPy1.26/timeseries-cv0.1.5; los entrenamientos usan el entorno PyTorch registrado. `analyze_filipe_multihistory.py --phase cv` y `--phase test` reutilizan predicciones guardadas, sin entrenar. `export_filipe_multihistory_book.py --phase complete` valida firmas y exporta solo resúmenes, sin abrir brutos o publicar vectores y pesos. La secuencia de un experimento nuevo es preparar índices, registrar y entrenar CV, cerrar configuraciones, ajustar y firmar modelos finales, abrir el test autorizado una vez y analizar las predicciones. No se deben modificar o sobrescribir los registros firmados de este experimento para probar otra decisión después de observar test.
