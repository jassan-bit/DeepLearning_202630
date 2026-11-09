# Síntesis de selección de modelos · 06/11/2026

Revisión documental de Codex sobre evidencia ya incorporada a E1, posterior a sus dos fases experimentales. No ejecuta búsquedas, recupera originales nuevos, modifica recetas ni selecciona usando resultados de test. No es prerregistro ni una revisión humana independiente.

`decision_matrix.csv` compara siete familias con evidencia favorable, objeciones y condiciones para reconsiderar cada puesto. `summary.json` conserva los criterios y contrafactuales, con el corpus histórico N=7 separado de los originales CVI/GARCH. `search_evidence_ledger.csv` diferencia evidencia recuperada, declaración del autor, propuesta y pendiente.

Los numeradores se cotejan con `crypto_extension_20261105/family_frequencies.csv`. RNN simple registra cero comparadores verificados en los textos actuales; no es una frecuencia tabulada originalmente ni prueba de inexistencia. CNN genérica y CNN1D no se suman como estudios diferentes.

La conclusión es una terna operativa condicionada. LSTM/GRU son las dos familias más recurrentes del corte; TCN añade un contraste causal convolucional respaldado directamente por S1. La selección no es el top3 de frecuencia y comparte dos mecanismos recurrentes, por lo que el cumplimiento conjunto de recurrencia, simplicidad y diversidad continúa sujeto al cierre del corpus. La autorización para publicar no constituye evidencia académica.

Las arquitecturas recurrentes del código reinician el estado al procesar cada ventana. La memoria separada LSTM no introduce información anterior a D ni permite atribuir resultados a memoria larga fuera de la ventana. Los parámetros de fase II: 1.121/1.489/1.185 describen las recetas ajustadas, sin probar simplicidad universal o superioridad de la terna.

Los metadatos públicos no contienen resúmenes exportados, contactos ni textos completos. Los hashes acreditan integridad de este expediente y sus fuentes; no prueban una ejecución de búsqueda ni una fecha experimental anterior.
