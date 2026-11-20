# Piloto controlado de desarrollo de V3

Este paquete cierra la preparación de la comparación de V3; no contiene resultados de mercado nuevos. El expediente bibliográfico está en `book_e3/_static/e3_r1_close_20261120/` y los capítulos 6.15–6.16 y 7.12–7.13 explican antecedentes, límite, hipótesis y controles. La referencia aditiva y los 796 trabajos anteriores se conservan.

## Ejecutar

Abrir `notebooks/entregable3_v3_controlled_pilot_colab.ipynb` en Colab con GPU y ejecutar su celda. Esa celda fija el código por commit y comprueba SHA-256 antes de ejecutarlo. Se necesita un token GitHub con acceso a este repositorio y Contents: Read and write; el lanzador lo consulta como secreto `GITHUB_TOKEN` o lo solicita con entrada oculta. Nunca se pega en el código.

La primera ejecución admite la GPU realmente asignada, prepara Python 3.12.12/Torch 2.8.0+cu126 aislados y publica el plan y su procedencia antes de ajustar. Guarda el identificador de ejecución que imprime. Después de un reinicio se repite la misma celda con ese identificador en `RESUME_RUN_ID`; se exige el mismo perfil GPU/software. Una rama previa sin identificador explícito detiene un inicio accidental.

## Alcance

Cinco recetas neuronales, dos controles analíticos, 15 ajustes HPO y 15 finales: 30 ajustes previstos. Sólo desarrollo BTC; 532 ventanas de ajuste, 54 de selección, siete purgadas y 55 de evaluación del piloto. El calendario ya estuvo expuesto. El análisis espera a completar todas las recetas; la heurística del 2 % no es significancia, margen confirmatorio ni acreditación R1/R3.

La oportunidad nominal de 120 segundos empieza en el marcador de cada fase de optimización; su guardado y las desconexiones consumen ese plazo. Se mide aparte la construcción del modelo y los tensores. El guardado remoto conserva inicio, primer checkpoint validado y cierre. Se recupera sólo lo publicado: una pérdida anterior al primer estado validado puede dejar un ajuste incompleto sin renovar su presupuesto. Ver [especificación científica](ENGINE.md), [control de agregación](PLUGIN_CONTROL.md) y [guardado y reanudación](README_launcher.md).

## Verificación y atribución

`verification.json` y `test_results.log` documentan pruebas con datos artificiales y repositorios Git bare locales. No se ejecutó una GPU de Colab ni se descargaron o entrenaron datos de mercado durante esta preparación. Se declara asistencia de IA en revisión, adaptación, código y pruebas; el estudiante debe comprender y defender la entrega. La revisión por IA sigue la indicación docente comunicada por el autor. El pilotaje futuro no sustituye una defensa del estudiante.

```bash
python -m unittest discover -s experiments/e3_v3_development_pilot_20261120 -p 'test_*.py' -v
```

Requiere Python 3.12, NumPy 2.3.5 y Torch 2.8.0 para las pruebas CPU. El flujo de integración continua ejecuta estas pruebas y verifica el expediente analítico; no ejecuta el piloto de mercado.
