# E3: datos, servicio y piloto común de desarrollo

Preparado con asistencia de IA declarada, 18 de noviembre de 2026. Las ejecuciones históricas y sus tablas conservan sus bytes. Esta etapa no acredita R1/R3 ni contiene resultados de mercado de V3.

1. **Render:** tras publicar esta versión en GitHub, abrir el servicio `deeplearning-e3-dashboard` → **Manual Deploy → Deploy latest commit**. Esperar `Live` y conservar el SHA completo del log. El blueprint usa `autoDeploy: false`. La prueba exige que `/healthz` comunique ese mismo SHA, además de comprobar recursos y callbacks; es identidad declarada por el servicio, no una atestación criptográfica.
2. **CPU:** ejecutar [el cuaderno de admisión y humo](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/e3_cierre_datos_render_colab.ipynb). Verifica D01 y descarga los 100 ZIP mensuales de D02/D03 con sus CHECKSUM oficiales. Comprueba las tres páginas y cinco modelos mediante navegador en Render. Guarda incluso fallos en una rama nueva `e3-closure-cpu-*`, sin cambiar `main`. Necesita `GITHUB_TOKEN` en Secretos de Colab, con Contents: Read and write para este repositorio. No pegar el token en código o chat.
3. **GPU:** ejecutar [el cuaderno de desarrollo común](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/e3_cierre_piloto_gpu_colab.ipynb). Reutiliza el entorno aislado Python 3.12.12/Torch 2.8.0+cu126, sin cambiar el kernel. Ajusta V3 y sus controles con 532/116 ventanas de BTC; no consulta prueba ni calibración. Guarda checkpoints y resultados incrementalmente en ramas nuevas `e3-closure-pilot-*`. La primera ejecución de tres semillas es de desarrollo, no dimensiona potencia ni certifica generalización.
4. Revisar los recibos reales antes de ampliar a la familia completa y diez semillas. Sólo el panel completo de 18 sistemas permite iniciar la planificación condicional de potencia. Faltan todavía protocolo final común de HPO/presupuesto, admisiones reales de D02/D03, estudios en los tres conjuntos, robustez y dictamen focal de V3.

Cada cuaderno fija un SHA completo antes de trabajar. Reejecutar la misma celda conserva el contexto local y los resultados para reintentar el guardado; reiniciar Colab elimina ese contexto. Los resultados confirmados en GitHub permanecen. No borrar ni editar checkpoints o recibos para forzar una reanudación: compartir el error y conservar la carpeta.

## Evidencia implementada y verificada localmente

- `data/`: recolector stdlib, fuentes oficiales, nueve pruebas offline; D01 3270 filas reales. D02/D03 siguen sin admitir hasta completar los bytes en Colab. El proxy de este entorno respondió 403 a esas fuentes.
- `mechanism/`: V3 de tres átomos y masas aprendidas, 1262 parámetros, teoría y siete pruebas sintéticas. CRPS y modelos de distribuciones tienen antecedentes citados; novedad y utilidad empírica no acreditadas.
- `campaign/`: piloto común con checkpoints de optimizador/RNG, fases SAE y predicción causal de clásicos; simulación condicional de potencia con Holm sobre la familia completa.
- `serving/`: intervalo posterior E3 seed271801, TSMixer quinto ONNX, pruebas unitarias, navegador y contenedor sin red. Nueve redes mantienen la exportación pendiente; los tres clásicos requieren un contrato histórico más amplio que siete barras.
- `acceptance/`: cliente V2 de cinco modelos, contratos y recorrido real local. Ambos indicadores públicos son falsos en loopback; no se ha ejecutado una aceptación pública desde este entorno.
- `orchestration/`: guardado por objetos Git en ramas nuevas, preservación completa del árbol fuente, verificación de bytes y reintentos. Rechaza secretos, archivos comprimidos inseguros y declaraciones de acreditación científica.
- `archive/`: ZIP seleccionado de 34 fuentes fijadas a `1bae73b`, manifiesto y formulario Zenodo. Excluye datos brutos, pesos y CSV de referencia. Licencia propia y depósito real pendientes; DOI no conocido. No es un archivo integral del repositorio.

La excepción docente comunicada permite revisión de R1 por IA declarada. No convierte las pruebas de software, un DOI o un buen piloto en evidencia de novedad o superioridad, y no firma por el autor ni por un reproductor humano.
