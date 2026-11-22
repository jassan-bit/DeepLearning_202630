Inicio de planificación CPU existente — s2-existing-cpu-planning-start-1

Este helper conserva el intérprete CPU, la fuente S2 db74afd0763596f71bcbafb4b9150a44a2e83339, sus 304 archivos, la candidata, el protocolo y el banco de semillas. Usa las cuatro admisiones que el usuario ya preparó en su Colab y el corrector COIN-M 6c247cf0914cf28532bcfa5ceb07267a35c28c48. No adapta modelos ni crea un diseño algorítmico nuevo.

La celda publicada verifica SHA256 antes de ejecutar, conserva los bytes como CPU_PLANNING_START_BYTES y llama start_cpu_planning(globals()). Se usa en la misma sesión. Comprueba fuente, corrector, cuatro particiones de desarrollo por panel, intérprete PyTorch CPU y recibos operacionales completos. Prepara el caso sólo si aún no está preparado; verifica el plan de 340 trabajos y cinco semillas sin reescribirlo.

Antes del primer ajuste guarda una cápsula JSON bajo case/receipts/operational_addons/. Incluye los bytes base64 y SHA256 del propio helper, los archivos del corrector y sus recibos operacionales. La cápsula cumple la lista de archivos permitidos del respaldo cifrado. No se copian archivos .py sueltos ni datos brutos de mercado. El comando sync original confirma por relectura de GitHub el caso, plan, fuente y commit antes de permitir el entrenamiento. Un fallo o una cancelación conserva el caso y no activa nuevos ajustes.

La restauración conserva la cápsula JSON; no la instala ni ejecuta automáticamente. Para volver a leer los CSV COIN-M al recuperar el caso, habrá que extraer y verificar el corrector de esa cápsula. Su inclusión en el respaldo no se presenta como una nueva restauración automática ya ejecutada.

Después del respaldo, el usuario escribe exactamente PLANIFICAR 340 para activar los trabajos. Presupuesto: 340 trabajos, incluidos 320 ajustes neuronales, cuatro ARIMA y 16 controles analíticos. Los 320 × 120 segundos son 10 h 40 min nominales; no incluyen ARIMA, E/S, transferencias e interrupciones ni garantizan un límite histórico estricto. La ejecución conserva checkpoints y omite trabajos anteriores sólo tras verificarlos.

El token GitHub y la clave independiente de al menos 32 caracteres se solicitan mediante las entradas privadas existentes o los secretos de Colab E3_R3_GITHUB_TOKEN y E3_R3_BACKUP_SECRET. Nunca van en argumentos, celdas guardadas o mensajes públicos. Conservar la clave de cifrado fuera de Colab.

No activa exportación, dimensionamiento, entrenamiento confirmatorio ni apertura de prueba final. La planificación sigue siendo desarrollo expuesto; no acredita R3. Las limitaciones inferenciales documentadas de métrica primaria y decisión conjunta conservan su estado. No se ejecutaron entrenamientos reales durante la preparación y publicación de este helper. Las 20 pruebas usan mocks y metadatos generados, incluyendo orden de activación, cancelación, fallos de respaldo, integridad e idempotencia de la cápsula y compatibilidad con la lista real de archivos permitidos por GithubStore.

Asistencia de IA declarada para implementación y revisión de software. No se generan nuevas declaraciones del autor ni firmas humanas. Se mantienen los términos de código y datos de los recursos originales.
