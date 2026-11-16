# Reanálisis integrado y autónomo de E2

Descargue **reanalisis_integrado_v1.zip**, extraiga todo su contenido e ingrese en `E2_reanalisis_integrado`. El archivo contiene las trece entradas completas y el ejecutor; no necesita el repositorio, conexión a Internet durante el reanálisis, pesos de redes ni GPU.

Con Python 3.12:

```bash
python -m pip install -r requirements.txt
python run.py
```

La instalación inicial de NumPy 2.3.5 y SciPy 1.17.0 necesita el índice de paquetes o sus ruedas locales. La ejecución posterior usa exclusivamente los archivos incluidos. `python run.py --check` comprueba los hashes de los trece archivos sin ejecutar cálculo científico.

El comando extrae cada módulo en una copia nueva y ejecuta secuencialmente el inventario de fuentes D02 —sin forwards—, datos D03, registros/auditoría D04, A2 D06, fichas D07, errores/incertidumbre D08, justificación NAS D09, agregados del paper D10 y los suplementos de controles negativos, OOF–prueba/variabilidad, dependencia de entradas, HPO descriptivo y cierre Scopus7500. Los ocho paquetes v3 conservan su fecha de corte; los cinco suplementos tienen su identidad propia. El formulario bibliográfico histórico del módulo HPO queda sustituido por la decisión de cierre Scopus7500 y no genera una tarea activa.

La salida se crea junto a la carpeta del paquete como `resultado_reanalisis_integrado`, con sufijo si ya existe. Se puede elegir otra ubicación nueva con `--output RUTA`. Abra `informe.html`: enlaza las tablas, gráficas e informes regenerados. `execution_report.json` y `task_statuses.csv` conservan los estados terminales de las trece tareas, sus comandos, retornos, duraciones, verificaciones de resultados y hashes de las salidas. Un módulo fallido no se declara aprobado; las demás tareas se registran por separado. Los resultados anteriores se conservan.

Las diferencias numéricas se informan por módulo y unidad. D06/D08 conservan su discrepancia aritmética del orden de **2,22×10⁻¹⁶** frente a los valores publicados; no se reemplaza por cero. Los dos CSV de inferencia se cotejan byte a byte con sus referencias. Las verificaciones de esquema o el éxito de un comando no certifican ausencia de fuga: los controles negativos mantienen sus resultados científicos y límites.

Este ejecutor cierra la **integración portátil de los trece reanálisis existentes indicados**, con un comando y entradas autosuficientes. **No repite entrenamiento, forwards, HPO, ablaciones, todas las perturbaciones o todas las tablas y figuras del libro.** No se presenta como una campaña histórica reentrenada ni recupera datos o tiempos originales no conservados. NAS sigue sin aplicar a las arquitecturas definidas.

El límite predeterminado es de 600 segundos del equipo local y puede cambiarse con `--cap-seconds`. La prueba del proyecto tiene un plan y recibo presupuestario propios dentro de las 84 horas acumuladas; el programa portátil no accede al diario histórico ni crea un presupuesto nuevo del proyecto.
