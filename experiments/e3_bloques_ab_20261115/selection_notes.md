# Selección de configuraciones para el Bloque B

Se reconstruyeron las trece elecciones neuronales a partir de sus recibos originales de selección, recuperados con bytes originales de `results/e2-reduced-delivery-v1` en el commit `d604ede6bf0978e14a1b1d7bafaae6250fb8ed07`. El archivo `selection_receipt_provenance.json` vincula sus rutas y SHA-256. Los tres comparadores clásicos conservan los registros publicados en E2. No se entrenó, cambió o seleccionó ninguna receta a partir de prueba.

## Criterio y dispersión

Cada red conserva seis configuraciones registradas y tres folds de validación de 83 fechas: enero, julio y octubre de 2024. El criterio histórico es la RMSE agrupada de las 249 fechas, equivalente a la raíz de la media de los cuadrados de las tres RMSE porque tienen el mismo tamaño. La media aritmética de RMSE por fold es un diagnóstico distinto. Ambos criterios eligen la misma configuración en las trece redes.

La tabla registra la media y la desviación muestral descriptiva entre esos tres folds. No calcula un IC iid de tres repeticiones ni confunde esos folds temporales con semillas independientes. El HPO ejecutado utilizó una semilla por configuración; la dispersión entre semillas de validación y el sesgo del ganador no se estimaron. Los IC de prueba se copian de la tabla vigente de E2 y conservan su interpretación condicional de variabilidad de optimización.

## Liderazgos y MCS

TimeMixer C01 tiene la menor RMSE agrupada de validación entre las quince recetas con score recuperado: 0,0041445906771571915. Su media por fold es 0,0041047951956858 y su DE descriptiva es 0,0007017352948082521. Persistence tiene una receta fija lag-1 cuyo recibo no contiene score CV; por ello no se certifica aquí un líder global de validación de los dieciséis sistemas.

El LSTM-autoencoder es el líder descriptivo de prueba histórica de A2 y pertenece al MCS al 90 %. TimeMixer queda excluido del MCS al 90 % de E2, con p=0,015798420157984203. Esta comprobación de pertenencia se reporta sin sustituir al ganador de validación ni convertir el MCS retrospectivo de MSE en evidencia de superioridad de RMSE.

## Exportación y campos ausentes

La tabla conserva todas las semillas y referencias congeladas. Para una exportación representativa se identifica la menor semilla registrada, 271801, por una regla de metadatos que nunca consulta su RMSE de prueba. No se elige la mejor semilla ni se convierte una exportación en seis predictores nuevos.

Parámetros, FLOPs core y latencia GPU existentes se vinculan a su fuente E2. Los FLOPs totales, las latencias CPU, el percentil 95, la memoria pico CPU, la métrica del modelo exportado y el round-trip no se inventan en esta tabla. Los ensayos de exportación que se ejecuten en E3 deben registrarse por separado con hardware, tolerancias, muestras y hashes.

## Reproducción sin red

```bash
python scripts/e3_select_e2_configurations.py --root .
python scripts/e3_select_e2_configurations.py --root . --check
python -m unittest discover -s tests -p test_e3_select_e2_configurations.py -v
```

Los trece tests verifican selección independiente de prueba, rechazo de CV incompleto, NaN, ganadores alterados, uso de calibración, hashes modificados y reproducción determinista de los tres productos generados. Los campos faltantes siguen siendo `null` en JSON; no son ceros medidos ni aprobaciones del requisito.
