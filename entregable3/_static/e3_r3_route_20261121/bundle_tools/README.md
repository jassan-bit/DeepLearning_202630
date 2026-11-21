# Verificación del suplemento de la ruta R3

Este verificador usa sólo la biblioteca estándar de Python y lee los archivos guardados. No importa los programas de validación estadística, no genera muestras y no ejecuta modelos, predicciones, descargas ni despliegues. Se declara asistencia de IA en su preparación y revisión.

Desde el repositorio publicado:

```bash
export PYTHONDONTWRITEBYTECODE=1
python -m unittest discover -s experiments/e3_r3_route_20261121/bundle_tools -p 'test_verify_package.py' -v
python experiments/e3_r3_route_20261121/bundle_tools/verify_package.py --packet experiments/e3_r3_route_20261121
```

`--report` permite escribir el recibo en una ruta externa al paquete. El programa rechaza escribir su propio recibo dentro del inventario que verifica.

Comprueba el inventario exacto y los hashes del paquete y de sus cinco componentes; rechaza archivos adicionales, ausentes, modificados, enlaces simbólicos y rutas inseguras. Los extractos completos del PDF, los pesos, datos de modelos y cachés quedan fuera de este inventario público.

Las comprobaciones semánticas distinguen candidata propuesta de candidata admitida, método candidato de procedimiento acreditado y borrador de registro prospectivo. Conservan los resultados adversos y las decisiones V3/V4, la condición pendiente de R3 y el valor de sustentación proporcionado por el autor, sin atribuirle una defensa observada. Los dos preflights técnicos siguen bloqueados; los costes no medidos y las dimensiones prospectivas quedan sin asignar.

Cuando existe una ejecución controlada, comprueba el vínculo con el código y plan publicados antes de ejecutarla, los estados completo/parcial/error, sus recuentos y el archivo de réplicas. Recuenta los eventos ya guardados sobre los verdaderos nulos del plan, los intervalos puntuales y los conjuntos MCS retenidos. No vuelve a calcular p-valores, cuantiles, intervalos estadísticos ni remuestreos. Si una ejecución falla, los draws de los intentos fallidos se mantienen separados de los draws demostrados por réplicas completas. Una ejecución no realizada requiere un estado y motivo explícitos.

Las pruebas de software alteran metadatos y archivos temporales para comprobar que los errores bloquean afirmaciones indebidas. Sus fixtures no son réplicas científicas ni evidencia de calibración.

Un resultado `PASS` acredita consistencia de bytes y de las condiciones documentadas. La pertinencia de los procedimientos, sus supuestos, la calibración general, la potencia del diseño y la aceptación académica requieren sus pruebas correspondientes. El programa conserva `R3_accredited=false` y `confirmation_enabled=false`.
