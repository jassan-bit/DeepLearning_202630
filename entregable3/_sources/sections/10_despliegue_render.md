# 10 Bloque F Despliegue en Render

**Estado: aplicación CPU preparada y probada localmente; URL Render y prueba de humo pública pendientes.** El [capítulo 9](09_dashboard_predictivo.md) documenta la aplicación multipágina, cuatro artefactos ONNX y sus controles. El repositorio contiene `app.py`, dependencias fijadas, `Dockerfile` y `render.yaml`; no se ha activado una cuenta o servicio de Render. GitHub Pages publica el informe y las evidencias, y no ejecuta el servidor Dash.

El objetivo es una URL pública HTTPS en Render que permita usar las tres páginas del dashboard sin instalación local. El despliegue conservará las modalidades reales: cuatro modelos con inferencia CPU y trece con resultados precalculados, con sus límites visibles.

## 10.1 Pasos de despliegue

Los pasos siguientes son condiciones de cierre, no operaciones declaradas como realizadas. La aplicación y sus artefactos de producción deben verificarse antes de conectar el despliegue automático.

**1. Conservar la verificación local y del servidor.** `app.py` crea la aplicación con `use_pages=True` y expone `server = app.server` para Gunicorn. La evidencia de navegación, callbacks, entradas inválidas y predicción se conserva en el capítulo 9. El comando de arranque está en el README. Esa comprobación local debe repetirse sobre la versión efectivamente desplegada en Render.

El arranque no entrenará, hará HPO ni reajustará normalización o calibradores. El servicio importará sólo los módulos de interfaz, datos, inferencia y explicación admitidos, sin importar el pipeline de entrenamiento. Un cambio de página no cargará todos los modelos.

**2. Mantener el perfil de inferencia, Python y contenedor.** `requirements.txt` usa el lock de ejecución CPU con versiones exactas, incluidos Gunicorn y ONNX Runtime; no instala Torch ni el framework de entrenamiento. Las evidencias tienen alcances distintos:

| Evidencia actual | Dependencia y alcance | Trabajo previo a producción |
|---|---|---|
| Tres TorchScript históricos de E2 | Requieren Torch y conservan su paridad sobre 53 fechas de calibración. | Mantener el recibo histórico; el servicio actual utiliza sus exportaciones ONNX. |
| Cuatro ONNX y `e3_dashboard.inference` | CPUExecutionProvider, contrato raw D7×5, escaladores fijos, una sesión en caché y paridad sobre 608 fechas con tolerancia 10⁻⁶. | Repetir la prueba de humo en el entorno público y vigilar recursos con su configuración efectiva. |
| Metadatos, rangos de ajuste y cuantiles guardados | E2 dispone de q por predictor; E3 mantiene intervalo sin calibrar. Los controles de entrada y dominio están separados. | Conservar hashes y límites de interpretación; la sensibilidad por oclusión no acredita fidelidad o estabilidad explicativa. |

La ruta implementada es **ONNX + ONNX Runtime con `CPUExecutionProvider`** para MLP16, LSTM32, GRU32 y CNN-LSTM aditiva E3, semilla representativa 271801. La paridad de las 608 ventanas está verificada para esos cuatro; no para los otros trece sistemas. Los resultados precalculados no se presentan como inferencia en vivo. El contrato y la escala se verifican por predictor, con la diferencia entre memoria y exportación documentada.

El runtime de servicio excluye PyTorch y las dependencias de entrenamiento. La reconstrucción científica conserva un entorno separado con el framework. El Dockerfile copia sólo código de servicio, datos permitidos y artefactos ONNX, utiliza usuario sin privilegios y no inicia el pipeline científico. La comprobación de la imagen y del navegador se distingue de una reproducción humana independiente de los experimentos.

El perfil local usa **Python 3.12.14**, ONNX Runtime 1.23.2 y locks separados de ejecución y desarrollo. El Dockerfile fija el digest de su imagen base; sus recibos identifican el entorno y la forma de construcción. Python 3.11.9 sigue siendo un ejemplo de la guía, no el perfil probado aquí.

**3. Versionar el código y resolver el almacenamiento de artefactos.** El repositorio conservará código, dependencias, metadatos, hashes y procedencia. Los pesos que superen el límite vigente por archivo se gestionarán con Git LFS o almacenamiento externo documentado, con versión inmutable y comprobación de integridad. El despliegue debe obtener el contenido binario real, no sólo un puntero LFS o un nombre de archivo.

La descarga/caché de pesos verificará SHA-256 y fallo explícito por ausencia o corrupción. Los identificadores de modelo procederán de un registro permitido; no se aceptarán rutas, URLs o pesos suministrados por el usuario. Las licencias de código y datos mantendrán su alcance propio. Un modelo sin artefacto de inferencia validado seguirá mostrando los resultados precalculados y la justificación documentada.

**4. Configurar el Web Service de Render.** Una vez aprobada la construcción local, el servicio se conectará a la rama y versión revisadas. Para la vía nativa Python, los comandos previstos son:

```text
Construcción: pip install -r requirements.txt
Inicio: gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 120
```

`$PORT` lo proporciona Render; el enlace a `0.0.0.0` evita que el proceso quede accesible sólo desde la interfaz local. El repositorio incluye un blueprint Docker en `render.yaml`, con `autoDeploy: false` y healthcheck `/healthz`; el archivo no crea un servicio por sí mismo. El proceso local se comprueba con un worker y un hilo. La concurrencia, tiempos y recursos efectivos de Render quedan por medir.

El diseño inicial usa **un worker, carga serial y un modelo activo bajo demanda**. El máximo de dos modelos por proceso de la sección 9 es un techo de caché, no una obligación de mantener dos simultáneamente. Las sesiones de inferencia se obtendrán después de validar el modelo solicitado, sin una `InferenceSession` global que cargue un modelo al importar el módulo. Si el servicio se implementa con runtime Docker, su configuración deberá reflejar esa vía y el target de servicio probado, sin mezclarla con un build nativo no verificado.

**5. Revisar construcción, arranque y funcionamiento público.** Se inspeccionarán logs y dependencias instaladas, URL HTTPS, recuperación ante errores, disponibilidad de artefactos y comportamiento de las tres páginas. Se medirán arranque frío, carga/evicción, RSS conjunto y transitorio, latencia completa y coste de explicaciones con la configuración efectiva. Una carga inválida no debe detener el servicio, revelar una traza interna o producir un pronóstico sustituido por cero.

El ejemplo de clasificación con Softmax de la guía ilustra un patrón de callback, pero no es la salida de este problema. Aquí el modelo produce una predicción continua de σ₇ con h=1, la escala se invierte exactamente una vez y el intervalo, cuando se habilite y verifique, usará el calibrador del mismo predictor. No se convertirán salidas de regresión en probabilidades de clase.

**6. Ejecutar y conservar la prueba de humo pública.** El guion deberá recorrer la URL Render y cada modelo habilitado. La lista procede del registro real de cuatro ONNX; los trece precalculados mantienen sus fuentes y su justificación. La verificación local del capítulo 9 no se renombra como prueba de la URL pública.

| Comprobación del servicio | Evidencia esperada | Estado |
|---|---|---|
| Contexto, EDA y Modelos | URL/versión, navegación y callbacks de cada página. | Implementados en local; verificación pública pendiente. |
| Filtros y datos insuficientes | Todas las figuras e insights actualizados, n efectivo y mensajes claros. | Pendiente. |
| Cada modelo en vivo | Un ejemplo permitido, contrato temporal, salida finita y procedencia del artefacto. | Cuatro ONNX verificados localmente; prueba pública pendiente. |
| Preprocesamiento y escala | Coincidencia con la referencia validada en tolerancia declarada. | Pendiente de integración completa. |
| Intervalo y explicación | Mismo predictor/caso, unidades, nominal y límites de cobertura/fidelidad. | Pendiente. |
| Entradas inválidas y OOD | Rechazo o aviso correcto, sin caída ni exposición de rutas/secretos. | Pendiente. |
| Recursos y tiempos | Arranque frío/caliente, carga bajo demanda y latencias frente a umbrales prefijados. | Pendiente: sin mediciones Render. |
| Modalidades precalculadas | Catálogo completo, fuentes y justificación de restricciones. | Diecisiete sistemas en la interfaz local; comprobar en Render. |

El informe de humo conservará fecha, commit, entorno, URLs visitadas, identificadores de casos permitidos, modelo/versión, estados HTTP, resultados, tiempos y errores, con extractos de logs identificados por versión e instante. Esos extractos excluirán secretos, rutas internas e inputs privados. Sus umbrales de aceptación se fijarán antes de medir. Actualmente no hay una URL Render, captura de funcionamiento o resultado de humo que pueda declararse aprobado.

### Recursos del plan y contingencia

Antes de desplegar se verificarán los límites vigentes del plan en la documentación oficial de Render. Se contemplarán suspensión por inactividad, arranque frío, memoria/CPU, almacenamiento y persistencia. Los RSS de procesos CPU de exportación y las latencias forward de las tres referencias no acreditan capacidad del servicio completo ni su comportamiento concurrente.

Si se requiere reducir tamaño, se evaluarán compresión o cuantización con paridad, pérdida de métrica frente al margen heredado y equivalencia cuando proceda. Cargar resultados precalculados o elegir un plan con más recursos exige justificación y contabilidad; este documento no selecciona ni activa un plan de pago. El entrenamiento seguirá fuera del servidor.

Para la sustentación se comprobará la URL con antelación y se preparará un vídeo breve de esa versión, con fecha. Las capturas locales del capítulo 9 son evidencia de la aplicación local; no prueban un servicio público. Ante una interrupción se explicará el estado y se usará el respaldo como contingencia; la URL Render funcional y su prueba de humo siguen siendo entregables propios.
