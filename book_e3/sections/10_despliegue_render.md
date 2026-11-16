# 10 Bloque F Despliegue en Render

**Estado: planificación, sin aplicación Dash, URL Render ni prueba de humo verificadas.** El [diseño del dashboard](09_dashboard_predictivo.md) define el producto por construir. La publicación actual del Jupyter Book permite consultar el informe; el servicio predictivo deberá producir sus propias evidencias de construcción, acceso e inferencia. Esta actualización documental no crea `app.py`, `Dockerfile`, `render.yaml` ni una cuenta o servicio de pago.

El objetivo es una URL pública HTTPS en Render que permita usar las tres páginas del dashboard sin instalación local. El despliegue incluirá las modalidades reales de cada modelo —inferencia habilitada, resultados precalculados o candidato sin evaluación— y mantendrá sus límites visibles.

## 10.1 Pasos de despliegue

Los pasos siguientes son condiciones de cierre, no operaciones declaradas como realizadas. La aplicación y sus artefactos de producción deben verificarse antes de conectar el despliegue automático.

**1. Verificar la aplicación local y el servidor.** El futuro `app.py` deberá crear la app multipágina y exponer `server = app.server` para que Gunicorn resuelva `app:server`. Se comprobarán navegación, callbacks, entradas inválidas, gestión de errores y el recorrido de inferencia con ejemplos autorizados. `python app.py` será la prueba de arranque local cuando ese archivo exista; hoy no se presenta como un comando ya ejecutado del dashboard.

El arranque no entrenará, hará HPO ni reajustará normalización o calibradores. El servicio importará sólo los módulos de interfaz, datos, inferencia y explicación admitidos, sin importar el pipeline de entrenamiento. Un cambio de página no cargará todos los modelos.

**2. Cerrar el perfil de inferencia, Python y el contenedor.** La guía 10.1 exige un `requirements.txt` con versiones exactas, incluidos **Gunicorn y ONNX Runtime**, y sin el framework de entrenamiento. Esa condición aún no está satisfecha por los tres artefactos actuales:

| Evidencia actual | Dependencia y alcance | Trabajo previo a producción |
|---|---|---|
| TorchScript de MLP16, LSTM32 y GRU32 | Requieren PyTorch para cargar y ejecutar. Su paridad puntual se verificó en CPU sobre 53 fechas históricas de calibración. | Conservar esta evidencia de desarrollo y exportar a ONNX para el perfil previsto de servicio. |
| Wrapper `predict_exported_reference` | Importa Torch, recibe ventanas ya estandarizadas y devuelve σ₇ en unidades originales. | Implementar el adaptador de producción con ONNX Runtime CPU y verificar el mismo contrato y escala. |
| Metadatos y calibradores de E2 | Identifican pesos, escalas y q; no son un servidor ni una exportación ONNX. | Comprobar hashes, entradas/salidas, preprocesamiento, intervalos y explicaciones del flujo final. |

La ruta de producción prevista es **ONNX + ONNX Runtime con `CPUExecutionProvider`**. No existen aquí exportaciones ONNX verificadas del catálogo. La conversión deberá conservar receta, semilla, esquema D7×5 de las tres redes, estadísticas de entrenamiento y escala objetivo, y repetir la ida y vuelta con tolerancias explícitas. La comprobación de las 608 entradas de prueba A2 sigue pendiente; la paridad actual sobre calibración no la sustituye. También faltan exportaciones y contratos de servicio del resto del catálogo.

El runtime de servicio excluye PyTorch y las dependencias de entrenamiento. La reconstrucción científica del experimento necesitará un entorno separado que sí contenga el framework y sus versiones. El diseño del `Dockerfile` deberá distinguir ambos perfiles o targets y demostrar sus dependencias e importaciones efectivas; este capítulo no entrega una imagen construida.

Python **3.11.9 es un ejemplo de la guía**, no una versión validada para este servicio. Las exportaciones heredadas se ejecutaron con Python 3.12.14 y Torch 2.8.0+cpu; ese antecedente no verifica compatibilidad de una imagen ONNX Runtime. La versión de Python de producción se fijará después de comprobar instalación, arranque, exportación, inferencia y pruebas con el perfil exacto. Se guardarán versiones, arquitectura y digest de la imagen probada.

**3. Versionar el código y resolver el almacenamiento de artefactos.** El repositorio conservará código, dependencias, metadatos, hashes y procedencia. Los pesos que superen el límite vigente por archivo se gestionarán con Git LFS o almacenamiento externo documentado, con versión inmutable y comprobación de integridad. El despliegue debe obtener el contenido binario real, no sólo un puntero LFS o un nombre de archivo.

La descarga/caché de pesos verificará SHA-256 y fallo explícito por ausencia o corrupción. Los identificadores de modelo procederán de un registro permitido; no se aceptarán rutas, URLs o pesos suministrados por el usuario. Las licencias de código y datos mantendrán su alcance propio. Un modelo sin artefacto de inferencia validado seguirá mostrando los resultados precalculados y la justificación documentada.

**4. Configurar el Web Service de Render.** Una vez aprobada la construcción local, el servicio se conectará a la rama y versión revisadas. Para la vía nativa Python, los comandos previstos son:

```text
Construcción: pip install -r requirements.txt
Inicio: gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
```

`$PORT` lo proporciona Render; el enlace a `0.0.0.0` evita que el proceso quede accesible sólo desde la interfaz local. Los valores de hilos y timeout son una propuesta por probar, no tiempos o concurrencia ya certificados. La configuración se documentará en el futuro `render.yaml`, junto con Python validado y variables necesarias, sin exponer credenciales.

El diseño inicial usa **un worker, carga serial y un modelo activo bajo demanda**. El máximo de dos modelos por proceso de la sección 9 es un techo de caché, no una obligación de mantener dos simultáneamente. Las sesiones de inferencia se obtendrán después de validar el modelo solicitado, sin una `InferenceSession` global que cargue un modelo al importar el módulo. Si el servicio se implementa con runtime Docker, su configuración deberá reflejar esa vía y el target de servicio probado, sin mezclarla con un build nativo no verificado.

**5. Revisar construcción, arranque y funcionamiento público.** Se inspeccionarán logs y dependencias instaladas, URL HTTPS, recuperación ante errores, disponibilidad de artefactos y comportamiento de las tres páginas. Se medirán arranque frío, carga/evicción, RSS conjunto y transitorio, latencia completa y coste de explicaciones con la configuración efectiva. Una carga inválida no debe detener el servicio, revelar una traza interna o producir un pronóstico sustituido por cero.

El ejemplo de clasificación con Softmax de la guía ilustra un patrón de callback, pero no es la salida de este problema. Aquí el modelo produce una predicción continua de σ₇ con h=1, la escala se invierte exactamente una vez y el intervalo, cuando se habilite y verifique, usará el calibrador del mismo predictor. No se convertirán salidas de regresión en probabilidades de clase.

**6. Ejecutar y conservar la prueba de humo.** El guion posterior al despliegue deberá recorrer la URL pública y cada modelo que tenga inferencia en vivo habilitada. No basta comprobar una respuesta HTTP de la portada. La lista de modelos elegibles se obtendrá del registro real; los precalculados y el candidato sin pesos BTC tendrán su estado comprobado, sin generar predicciones ficticias.

| Comprobación del servicio | Evidencia esperada | Estado |
|---|---|---|
| Contexto, EDA y Modelos | URL/versión, navegación y callbacks de cada página. | Pendiente: aplicación sin construir. |
| Filtros y datos insuficientes | Todas las figuras e insights actualizados, n efectivo y mensajes claros. | Pendiente. |
| Cada modelo en vivo | Un ejemplo permitido, contrato temporal, salida finita y procedencia del artefacto. | Pendiente: perfil ONNX/ORT sin verificar. |
| Preprocesamiento y escala | Coincidencia con la referencia validada en tolerancia declarada. | Pendiente de integración completa. |
| Intervalo y explicación | Mismo predictor/caso, unidades, nominal y límites de cobertura/fidelidad. | Pendiente. |
| Entradas inválidas y OOD | Rechazo o aviso correcto, sin caída ni exposición de rutas/secretos. | Pendiente. |
| Recursos y tiempos | Arranque frío/caliente, carga bajo demanda y latencias frente a umbrales prefijados. | Pendiente: sin mediciones Render. |
| Modalidades precalculadas | Catálogo completo, fuentes y justificación de restricciones. | Pendiente de interfaz. |

El informe de humo conservará fecha, commit, entorno, URLs visitadas, identificadores de casos permitidos, modelo/versión, estados HTTP, resultados, tiempos y errores, con extractos de logs identificados por versión e instante. Esos extractos excluirán secretos, rutas internas e inputs privados. Sus umbrales de aceptación se fijarán antes de medir. Actualmente no hay una URL Render, captura de funcionamiento o resultado de humo que pueda declararse aprobado.

### Recursos del plan y contingencia

Antes de desplegar se verificarán los límites vigentes del plan en la documentación oficial de Render. Se contemplarán suspensión por inactividad, arranque frío, memoria/CPU, almacenamiento y persistencia. Los RSS de procesos CPU de exportación y las latencias forward de las tres referencias no acreditan capacidad del servicio completo ni su comportamiento concurrente.

Si se requiere reducir tamaño, se evaluarán compresión o cuantización con paridad, pérdida de métrica frente al margen heredado y equivalencia cuando proceda. Cargar resultados precalculados o elegir un plan con más recursos exige justificación y contabilidad; este documento no selecciona ni activa un plan de pago. El entrenamiento seguirá fuera del servidor.

Para la sustentación se comprobará la URL con antelación y se prepararán capturas y un vídeo breve del dashboard realmente funcionando, con versión y fecha. Ese material está pendiente de producir y no se sustituye por maquetas presentadas como funcionamiento real. Ante una interrupción se explicará el estado y se usará el respaldo como contingencia; la URL pública funcional y su prueba de humo siguen siendo entregables propios.
