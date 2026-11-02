# 2. Análisis Exploratorio de Datos (EDA) Mejorado

El EDA de este proyecto debe ser reproducible y orientar decisiones de modelado, preprocesamiento y validación. Cada resultado deberá acompañarse de una interpretación técnica y una decisión justificable.

```{important}
Esta sección establece el protocolo de análisis. La fuente será Binance Spot directamente, según §2.3. La adquisición verificada y los resultados del EDA siguen pendientes. La fuente, el horizonte y la frecuencia deben concretarse según la sección 1 antes de construir las etiquetas.
```

## 2.1. EDA general

### 2.1.1. Análisis de la variable objetivo

El objetivo propuesto es continuo: la dispersión de retornos futuros definida en §1.3. No corresponde calcular desbalance de clases ni entropía de clases, porque no se ha definido una clasificación.

Se producirán histogramas y funciones de distribución empírica del objetivo, con conteos, proporción de ceros, cuantiles, media, mediana, desviación estándar, asimetría y curtosis. Se especificará la convención de curtosis y se contrastarán escalas lineal y logarítmica cuando sea apropiado. Las colas se examinarán sin eliminar automáticamente episodios de volatilidad elevada.

**Decisión que informa:** comprobar si MSE concentra el ajuste en episodios extremos y contrastar, únicamente en desarrollo, pérdidas robustas o una transformación del objetivo. Una transformación logarítmica requiere resolver explícitamente los ceros y estimar sus parámetros solo con entrenamiento; las métricas finales se calculan en la escala original. No se convertirán intervalos de alta volatilidad en clases para justificar resampling.

### 2.1.2. Calidad del dataset

| Comprobación | Evidencia que debe generarse | Implicación técnica |
|---|---|---|
| Faltantes | Conteos por columna y periodo; minutos ausentes respecto a la cuadrícula temporal; duración y agrupación de huecos | Determinar qué ventanas pueden construirse sin confundir retornos de diferente duración |
| Mecanismo de ausencia | Relación de los huecos con horario, actividad y variables observables; documentación de incidencias del proveedor | Discutir compatibilidad con MCAR, MAR o MNAR como hipótesis; los conteos por sí solos no identifican el mecanismo |
| Corrupción e inconsistencias | Errores de lectura, timestamps inválidos, cierres no positivos y, si hay OHLC, comprobaciones de orden y coherencia | Separar errores verificables de eventos de mercado; registrar cada corrección |
| Duplicados | Filas idénticas y timestamps repetidos con precios iguales o contradictorios | Resolver según procedencia antes de crear ventanas; no eliminar precios repetidos legítimos en instantes diferentes |
| Casi duplicados y fuga | Intervalos y ventanas compartidos entre particiones; coincidencias de archivos por hash | Evitar que etiquetas futuras entren en entrenamiento; conservar el uso legítimo de historia ya observable |
| Ruido de etiqueta | Inspección de intervalos seleccionados de desarrollo, trazabilidad del cálculo y sensibilidad a errores de precios | Evaluar la fiabilidad del proxy; no afirmar acuerdo entre anotadores porque el etiquetado es automático |

No se imputarán precios ni se eliminarán extremos por defecto. Cada política deberá justificarse y ajustarse dentro del entrenamiento de cada fold. Los intervalos cuya etiqueta no pueda calcularse con la frecuencia aprobada se identificarán antes del modelado.

### 2.1.3. Sesgos y desplazamientos de distribución

El alcance de un solo activo, un par y un proveedor limita la representación del mercado. Se documentarán cobertura temporal, moneda de cotización, exchange y periodos de baja actividad o cambios de mercado; todavía no se conocen esos atributos del dataset.

Se examinarán cambios de distribución de retornos y volatilidad por periodos de desarrollo, su autocorrelación y posibles cambios de régimen. Los horarios o índices temporales pueden actuar como proxies de régimen: su asociación con el objetivo no demuestra causalidad ni estabilidad predictiva.

Para comparar folds de entrenamiento y validación se mostrarán distribuciones, diferencias de cuantiles y tamaños de efecto. Una prueba de dos muestras o un clasificador adversarial deberá respetar la dependencia temporal mediante bloques y evaluación cronológica; no se aplicarán valores p basados en independencia de filas solapadas. La longitud de bloque y sus análisis de sensibilidad deben fijarse usando desarrollo.

**Decisión que informa:** elegir ventanas de entrenamiento, escalamiento y validación que representen el uso previsto, sin mezclar aleatoriamente periodos. El desplazamiento observado se reporta y no se elimina artificialmente haciendo coincidir distribuciones.

### 2.1.4. Partición del dataset

Se conserva el calendario de §1.6: historia inicial en 2023, validación forward chaining en 2024, ajuste final con datos elegibles de 2023–2024 y test desde el 1 de enero de 2025 hasta el 31 de agosto de 2026. Se informarán 2025 y 2026 por separado.

Las proporciones y cantidades reales se calcularán después de adquirir y auditar los datos; no se impondrá un reparto aleatorio 80/20. La partición cronológica es determinista y no necesita semilla de barajado. Las operaciones estocásticas que se incorporen sí deberán registrar sus semillas.

Todo el intervalo futuro de una etiqueta debe pertenecer a su etapa. En cada origen de validación solo se entrenará con etiquetas ya conocidas. El contexto histórico anterior a validación o test puede usarse si estaba disponible en el momento del pronóstico. Escalamiento, imputación y selección de hiperparámetros se ajustan sin consultar el test.

Las distribuciones de entrenamiento y validación se contrastarán durante desarrollo. La inspección del test y su comparación con desarrollo se reservan para la evaluación final, una vez congelado el protocolo; no se usarán para cambiar modelos o preprocesamiento. Así se compatibiliza el diagnóstico entre particiones con la prohibición de consultar el test durante el ajuste.

### 2.1.5. Congelación del conjunto de prueba

Se conservarán archivos brutos inmutables con SHA-256, fuente, fecha de adquisición, periodo y versión. El manifiesto de test incluirá identificadores temporales, límites, esquema, reglas de elegibilidad, parámetros de construcción del objetivo y hashes de los archivos derivados. Un hash permite detectar cambios, pero no demuestra por sí solo ausencia de consulta ni calidad de los datos.

El pipeline de desarrollo no cargará los archivos de test. La evaluación definitiva verificará los hashes antes de leerlos y registrará la versión de código y del protocolo utilizado. Cualquier cambio posterior deberá declararse como una nueva versión del experimento.

## 2.2. Evidencias necesarias para completar la sección

| Producto reproducible | Estado |
|---|---|
| Fuente, términos de uso, archivo y manifiesto de adquisición | Binance Spot BTCUSDT identificado; descarga, condiciones y hashes pendientes |
| Frecuencia, horizonte, ventana de entrada y cadencia definidos | Pendiente |
| Código o notebook de auditoría y EDA de desarrollo | Pendiente de adquisición directa |
| Tablas, figuras e interpretación vinculadas a decisiones | Pendientes de ejecución sobre brutos verificados |
| Recuentos y límites exactos de particiones y folds | Pendiente de datos |
| Manifiesto y hashes del test congelado | Pendiente de adquisición |

Esta sección cubre los requisitos generales mostrados en las imágenes. No sustituye los requisitos específicos para series de tiempo de la guía completa ni presenta el EDA como terminado.


## 2.3. Adquisición directa desde Binance Spot

### 2.3.1. Fuente y alcance

La fuente del proyecto es **Binance Spot**, par **BTCUSDT**, mediante los [archivos públicos oficiales](https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1m/). Se utilizarán velas de un minuto y el precio de cierre; USDT es la moneda de cotización. La documentación técnica está en [Binance Public Data](https://github.com/binance/binance-public-data).

El patrón de adquisición es:

```text
https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-AAAA-MM.zip
https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-AAAA-MM.zip.CHECKSUM
```

Se descargarán los archivos y sus checksums directamente del proveedor, se verificará SHA-256 antes de procesarlos y se registrarán URL, fecha de adquisición, tamaño, hash, esquema y unidades temporales. No se asumirá que todos los archivos usan la misma unidad de timestamp: se verificará la documentación y cada periodo, incluida la transición a microsegundos.

El calendario objetivo sigue siendo desarrollo 2023–2024 y test 2025–agosto de 2026. La cobertura real se comprobará con los archivos adquiridos. El test se conservará por separado y no se explorará para elegir preprocesamiento o hiperparámetros.

### 2.3.2. Estado de adquisición y resultados

**La adquisición directa aún no se ha completado.** El intento de acceder al checksum oficial fue bloqueado por el proxy del entorno con HTTP 403. Hace falta permitir `data.binance.vision` en la configuración de red del entorno. Este bloqueo no demuestra que Binance haya rechazado el acceso.

Las estadísticas y la figura calculadas anteriormente a partir de un archivo intermediario se retiraron de este avance. No se presentan como resultados obtenidos directamente del proveedor. El EDA se ejecutará de nuevo sobre los archivos oficiales verificados, sin inventar resultados ni reutilizar etiquetas de otro estudio.

### 2.3.3. Condiciones para ejecutar el EDA

1. Habilitar acceso al dominio oficial y comprobar una descarga con verificación TLS.
2. Documentar las condiciones de uso y redistribución aplicables; acceso público no equivale a licencia abierta de datos de mercado.
3. Descargar y verificar los archivos de desarrollo, conservar los brutos y generar el manifiesto.
4. Auditar timestamps, unidades, duplicados, huecos, cierres y coherencia OHLC antes de crear retornos.
5. Fijar horizonte, frecuencia de los retornos objetivo, longitud de entrada y cadencia conforme al Entregable 1.
6. Ejecutar el EDA general y el específico de series de tiempo; vincular cada resultado a una decisión de modelado o validación.
