# E3 · hipótesis, adecuación y preparación cripto, versión de datos 2

Extensión explícita del motor histórico A
`c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c`. El modelo
`input-only-overlap-mse-1`, el banco de 4096 semillas, los 21 sistemas, recetas,
entrenador y reglas estadísticas conservan sus bytes. Se versionan los
adaptadores de datos, las propuestas de fuentes y la operación Colab; no se
cambia una regla de decisión después de observar resultados.

La versión distribuida **no contiene admisiones reales nuevas, predicciones
nuevas, dimensionamiento de mercado ni una apertura de prueba reservada**.
Las pruebas nuevas usan archivos generados y red simulada. Los archivos bajo
`qa/` y `DELIVERY_STATUS.json` copiados de A mantienen su identidad histórica;
no son recibos de esta extensión. `HISTORICAL_SOURCE_A_MANIFEST.json` y
`SOURCE_A_PRESERVATION.json` permiten verificar esa separación. La revisión
metodológica condicional del motor A conserva su alcance de DGP y ajustes, sin
convertirse en potencia o superioridad de mercado.

## Fuentes propuestas y siguiente paso

El problema principal sigue siendo Bitcoin Spot. `configs/data_panels.json`
fija tres propuestas adicionales de contratos perpetuos frente al mercado
Spot, con tamaños de desarrollo distintos y elegidos antes de adquirir sus
precios para esta extensión. La distinción económica de producto/colateral y
el tamaño son los dos ejes justificados; no se declara que un ticker sea por sí
solo un dominio nuevo ni que los conjuntos sean económicamente independientes.
La guía no impone Nasdaq, acciones o una razón de tamaño de 1,5×.

El cuaderno nuevo usa ocho etapas y fuentes/hash fijados antes de ejecutarse.
La instalación y la ejecución de todas las banderas por defecto no descargan
observaciones ni activan entrenamientos o prueba. **La primera acción después
de instalar es activar sólo `PREPARE_CRYPTO_DEVELOPMENT=True` en la etapa 2 y
ejecutar esa celda.** Pide nombre, contacto, función y aceptación real de uso,
licencia y exposición. No requiere editar cuatro JSON, montar Drive, usar
Nasdaq ni solicitar una GPU específica. Binance Vision ofrece los archivos
públicos; el acceso efectivo y sus bytes se verifican en la ejecución del autor.

La aceptación declara todo el histórico de esos paneles expuesto/desarrollo y
reconoce la exposición anterior D00–D03. Vincula los bytes de la evidencia
primaria de licencia y el propio formulario con SHA-256. No autentica identidad,
no crea una firma y no concede permiso para preparar prueba final. Uso
académico no comercial, atribución, condiciones de licencia de derivados y
ausencia de redistribución raw se presentan antes de aceptar. Las condiciones
de las fuentes prevalecen sobre la licencia del código de herramientas.

## Admisión y reanudación

Spot y USD-M usan volumen en activo base de la columna 5; COIN-M conserva el
precio USD y usa volumen base de columna 7, no el número de contratos. Los
proveedores son identidades distintas y verifican URLs, epoch/unidades,
apertura/cierre UTC, checksums, duplicados, calendario, formas y soporte causal
de quince cierres. La etiqueta continúa siendo σ7 poblacional del siguiente
día y la entrada OHLCV conserva siete barras.

El colector conserva una identidad pendiente de configuración/derechos/
exposición antes del primer archivo. Reanuda sólo los mismos bytes con sus
checksums y no sobreescribe payloads o evidencia existentes. Los intentos y la
finalización tienen UTC auténticos separados. No convierte una descarga
incompleta en una admisión. El helper valida estado, manifiesto, cuatro NPZ de
desarrollo y bundle de recuperación antes de omitir datos completos.

La adquisición prepara únicamente ajuste, selección, planificación y
calibración. Una declaración de desarrollo no produce test ni una reserva
independiente. Para una prueba futura, cada soporte completo debe satisfacer
la regla temporal y autorización auténtica del protocolo cerrado. Los datos
históricos expuestos no se convierten en confirmación al volver a descargarlos.

## Planificación y coste

La etapa 4 permanece desactivada. Su activación requiere cuatro admisiones
verificadas y escribir `PLANIFICAR 340`. Son 340 trabajos: 320 ajustes
neuronales (16×5×4), cuatro ARIMA y 16 controles analíticos. El presupuesto
neuronal nominal es 320×120 s = **10 h 40 min**, excluyendo ARIMA, instalación,
adquisición, E/S y GitHub; no es una medición ni un límite garantizado del
histórico bajo interrupciones. La cola de cinco semillas de planificación es
distinta del prefijo reservado de al menos diez semillas finales.

GitHub se verifica antes del primer paso y de continuar cada checkpoint.
Cada job completo se verifica antes del siguiente y hay checkpoints
intermedios aproximadamente cada minuto. Un fallo de respaldo detiene nuevos
ajustes. Al reanudar se conservan pesos, Adam, RNG, tiempos inciertos y trabajos
terminados; no se modifican calendarios o semillas para completar un resultado.
Los CSV completos y checkpoints quedan cifrados; el índice público contiene
metadatos/recibos acotados. Los raw no se suben.

Las credenciales se introducen sólo al activar una operación remota: secretos
Colab `E3_R3_GITHUB_TOKEN` y `E3_R3_BACKUP_SECRET` (clave independiente de al
menos 32 caracteres, que el autor debe conservar fuera de Colab). El cliente
las entrega por stdin y las redacta antes de mostrar/guardar mensajes. Nunca
se incluyen en el código, argumentos o manifiestos.

El dimensionador consume las predicciones y recibos completos de planificación.
La ruta activa `configs/power_design_crypto_metadata.json` cambia únicamente
las etiquetas de las propuestas y metadatos de versión; no altera ningún número,
semilla, método, margen, presupuesto o regla. El original
`configs/power_design.json` conserva sus bytes A, incluidas etiquetas históricas
que no se usan como fuentes activas. La auditoría verifica que ambos diseños
son científicamente idénticos al retirar esos campos de metadatos.
No dimensiona con fixtures, el piloto de 55 fechas o una familia incompleta.
Con entradas válidas estudia potencia condicional empírica para MSE; no convierte
sus p-valores en inferencia para la media histórica de RMSE individuales. Un
estado `NOT_DIMENSIONED` conserva R/T/B nulos. El auditor ejecutable de entradas
se entrega en `package/audit_power_inputs.py` y reutiliza los códigos y ledgers
metodológicos, sin nuevas simulaciones automáticas.

**La confirmación no está habilitada por esta versión.** La auditoría identifica
dos requisitos científicos anteriores a una ejecución final:
`PRIMARY_ENDPOINT_ALIGNMENT_NOT_ESTABLISHED` y
`JOINT_DECISION_VALIDATION_NOT_ESTABLISHED`. No se cambia el estimando, α o la
regla conjunta para evitarlos. Las celdas y la CLI bloquean congelar un protocolo,
preparar/entrenar/sellar un caso final y abrir prueba antes de credenciales,
red o etiquetas. Editar una casilla booleana no habilita esta versión: hace
falta otra versión justificada y validada antes de resultados finales. Las APIs
internas históricas conservan sus fixtures explícitos de software para probar
la mecánica; no son un atajo CLI para una confirmación de mercado.

Asistencia de IA declarada: implementación, pruebas y revisión de la herramienta.
El autor conserva responsabilidad por los datos auténticos, la comprensión y
las decisiones de ejecución.
