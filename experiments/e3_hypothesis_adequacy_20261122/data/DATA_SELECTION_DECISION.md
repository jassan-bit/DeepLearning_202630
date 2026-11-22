# Selección de datos: Bitcoin y tres mercados de contratos perpetuos

Propuesta `crypto-products-calendar-2.1`, fijada por metadatos y calendario antes
de adquirir precios nuevos o ejecutar modelos en esta tarea. **No es una
admisión real, un protocolo confirmatorio ni una declaración de reserva.**
Se conserva la propuesta inicial y su enmienda de F03 por incertidumbre de
listing temprano; no hubo selección por resultados de una descarga nueva.

La guía original, §8.3, p. 23, pide tres adicionales de la misma tarea, cada uno
distinto del principal en al menos dos ejes. No exige acciones, Nasdaq, mercados
independientes ni que cada adicional difiera de los otros adicionales. Sí exige
documentar dependencia, fuentes, licencias, versiones, particiones y desempeño
por conjunto. El dominio no es obligatorio como eje, pero aquí se propone un
cambio económico de producto y un tamaño distinto. **Son argumentos sujetos a
revisión, no dos casillas acreditadas automáticamente.**

| Panel | Producto y fuente explícita | Desarrollo propuesto, UTC | Barras por calendario | Ajuste / selección / planificación / calibración, después de purgar soporte |
|---|---|---|---:|---|
| F00 principal | BTCUSDT Spot, `binance_vision` | 2017-08-17–2023-12-31 | 2328 | 1382 / 219 / 452 / 219 |
| F01 adicional | BTCUSDT perpetuo USD-M, `binance_vision_um` | 2020-01-01–2022-12-31 | 1096 | 643 / 96 / 205 / 96 |
| F02 adicional | BTCUSD_PERP perpetuo COIN-M, `binance_vision_cm` | 2021-01-01–2022-12-31 | 730 | 424 / 58 / 133 / 59 |
| F03 adicional | ETHUSDT perpetuo USD-M, `binance_vision_um` | 2021-01-01–2023-12-31 | 1095 | 643 / 95 / 205 / 96 |

Los conteos provienen exclusivamente del calendario 24/7 y la misma regla
computable 60/10/20/10 del motor; cada partición excluye los primeros 14 targets
para conservar sus 15 cierres de soporte dentro del intervalo asignado. Las
fechas y el código de redondeo están fijados. Los conteos reales requieren
descarga completa y validación; no se imputan días ni se sustituyen archivos
ausentes. Más de 80 fechas de planificación no demuestra que el piloto tenga
potencia suficiente o que permita dimensionar toda la familia.

## Justificación de los dos ejes, y límites

**Dominio de mercado/producto:** una operación spot representa intercambio del
activo, mientras un perpetuo es un contrato derivado con financiación, margen,
liquidaciones y posible base respecto al spot. USD-M y COIN-M tienen colateral y
unidades contractuales diferentes. Se cambia el mecanismo económico del mercado
generador de precios, no sólo el ticker. Los dos productos BTC siguen teniendo
el mismo subyacente; por eso no se afirma independencia económica.

**Tamaño:** cada adicional usa una ventana histórica completa y más corta,
elegida aquí por calendario y régimen de longitud de desarrollo antes de nuevas
observaciones. La razón de barras principal/adicional es aproximadamente
2,1241, 3,1890 y 2,1260. No se exige que el adicional sea 1,5 veces mayor; esa
regla pertenecía a la propuesta anterior de WIKI y nunca fue una cifra de la
guía. Tamaño y producto cambian conjuntamente: estos paneles no aíslan un efecto
causal del tamaño ni sustituyen un análisis de sensibilidad controlado.

**Dependencias:** los cuatro comparten proveedor, calendario cripto y factores
macroeconómicos; dos comparten BTC y los tres adicionales son perpetuos. La
generalización evaluable es estrecha: otros productos cripto, no todos los
mercados financieros. La prueba entre pocos conjuntos tiene potencia limitada;
deben conservarse resultados y efectos por panel, sin mezclar métricas crudas de
escalas distintas ni tratar las series como cuatro réplicas independientes.

**Exposición:** F02 se solapa con D03 ya archivado y sus resultados históricos son
conocidos. F00 y los subyacentes BTC/ETH también tienen antecedentes expuestos.
No se afirma que esta elección precediera a todo resultado histórico. Estas
ventanas son exclusivamente desarrollo y planificación. Una futura confirmación
requiere reserva histórica auténtica o datos cuyo soporte completo ocurra
después del cierre del protocolo y del dimensionamiento; no se fabrican sus
fechas ni un tamaño final. El backend de esta versión sólo habilita desarrollo;
la CLI final está bloqueada por el gate metodológico de la entrega.

## Misma tarea y unidades

Se mantienen las siete barras OHLCV conocidas y la volatilidad poblacional de
siete retornos logarítmicos de cierre hasta el próximo cierre diario UTC. No se
incluyen financiación o PnL de una posición apalancada en el objetivo. Las
ventanas y la métrica primaria de la candidata permanecen intactas.

El volumen de Spot y USD-M se toma de la columna 5, en unidades del activo base.
En COIN-M la columna 5 cuenta contratos: **se usa la columna 7**, volumen del
activo base. No se llama BTC a un número de contratos ni se reutiliza el parser
Spot sobre COIN-M sin declarar esa diferencia. Dentro de cada panel todos los
modelos reciben los mismos arrays, particiones y presupuesto de ajuste.

## Fuentes y derechos

Se reutilizan los bytes primarios ya conservados del repositorio oficial
`binance/binance-public-data`, commit
`d571690eb66760763b0bd6e27e8cd35603c7e244`: README de esquemas y términos de
Binance Vision. El README describe separadamente Spot, USD-M y COIN-M; su SHA256
es `2e133d9945a9263a02781369078e72805eff074680f305cd5460016471c6caad`.
Los términos conservados tienen SHA256
`99c4921dbe023cd68c1a447775707d621453cc78aec6ac0bfe2e6c06a5114ee0`.
Las licencias aplican a datos y código con alcances distintos: el MIT del README
no licencia los datos. Los datos están sujetos a CC BY-NC-SA 4.0 y los términos
de Vision; el actor debe aceptar el uso académico no comercial, atribución y
condiciones de derivados. Esta entrega no redistribuye datos raw.

No se requiere una cuenta de trading ni una clave Nasdaq para este adaptador.
La disponibilidad efectiva de cada archivo oficial **no se ha probado aquí con
datos reales**; la adquisición autorizada debe detenerse ante errores o datos
incompletos, sin escoger una fuente por resultados de pronóstico.

## Ejecución, coste HTTP y recuperación

`binance_archive_plan(config)` produce un itinerario de metadatos puro. Usa ZIP
mensuales de barras 1d sólo cuando el mes entero está dentro del rango autorizado
y diarios en bordes parciales. F00 adquiere agosto de 2017 desde el día 17 con
15 archivos diarios; no descarga ni inventa los días 1–16. El plan total es
**187 ZIP y 187 CHECKSUM: 374 solicitudes sin reintentos**, frente a 10498 en una
estrategia diaria. Son 91/36/24/36 archivos por panel. No se descargaron esos
precios al construir o probar el adaptador.

La CLI conserva `record-owner-evidence`, `collect-development`,
`prepare-development`, `restore-development-metadata` y `reacquire-development`.
La fuente nueva incluye una versión propia de `engine.data`; la fuente A y sus
identidades antiguas permanecen fijadas. El notebook nuevo prepara formularios
auténticos y deja adquisición y entrenamiento desactivados hasta activación.

La adquisición parcial conserva `acquisition_pending.json`, con configuración,
versión y hashes de recibos. `inspect_binance_pending(...)` valida ese vínculo y
los pares existentes sin red ni escrituras. El colector activado reutiliza bytes
idénticos y sólo completa los archivos o compañeros faltantes; ante cambios
detiene y preserva evidencia. El primer intento conserva su UTC explícitamente
como inicio; `completed_utc` y cada ledger de intentos tienen sus fechas reales.
No se fecha una recuperación como si hubiera ocurrido en el evento original.

`validate_binance_manifest(record, manifest_path=None)` verifica instrumento,
producto, URL oficial, nombre, frecuencia, periodo, itinerario y checksums; con
una ruta también verifica los bytes. Se ejecuta antes de preparar etiquetas y
antes de recuperar fuentes. Un manifiesto con SHA propio recalculado no permite
relabelar BTCUSDT como ETHUSDT dejando archivos BTCUSDT. La recuperación vuelve a
producir los mismos NPZ, admisión y procedencia, incluyendo Spot legado A sin
añadir campos que alterarían su identidad.

La recuperación y la preparación final de bajo nivel se probaron sobre números
generados y autorizaciones de fixture explícitas. No representan firma real,
precio observado, reserva o apertura confirmatoria. La CLI final de esta edición
no puede habilitarse cambiando una casilla del JSON: la bloquea el módulo de
gates incluido por el runtime.

Asistencia de IA declarada en propuesta, implementación, pruebas y revisión.
