# Declaración de datos del dashboard E3

**Actualización documental: 19 de noviembre de 2026, America/Bogota.** Esta declaración describe
los datos derivados y pronósticos históricos del prototipo académico. No
otorga una licencia a los datos de un tercero ni certifica un cierre integral
de la auditoría.

## Fuente, tarea y unidades

La procedencia declarada es **Binance Spot BTCUSDT**, barras OHLCV diarias en
UTC. La tarea es pronosticar a un día sigma7: desviación estándar poblacional
de siete retornos logarítmicos diarios de Close, sin anualización. El objetivo
del día t sólo se conoce al cerrar ese día; la entrada usa siete barras hasta
t−1. No son datos personales ni registros de cuentas: el insumo documentado
son agregados de mercado y ventanas derivadas.

El bundle del dashboard deriva del archivo auditado:

`book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz`

Commit fuente: `af3f1223ad5899ed63016d2af2d09bae8470ab00`.

SHA256: `186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683`.

El archivo contiene **1316 ejemplos de ventana–objetivo D7×5**. No son 1316
barras originales independientes ni se reconstruye una serie única pegando
ventanas solapadas.

| Cohorte heredada E2 | Ejemplos | Fechas objetivo |
|---|---:|---|
| Entrenamiento | 655 | 2023-01-08 a 2024-10-31 |
| Calibración | 53 | 2024-11-08 a 2024-12-30 |
| Evaluación histórica | 608 | 2025-01-01 a 2026-08-31 |

Los huecos de las particiones se conservan y no se imputan silenciosamente.
La partición E3 de ajuste/validación es distinta: 532 ejemplos de ajuste, siete
purgados y 116 de validación de desarrollo; su receta de entrenamiento no usó
el bloque de calibración de E2. El corte posterior de servicio del 18 de
noviembre adopta para el representante E3 un cuantil residual ya calculado
con esas 53 fechas, separado del ajuste y la selección de época. Las fechas de
evaluación ya tienen exposición histórica y no se presentan
como confirmación ciega nueva.

## Datos que muestra la aplicación

El explorador distingue **fecha objetivo t** de **fecha de la última barra
conocida t−1**. Las variables `last_open`, `last_high`, `last_low`, `last_close`
y `last_volume` proceden de esa última barra de la ventana. `known_return6`
es el último de los seis retornos conocidos y `known_floor` usa exclusivamente
esa historia. Estas covariables no incorporan el OHLCV del día objetivo.

Los filtros seleccionan ejemplos disponibles y el mismo subconjunto debe
alimentar gráficos, interpretaciones y conteos. Un subconjunto con huecos no
se interpreta como una serie diaria continua para pruebas temporales. La
tabla científica global conserva su cohorte histórica y no cambia porque se
filtre una vista EDA.

El catálogo representa **17 sistemas**. Sus métricas medias proceden de
**143 predictores conservados**: 13 redes E2 con diez semillas, tres
comparadores deterministas con una trayectoria y diez réplicas E3. Las curvas
del predictor representativo usan la semilla **271801** para las redes,
elegida por identificador, y la única trayectoria de cada clásico. La media
de RMSE de diez ejecuciones no se confunde con la RMSE de esa curva.

Los intervalos históricos E2 se reconstruyen de sus cuantiles residuales
guardados, sin recalibración. En el corte original de E3 sus campos de
cobertura/anchura estaban ausentes. El servicio actualizado incorpora un
perfil posterior exclusivo de la semilla **271801**, con
`q = 0.005402463770651868` sigma7 y la regla
`[max(0, predicción−q), predicción+q]`. La calibración comprende las 53 fechas
objetivo del 8 de noviembre al 30 de diciembre de 2024, posteriores a su ajuste
y validación. Para objetivos anteriores o iguales al 30 de diciembre de 2024,
el runtime no aplica ese intervalo. No se modificaron los datos históricos,
pronósticos puntuales, pesos o cuantil.

La cobertura nominal es 90 %. La cobertura histórica **individual** del
representante E3 es 94,078947 % en las 608 fechas y 85,555556 % en 2026Q1;
la anchura media es 0,01080287256 sigma7. Son cantidades diferentes de la media
de cobertura entre diez semillas y del IC entre semillas. La comprobación de
661 ventanas ORT conserva la cobertura de evaluación; una diferencia numérica
en el residuo situado en el cuantil cambia la inclusión de una observación de
calibración (48/53 en ORT frente a 49/53 en el cálculo original), sin retocar q.
El nominal no garantiza cobertura individual o por grupo bajo dependencia
temporal o cambio de distribución.

La inferencia actual admite cinco representantes ONNX CPU: MLP16, LSTM32,
GRU32, E3 y TSMixer. Este último conserva la calibración E2 de su semilla
271801, sin recalibración. Las otras nueve redes siguen pendientes de
exportación y paridad. Persistence y SeasonalNaive requieren más historia de
cierres que las siete barras admitidas, y ARIMA requiere su estado causal
fechado; se muestran como históricos hasta definir y verificar ese contrato
adicional. Esta distinción no elimina sus resultados del catálogo de 17
sistemas ni demuestra una imposibilidad de ejecutarlos en CPU.

## Calidad, sesgos y disponibilidad

La auditoría posterior conserva **12 controles PASS, 3 PARTIAL y 1 NA**.
Verificar hashes, fechas y fórmulas no cierra todos los controles de fuga,
portabilidad o cambio de distribución. Las ventanas, objetivos solapados y
semillas de una misma cronología tienen dependencia; no se cuentan como
mercados independientes.

La cobertura corresponde a un activo, una fuente, días UTC y los periodos
indicados. No representa automáticamente otros mercados, fuentes, regímenes
o frecuencias. La volatilidad realizada es un proxy retrospectivo; sus errores
de medición o límites económicos no se eliminan al reproducir su aritmética.
La [auditoría exploratoria posterior](https://jassan-bit.github.io/DeepLearning_202630/entregable3/sections/08_verificacion_estadistica.html#e3-final-extended-audit)
ya evaluó 19 sistemas en D00 BTC, D01 WIKI/AAPL, D02 ETHUSDT Spot y D03
BTCUSD_PERP, con 796 trabajos completos. Esa auditoría conserva sus propias
cohortes, pesos y calibradores: no amplía automáticamente el dominio de estos
artefactos históricos ni convierte cuatro archivos en cuatro mercados
independientes. La potencia prospectiva y la validación confirmatoria de R3
siguen abiertas.

La documentación disponible no acredita una **licencia de redistribución
final** para los datos de Binance. Sus condiciones y las de cada fuente deben
verificarse antes de una distribución o uso nuevo. No se etiqueta el dataset
como CC0, abierto sin restricciones o perteneciente al autor. Los derechos
del código y los de los datos son asuntos distintos.

## Entradas del usuario y privacidad

El formulario sólo espera las siete barras necesarias con las columnas
documentadas, sin datos personales. El diseño de servicio procesa la carga
en memoria para validar y predecir; no debe guardar su contenido en archivos,
bases de datos o logs de aplicación ni usarlo para entrenar o recalibrar.
Los registros técnicos de acceso del servidor pueden identificar rutas y
respuestas; no deben incluir el contenido de la carga. No se incorpora una
recogida de datos para analítica o uso secundario en este prototipo.

La validación de formato, tamaño, fechas y coherencia física es distinta del
control de extrapolación. Cumplir el esquema no acredita pertenencia al dominio
de entrenamiento, y un aviso OOD no es una garantía de error futuro.

## Procedencia verificable

- [Cohorte y auditoría fuente](https://github.com/jassan-bit/DeepLearning_202630/tree/af3f1223ad5899ed63016d2af2d09bae8470ab00/book_e2/_static/e2_scientific_closure_20261116/data_audit).
- [Predictores E2 ampliados sin retunear recetas](https://github.com/jassan-bit/DeepLearning_202630/tree/5e9023ea821bfa9fb0b92340d518413f6ab89161/runs/e2_extension_10_semillas_v1).
- [Diez réplicas E3](https://github.com/jassan-bit/DeepLearning_202630/tree/510c3bddfdc9926322e3210ac61ee1f5eeffad2e/runs/estabilidad_aditiva_v1).
- [Definiciones de las métricas descriptivas](https://github.com/jassan-bit/DeepLearning_202630/tree/b3b7f5210e8a81fe3e0f35493221bcb0f6f8c4e4/experiments/e3_revision_bloque_d_20261117).
- [Correcciones y límites de la auditoría](https://jassan-bit.github.io/DeepLearning_202630/entregable3/sections/04_correcciones_e2.html).
- [Perfil residual posterior E3](artifacts/e3_serving_calibration_v2/e3_additive_271801.json) y [verificación de su runtime](experiments/e3_cierre_integral_20261118/serving/calibration_runtime_verification.json).
- [Procedencia de la extensión TSMixer](artifacts/e3_serving_extensions_v2/tsmixer/provenance.json) y [contratos de los sistemas pendientes](experiments/e3_cierre_integral_20261118/serving/export_capabilities/capability_matrix.csv).

El manifiesto del bundle debe vincular cada archivo servido con su SHA256,
fuente y versión. La asistencia de IA en la preparación y revisión está
declarada; la persona autora conserva la responsabilidad de comprobar
procedencia, derechos, interpretación y uso.

Las comprobaciones locales no equivalen a aceptación de la URL pública en
Render ni a reproducción humana independiente de la entrega completa. El
candidato V3 de 1262 parámetros completó un piloto de 30 ajustes con datos de
desarrollo BTC previamente expuestos: 532 ventanas de ajuste, 54 de selección,
siete purgadas y 55 de evaluación del piloto. No abrió los miembros de prueba
ni calibración del archivo fuente y no usó D01–D03. La variante completa no
superó su heurística frente a los controles. Su [recibo verificable](https://github.com/jassan-bit/DeepLearning_202630/blob/8eaa581bc34b615a8507c9e384a3e4be29084720/runs/e3_v3_development_pilot/20261120T031424176339Z/payload/pilot_receipt.json)
conserva ese alcance; no utiliza las métricas o intervalos del servicio como
evidencia propia ni incorpora V3 al catálogo.
