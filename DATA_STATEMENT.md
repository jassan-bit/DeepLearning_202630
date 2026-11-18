# Declaración de datos del dashboard E3

**Fecha: 17 de noviembre de 2026, America/Bogota.** Esta declaración describe
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
purgados y 116 de validación de desarrollo; no usó el bloque de calibración de
E2. Las fechas de evaluación ya tienen exposición histórica y no se presentan
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
guardados, sin recalibración. E3 no tiene un intervalo calibrado congelado;
sus campos de cobertura/anchura siguen ausentes. Un nominal de 90 % no
garantiza cobertura individual bajo dependencia temporal o cambio de
distribución. Los IC entre semillas son otra cantidad.

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
Las pruebas de tres conjuntos adicionales siguen pendientes.

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

El manifiesto del bundle debe vincular cada archivo servido con su SHA256,
fuente y versión. La asistencia de IA en la preparación y revisión está
declarada; la persona autora conserva la responsabilidad de comprobar
procedencia, derechos, interpretación y uso.
