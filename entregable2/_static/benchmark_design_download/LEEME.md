# Entregable 2, sección 4: propuesta de diseño y auditoría documental

Paquete autocontenido del capítulo 4 de
https://github.com/jassan-bit/DeepLearning_202630.
Extraer conservando todos los directorios. No exige datos de mercado, paquetes de
Deep Learning, acceso de red ni una copia adicional del repositorio para ejecutar
el auditor y sus pruebas; utiliza Python 3.12 y la biblioteca estándar.

**Archivo de la propuesta v1:** este suplemento conserva las quince filas y los bloqueos de la propuesta inicial. El protocolo vigente, con guía completa y veinte configuraciones, está en `experiments/e2_remediation_20261106/` y en el paquete Remediacion_E2_protocolo_codigo_evidencias.zip. Las cifras de este archivo no son el estado global de experimentos posteriores.

## Estado de esta entrega

Estado: proposed_not_frozen. Hay 17 prerrequisitos bloqueantes y cero
ajustes de Deep Learning nuevos. La propuesta no está lista para entrenar ni
acredita un test confirmatorio nuevo. No contiene adaptadores D7 validados,
pipeline de entrenamiento, pesos ni predicciones individuales. El test de E1 ya
era conocido; esta cronología se conserva.

La propuesta distingue diez modelos nuevos de las tres recetas E1 fijas, una
referencia no profunda y una trivial: al menos 15 configuraciones. Ocho candidatos
de hiperparámetros por modelo no equivalen a ocho entrenamientos. La contabilidad
propuesta es de 660 evaluaciones de validación y 1320 ajustes del
optimizador por modelo nuevo, antes del piloto y del ajuste definitivo; para los
diez modelos nuevos son 13200 ajustes propuestos, no ejecutados. El presupuesto
común de cómputo, los adaptadores, el piloto, la potencia y el cierre del protocolo
siguen pendientes. No se atribuye HPO nuevo a las tres anclas E1 sin volver a
ajustarlas y registrar esas variantes por separado.

El calendario anidado contiene 30 filas de fechas propuestas, no índices tsxv ni
ventanas supervisadas auditadas. Se conserva el embargo, la escala de la métrica,
el margen absoluto y las demás condiciones heredadas de E1.

## Ejecutar la auditoría y las pruebas

Desde la carpeta extraída:

    python scripts/audit_e2_benchmark_readiness.py
    python tests/test_e2_benchmark_readiness.py
    python scripts/audit_e2_benchmark_readiness.py --require-ready

La primera orden valida la coherencia documental y escribe seis derivados
JSON/CSV en book_e2/_static/benchmark_design. Su éxito no significa que se pueda
entrenar. La opción --require-ready devuelve código 2 mientras existen bloqueos;
es el resultado esperado para esta propuesta válida y bloqueada. Una incoherencia
del contrato o un hash alterado devuelve código 1. El auditor no importa librerías
de DL, no abre archivos de mercado y no entrena ningún modelo.

Las secciones de la guía pendientes de recibir se identifican en el capítulo y en
la propuesta. Este paquete permite revisar lo ya recibido, sin inventar el
contenido restante ni declarar completada toda la guía de E2.

## Verificar y regenerar el paquete

MANIFIESTO.json contiene tamaño y SHA-256 de cada archivo. Se incluyen seis
fuentes públicas de E1 identificadas por los recibos fijados en proposal.json:
recetas, resumen agregado, dimensionamiento, diseño, manifiesto de datos y
capítulo metodológico. El manifiesto de datos describe archivos; no contiene los
datos de mercado. No se incluyen archivos crudos, pesos ni predicciones individuales.

    python scripts/package_e2_benchmark_design.py --check
    python scripts/package_e2_benchmark_design.py

--check comprueba CRC, inventario, hashes, fechas internas fijas y correspondencia
exacta con las fuentes presentes. Si se ha vuelto a ejecutar el auditor con otra
versión de Python, su observación de entorno puede cambiar; regenerar el paquete
después de esa actualización. El mismo conjunto de archivos genera el mismo ZIP.
SHA256SUMS, publicado junto al ZIP, identifica el contenedor completo.

Las fuentes públicas E1 se conservan como evidencia heredada y no se seleccionan
modelos con sus resultados. Su inclusión no constituye una repetición del
experimento ni transforma el test histórico en un test desconocido.
