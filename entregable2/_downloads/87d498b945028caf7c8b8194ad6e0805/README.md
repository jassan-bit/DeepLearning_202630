# Ejecutor común: evidencia sintética y disponibilidad separada

Este corte no ajusta modelos con mercado ni ejecuta HPO/NAS. El catálogo sigue siendo once candidatos, comparados mecánicamente con siete referencias separadas. Las métricas sintéticas no son resultados predictivos ni un ranking científico.

`synthetic_receipt.json.gz` y `events.jsonl.gz` contienen el recibo y diario completos, comprimidos con gzip determinista. No se eliminan fases ni eventos. Hay 18 sistemas, 36 etapas de ajuste mínimo y 91 días con pronósticos persistidos antes de la observación. La fase interna tiene 83 fechas; la evaluación del refit, ocho. Una receta única por sistema no acredita selección entre alternativas.

Reproducción: instalar las versiones directas observadas de `requirements-e2-common-synthetic.lock.txt` en entorno nuevo. Es una lista de dependencias directas, no un lock transitivo ni una imagen. En particular torch-einops-utils=0.2.5 en esta ejecución; no se altera el registro histórico que indica otra versión. Ejecutar `python scripts/smoke_e2_common_executor.py --source-root <fuentes_verificadas> --output <directorio_nuevo_synthetic>`. La API no acepta datos de mercado. Fuentes externas: subconjunto de 70 archivos verificados, sin pesos de autor.

Pruebas completas: `E2_AUTHOR_SOURCE_ROOT=<fuentes_verificadas> python -m pytest -q tests/test_e2_common_executor.py`. Sin fuentes, las omisiones de integración no cuentan como aprobaciones. La auditoría usa `python tests/test_e2_common_availability.py`; su script sólo admite el snapshot de desarrollo registrado y exporta conteos, fechas, causas y hashes.

`availability.json` es una auditoría separada: cero modelos, escaladores, pronósticos o métricas. Desarrollo común: 666; CV: 305; piloto E1: 60; piloto propuesto: 53, intactos. Las 608 fechas finales no se verificaron: no se abrió payload ni ZIP protegido. La reserva descriptiva no certifica elegibilidad de ventanas.

Integridad pública: `python scripts/publish_e2_common_executor.py --check`. Verifica hashes, cadena del diario, orden predicción→observación, correspondencia del recibo y espejo. La lectura de bytes no prueba por sí sola fsync; el escritor y las pruebas lo comprueban. No hay protección contra un adversario que pueda modificar todos los archivos.

Límites: fixture completo y finito; aún no es ejecutor general de huecos/calidad de mercado. El control ARIMA maneja faltantes, pero eso no certifica todo el flujo. MASE usa adyacencias del entrenamiento contiguo artificial. Se debe implementar el manejo causal de faltantes y denominadores antes de mercados. NAS, HPO completo, recursos, acceso final e inferencia siguen pendientes.
