# Ejecución del procedimiento E2

Esta enmienda completa el código del procedimiento. Los resultados P0 ya
publicados pertenecen a otro experimento. La presencia de este documento, de un
adaptador o de un candidato no acredita que el benchmark se haya ejecutado.
El test E1 es conocido y las conclusiones E2 siguen siendo exploratorias.

## Preparación reproducible

Usar un checkout limpio del commit de registro y Python 3.12. El lock define
el mismo entorno CPU para las trece redes ajustables y las referencias E1.
Una migración de hardware exige registrar la modificación antes de ajustar;
no mezclar tiempos obtenidos en máquinas distintas en la comparación de coste.

```bash
python3.12 -m venv ../e2-runtime
../e2-runtime/bin/python -m pip install -r requirements-e2-completion.lock.txt
../e2-runtime/bin/python -m pip check
../e2-runtime/bin/python scripts/download_e2_completion_sources.py \
  --output-dir ../e2-private/author_sources
../e2-runtime/bin/python scripts/run_e2_final_benchmark.py --phase plan
```

El descargador verifica 112 archivos de once árboles de autores, incluidas
dependencias transitivas, presets y licencias. No descarga pesos, artículos ni
datos de mercado. Los dos CSV diarios de desarrollo del repositorio deben
coincidir con `design.json`. Los ZIP originales de Binance se conservan fuera
del repositorio; `data/README.md` describe su adquisición y verificación.

## Orden obligatorio

En los comandos siguientes, sustituir `../binance-spot-data` por el directorio
privado que contiene `BTCUSDT/1m/monthly` y los archivos diarios declarados en
el manifiesto. El directorio de salida debe ser nuevo y externo al checkout.

```bash
../e2-runtime/bin/python scripts/run_e2_final_benchmark.py \
  --phase all --workers 4 \
  --source-roots-json ../e2-private/author_sources/source_roots.json \
  --raw-root ../binance-spot-data --output-dir ../e2-private/run
```

El flujo ejecuta preflight sintético, registro, HPO anidado, auditoría OOF,
P1, potencia, ajustes finales, calibración A53, medición de coste, ablaciones
de desarrollo, una fase final de score y análisis. El registro exige un
checkout limpio y fija hashes del código, configuraciones, fuentes de autores
y desarrollo. La reserva inicial comprueba metadatos y tamaños de los ZIP;
abre precios futuros únicamente tras el cierre de todos los predictores y
calibradores exigidos. Los controles con etiquetas permutadas se congelan
antes del score y nunca seleccionan candidatos.

Cada fase puede invocarse por separado con los mismos argumentos. Para
continuar un proceso interrumpido, conservar el mismo checkout, entorno,
fuentes y directorio privado; el ejecutor acepta únicamente artefactos cuya
cadena de hashes y recibos esté cerrada. Un ajuste interrumpido sin recibo de
cierre requiere revisar su incidencia; no se convierte en un candidato válido
ni se sustituye silenciosamente por otro experimento.

El inventario de `--phase plan` es un máximo programado. El tiempo asignado
de 720 horas CPU a los quince procedimientos ajustables no es tiempo observado.
Se conservan los tamaños publicados, doce candidatos por procedimiento y el
límite común. El código no reduce modelos, semillas o candidatos para obtener
una tabla rápida. La potencia puede quedar por debajo de 0,80 incluso en el
límite de 200 semillas; en ese caso debe informarse insuficiencia.

## Interpretación y publicación

P1 y calibración A usan 53 fechas, del 8 de noviembre al 30 de diciembre de
2024, después de selección hasta el 31 de octubre y embargo de siete días.
La rama A usa cinco miembros fijos antes de P1; su predictor de calibración
es el mismo que se evalúa en 2026. No sustituye al predictor primario.
Las tres recetas E1 conservan su exposición histórica al desarrollo al
interpretar OOF; las otras diecisiete requieren selección mensual estricta.

Publicar únicamente agregados revisados, registros sanitizados, configuraciones
y código propio. Mantener privados los pesos, precios originales, predicciones
individuales, muestras de errores y textos completos. Los resultados no cierran
por sí mismos la revisión sistemática: siguen pendientes la cobertura Scopus
restante, búsquedas nativas y elegibilidad de cuerpos aplicados documentadas
en el capítulo de literatura.
