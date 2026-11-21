# Reproducción externa del libro E3 y su tabla V3

Este suplemento CPU añade un flujo material al kit A/B/C anterior. Su alcance
exacto está en GUIDE_SCOPE.md. No ejecuta modelos, entrenamiento, inferencia,
descargas de mercado, remuestreos ni una campaña nueva. Una ejecución automática
de software no se presenta como una persona independiente.

## 1. Preparar fuentes completas y el caso

Se necesita Docker y Git. Obtenga este kit desde el commit/manifiesto publicado
por el autor y compruebe su MANIFEST.json contra ese identificador externo. El
manifiesto interno sólo verifica integridad, no la identidad de su emisor.

```bash
git clone --filter=blob:none --no-checkout https://github.com/jassan-bit/DeepLearning_202630.git e3-source
git -C e3-source fetch --depth=1 origin f5c960eefedb5fc49b313dd795b1ec84e784bd78
git -C e3-source checkout --detach f5c960eefedb5fc49b313dd795b1ec84e784bd78
python3 human_reproduction/prepare_external.py --inspect-kit
python3 human_reproduction/prepare_external.py --source-root e3-source --case-root e3-human-case-001
```

El preparador sólo crea un caso nuevo: verifica todos los blobs del árbol Git,
registra SHA256/tamaño por archivo y copia únicamente los archivos versionados.
Rechaza fuentes modificadas/incompletas, enlaces simbólicos, casos existentes y
falta de disco. No copia HTML/cachés no versionados. El original no se modifica.
Presupuesto: hasta3GB de fuentes más1GB libre para salidas; construcción de imagen
requiere espacio adicional. La edición se fija antes del resultado y no se
actualiza a main automáticamente. Los hooks de Sphinx reconstruyen archivos
documentales congelados dentro de la copia propia; por eso el libro necesita un
checkout completo y no solamente la cápsula filtrada0.3.1.

Archivo previo que se reutiliza, sin reexportar ningún grafo:
https://github.com/jassan-bit/DeepLearning_202630/releases/download/e3-source-evidence-8d03a73-v0.3.1-review/e3-source-evidence-8d03a73-v0.3.1-REVIEW.tar.gz
72823602bytes,SHA256 `2c9b5d0ee865683ee1555ee7f745ade79aaf99446a93c3958e3943935b0d49eb`.
Si ya está descargado, añada `--review-archive /ruta/archivo.tar.gz` al preparador;
se verifica sin extraer grafos ni cargar pesos. No es obligatorio descargarlo
otra vez para este alcance del libro y predicciones. C opcional sigue en su kit
congelado, `tools/reproduction_C_preparation`, dentro de esa cápsula.

## 2. Declarar alcance y tolerancias antes de ejecutar

El tercero completa `e3-human-case-001/HUMAN_REQUEST.json`: identidad propia,
relación con el código, independencia real, alcance acordado, justificación,
fecha UTC previa y tolerancia absoluta/relativa por cada campo. Los valores
vacíos o nulos bloquean la ejecución humana. No copie la identidad del autor ni
considere SOFTWARE_PROFILE.json una aprobación humana. Los campos de soporte,
nombre, tipo, número de trayectorias, fechas y ranking se comparan exactamente.
Cada campo flotante usa `|observado−referencia|≤absoluta+relativa×|referencia|`.
La DE usa ddof1; en controles analíticos es «no aplica», nunca cero imputado.

## 3. Construir el contenedor y ejecutar sin red

```bash
docker build --pull -f human_reproduction/Dockerfile -t e3-book-reproduction:f5c human_reproduction
docker image inspect e3-book-reproduction:f5c > e3-human-case-001/IMAGE_INSPECT.json
docker run --rm --read-only --network none --cap-drop=ALL --security-opt no-new-privileges --tmpfs /tmp:rw,nosuid,size=512m --memory=2g --cpus=2 --pids-limit=256 --user "$(id -u):$(id -g)" --mount "type=bind,src=$(pwd)/e3-human-case-001,dst=/case" e3-book-reproduction:f5c --mode human-external
```

La base Python3.12.14-slim se fija por digest; todos los paquetes del libro por
versión en el lock exacto de f5c. La instalación se hace sin resolver dependencias
adicionales y `pip check` debe pasar. El informe `/opt/e3/pip-install-report.json`
en la imagen registra URLs y hashes de wheels realmente usados. Conserve imagen
por digest/ID e informe pip para repetir los mismos binarios; el lock original
fija versiones, no promete por sí solo un wheel idéntico entre plataformas.
La red de preparación sólo sirve fuentes/paquetes; la ejecución es offline.
El kernel, la selección del piloto y los artefactos originales permanecen intactos.

Se reconstruyen las7 filas principales desde935 puntos existentes; se comprueban
contra referencias completas y valores visibles a9decimales. Las columnas
puntuales regeneradas reemplazan el CSV descargable en la copia del libro; CRPS_U
se conserva explícitamente como fuente no recalculada. Después se construye E3
con `--all --warningiserror --keep-going` y notebooks desactivados. Se comprueban
enlaces locales, anclas y que el HTML sirva exactamente el CSV regenerado.
No se omiten recetas que perdieron; atom_crps conserva2,7043%peorRMSE y NO-GO.

Una falla conserva logs y recibos; cree un caso nuevo al corregirla. No se
relajan tolerancias después del resultado. PREPARATION.json no es un recibo de
ejecución. EXECUTION_RECEIPT.json tampoco autentica la identidad de una persona.

## 4. Reporte real y anexo

El tercero revisa los CSV, el libro y recibos en output, completa
HUMAN_REPORT.template.md y adjunta su constancia propia. Debe informar la máxima
discrepancia observada por campo, soporte, unidad y limitaciones. La firma se
mantiene vacía hasta esa intervención real. No se amplía la excepción de IA de
R1 a este requisito ni a la sustentación R2. Otros componentes de R3, la DOI y
la reproducción de todas las tablas no se acreditan por este procedimiento.

## Validación de software por el equipo

Para una prueba automática use un caso NUEVO y reemplace sólo el último argumento
por `--mode software-validation`. Ese modo usa SOFTWARE_PROFILE.json, declarado
antes, con tolerancias1e-15absoluta y1e-12relativa, sin nombre ni firma inventados.
El recibo queda `human_execution_claimed=false` incluso si todo pasa. No ejecute
el modo humano del operador para obtener un recibo de CI. Las validaciones
stdlib están en test_reproduction.py; ningún fixture representa mercado nuevo.
