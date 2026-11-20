# Entregable 3: suplemento documental y herramientas de verificación

Versión **0.2.0**. Autor indicado por el proyecto: **Jassan Arteta**. El propietario confirmó su autorización sobre los archivos propios seleccionados y las licencias: **MIT para código propio; CC BY 4.0 para documentación, recibos y capturas propios**. Véanse `OWNER_DECISION.json`, `RIGHTS.csv` y los dos avisos de licencia. No son licencias alternativas globales para todo el ZIP.

**Estado al preparar el paquete:** listo para depósito, aún no publicado en Zenodo; DOI reservado `10.5281/zenodo.24356217`. Incluir el DOI reservado no declara que ya resuelva públicamente. La publicación y sus bytes se comprueban después de la acción del propietario. Este estado es un corte previo a la publicación, no una afirmación sobre consultas futuras.

## Fuente y resultado conservado

El corte fuente es `60a9d988af9608b1037c10754e68cac3287277a9` de `jassan-bit/DeepLearning_202630`. Render corresponde a otro commit, `feb83dbc08197bde7076857bd8ff475de197efd2`. Las pruebas públicas del 20 de noviembre de 2026 aprobaron el cliente original de cinco modelos y el perfil de catorce: Actions `44801411372`, job `136514212866`. V3 está deshabilitada. Las cuatro entradas públicas del libro fueron verificadas por HTTP.

Se incluyen capítulos seleccionados, herramientas de verificación y reproducción automática B, recibos, matrices y tres capturas públicas representativas. `SOURCE_INVENTORY.json` identifica 53 copias de fuente por ruta, commit, URL, tamaño y SHA256. Los materiales nuevos de empaquetado/licencia se identifican en `MANIFEST.json`.

## Límites del depósito

Los 796 trabajos de la aditiva y los treinta ajustes V3 se conservan en sus fuentes originales: **no están íntegramente incluidos en este ZIP**. El suplemento no contiene el repositorio completo, los 4,27 GB del maestro, pesos/checkpoints/ONNX, datasets completos, matrices de ventanas, ruedas, imágenes Docker o cuerpos de artículos. Un localizador externo no equivale a un archivo depositado.

Las capturas muestran el breve ejemplo OHLCV de siete barras; los recibos registran sus referencias, fecha y hash. Los datos de origen, dependencias y obras citadas conservan condiciones y atribuciones: no se concede aquí propiedad o licencia global sobre contenidos ajenos. La decisión del propietario se refiere a su material propio, con estas excepciones.

Los archivos bajo `repository/` son instantáneas de rutas originales. No constituyen una aplicación autónoma ni un libro completo construible desde el ZIP. Algunas herramientas necesitan entradas externas y el checkout indicado en sus instrucciones. Se distinguen A/tablas, B/predicciones y C/entrenamiento: **no se afirma reproducción integral o humana independiente**. V3 mantiene el resultado adverso, el NO-GO confirmatorio y R3 no acreditada.

Los formularios y recibos históricos conservan sus estados. Una plantilla humana sin completar no se convierte en firma; un corte anterior de preparación no se convierte en autorización posterior. La decisión de licencia efectiva del suplemento está registrada por separado y con fecha real. No es una licencia retroactiva global para todo el repositorio.

## Verificación y reconstrucción

```bash
python verify_archive.py e3-entregable3-documentary-60a9d988-v0.2.0.zip
```

Después de extraer el ZIP:

```bash
python verify_archive.py --directory .
python build_archive.py --output ../e3-entregable3-documentary-60a9d988-v0.2.0.zip
```

Se comprueban rutas, CRC, tamaños, SHA256, licencias, recibos de aceptación y límites científicos. La reconstrucción usa miembros y metadatos fijados. No entrena, infiere, calibra, descarga datos, publica o modifica GitHub/Zenodo.

## Atribución y referencias

Se declara asistencia de IA en código, documentación, investigación y auditoría automática. No equivale a firma humana independiente ni aceptación institucional. No se inventan ORCID, afiliación o coautores.

Libro: https://jassan-bit.github.io/DeepLearning_202630/entregable3/intro.html

Fuente: https://github.com/jassan-bit/DeepLearning_202630/tree/60a9d988af9608b1037c10754e68cac3287277a9

Aplicación: https://deeplearning-e3-dashboard.onrender.com/perfil-14
