# Entregable 3: suplemento documental y herramientas de verificación

**Borrador para revisión del propietario. No publicado en Zenodo. Derechos, licencias y autorización de publicación pendientes.** DOI reservado: `10.5281/zenodo.24356217`. Versión propuesta: `0.2.0`.

El corte fuente es `60a9d988af9608b1037c10754e68cac3287277a9` de `jassan-bit/DeepLearning_202630`. El servicio Render corresponde a otro commit, `feb83dbc08197bde7076857bd8ff475de197efd2`. Las pruebas públicas del 20 de noviembre de 2026 aprobaron el cliente original de cinco modelos y el perfil de catorce: Actions `44801411372`, job `136514212866`. V3 está deshabilitada. El libro y sus cuatro entradas públicas fueron verificados por HTTP.

## Qué contiene

- Copias seleccionadas de capítulos actuales, herramientas de verificación y reproducción automática B, recibos, matrices y tres capturas públicas representativas.
- Procedencia por archivo en `SOURCE_INVENTORY.json`, con commit, URL permanente, tamaño y SHA256.
- Tres capturas adicionales, modelos, datos y otros materiales disponibles por los localizadores del repositorio original. Un enlace no equivale a un archivo depositado.
- Cierre científico NO-GO de V3, resultado adverso conservado y distinción entre reproducción A/tablas, B/predicciones y C/entrenamiento.

Los 796 trabajos de la aditiva y los treinta ajustes V3 se conservan en sus fuentes originales: **no están íntegramente incluidos en este ZIP**. El suplemento no contiene los 4,27 GB del maestro, pesos/checkpoints/ONNX, datasets completos, matrices de ventanas, ruedas, imágenes Docker o cuerpos de artículos. Los recibos y las capturas sí muestran el breve ejemplo OHLCV de siete barras usado por la aplicación pública; conservar su atribución no concede derechos sobre fuentes de datos ajenas.

El código seleccionado y los capítulos son instantáneas de sus rutas originales bajo `repository/`. No constituyen todo el repositorio, una aplicación autónoma o un libro completo que pueda construirse desde este ZIP. Algunas herramientas requieren entradas externas y el checkout indicado en sus instrucciones originales. No se promete una reproducción integral de entrenamiento ni una constancia humana independiente.

## Estado de derechos y licencias

`RIGHTS.DRAFT.csv` distingue los componentes. La propuesta es **MIT para el código propio** y **CC BY 4.0 para la documentación, recibos y capturas propios**. Ninguna está adoptada en este borrador. El propietario debe confirmar su autoridad, los autores/coautores reales, las licencias y el archivo exacto que autoriza publicar. No se inventan firmas, ORCID, afiliación ni permisos sobre contenidos externos. Las licencias de terceros y los ejemplos de datos conservan sus condiciones y atribuciones.

Los formularios históricos copiados mantienen su fecha y estado originales; no se convierten en autorizaciones posteriores. La futura decisión efectiva se registrará por separado y con fecha real. Publicar el suplemento permitirá acreditar un DOI para este subconjunto; no acreditará el archivo científico integral ni R3.

## Verificación

```bash
python verify_archive.py e3-entregable3-documentary-60a9d988-v0.2.0-DRAFT.zip
```

También puede ejecutarse con `--directory` sobre los archivos extraídos. Comprueba rutas, CRC, tamaños, SHA256, los recibos de aceptación pública y los límites científicos. No entrena, infiere, calibra, descarga datos, publica o modifica GitHub/Zenodo.

## Autoría y asistencia

El autor observado en la cita del proyecto es **Jassan Arteta**, pendiente de confirmación del propietario para este depósito y de cualquier coautoría real adicional. Se declara asistencia de IA en código, documentación, investigación y auditoría automática. No equivale a firma humana independiente o aceptación institucional.

Libro: https://jassan-bit.github.io/DeepLearning_202630/entregable3/intro.html

Fuente inmutable: https://github.com/jassan-bit/DeepLearning_202630/tree/60a9d988af9608b1037c10754e68cac3287277a9

Aplicación: https://deeplearning-e3-dashboard.onrender.com/perfil-14
