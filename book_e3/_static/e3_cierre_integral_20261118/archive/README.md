# Depósito E3: paquete concreto y metadatos de borrador

Se preparó una instantánea pequeña de código y auditoría del commit `1bae73b5373c11466b226fb3efab706237b5e68b`. El autor procede de `CITATION.cff`: **Jassan Arteta**. No se infieren afiliación u ORCID de la cuenta mostrada en Zenodo.

**Estado: paquete local de borrador; DOI no reservado y depósito no publicado por este ejecutor.** Este paquete no es el archivo completo del repositorio ni la campaña confirmatoria pendiente. Se conserva como versión `0.1.0-DRAFT`, para revisar exactamente qué se pretende depositar.

## Archivos preparados

- `e3-source-audit-1bae73b-v0.1.0-DRAFT.zip`: código del candidato histórico, generadores del reanálisis, locks, manifiestos y dictamen de IA, sin datos de mercado o pesos.
- `package_manifest.json`: SHA-256, tamaño y blob Git de las 34 fuentes fijadas, además de las instrucciones del paquete.
- `zenodo_metadata.DRAFT.json`: campos para el depósito, con autor verificado y licencia/DOI pendientes.
- `ZENODO_FORM.md`: texto para completar el formulario en la interfaz.
- `rights_matrix.csv` y `LICENSE_STATUS.md`: alcance de derechos y exclusiones concretas.
- `build_package.py`: constructor determinista. `--check` verifica el ZIP existente sin consultar la red ni alterar las fuentes.

Los campos de una licencia nueva se dejan vacíos porque el repositorio de partida no contiene un archivo `LICENSE` que autorice asignar MIT, CC0 u otra licencia propia. La decisión pendiente debe tomarla el autor. No se extiende una licencia de código a datos o modelos de terceros.

## Alcance reproducible

La instantánea permite inspeccionar el candidato de 1210 parámetros y reconstruir los once agregados mediante el contenedor científico, tras adquirir por separado las referencias y las once entradas fijadas. Los CSV derivados se excluyen del ZIP y se conservan como localizadores, hashes y tamaños. El paquete contiene una herramienta para recuperar esas referencias en una ejecución académica autorizada; no concede nuevos derechos de redistribución.

La reconstrucción automática previa de once CSV ya se documentó en el proyecto. Esta preparación no ejecuta una reproducción humana independiente, entrenamientos, inferencias o un experimento confirmatorio. Tampoco acredita novedad, R2 o R3. La excepción docente para el revisor de R1 se mantiene en su propio alcance.

## Paso inmediato en Zenodo

Con la sesión ya iniciada, pulsar **+ → New upload** y guardar un **borrador**. En Digital Object Identifier, elegir **No** y **Get a DOI now!** permite reservarlo. El DOI sólo se registra al publicar; borrar el borrador pierde la reserva. Se puede compartir con el proyecto la URL del borrador o el DOI reservado, sin contraseñas o tokens.

La cuenta GitHub se conecta desde el menú de perfil → **Linked accounts → Connect** junto a GitHub. La vinculación de cuenta es distinta de activar el archivo automático de un repositorio por releases. El paquete preparado aquí es una carga manual seleccionada; no afirma que toda la historia, datos y pesos del repositorio tengan una licencia única de redistribución.

Fuentes oficiales consultadas el 18 de noviembre de 2026:

- https://help.zenodo.org/docs/deposit/create-new-upload/
- https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/
- https://help.zenodo.org/docs/profile/linking-accounts/

## Antes de la publicación del depósito

La decisión concreta que falta es la licencia del código y los textos propios del paquete. La elección puede registrarse después de revisar este ZIP y su manifiesto. El depósito debe usar la versión realmente seleccionada; si la nueva campaña cambia el contenido, se prepara una versión nueva y se conserva esta instantánea. No se inventa un DOI ni se presenta una reserva como publicación.

## Verificar el paquete

```bash
python experiments/e3_cierre_integral_20261118/archive/build_package.py --check
```

Para reconstruir el mismo ZIP a partir de fuentes, usar un checkout separado del commit fijado y conservar esta carpeta de herramientas. El constructor rechaza fuentes distintas y no reemplaza un archivo de la misma versión con bytes diferentes.

La ficha del modelo incluida conserva sus cifras descriptivas publicadas. La exclusión de tablas se refiere a los once CSV de referencia y las trayectorias: no se oculta que la documentación describe resultados agregados. Los derechos de los textos propios también requieren la decisión de licencia del autor.
