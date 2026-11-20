# Cierre del archivo: suplemento documental limitado, no depósito integral

Este paquete actualiza la preparación de Zenodo a partir del corte público
`7db8040619f602d7c52ae0c8bf54f7b6f5b5110e`. Fue preparado con asistencia de IA.
El nombre de cita observado es **Jassan Arteta**; no se inventan coautorías,
afiliaciones, ORCID, firmas ni licencias. La fecha de publicación se completará
cuando exista una publicación efectiva.

El DOI `10.5281/zenodo.24356217` estaba reservado en el borrador mostrado por
el autor. Este paquete no afirma que esté publicado. La comprobación pública
actual y su hora se registran por separado en el nuevo recibo de auditoría del
libro; un enlace privado `/uploads/24356217` no acredita un registro público.

## Qué incluye el candidato

- Documentación del alcance, borrador de cita y metadatos, formulario sin firmar
  de derechos y autorización y matriz de derechos por componente.
- Copias exactas de recibos y matrices ya publicados en el corte indicado,
  con su procedencia fijada por commit y SHA-256 en `source_inventory.json`.
- Localizadores y huellas de los nueve ONNX históricos publicados y del ONNX
  experimental V3. Se usan los binarios aceptados; no se reexporta ninguno.
- Manifiesto de todos los miembros y un verificador de depósito de sólo lectura.

**No incluye** datos OHLCV, checkpoints, predicciones masivas, ruedas Python,
imágenes Docker, cuerpos de artículos ni los diez ONNX. Los enlaces externos
no significan que esos archivos estén depositados en Zenodo. El índice de 193
deltas sigue siendo un catálogo: no equivale a hidratar los 8.582 caminos
lógicos ni los 4.268.147.881 bytes declarados del maestro aditivo. El paquete
documental congelado anterior de 375 archivos conserva sus propios límites y
no se reescribe.

Los 796 trabajos pertenecen a la aditiva; V3 tiene 30 ajustes de desarrollo.
`atom_crps` fue 2,7043119888 % peor que `atom_mse_only` y falló la heurística
de continuación original. Las 51 trayectorias/20.400 filas posteriores son
transferencia retrospectiva sobre calendarios expuestos, no confirmación.
Este archivo no acredita R3, reproducción integral de entrenamiento, revisión
humana independiente ni aceptación pública de Render.

## Crear y comprobar el candidato sin publicar

Desde la raíz del repositorio, con Python 3.12 o posterior:

```bash
python experiments/e3_archive_followup_20261120/build_candidate.py \
  --output-dir /tmp/e3-zenodo-candidate
python experiments/e3_archive_followup_20261120/verify_candidate.py \
  --archive /tmp/e3-zenodo-candidate/e3-documentary-followup-20261120-DRAFT.zip \
  --deposit-manifest /tmp/e3-zenodo-candidate/deposit_manifest.json
```

El constructor usa los archivos incluidos en este directorio. No descarga datos,
no entrena ni publica, y produce siempre los mismos bytes para una misma lista
de archivos. El ZIP incorpora `MANIFEST.json`, que enumera cada miembro excepto
el propio manifiesto. `deposit_manifest.json` conserva tamaño y SHA-256 del ZIP
completo. Mantener ese manifiesto externo permite comprobar los bytes que se
suban; no debe sustituirse después por el hash de un ZIP diferente.

## Intervención concreta del propietario

1. Revisar el candidato y completar `AUTHOR_RIGHTS_AND_AUTHORIZATION.md`:
   autorías reales, autoridad para redistribuir las copias documentales,
   licencia para cada clase y autorización específica para Zenodo.
2. Conservar el alcance **suplemento documental limitado** en título,
   descripción y versión. Si se añaden pesos o datos, revisar primero sus
   derechos y regenerar los manifiestos. Una licencia del código propio no
   relicencia los datos, dependencias, artículos ni un mecanismo transferido.
3. Tras resolver esos campos, adaptar el borrador de metadatos al formulario
   vigente de Zenodo, subir el ZIP exacto y su manifiesto externo y revisar la
   vista previa. La licencia que muestra la interfaz por defecto no es una
   confirmación de derechos.
4. Publicar sólo después de esa revisión del propietario. Registrar el número
   de versión efectivo y el DOI realmente asignado/resuelto; el reservado
   `24356217` debe corresponder al registro finalmente publicado.
5. Ejecutar la comprobación siguiente. El verificador **no publica** ni usa un
   token; sólo consulta y descarga archivos del registro público.

```bash
python experiments/e3_archive_followup_20261120/verify_zenodo_deposit.py \
  --deposit-manifest /tmp/e3-zenodo-candidate/deposit_manifest.json \
  --output /tmp/e3-zenodo-candidate/public_deposit_receipt.json
```

El éxito exige metadatos públicos del registro esperado, resolución del DOI al
registro, el ZIP esperado por nombre y tamaño y coincidencia SHA-256 tras su
descarga completa. También comprueba el checksum MD5 publicado por Zenodo
cuando se suministra. No interpreta la existencia pública como validación
científica ni como comprobación jurídica de los derechos declarados.

## Archivos preparados y pruebas ejecutadas

`archive_preparation_receipt.json` contiene las comprobaciones locales realmente
ejecutadas, los hashes y las limitaciones. Los campos de autorización permanecen
sin completar. Ningún archivo de este paquete es un acta humana ni una
autorización para publicar en Zenodo.
