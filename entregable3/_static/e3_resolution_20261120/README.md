# Resolución verificable del Entregable 3

Preparado y revisado con asistencia de IA. Fuente inicial del encargo:
`8d03a73d529eb3f9b2e89992b4488cfeb88a99e6`. Los recibos conservan las fechas,
fuentes y diferencias entre preparación, ejecución automática y acción humana.

## Qué quedó ejecutado

- Auditoría de los 486 escenarios existentes y 71 insumos distintos; familia de
  19 rivales conciliada y 16 referencias de varianza ausentes identificadas.
  No se calculó potencia científica nueva ni se ejecutó una batería de mercado.
- Integridad de 9.227 rutas históricas —8.841 payloads y 386 metadatos— y 9.217
  objetos SHA únicos. Los 4.268.147.881 bytes del maestro aditivo se recuperaron
  y verificaron en una instantánea temporal; no se depositaron como master.
- Archivo seleccionado **0.3.1-REVIEW**: 72.823.602 bytes, 18.659 miembros,
  300.901.185 bytes extraídos y comprobados; catorce ONNX históricos y V3
  experimental, sin cambios en pesos/calibradores ni habilitación de V3.
- [Release de revisión](https://github.com/jassan-bit/DeepLearning_202630/releases/tag/e3-source-evidence-8d03a73-v0.3.1-review)
  con cuatro assets descargados por HTTP público y SHA256 verificados. El tar
  tiene SHA256 `2c9b5d0ee865683ee1555ee7f745ade79aaf99446a93c3958e3943935b0d49eb`.
- Recomprobación del DOI documental existente y sus bytes publicados: no se
  modificó el ZIP 0.2.0 ni se publicó otra versión de Zenodo.

El primer intento de consolidación rechazó tres recibos de validación del kit
C. El fallo permanece documentado; la corrección sólo admite esos tres paths
con tamaño y SHA fijados y se probó contra los 24 miembros reales del kit.
La ejecución posterior reutilizó el archivo anterior, sin recuperar otra vez
el master. Los tests con fixtures comprueban software, no resultados de modelos.

## Decisión científica

**NO-GO de confirmación para la receta V3 actual; R3 no acreditada.** Se
conservan 796 trabajos aditivos, 30 ajustes V3 y 51 trayectorias/20.400 filas de
transferencia retrospectiva. `atom_crps` mantiene 2,7043119888 % más RMSE que
`atom_mse_only` y no superó el mejor control en los tres dominios de transferencia.
Esto no prueba inferioridad universal; tampoco justifica confirmar la receta
actual. No se cambió la regla ni se promovió automáticamente una ablación.

El [dictamen](science/SCIENCE_RESOLUTION.md), la [revisión de métodos](science/METHODS_REVIEW.md)
y el [diseño futuro DRAFT](science/prospective_design.DRAFT.json) precisan las
puertas pendientes: fundamento científico, panel principal y tres adicionales
admitidos sin exposición, márgenes/alternativas, covarianza entre semillas y
calendario, tres procedimientos validados y dimensionamiento final. Los cuatro
paneles futuros son clases por admitir, no datasets nuevos ya identificados.

`science/power_contract.py` prepara y valida entradas y resume **futuras**
simulaciones aportadas por quien planifique: Holm sobre 57 componentes,
intersección de tres métodos por rival, potencia conjunta y límites Monte Carlo.
No implementa una DGP ni la batería científica. Se requieren al menos cinco
semillas de planificación y diez finales, o más según el dimensionamiento;
potencia por contraste de al menos 0,90 y reporte de la conjunta. No se creó
preregistro ni Colab confirmatorio porque las puertas siguen bloqueadas.

## Material ejecutable para la persona externa

- [A/B: instrucciones y comparador por campo](archive/human/README.md).
  Se reutilizan los recibos automáticos existentes; la persona debe ejecutar
  sus propios comandos y emitir su concepto con logs y discrepancias reales.
- [C: controlador de reproducción histórica parcial](reproduction_C_preparation/README.md).
  Dos futuros ajustes fijados, una semilla por receta y 110 comparaciones
  posteriores. La plantilla está bloqueada por defecto; requiere identidad,
  relación, alcance, tolerancias y recursos declarados antes de ejecutarlos.
  El kit incluye 935 referencias históricas, no predicciones nuevas. No
  reproduce los 796 trabajos ni los 30 ajustes completos.
- [Formulario sin firma](archive/human/INDEPENDENT_REPRODUCTION_FORM.md).
  No se observó ejecución humana ni se inventó un acta. La excepción de IA de
  R1 no se extiende automáticamente a la reproducción humana o R2.

## Material concreto para el titular de Zenodo

[Acciones y campos preparados](ZENODO_OWNER_HANDOFF.md): usar **Edit** para
corregir metadatos del suplemento conservando DOI/ZIP. Para el archivo ampliado,
revisar [derechos y exclusiones](RIGHTS_REVIEW.md) antes de **New version**. La
licencia de los miembros propios seleccionados del suplemento no se extiende
al repositorio, datos o dependencias completos. La release REVIEW no es otro
depósito de Zenodo y no se inventa un DOI nuevo.

## Integridad y estados

`PACKET_MANIFEST.json` enumera cada payload de este paquete por bytes/SHA256.
Desde su directorio, ejecutar:

```bash
python3 bundle_tools/verify_packet.py --root .
```

El verificador sólo lee archivos: no descarga datos ni ejecuta scripts
archivados, modelos, inferencias o remuestreos. La matriz de diecisiete
requisitos está en `requirements_matrix_current.json` y su CSV; cada fila
declara su base de evidencias. Los JSON históricos no se reescriben para
actualizar estados. La comprobación posterior del libro se conserva en el
[acta de publicación](https://github.com/jassan-bit/DeepLearning_202630/tree/e3-resolution-book-publication-8d03a73d/proof)
con su commit efectivo, CI y respuesta pública.

Siguen pendientes: decisiones científicas que justifiquen reabrir desarrollo,
admisión y planificación prospectiva suficiente; ejecución y concepto de una
persona externa con alcance acordado; C completa si se exige todo el histórico;
revisión de derechos ampliados y acciones del titular en Zenodo. La publicación
del producto, el archivo y un DOI no sustituyen esas evidencias.
