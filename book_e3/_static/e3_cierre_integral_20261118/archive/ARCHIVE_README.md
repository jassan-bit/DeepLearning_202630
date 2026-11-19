# E3: instantánea de código y auditoría `1bae73b`

Autor indicado en `CITATION.cff`: Jassan Arteta. Versión del paquete: `0.1.0-DRAFT`. La preparación y revisión usan asistencia de IA declarada.

Esta instantánea seleccionada conserva fuentes del commit `1bae73b5373c11466b226fb3efab706237b5e68b`. No incluye el repositorio completo, datos de mercado, pesos, trayectorias, CSV de referencia o una campaña confirmatoria nueva. La licencia propia y la publicación del depósito siguen pendientes.

## Inspección del candidato

Los módulos `src/original_model.py`, `src/e3_development.py` y `src/e3_innovation_model.py` conservan la identidad del candidato aditivo de 1210 parámetros. Los diseños históricos de atención y momentos se distinguen de esa referencia. Los parámetros y fórmulas pueden inspeccionarse; sin pesos o datos, el archivo no restaura un predictor entrenado.

## Reproducción de los once agregados históricos

1. Recuperar las once tablas CSV de referencia mediante la herramienta del paquete. Verifica tamaño/SHA-256 y rechaza archivos existentes distintos:

```bash
python archive_tools/restore_references.py --dry-run
python archive_tools/restore_references.py
```

2. Aplicar el procedimiento de `experiments/e3_remediacion_20261118/reproduction/README.md`. Sus once entradas de pronósticos se descargan por separado mediante `reproduce.py fetch`, con commits, tamaños y hashes fijados.

3. Construir `deployment/science/Dockerfile` y ejecutar el contenedor sin red, con entradas de sólo lectura y una salida nueva. Los generadores comparan once CSV, sus campos/unidades y las tolerancias conservadas. No entrenan modelos ni recalibran.

La adquisición separada es para una ejecución académica que respete los derechos de las fuentes; no incorpora esos archivos a este ZIP ni concede permiso para redistribuirlos. La reproducción automática previa no se presenta como un acta humana independiente.

## Integridad y DOI

`SOURCE_MANIFEST.json` identifica cada entrada y su procedencia. El archivo externo `package_manifest.json` identifica el ZIP. El DOI se deja sin valor: sólo un registro realmente publicado en Zenodo permite citar el DOI como depósito persistente. Una reserva de borrador se registra separadamente.
