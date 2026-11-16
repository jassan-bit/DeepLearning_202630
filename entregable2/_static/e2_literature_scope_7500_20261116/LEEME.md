# Cierre operativo de Scopus sobre los 7.500 registros originales

Por instrucción expresa del usuario —«pues cierra eso con los 7500 registros»— se fija como corte de Scopus inicial el archivo original `export_dcb22fe7-ce2a-4457-ab38-2a1021da5766_2026-11-06T011951.464806988.csv`, SHA-256 `387f673a2146902ce12888ab3d662c095abc33d2cf5b54a168e9e2a12099f7de`. La decisión es posterior al análisis y no se presenta como preregistro original.

**Estado: cerrado operativamente dentro del alcance congelado; cero tareas activas de recuperación de exportaciones Scopus para esta versión.** Los 7.500 registros son el componente inicial Scopus, no el número total de estudios o grupos. Las otras rondas ya admitidas de WoS, Scopus, IEEE y arXiv conservan exactamente su función. El flujo multibase mantiene **9.559 grupos, 42 estudios aplicados y 174 informes no recuperados**, sin nuevo cribado, adjudicación, deduplicación o cálculo del PRISMA.

## Alcance y límites conservados

- La diferencia histórica entre 7.874 resultados mostrados y 7.500 recibidos sigue siendo 374. No se afirma haber recuperado esos identificadores ni se los convierte en exclusiones científicas. Su recuperación no condiciona el cierre de este corte.
- El total actual de 7.936 mostrado en la conversación no equivale a un conjunto recibido íntegramente. El nuevo CSV de 7.500 comparte 7.436 EID con el original; tiene 64 diferentes y omite 64 originales. El archivo posterior de 436 repite EID ya presentes en ese nuevo CSV. Son actualizaciones **no incorporadas** a esta versión. No se afirma haber cribado nuevamente esas actualizaciones ni haber sumado sus EID al flujo.
- La cadena nativa exacta, todos los filtros, la fecha de ejecución y los rangos de la consulta Scopus original no están íntegramente acreditados. Las cadenas WoS Q1/Q2 y las consultas arXiv conservadas no sustituyen esa cadena Scopus. Los límites históricos de correspondencia y cobertura WoS se mantienen.
- Este cierre no certifica exhaustividad de las bases, cobertura total de consultas, elegibilidad conocida de los 174 informes inaccesibles, mérito global óptimo del Top10 ni cumplimiento metodológico integral de la guía. Tampoco cambia A2, modelos, inferencia, datos de mercado o la decisión de NAS no aplicable.

Los formularios, colas y estados anteriores de recuperación se conservan como **archivos históricos**. No deben interpretarse como una solicitud vigente de más exportaciones Scopus para esta versión. Una ampliación futura requeriría otra versión con su recepción, identidad, cribado y flujo propios.

## Archivos y verificación independiente

`inventory_7500.csv` contiene sólo índice original, EID, título, DOI, año y fuente. No redistribuye el CSV nativo, resúmenes, referencias completas o PDF. Los campos y el orden proceden exclusivamente del archivo original admitido por hash.

Descargar **`cierre_scopus_7500.zip`**, extraerlo y ejecutar:

```bash
python run.py
```

Sólo requiere Python estándar. No necesita repositorio, Git, red, cuentas, Torch ni datos de mercado. Verifica los hashes del paquete, los 7.500 EID únicos, el orden y los conteos del flujo histórico copiado; escribe `verification_result.json`. No vuelve a ejecutar el cribado ni el PRISMA.

El archivo privado original no se incluye. El modo predeterminado verifica el inventario derivado y el flujo incluido; no afirma recalcular el hash de un archivo original ausente. Quien disponga de ese original puede comprobar además sus bytes y su proyección exacta:

```bash
python run.py --original-export /ruta/al/export_dcb22_original.csv
```

`verification_report.json` acredita la admisión local del original por SHA y la conservación de los archivos previos antes/después de esta operación de metadatos. `standalone_qa.json` acredita la ejecución del ZIP extraído fuera del repositorio. Los manifiestos nuevos no reescriben los recibos históricos.
