Esta reparación se ejecuta en la misma sesión Colab S2, después de que el autor
ya haya aceptado desarrollo y se hayan admitido F00/F01. No vuelve a preguntar
por identidad, consentimiento, GitHub o claves. Conserva los pins S2 y ejecuta
el corrector en un proceso CPU separado. Sólo puede completar F02/F03 y verificar
las cuatro admisiones; no entrena ni abre una prueba.

La celda publicada se obtiene de `resume_cell.TEMPLATE.py` fijando el commit del
addon y SHA256 de `ops/REPAIR_MANIFEST.json`. Primero verifica ese manifest y
todos sus archivos. El worker comprueba los 304 archivos S2 y su ancla; instala
únicamente el parser fijado y usa los colectores, comprobadores y preparadores
originales. No escribe en SOURCE. La fuente S2 tiene commit
`db74afd0763596f71bcbafb4b9150a44a2e83339` y manifest
`2c7dc052d2a53f212688340b50b65bed5fa367ff16fb5ed0a2474d417696715b`.

El codec físico conserva la semántica v2.1 y columna 7 de volumen base COIN-M.
El encabezado genérico permitido se adapta sólo en memoria; ZIP, checksums,
manifests, formularios y admisiones previas se preservan. Esto es ejecución S2
**con un overlay explícito**, y no una ejecución prístina de S2. Los archivos
existentes se verifican por SHA y mtime; los intentos de adquisición F02/F03
sólo pueden agregar contenido a su registro. Cada intento guarda recibos nuevos
en `private/operational-repairs/coinm-header-operational-resume-v1` con fuente,
hashes del addon, evidencia existente, eventos de encabezado normalizado y los
hashes de las admisiones/recovery. En interrupciones conserva datos y repite la
misma celda con los mismos pins. Si aparecen salidas parciales de admisión, se
detiene para conservarlas; no las borra ni sobrescribe.

Tras terminar, ejecuta la etapa 3 original. Esta verifica las particiones NPZ;
el worker ya habrá verificado, con el overlay, también raw, evidencia y recovery.
La futura reproducción desde el ZIP COIN-M requiere conservar el addon, sus
pins y los recibos operacionales junto con los bundles S2. El `recovery_metadata`
original no incorpora retrospectivamente el corrector; no se debe describir
como suficiente para recuperar este caso sin el addon. Esta entrega no activa
ni implementa un nuevo procedimiento automático de restauración remota.

Pruebas: `PYTHONDONTWRITEBYTECODE=1 <python-CPU> -B -m pytest -q ops/test_resume_preflight.py ops/test_resume_end_to_end.py`.
Usan ZIP y declaraciones generados exclusivamente para probar software,
red simulada y ninguna cotización de mercado. No constituyen admisiones
experimentales reales, consentimiento humano ni acreditación de R3.

Asistencia de IA: implementación y revisión de software. Las declaraciones y
licencias del autor no se generan ni firman durante esta reparación. Aplican
los términos de código del repositorio; el addon no concede derechos de datos.
