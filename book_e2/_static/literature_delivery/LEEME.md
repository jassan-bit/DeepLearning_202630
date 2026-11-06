# Literatura del Entregable 2: código, matrices y diagramas

Suplemento descargable del capítulo 3. Conserva la estructura de directorios del
repositorio https://github.com/jassan-bit/DeepLearning_202630 y permite auditar
la selección bibliográfica, las cifras publicadas y sus localizadores.

Contiene derivados bibliográficos mínimos de dos fuentes recibidas (WoS y
Scopus), la selección propuesta de diez arquitecturas, diez diagramas propios,
las fichas y matrices extraídas de diez PDF completos (215 páginas), bibliografía,
recibos SHA-256 y los scripts de importación, compilación y empaquetado.

Alcance: 8.802 ocurrencias recibidas producen 7.833 grupos bibliográficos después
de deduplicar. El CSV Scopus recibido contiene 7.500 de los 7.874 resultados
mostrados; los 374 no exportados no son exclusiones. La lectura dirigida de los
diez originales no equivale al cribado completo de esos 7.833 grupos. Las cifras
MSE/MAE publicadas mantienen dataset, variante, horizonte, contexto, página y
tabla; no forman un ranking comparable de Bitcoin.

Los diez modelos son adicionales a GRU, LSTM y TCN heredados, una referencia no
profunda y una trivial: el benchmark completo requiere al menos 15
configuraciones. Este paquete no entrena modelos ni contiene nuevos resultados
de Bitcoin, implementaciones D7 ya validadas o reproducciones numéricas.

## Verificar y reconstruir los derivados congelados

Extraer conservando directorios sobre una copia del repositorio. El compilador
de lectura completa y los tests utilizan Python 3.12 y biblioteca estándar:

    python scripts/compile_e2_fulltext_review.py --check
    python tests/test_e2_wos_import.py
    python tests/test_e2_scopus_import.py
    python tests/test_e2_fulltext_review.py

Para regenerar las fichas y los diagramas, crear un entorno con las dependencias
de requirements-eda.lock.txt (incluye matplotlib) y ejecutar:

    python scripts/build_e2_literature_selection.py
    python scripts/package_e2_literature.py
    python scripts/package_e2_literature.py --check

Los importadores de WoS y Scopus están incluidos con sus tests sintéticos y su
interfaz --help. Volver a importar los datos nativos exige los archivos originales
obtenidos mediante acceso autorizado; no se incluyen en este ZIP. Los derivados
versionados permiten auditar la recepción, la vinculación y el recuento sin
redistribuir exportaciones completas de las bases.

MANIFIESTO.json identifica cada archivo incluido por tamaño y SHA-256. El archivo
SHA256SUMS publicado junto al ZIP permite verificar además el contenedor. Las
fechas internas del ZIP son fijas para permitir regeneración determinista.

## Procedencia y derechos

Las fuentes bibliográficas son WoS, Scopus, los originales suministrados por el
usuario y los repositorios públicos identificados en sus recibos. Se incluyen
paráfrasis propias, metadatos mínimos, métricas y localizadores; no se incluyen
PDF, resúmenes completos, listas CR, direcciones, exportaciones nativas ni copias
del código de autores. Consultar los originales mediante sus DOI/enlaces y sus
condiciones de acceso. Una URL pública no implica permiso de reutilización.

El registro de código conserva el estado histórico de esa etapa; la matriz
fulltext_20261105 documenta la revisión actual de los diez originales. NR
significa no reportado en el PDF y alcance examinados; NV significa no verificado.
No se equiparan ausencia de reporte y cero, ni código disponible y reproducción.
