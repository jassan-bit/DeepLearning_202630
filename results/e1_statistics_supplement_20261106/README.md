# Suplemento estadístico posterior de E1: fase II D7

Este análisis del 06/11/2026 consume las métricas RMSE por semilla ya publicadas.
No fue prerregistrado, no cambia las pruebas primarias, el baseline GRU, las
recetas congeladas o el dimensionamiento. No entrena ni produce pronósticos.

- `friedman_global.json`: comparación global bloqueada por semilla, rangos y W de Kendall.
- `pairwise_sensitivity.csv`: tres Wilcoxon bilaterales asintóticos y Holm dentro de esta
  familia suplementaria; sumas de rangos y efecto biserial pareado con signo A−B.
- `qq_diagnostics.csv`, `qq_coordinates.csv`, `qq_differences.png`: normalidad y Q–Q
  reproducibles de las mismas 465 diferencias por par.
- `artifact_availability.json`: comprobación de disponibilidad local de evidencia privada;
  sólo existencia, sin cargar o exportar sus contenidos.
- `summary.json`, `source_manifest.json`: alcance, versiones, entradas y SHA-256.

Friedman prueba diferencias de distribuciones/rangos bajo intercambio de etiquetas
de modelo dentro de bloques independientes. Una interpretación sólo de medianas o
localización requiere formas comparables. W cuantifica concordancia del orden entre
semillas, no ganancia en RMSE ni relevancia económica.

Wilcoxon usa las diferencias como están guardadas, sin redondearlas, descarta ceros
exactos y asigna rango medio a empates absolutos. Emplea aproximación normal sin
corrección de continuidad, no p exacta. Su interpretación de localización requiere
simetría de la distribución de diferencias; no soluciona automáticamente el rechazo
de normalidad. Las asimetrías y Q–Q permanecen visibles para juzgarlo. La magnitud
biserial no es la d_z de la prueba primaria ni su misma estimación de efecto.

Los p suplementarios y los pares primarios reutilizan los mismos datos. No son
corroboraciones independientes; Holm3 sólo cubre los tres Wilcoxon de este archivo,
no todos los contrastes del informe. La comparación global es una prueba posterior
individual. No se selecciona el método con menor p ni se modifica el ranking fijado
con desarrollo. Ningún resultado demuestra independencia temporal, equivalencia,
potencia temporal o mejora práctica superior a δ.

Reproducción desde la raíz del repositorio:

```bash
python scripts/supplement_e1_planned_statistics.py
```

Dependencias numéricas y de gráficos efectivamente usadas constan en `summary.json`;
son el entorno de este suplemento posterior, distinto del entorno de entrenamiento
registrado. Sólo se necesita la tabla pública y las fuentes de identidad verificadas.
Para instalar este entorno: `python -m pip install -r requirements-e1-supplement.lock.txt`.
