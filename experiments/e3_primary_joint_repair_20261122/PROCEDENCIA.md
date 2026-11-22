# Procedencia y alcance de la reparación

Este paquete fue preparado para `jassan-bit/DeepLearning_202630` por instrucción
del autor, con implementación, derivaciones y revisión asistidas por IA.
No atribuye una evaluación, defensa, firma o reproducción a una persona que no
la haya realizado. La candidata permanece `input-only-overlap-mse-1`.

## Fuentes del proyecto

- Fuente histórica S2: commit
  `db74afd0763596f71bcbafb4b9150a44a2e83339`, manifiesto SHA256
  `2c7dc052d2a53f212688340b50b65bed5fa367ff16fb5ed0a2474d417696715b`.
  Los 304 miembros originales se verifican con `audit_preservation.py`.
- La infraestructura HAC/Bartlett, referencias gaussianas fixed-b, Holm y
  la política de decisión parten de `engine/statistics.py` y `engine/power.py`
  del mismo proyecto. El SHA256 de ese `statistics.py` es
  `a14e20b74f668404835f0f15e1fe555c5ac718ab886125b2dea20d531df06e95`.
- Las derivadas de la media de raíces, el ajuste de parámetros auxiliares del
  GMM, el remuestreo del funcional y la nueva validación conjunta son cambios
  versionados de este paquete. No se presentan como algoritmos estadísticos
  inéditos ni como validación predictiva de la candidata.
- El expediente R1/R2 reutiliza antecedentes y evidencia auténtica previamente
  archivados. Su inventario indica las fuentes que fueron revisadas; este
  paquete no afirma haber repetido búsquedas bibliográficas que no ejecutó.

## Referencias metodológicas

- Diebold y Mariano (1995), DOI `10.1080/07350015.1995.10524599`.
- Harvey, Leybourne y Newbold (1997), DOI `10.1016/S0169-2070(96)00719-4`.
- Politis y Romano (1994), DOI `10.1080/01621459.1994.10476870`.
- Giacomini y White (2006), DOI `10.1111/j.1468-0262.2006.00718.x`.
- Kiefer y Vogelsang (2005), CAE05-08, teorema 3 y sección 3.3,
  <https://cae.economics.cornell.edu/05-08.pdf>. Se reutiliza la lectura y
  procedencia del expediente anterior; no se inventa una nueva descarga.

Los nombres de esos procedimientos no convierten las extensiones suaves de
este paquete en aplicaciones literales de sus teoremas sin supuestos. Las
derivaciones, la revisión y las simulaciones declaran su ámbito explícito.

## Dependencias y derechos

La validación estadística usa NumPy 2.3.5 y SciPy 1.17.0, bajo sus licencias BSD
y avisos incluidos en las distribuciones oficiales, y un compilador C++ para
el núcleo local. La recuperación opcional de pesos añade las dependencias
indicadas por su propia fuente fijada; no se instalan para la simulación.

Se citan los artículos y repositorios externos; no se redistribuyen aquí sus
cuerpos ni se copia el núcleo de un modelo externo. Este paquete no asigna
una licencia abierta nueva al repositorio ni hereda la licencia del suplemento
de Zenodo para archivos ajenos a ese depósito. Las condiciones de datos,
artículos y dependencias siguen teniendo su alcance original.

## Cambio prospectivo explícito

La métrica primaria se conserva como media de RMSE individuales, pero se
corrige el objeto de la inferencia: las diferencias, ganancias, intervalos y
efectos pasan al funcional de raíces por semilla. El efecto descriptivo se
expresa en unidades de su influencia temporal. Se mantienen alfa 0,025, margen
relativo 2 %, umbral descriptivo 0,2, familia completa y regla adicional dos de
tres. Esto es una reparación estadística prospectiva identificada, no una
reescritura de las pruebas antiguas ni una aplicación automática de su potencia
o calibración a la versión nueva.
