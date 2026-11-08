# Plan del nuevo núcleo, antes del ajuste definitivo

La fuente científica inmutable es 91acdeae1efd92f098ff2ce9a32f90c5b0a4b98a.
El HPO nuevo completó 48 trayectorias y 960 etapas; su baseline es GRU.
El piloto nuevo completó 30 trayectorias y 60 etapas, sin leer precios de evaluación.
El R requerido por sus tres pares es 331; el piso registrado produce R final=465
por familia, semillas 8101–8565. Son 1.395 trayectorias y 2.790 etapas definitivas.

plan.json es una copia byte a byte del cálculo privado; su SHA-256 es
5c1ce3d67c72def7c337b92d846df9ea5589bc6f42802ed000dc0144b390568b.
Se publica antes del primer ajuste definitivo. No es un informe de ejecución final.
El ejecutor requiere un recibo de GET de este archivo y su commit antes de fit.

Los CSV contienen métricas y configuración, sin precios, predicciones diarias,
pesos o escaladores. Se excluye la escala MASE aprendida del piloto.
La adquisición recuperó los 32 ZIP originales por sus hashes, sin decodificar
filas de mercado. Su lectura y puntuación siguen bloqueadas hasta cerrar pesos.

El periodo septiembre–octubre de 2026 ya se evaluó en el proyecto: esta extensión
es histórica y exploratoria, sin recuperar un holdout confirmatorio independiente.
