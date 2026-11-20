# Dictamen científico del cierre

**La decisión NO-GO se conserva.** No hay nueva evidencia que justifique abrir
confirmación de `atom_crps`. El resultado adverso del piloto (2,7043119888 % más
RMSE que `atom_mse_only`) y los tres dominios expuestos sin superar al mejor
control siguen siendo el fundamento observado. Esto no prueba inferioridad
universal ni equivalencia, y no modifica el dictamen documental acotado de R1.

Se cerró la auditoría de los 486 escenarios guardados: 162 corresponden a
sensibilidad entre semillas y 324 al calendario. Las referencias condicionales
van de 10 a 1032 pares y de 4,73 a 38.800,24 fechas, según supuestos y objetivos.
Esos rangos no recomiendan una configuración: mezclan escenarios ilustrativos,
incluidos límites de dispersión muy inciertos con tres seeds. La disponibilidad
de GPU no convierte más seeds en más calendarios independientes.

La conciliación nueva identifica nominalmente los 19 rivales. Sólo la aditiva
y dos controles analíticos aportan referencias de varianza en el piloto; las
otras tres referencias son ablaciones, y faltan las de los 16 rivales de E2.
Además, las referencias disponibles cubren sólo el desarrollo BTC expuesto,
no las 76 celdas futuras de cuatro paneles y 19 rivales. Los valores ausentes
se conservan como tales en la plantilla entregada.

La revisión del diseño detecta una decisión metodológica pendiente concreta:
el estimando de media de RMSE por seed tiene incertidumbre entre seeds y
calendario. Un bootstrap de fechas o HAC temporal no cubre por sí solo ambas;
un t-test entre seeds tampoco aporta generalización temporal. La propuesta
anterior GW omnibus no establece automáticamente la dirección y el margen
unilateral de ese funcional. Se documenta P3 subsampling como alternativa de
un futuro borrador, sin ejecutarla ni declarar validada la batería.

Está implementada la agregación futura de potencia con Holm sobre 57 p-valores
por réplica, intersección de los tres procedimientos por rival, decisión
conjunta y límites binomiales Monte Carlo. Los tests son fixtures de software;
no se calculó potencia científica nueva ni se aplicaron pruebas al mercado.

| Estado | Producto |
|---|---|
| Ejecutado y verificado | Auditoría de insumos y 486 escenarios, inventario de 19 rivales, exposición y hashes de procedencia, tests y preflight bloqueado. |
| Preparado | Diseño futuro con estimando, familias, tratamiento, márgenes por fundamentar, piloto de planificación, plantilla de 76 celdas y lector de simulaciones. |
| Bloqueado | Confirmación, potencia final, R/T, calendario, datos no expuestos y validación de las tres pruebas. |
| No generado | Tag de protocolo y Colab confirmatorio, porque las puertas científicas siguen sin satisfacerse. |

La guía exige al menos cinco seeds de planificación por receta estocástica y
diez de confirmación, o más según dimensionamiento. Exige potencia por contraste
de al menos 0,90 y reportar la conjunta; no fija un mínimo 0,90 para la conjunta.
El margen práctico, la alternativa de planificación y el efecto nulo son
parámetros distintos. Elegir una alternativa favorable a partir de los resultados
observados no cierra la planificación.

Los datos D00–D03 continúan válidos para sus usos exploratorios admitidos.
Ninguno se certifica como confirmación no expuesta; los futuros dominios son
clases por evaluar y no datasets seleccionados. Una admisión nueva requiere
fuente, licencia, fechas, hashes, exposición, tarea y dependencia documentadas.

Se preservan 796 trabajos de la aditiva, 30 ajustes V3 y 51 trayectorias con
20.400 predicciones retrospectivas. No se entrenó, infirió, recalibró, simuló
mercado ni remuestreó una batería científica. **R3 permanece no acreditada.**
