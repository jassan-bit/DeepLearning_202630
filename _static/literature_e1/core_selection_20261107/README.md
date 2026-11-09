# Selección vigente de los tres modelos base de E1

La selección vigente es **LSTM–MLP–GRU**, derivada del estrato cripto con reporte
suficiente: **10/10, 4/10 y 2/10**, respectivamente. TCN registra 1/10 y permanece
como comparación complementaria histórica. La actualización del 7 de noviembre de 2026
se realiza después de conocer los resultados anteriores y por solicitud del autor:
no convierte la selección previa ni la reserva conocida en un estudio prospectivo.

## Criterio conjunto de la guía §3.2

La frecuencia se extrae del corpus congelado `homogeneous_expansion_20261107`.
Se consideran candidatos DL; RBFN/GFN/MFN y otras ANN contextuales no se convierten
automáticamente en modelos Deep Learning elegibles. Los tres puestos tienen frecuencias
10, 4 y 2; el siguiente grupo tiene 1, por lo que no hace falta resolver un empate
en el límite para favorecer una terna. Las filas empatadas posteriores se muestran
alfabéticamente y no como un ranking de complejidad desconocida.

La simplicidad se concreta en **una capa oculta MLP** y **una capa recurrente
unidireccional GRU/LSTM**, con cabezas escalares. Dudek, Brauneis y García-Medina
aportan MLP autónomas de una capa oculta; Korkusuz publica una búsqueda con una/dos
capas y anchos 32/64/128. Zhang añade una variante MLP ReLU de una capa en su
Apéndice B, como transferencia financiera y sin aumentar la frecuencia cripto.
GRU/LSTM se apoyan en comparadores publicados y se adaptan al mismo contrato.
No se afirma réplica exacta de las configuraciones ganadoras de los autores.

Las anchuras MLP 32/128 se prefijan como candidatos por su orden de capacidad bajo
D7 y cinco entradas: **1.185/4.737 parámetros**, frente a GRU 16/32 **1.121/3.777**
y LSTM 16/32 **1.489/5.025**. La antigua MLP 16 de 593 parámetros conserva su propia
historia; no es la receta final de esta actualización. Los candidatos no son una
configuración seleccionada ni una puntuación observada. La búsqueda experimental
separada debe seleccionar por desarrollo y publicar su estado y resultados.

MLP mezcla rezagos con pesos específicos por posición, GRU comparte una transición
recurrente con reset/actualización y LSTM añade una celda separada y compuerta de
salida. Son diferencias de sesgo funcional, sin exigir tres macrofamilias ni
inferir errores independientes. Todas reciben los mismos siete días; GRU/LSTM
reinician estado en cada ventana y no acceden a memoria anterior.

## Alcance y cronología

Este cambio resuelve la discrepancia documental de conservar TCN mientras la regla
de recurrencia favorecía MLP. No demuestra mínimos universales, prevalencia global
o superioridad predictiva. La búsqueda previa era dirigida por arquitectura; las
frecuencias son las del corpus evaluado, con sus límites de cobertura y calidad.
La versión anterior GRU–LSTM–TCN, sus recetas, curvas y métricas permanecen intactas.

La ejecución nueva tiene protocolo y planificación propios en el capítulo 10.
La reserva de septiembre–octubre de 2026 ya fue observada: una extensión sobre ella
es exploratoria respecto de selección/modelado, aunque las nuevas reglas de ajuste
se publiquen antes de sus ajustes. Este compilador no entrena ni abre datos de
mercado, no publica pesos/predicciones y no inventa resultados MLP.

`python scripts/build_e1_core_selection.py --check` verifica hashes de cinco
insumos documentales, ranking, procedencia de cada ocurrencia, parámetros y
salidas públicas. El manifiesto aprueba exclusivamente los metadatos de la selección.
