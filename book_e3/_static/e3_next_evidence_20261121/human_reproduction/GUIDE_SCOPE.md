# Alcance preciso de la reproducción independiente

Fuente: *Entregable3DeepLearningProject.pdf*, 60páginas,922729bytes,
SHA256 `d6048aebd7b3db85a59231a5a89b01bc3f8ad8cb581a96b462e341d21ff1f364`.
Los números son páginas físicas del PDF.

En p38,§11.2: «Para entrenamientos costosos se aceptan resultados precalculados
y versionados, siempre que el código que los genera esté incluido y documentado».
En la misma página: «Reproducción independiente: una persona ajena al código
ejecuta el libro desde cero en el contenedor y regenera la tabla principal de
resultados; se reporta la discrepancia máxima observada».

P46,§14.5.7 reitera «Reproducción independiente desde cero dentro del contenedor,
comprobada por una persona ajena al código, con la discrepancia máxima reportada»
y exige entorno fijado, DOI, tabla maestra, pruebas, datos o instrucciones exactas
y licencias verificadas. P40,item13 pide anexar el reporte y el humo del despliegue;
p52,A5 reitera Docker funcional y máxima discrepancia. Son condiciones relacionadas,
no una licencia para declarar una persona ficticia o una validación inexistente.

A/B/C son una clasificación del proyecto, no tres reentrenamientos exigidos por
esas cláusulas. A reconstruye tablas desde resultados costosos versionados; B
verifica la ruta de inferencia de binarios aceptados; C reproduce ajustes nuevos
en un caso separado. Los recibos automáticos previos A/B se conservan y no se
repiten aquí. C tenía dos ajustes prefijados y sigue siendo una preparación
parcial de entrenamiento histórico; no es una obligación universal de repetir
796 trabajos ni30 ajustes. El coste nominal histórico completo y el de ese
subconjunto no se confunden con este procedimiento CPU.

El vacío material previo era la combinación **libro limpio en contenedor +tabla
principal realmente regenerada +reporte del tercero**. Este kit añade las dos
primeras rutas ejecutables, y deja el tercero preparado para una persona real.
La ruta usa la edición `f5c960eefedb5fc49b313dd795b1ec84e784bd78`, una copia nueva
del checkout completo y el generador de siete recetas V3 de la sección8.19.
Recalcula RMSE/DE/MAE/sesgo desde935 predicciones ya guardadas, verifica sus17
trayectorias, conserva los55 días y todas las recetas, mantiene la decisión
adversa, integra las columnas regeneradas en el CSV descargable del libro y
comprueba sus siete filas publicadas redondeadas. El CRPS_U de fuente se conserva
etiquetado y no se reconstruye desde predicciones puntuales.

El libro completo se vuelve a construir; las demás tablas y figuras se renderizan
desde sus precálculos versionados. **No se declara que se regeneraron numéricamente
todas sus tablas**, la auditoría aditiva19×4, las predicciones de transferencia o
un entrenamiento V3. Tampoco se acredita confirmación independiente del modelo.
La persona y, cuando proceda, el docente deben aceptar antes que esta tabla V3 es
el alcance principal pertinente. Si exigen también la tabla19×4 o toda la maestra,
se amplía el alcance y sus insumos antes de ejecutar; no se sustituye esa obligación
por siete filas ni se exige C automáticamente. El informe deja esa decisión visible.

El archivo0.3.1-REVIEW conserva15 grafos aprobados y fuentes/evidencia con
exclusiones; no contiene por ello cada activo de la ediciónf5c ni el maestro
científico4,27GB. Se reutiliza su URL/hash para procedencia y C opcional. Para el
libro se requiere el checkout completo fijado, con sus licencias originales;
este kit no asigna una licencia nueva global ni redistribuye mercado crudo.
