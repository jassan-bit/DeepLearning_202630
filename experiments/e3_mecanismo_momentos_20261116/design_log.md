# Bitácora matemática posterior al cierre v2

**16 de noviembre de 2026. No se realizó selección por resultados nuevos de mercado.** Los pasos siguientes corresponden a formulación y revisión, no a recetas entrenadas. No se reconstruyen horas que no quedaron registradas.

| Paso | Decisión y motivo | Evidencia o corrección |
|---|---|---|
| 1 | Conservar la referencia aditiva v2 y cerrar el candidato NLL que no superó la heurística. | Cierre v2 ya publicado; no se sustituyen resultados ni se reabre su presupuesto. |
| 2 | Delimitar la limitación al posible conflicto de una misma escala bajo mala especificación. | Contraejemplo sintético exacto y exponencial continua. El mal resultado de BTC no identifica esa causa. |
| 3 | Proponer dos momentos de `V=U²` y una coordenada de la media transformada. | Identidad del objetivo; no se cambia el backbone ni se agregan entradas futuras. |
| 4 | Revisar la cota inferior antes de reclamar contribución. | Antecedente directo encontrado en Merhav; se atribuye. Scores de media y guardas también son conocidos. |
| 5 | Corregir el dominio de la media: `H` es supremo, no máximo cuando `q>m²`. | Secuencia de dos puntos con momentos fijos; rango `[L,H)`. Se descarta una parametrización que incluya `H` como realizable no degenerado. |
| 6 | Permitir igualdad en `q=m²` y en la frontera inferior de la media. | `q=m²(1+b²)` y `rho=d²/(1+d²)`. Una sigmoid estricta no alcanza `rho=0`; no se eligió por RMSE. |
| 7 | Hacer explícitas las trampas de derivada cero. | Inicialización propuesta `b=√8,d=1`; no se garantiza optimización. El borde verdadero `m0=q0=0` se excluye del teorema de score en vez de ocultarlo con epsilon. |
| 8 | Usar scores logarítmicos de medias en vez de MSE de `U⁴`. | Finitud de `E[U⁴]` para riesgo auxiliar; no garantiza varianza finita de gradiente si `E[U⁸]=∞`. Pareto α5 verifica la distinción. |
| 9 | Contrastar con Faithful y exigir control de igual capacidad. | La consistencia de riesgo no implica su garantía de trayectoria. El control libre tendrá las mismas tres lecturas activas. |
| 10 | Mantener la admisibilidad abierta. | Doce consultas web focales; tres intentos API bloqueados antes de respuesta, revisión nativa e intervención humana pendientes. |

Los pesos `lambda_m=lambda_q=0,1` y las inicializaciones son una especificación inicial propuesta. No hubo barrido, nuevas métricas de BTC, carga de NPZ/checkpoints ni entrenamiento/inferencia de mercado. Las optimizaciones escalares del verificador sí se realizaron y se declaran como **riesgos poblacionales sintéticos**, con su alcance y unidades.

**Recetas neuronales entrenadas en mercado: cinco** entre v1 y v2. El **`N_d` global sigue pendiente de conciliación**, incluidas variantes analíticas, parametrizaciones y ensayos evaluados. Este registro añade una propuesta matemática y documenta sus alternativas de parametrización descartadas. No se borra el esfuerzo de diseño por no haber entrenado ni se confunde ese inventario con el número de modelos confirmatorios.

**Decisión final de este corte:** no entrenar otro piloto ni declarar modelo original o superior. Conservar la propuesta para revisión de admisibilidad y la cabeza aditiva como referencia. Completar antecedentes y verificación humana, luego cerrar controles, presupuesto y protocolo antes de una eventual implementación.
