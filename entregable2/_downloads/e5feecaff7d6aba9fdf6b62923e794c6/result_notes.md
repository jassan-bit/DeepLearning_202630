# Resultado del diagnóstico de dependencia de entradas

Se completaron **142.272 forwards**: 78 predictores × 608 fechas × tres condiciones. No hubo nuevos ajustes, selección ni recalibración. Las 47.424 predicciones originales CPU se regeneraron con discrepancia máxima **0** frente al estudio CPU anterior. Todos los estados cargados permanecieron idénticos. Esta igualdad pertenece al entorno CPU admitido; no corrige ni sustituye la anterior puerta de paridad A100 fallida.

La tabla `study_v1/statistics/input_dependency_by_model.csv` contiene 26 comparaciones, una por sistema y grupo intervenido, con seis trayectorias fijas. El diagnóstico evidencia dependencia de entradas bajo las sustituciones especificadas; no establece que el volumen sea espurio, que exista fuga o que la eliminación de una variable mejore el modelo entrenado.

Como ejemplos descriptivos, sustituir volumen cambia el RMSE de iTransformer en **+0,0001773213**, con IC95 % marginal **[0,0000247184; 0,0003260195]**; en PatchTST, **+0,0000819208**, con IC **[0,0000160043; 0,0001456292]**. Estos son intervalos exploratorios marginales, sin decisión simultánea de una familia. Su comparación con δ no constituye una prueba de equivalencia.

La sustitución del grupo OHLC produce, por ejemplo, un cambio de RMSE **+0,0058787645** en iTransformer y **+0,0093803891** en SAE. Esto corresponde a una perturbación sustancial de señales relevantes para pronosticar volatilidad, no a evidencia de una fuga por esas variables.

Ambas intervenciones preservan caminos de siete días procedentes de train, pero rompen relaciones entre grupos y entre ventanas solapadas. Sus efectos pueden incluir extrapolación fuera del soporte conjunto. Los IC temporales son condicionales a una sola asignación de donantes; no incluyen incertidumbre de asignación, de particiones nuevas o entre mercados. A2 y RQ1 se conservan.

## Evidencia independiente descargable

`diagnostico_entradas_reanalisis_v1.zip` incluye pronósticos, plan, tabla, script de reanálisis, instrucciones y requisitos. Se extrajeron sus 15 miembros fuera del repositorio y se verificaron los hashes de los 14 archivos de contenido. Con NumPy, sin Torch, se reconstruyeron los 26 contrastes y **208 cifras**: discrepancia máxima **0**. El ZIP acredita reanálisis independiente de pronósticos guardados; no se denomina reproducción del entrenamiento ni de los nuevos forwards.

Los cargos acumulados de este componente fueron **321,419550806 s**, repartidos entre inferencia, estadística y reanálisis independiente. Sus tres recibos figuran en el ledger continuado, sin reiniciar el presupuesto de 84 horas.
