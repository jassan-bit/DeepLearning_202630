# Análisis posterior de errores con predicciones congeladas

Las tres redes y persistencia conservan sus predicciones y resultados finales. Este suplemento no entrena, no carga pesos ni escaladores, no vuelve a pronosticar y no cambia recetas o semillas. Lee exclusivamente predicciones cerradas y sus registros, además de sigma7 de desarrollo para fijar rangos comunes.

Se publican veinte medias mensuales observado/pronosticado y comparaciones agregadas por deciles del objetivo de desarrollo, junto a errores por mes, año y rangos Q25/Q75. Todos los rangos usan exactamente los mismos días para las redes y persistencia. Los errores se calculan primero por corrida y después se resumen entre semillas; las líneas de pronóstico medio ilustran agregados, no un nuevo ensemble evaluado.

La figura de densidad observado/pronosticado usa una rejilla común 30×30, fijada mediante desarrollo. Contiene todos los pares de las corridas, normalizados por familia. Los extremos fuera del rango se acumulan en la última celda y se contabilizan aparte. Repetir el mismo observado entre 309 semillas no genera fechas independientes. El CSV sólo contiene intervalos, conteos y porcentajes agregados.

El sesgo residual es pronosticado menos observado. La asociación entre error absoluto y objetivo es descriptiva y no tiene p-valor. La descomposición de sumas de cuadrados distingue variación entre medias de modelos y variación dentro de modelos; no constituye ANOVA inferencial.

Los rangos emplean el observado para analizar errores retrospectivamente: esa pertenencia no estaba disponible como entrada. Los límites provienen exclusivamente de desarrollo 2023–2024. Los IC son marginales entre inicializaciones y condicionados a los mismos días; no son intervalos temporales o simultáneos. No hay nuevas pruebas confirmatorias, selección a partir del test ni generalización independiente.

Los archivos públicos no contienen filas diarias de observaciones o pronósticos, datos de mercado, pesos o valores de escaladores. Las medias por bin se suprimen si contienen menos de cinco días. Los hashes de todas las fuentes registradas y los dos espejos originales se verifican antes y después. Las seis métricas completas se recalculan sólo para verificar que coinciden con las originales, sin sobrescribirlas.
