# Decisión nueva: diagnóstico de transferencia sin adaptación

Se autoriza exclusivamente un diagnóstico CPU con pesos ya existentes. La
pregunta es cómo transfieren las cinco recetas del piloto BTC y sus dos
controles analíticos a los paneles limpios D01, D02 y D03, **sin ningún ajuste,
HPO, recalibración o selección nueva**. La decisión y sus hashes deben quedar
en un commit inmutable antes de producir estas predicciones nuevas.

La confirmación sigue en **NO-GO**. El piloto original de treinta ajustes
conserva su heurística fallida: `atom_crps` fue 2,70 % peor que
`atom_mse_only`. El diagnóstico no modifica esa regla ni promociona al ganador.

Se usan las tres semillas finales originales —20261118, 20261119 y
20261120— de las cinco recetas, con sus quince checkpoints fijados en el
recibo. No se usan los pesos ajustados por dominio de la auditoría de 796
trabajos como si fueran V3. Los dos controles analíticos siguen siendo únicos,
no tres entrenamientos ficticios.

Las ventanas de prueba son exactamente las de la partición original de la
admisión/auditoría: D01, índices 2630:3263, 633 fechas; D02, 1183:1454, 271;
D03, 1280:1576, 296. Son índices de ventanas/objetivos, no índices directos
de barras. Conservan las purgas de siete ventanas. El origen y los SHA-256
de cada CSV normalizado están en el JSON. Esos calendarios ya se expusieron
durante la auditoría aditiva: **no son confirmación independiente**.

La estandarización y las escalas y/u proceden exclusivamente del ajuste
BTC de 532 filas. El exceso medio del control analítico es el mismo
`d_fit=0.002030102424887359`. No se vuelve a calcular con AAPL, ETH o
contratos BTC. El volumen mantiene sus unidades originales, por lo que su
cambio de distribución debe declararse; no se corrige retrospectivamente con
otro escalador, recorte o eliminación de filas. A sí se calcula causalmente
de las seis rentabilidades disponibles en cada entrada, tal como exige el
forward; eso no es un ajuste.

Se reportan por conjunto las RMSE individuales y su media por receta, MAE
y CRPS_U sólo cuando existan átomos/masas. No se usa RMSE de un ensamble, no
se ajustan intervalos ni se hereda el calibrador de la aditiva. No se
computan p-valores, remuestreos ni una batería nueva de R3. Las 51
trayectorias fijas suman 20.400 filas esperadas, incluidos los controles.
Los fallos se conservan, sin sustituir semillas o descartar observaciones.

El límite de inferencia es CPU de un hilo, 60 s por receta/semilla/conjunto
y 900 s agregados, con descarga registrada aparte. Se conservan los
consumos observados; no se llama memoria incremental al RSS del proceso.

La decisión está fechada después del piloto y antes de este diagnóstico:
registra únicamente la nueva operación. No retrofecha los resultados
antiguos, no prueba utilidad confirmatoria y no cierra R3 aunque alguna
transferencia descriptiva resulte favorable. La IA está declarada.
