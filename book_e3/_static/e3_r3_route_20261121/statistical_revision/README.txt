Revisión estadística R3 de desarrollo, 21 de noviembre de 2026

Se conserva el estimando E_semilla[sqrt(E_tiempo error²)]. La corrección nueva
reemplaza sólo P1 básico por un bootstrap-t pareado de semillas y calendario.
P2 delta-HAC y MCS bootstrap-t v2 son controles congelados. El MCS aporta una
decisión global y un conjunto de modelos; no aporta el tercer p por rival.

La derivación y los límites están en STATISTICAL_DERIVATION.json. El diagnóstico
DETERMINISTIC_DIAGNOSIS.json relee los resultados anteriores, incluida la FWER
14/128 del nulo de semillas. No borra ni vuelve a etiquetar aquella falla.

El plan y las ocho fuentes de CODE_MANIFEST.json quedaron publicados antes de
la única ejecución en el commit 374fc00f50da4454522c04a7973d0ef590d5e192.
PUBLIC_ANCHOR.json registra el readback real. El plan contiene seis escenarios
de nulo o nulo parcial de 256 réplicas y una alternativa de 128: 1664 réplicas,
B=999, hasta 600 segundos y dos hilos. No permite reintentos, otras semillas,
cambiar bandas ni redefinir las verdades después de observar resultados.

La familia principal sigue teniendo 38 posiciones: dos procedimientos por
19 rivales. Se reportan por separado cualquier falso rechazo de componente,
la intersección de ambos procedimientos en un rival y la decisión de relevancia
práctica del subconjunto de dos procedimientos. Esta última no es la batería
completa de tres métodos. Los intervalos puntuales no pasan a ser simultáneos
por aplicar Holm a las pruebas de igualdad.

Los controles deterministas y su revisión independiente preceden al experimento.
SOFTWARE_VALIDATION.json registra 23 pruebas de software; no son réplicas de
Monte Carlo ni certifican validez estadística. El código de covarianza es una
aproximación cruzada de primer orden, con límites de pocas semillas, sesgo de
raíz en calendarios finitos y dependencia temporal. La heterogeneidad temporal
entre modelos conserva theta poblacional cero y debe permanecer en la decisión.

Los resultados cerrados viven en validation_new/. Su execution_receipt.json
identifica estado, conteos, tiempo y hashes. RESULTS_DECISION.json dentro de ese
directorio aplica la pantalla de tolerancia de FWER predeclarada. Las pantallas
usan intervalos CP95 individuales; una pantalla aprobada no certifica nivel
exacto, cobertura simultánea ni aplicabilidad al mercado.

export_results.py exige recibo terminal e integridad de fuentes y registro antes
de leer resultados. Recuenta Holm y los eventos principales, y produce exports/
con tablas CSV, RESULTS_INTERPRETATION.json y EXPORT_MANIFEST.json. No usa RNG.
No debe ejecutarse contra un registro en curso. Las conclusiones efectivas,
incluidas fallas o resultados inconclusos, están en esos JSON y tablas.

No hubo entrenamiento, fits, forwards ni predicciones nuevas de mercado. R3,
el tercer procedimiento primario y la confirmación permanecen sin acreditar.

Resultado cerrado de la ejecución única

Se completaron 1664/1664 réplicas y 1.662.336 draws en 479.82273 s, con cero errores, cero réplicas no soportadas y las ocho fuentes intactas.
La batería experimental permanece INCONCLUSA Y NO APROBADA. La FWER
de componentes fue 6/256 (semillas), 9/256 (AR moderado), 12/256
(AR fuerte), 7/256 (nulo parcial), 6/256 (semillas heterogéneas)
y 2/256 (dependencia temporal heterogénea entre modelos). El límite
superior CP95 del AR fuerte, 0,080449, excede el umbral fijado 0,08.
Los otros cinco escenarios pasan sólo la pantalla de tolerancia acotada.

En semillas, el control antiguo produjo 31/256 eventos frente a 6/256
de la revisión; la comparación pareada tiene p Holm6=3,5763e-7.
Las otras cinco comparaciones no demuestran una mejora uniforme.
Hay nueve pantallas descriptivas de subcobertura puntual (ocho P1 y
una P2); sus intervalos individuales no constituyen un test global
ajustado por multiplicidad.

MCS también conserva límites: AR moderado resulta inconcluso en su
pantalla global, y bajo nulo parcial excluye al menos un óptimo en
30/256 réplicas, CP95 [0,08048; 0,16305]. No se acredita cobertura
de conjuntos bajo todos los nulos parciales. En la alternativa, la
intersección de las dos pruebas rechaza en 20/128, la regla práctica
del subconjunto en 16/128 y MCS deja sólo la candidata en 35/128.
Ninguna de esas tasas es potencia completa de tres métodos.

RESULTS_DECISION.json resume la decisión del componente y preserva
el RESULTS_DECISION.json original del ejecutor en validation_new/.
