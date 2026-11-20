# Bitácora de diseño de E3

Actualización de conciliación: **19 de noviembre de 2026, America/Bogota**.
Es una auditoría documental posterior a los resultados conocidos, con asistencia
de IA declarada; no es un preregistro, no modifica resultados históricos y no
reinicia presupuestos. La [versión anterior del índice](https://github.com/jassan-bit/DeepLearning_202630/blob/29f9a26f1fd05484c6581ea6538ad7cb5a4781cb/experiments/design_log.md)
y la [consolidación del 17](https://github.com/jassan-bit/DeepLearning_202630/blob/29f9a26f1fd05484c6581ea6538ad7cb5a4781cb/book_e3/_static/e3_diseno_riguroso_20261117/design_log.md)
se conservan por commit. El [corte inicial del 15](https://github.com/jassan-bit/DeepLearning_202630/blob/29f9a26f1fd05484c6581ea6538ad7cb5a4781cb/experiments/e3_bloque_c_20261115/design_log.md)
mantiene su Nd mercado=0 histórico.

## Inventario conciliado dentro del corpus registrado

El [expediente de conciliación](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/e3_evidence_close_20261120/inventory)
contiene variantes, alias, selección, intentos, incidentes, fuentes congeladas y
verificador. Sustituye la expresión genérica «Nd global pendiente» por unidades
observables y límites identificados:

| Unidad | Conteo | Qué incluye |
|---|---:|---|
| Variantes neuronales funcionales implementadas | **15** | Cinco estructuras iniciales; tres cabezas v2; tres por momentos; cuatro atómicas V3. |
| Variantes neuronales distintas con ajustes de mercado completos | **12** | Dos v1, tres v2, tres por momentos y cuatro atómicas. |
| Variantes sólo verificadas artificialmente | **3** | Sin penalización, sin saturación y sin contenido. |
| Controles analíticos/de transformación examinados | **4** | Piso, exceso medio, M00 proyectado y full proyectado. |
| Propuestas neuronales descritas sin implementación acreditada | **3** | Control inspirado en Faithful, plug-in Jensen por momentos y q=9m² fijo. |
| Entradas documentadas identificadas | **22** | 15 + 4 + 3, sin confundirlas con redes entrenadas. |
| Bosquejo conceptual con identidad incompleta | **1** | La referencia «su propia media sola» de Faithful requiere especificación para decidir si añade otro diseño. |
| Ensayos HPO conservados | **183** | 168 de la auditoría y 15 V3; no arquitecturas nuevas. |
| Trabajos completos en los alcances conciliados | **844** | 18 neuronales históricos, 796 de auditoría y 30 neuronales V3. |

Las 22 entradas —o 23 si el bosquejo ambiguo finalmente se distingue— son un
conteo **del corpus registrado**. No certifican el número de ideas informales
sin archivo, todas las oportunidades de selección humana/IA o todas las
invocaciones físicas. Los alias se deduplican: full/M11 cuenta una vez, M00
conserva su identidad y la aditiva no se vuelve otro diseño por cambiar de semilla,
cohorte o fase. Las pérdidas distintas sí se distinguen como recetas.

El control full proyectado se recupera de la continuación de v1 y corrige una
omisión del subtotal anterior de tres controles, sin borrar aquella fuente.
`floor_plus_fit_mean` y `floor_plus_mean_excess` son la misma regla; la media se
estima con ajuste. El plug-in atómico de V3 y el plug-in Jensen por momentos
son controles distintos.

## Selecciones y esfuerzo conservados

Las tres tasas 0,0003/0,001/0,003, las inicializaciones y las validaciones se
registran por ámbito. El igualado de sesgo de v2 no se supone ejecutado por el
constructor nativo de la auditoría o del nuevo piloto. Los órdenes de cuadratura
64/128/256 y las configuraciones de las pruebas artificiales se conservan como
decisiones de precisión/software, no se ocultan ni se presentan como HPO de
mercado. Un conteo único de todas esas unidades heterogéneas sería engañoso;
el expediente no calcula una corrección numérica del sesgo del ganador.

El ledger contiene **824 ajustes neuronales completos**: 18 históricos,
776 de auditoría y 30 V3; añade cuatro ARIMA y 16 trabajos sin optimizador.
Los 900 renglones de recursos de la auditoría son fases de **796 trabajos
únicos**, porque el autoencoder comparte un presupuesto entre tres fases.
La aditiva corresponde a 76 trabajos de auditoría: 12 HPO, 40 finales y
24 curvas reducidas. Sus 93 ajustes en todos los alcances son repeticiones de
una identidad funcional; no 93 diseños. No se suman otra vez las 52 corridas
históricas de extensión E2, que tienen contador y propósito separados.

La auditoría histórica conserva tres fases con cómputo de caída desconocido y
un sobrepaso máximo de plazo de 35283,630231 s que incluye desconexión.
No se convierte en horas GPU medidas ni se borra con cero. La tabla de
incidentes conserva fallos de calendario, transporte y trabajador aunque el
recibo final de la auditoría enumere cero fallos de modelo pendientes.

## Decisiones que permanecen

Se mantienen los desenlaces adversos de full frente a M00, de half-t con NLL
y de momentos. El piloto V3 `20261120T031424176339Z`, resultados
`8eaa581bc34b615a8507c9e384a3e4be29084720`, completó 15 HPO y 15 finales.
La receta primaria `atom_crps` quedó **2,7043 % peor que `atom_mse_only`**;
no superó la heurística del 2 %. El control ganador no se promociona
retroactivamente a candidato primario ni a confirmación.

La consolidación conserva los 796 trabajos y los 30 nuevos con fuentes y hashes.
No entrena, infiere, reoptimiza o modifica la regla después del resultado.
No acredita novedad R1, defensa R2 ni superioridad R3. Las hipótesis, familias,
datos realmente reservados, potencia y versión de una confirmación futura
deberán fijarse prospectivamente y añadirse como decisiones nuevas.

Cada entrada posterior conservará fecha real, identidad, cambio respecto de su
ancestro, fuentes, datos usados, métrica, ausencias, decisión, fallos y consumo.
Una corrección añadirá trazabilidad y mantendrá la versión anterior; las fechas
desconocidas no se rellenarán a partir de semillas o nombres de carpeta.
