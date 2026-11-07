# Síntesis homogénea E1 · 7 de noviembre de 2026

Este expediente consolida fichas ya revisadas. No ejecuta nuevas búsquedas,
recupera originales, entrena redes ni cambia los resultados del proyecto.
La unidad es la obra; la frecuencia utiliza experimentos base independientes.
El registro contiene 47 identidades, 26 originales primarios leídos y una
transcripción espejo completa separada. No son 47 artículos completos leídos.

## Frecuencia y sensibilidad

| Familia | Referencias extraíbles | Todo el constructo realizado/histórico | Reporte elegible §3.1 |
|---|---:|---:|---:|
| LSTM | 10/10 | 10/12 | 8/8 |
| GRU | 4/10 | 4/12 | 2/8 |
| MLP | 3/10 | 3/12 | 2/8 |
| TCN | 1/10 | 1/12 | 1/8 |
| CNN genérica | 1/10 | 1/12 | 0/8 |
| Encoder Transformer | 1/10 | 1/12 | 1/8 |

N12 reúne constructos operativos continuos realizados/históricos de cripto;
N10 contiene al menos una referencia autónoma verificable: es extracción
descriptiva, no diez inclusiones que cumplan íntegramente §3.1. Los dos restantes
de N12 no son negativos demostrados. El estrato N8 elegible por suficiencia de
reporte §3.1 excluye juntas S8 y PSO–GRU por descripción insuficiente.
La escala, las publicaciones/grises y el objetivo implícito
conservan estratos distintos en `frequencies.csv`. Dudek 2025 reutiliza la base
de 2024 y no suma voto. T3/DeepVol es una identidad. Los híbridos y componentes
internos no se descomponen en familias. N7 y sus tablas históricas se conservan.

## Decisión y límites

Las tres frecuencias mayores son LSTM, GRU y MLP. El estudio registrado
GRU–LSTM–TCN se conserva como contraste de reglas recurrentes de estado y
filtros temporales causales. La revisión actual es posterior a sus ajustes.
No prueba que esa terna sea top3, mínima ni única. La guía exige sesgos
inductivos diferentes, sin imponer tres macrofamilias: la diferencia entre
interpolación GRU y celda/salida LSTM se explica junto a su recurrencia común.
Ambas empiezan sin estado previo a cada ventana D7; TCN tampoco ve datos
anteriores a esos siete días aunque su campo receptivo teórico sea29.

Las recetas finales propias suman GRU1121, TCN1185 y LSTM1489 parámetros.
Existe una MLP histórica propia con593 y HPO declarado antes de sus ajustes;
no se descarta por falta de receta. Su menor número de pesos no demuestra
mejor pronóstico. Los tamaños de recetas publicadas no forman una comparación
común sin supuestos de entrada/salida, bloque y ancho. Se mantienen los
ausentes; no se rellenan defaults para decidir a favor de una familia.

Los14 informes nativos no recuperados al corte son un estado terminal de
acceso y una limitación de cobertura; no se eliminan del flujo ni se excluyen
por calidad sin leerlos. El estrato de17 registros parciales/no obtenidos del
universo ampliado es otra unidad y no17 tareas obligatorias de recuperación;
no equivale a todos los registros sin lectura primaria: también hay código y
otros niveles de acceso.
La bitácora nativa se acredita aparte: esta síntesis no prueba consultar IEEE.

## Reproducción documental

`python scripts/build_e1_homogeneous_synthesis.py --check`

El compilador verifica las fuentes históricas, identidades, dependencias,
denominadores,25 estratos y las sumas de parámetros. `adjudications.csv` permite
auditar lectura, objetivo, comparadores y razones en47 filas; los JSON de
`part_corpus` conservan localizadores y juicio crítico. `part_mechanisms`
contiene ecuaciones, recetas, contraste por pares y alternativas. El manifiesto
permite únicamente estos metadatos revisados; no redistribuye originales,
resúmenes exportados, datos de mercado ni predicciones.
