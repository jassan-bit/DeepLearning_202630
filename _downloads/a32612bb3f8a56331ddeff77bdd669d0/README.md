# Síntesis acumulada E1 · ampliación gratuita del 7 de noviembre de 2026

Este corte conserva las 47 obras del expediente anterior y añade diez obras nuevas:
los 13 trabajos seleccionados contienen tres obras ya registradas. El registro
acumulado tiene **57 identidades**, **35 lecturas de fuentes primarias completas**
y **1 transcripción espejo completa**. Los demás niveles de acceso
siguen separados; el original de Vo no se considera leído íntegramente a partir
de fragmentos editoriales indexados.

## Frecuencia por un criterio común

| Familia | Cripto elegible por reporte actual | Cripto descriptivo actual | Todo el constructo cripto actual |
|---|---:|---:|---:|
| LSTM | 10/10 | 13/13 | 13/15 |
| MLP | 4/10 | 6/13 | 6/15 |
| GRU | 2/10 | 4/13 | 4/15 |
| TCN | 1/10 | 1/13 | 1/15 |
| RNN_simple | 1/10 | 1/13 | 1/15 |
| Transformer_encoder | 1/10 | 1/13 | 1/15 |
| GFN | 1/10 | 1/13 | 1/15 |
| MFN | 1/10 | 1/13 | 1/15 |
| RBFN | 1/10 | 1/13 | 1/15 |
| CNN_generic | 0/10 | 2/13 | 2/15 |

El denominador principal es **N=10 elegible por suficiencia de reporte**;
N=13 es extracción descriptiva y N=15 es el constructo operativo
continuo realizado/histórico. No se exige mismo activo, exchange, frecuencia,
ventana o transformación que E1. La suficiencia se aplica simétricamente a
fuentes históricas y nuevas, sin convertir un ancho final ausente en ausencia
universal del candidato cuando estructura y protocolo son suficientes.
Los tres estudios cripto nuevos son García-Medina/Aguayo-Moreno,
Korkusuz/Sahiner y Lensky/Hao. Se revisan **todas** sus familias comparativas,
no únicamente MLP: RNN simple y CNN genérica conservan sus etiquetas propias.
Los módulos internos, variantes y componentes híbridos no suman bases autónomas.
Independiente significa experimento base no reutilizado; no independencia
estadística de mercados, muestras o errores compartidos entre publicaciones.
RBFN, GFN, MFN y otras redes comparativas permanecen como familias distintas;
sus ocurrencias no certifican que cada una satisfaga la condición de modelo
Deep Learning básico de la guía y se excluyen del ranking de candidatos DL.

Los cortes históricos N7, N12/N10/N8 permanecen en sus carpetas, con sus propios
identificadores y hashes. El N10 descriptivo anterior y el N actual elegible por
reporte pueden tener el mismo número y distinta composición: no se intercambian.
Las publicaciones/grises, unidades de varianza/desviación e índices implícitos
conservan estratos en `frequencies.csv`. Los trabajos financieros se registran
como transferencia; sus ocurrencias no incrementan los numeradores cripto.
El preprint de Ge no se transforma en publicación revisada, ni el preprint y
la publicación IEEE de Lensky se cuentan como dos estudios.

## Selección y alcance

La terna experimental registrada **GRU–LSTM–TCN** permanece como contraste entre
reglas de estado recurrente y filtros causales. Las frecuencias y el tamaño de
la MLP histórica se muestran con el mismo criterio; no se afirma que la terna
sea la más frecuente ni un mínimo global. Las recetas propias conservan
GRU 1.121, LSTM 1.489 y TCN 1.185 parámetros; MLP 593 pertenece a otro experimento
histórico y no a la reserva final de 465 semillas. La comparación de cantidades
no certifica igualdad de capacidad, coste, presupuesto ni error.

La búsqueda nueva fue **dirigida por arquitectura**, solicitando siete TCN y seis
MLP. Esa cuota organiza recuperación, sin estimar prevalencia ni convertir
los 13 trabajos en13 inclusiones homogéneas. No se declara revisión exhaustiva.
Los 14 informes nativos no recuperados siguen como disposición de acceso al
corte anterior, sin compras obligatorias ni evaluación de calidad inventada.
Este compilador no consulta bases nativas, entrena, emite pronósticos ni
modifica datos, selección o resultados experimentales.

## Verificación

`python scripts/build_e1_homogeneous_expansion.py --check`

El programa comprueba el expediente histórico, deduplicación por DOI,
47 identidades conservadas, diez adiciones, todas las familias verificadas,
subconjuntos de reporte, grupos de dependencia y ámbitos. `protocol.json`
conserva hashes de los insumos públicos. `adjudications.json` permite revisar
los juicios; `comparator_quality.csv` separa calidad, acceso y rol de referencia.
El manifiesto permite únicamente estos metadatos, sin artículos, resúmenes,
contactos, datos de mercado o predicciones individuales.
