# Auditoría independiente de identidad — corte 14/11/2026

El CSV contiene **9 registros únicos de Scopus**, todos finales, con títulos, años, autores, EID y citas presentes. Corresponden a **9 fuentes para 10 sistemas**: Wang es compartido por FCN y ResNet. Los títulos y autores coinciden con las fichas locales; los tres DOI explícitos coinciden con los alias editoriales. Seis filas carecen de DOI en el CSV; su enlace se sustenta en título, autores, fuente editorial y EID, sin completar DOI por inferencia.

| Fuente/sistemas | Año | Citas Scopus | EID |
|---|---:|---:|---|
| TSMixer | 2023 | 247 | 2-s2.0-86000299816 |
| iTransformer | 2024 | 1378 | 2-s2.0-85200562438 |
| TimeMixer | 2024 | 619 | 2-s2.0-85186508140 |
| TiDE | 2023 | 247 | 2-s2.0-86000040350 |
| PatchTST | 2023 | 2477 | 2-s2.0-85192521637 |
| LSTM-autoencoder con predictor | 2019 | 321 | 2-s2.0-85076460928 |
| FCN-1D, ResNet-1D | 2017 | 2248 | 2-s2.0-85030974623 |
| TimesNet | 2023 | 1886 | 2-s2.0-85196367497 |
| Inception-1D | 2020 | 1761 | 2-s2.0-85090433635 |

Se puede generar un **nuevo escenario de C** con este corte y conservar el anterior. No se deben sumar citas de preprint y publicación final. La misma fuente de Wang aporta idéntico C a sus dos sistemas; sigue siendo una sola fuente bibliográfica. TSMixer está correctamente identificado como el trabajo All-MLP de Chen y colegas.

Límites: la fecha de corte procede de la declaración del usuario y del nombre fechado del archivo; el CSV no registra hora de consulta de las citas. Los EID son nuevos respecto a los dos archivos v2 de contraste: se comprobó formato, unicidad y coincidencia con el ID de sus enlaces, sin consulta en vivo de Scopus. Para Inception, la ficha local resume autores con et al.; se validó el primer autor, no toda la lista contra esa ficha. La captura mencionada por el usuario no fue inspeccionada en esta auditoría. No se modificó el repositorio ni se recalculó C.
