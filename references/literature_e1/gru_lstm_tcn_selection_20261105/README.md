# Selección vigente: GRU, LSTM y TCN

La decisión de publicar esta terna se tomó el 05/11/2026, después del benchmark histórico y de revisar la ejecución privada GRU–LSTM–TCN. No se presenta como una elección bibliográfica preregistrada antes de los experimentos. El protocolo privado había fijado sus recetas antes de sus nuevos ajustes; esa congelación es distinta de la decisión posterior de publicación. La comparación usa desarrollo previamente observado y sigue siendo exploratoria.

| Familia | Referencias independientes, N=7 | Publicaciones identificadas en originales | Receta propia | Parámetros |
|---|---|---|---|---:|
| GRU | S1; S2; S8: 3/7 | Akbiyik (2023); Miura (2019) | Una capa H32, dos compuertas, estado integrado y lectura final | 3.777 |
| LSTM | S1; S2; S5; S7; S8: 5/7 | Akbiyik (2023); Miura (2019) | Una capa H32, tres compuertas, celda separada y lectura final | 5.025 |
| TCN | S1: 1/7 | Akbiyik (2023) | Tres bloques residuales, seis convoluciones causales H16/k3/d1,2,4 y lectura final | 4.289 |

Akbiyik et al. (2023), ACM WSDM, DOI [10.1145/3539597.3570387](https://doi.org/10.1145/3539597.3570387), Tabla 2, comparan las tres familias como referencias independientes para volatilidad de Bitcoin. D-TCN es su propuesta avanzada. Miura, Pichl y Kaizoji (2019), ISNN, LNCS 11554, pp.165–172, DOI [10.1007/978-3-030-22796-8_18](https://doi.org/10.1007/978-3-030-22796-8_18), Tabla 1, añaden GRU y LSTM independientes. Sus MLP/CNN siguen en la extracción de autores, sin ocupar puestos experimentales actuales. El JSON conserva URL fijadas, ubicación de evidencia y límites.

La elección combina apoyo aplicado directo, capacidad pequeña explícita y contraste entre dos memorias recurrentes distintas y convolución temporal causal. No son los tres primeros puestos por frecuencia: el encoder tiene 2/7 frente a TCN 1/7. **TCN no acredita amplia recurrencia.** La terna se adopta con ese límite visible; no se modifica el corpus de siete estudios, sus frecuencias ni sus estratos de calidad. Publicación identificada no acredita por sí sola un registro individual WoS/Scopus o Q1.

Las recetas propias son adaptaciones compactas, no réplicas de los ganadores. La TCN de S1 usa seis niveles, F=287 y k=5. La propia usa tres bloques residuales, dos convoluciones por bloque, H16, k3, dilataciones 1/2/4 y lectura de la última posición, sin pooling global. RF=1+2(k−1)(1+2+4)=29; para D28 hay como máximo 28 días reales y relleno anterior cero. No se renombra la CNN1D histórica como TCN.

El contrato común recibe (B,D,5), D=7/14/21/28, OHLCV y σ7/h=1, con AdamW, dropout 0,1, weight decay 0,0001, batch 32, clipping 1, máximo de 100 épocas internas y paciencia 12. GRU/LSTM usan learning rate 0,003; TCN, 0,001. Diez semillas por receta/historia y doce bloques mensuales de 2024 forman 120 trayectorias, 1.440 segmentos y 2.880 ajustes. No hubo HPO nuevo. Las métricas propias permanecen en el export de ejecución; este expediente no reasigna cifras antiguas MLP/CNN ni resultados de autores como resultados propios.

Los adaptadores públicos `scripts/dl_models_gru_lstm_tcn.py` y `scripts/run_gru_lstm_tcn.py` se prepararon después de ejecutar en privado. El JSON registra hashes del protocolo, módulo y runner originales, sin atribuir retrospectivamente sus pesos al adaptador público. No se consultaron bases ni se ejecutaron entrenamientos al redactar esta selección. Los comprobantes históricos conservan sus identidades.
