# Revisión ampliada a criptomonedas

Corte documental posterior al benchmark: 40 referencias, siete textos completos sobre criptomonedas y una matriz de diez registros. S9 estudia 20 activos y se cuenta una sola vez. Incluye BTC: los subconjuntos de estudios con BTC y con otros activos se solapan. La frecuencia de comparadores independientes se distingue de la presencia de componentes propuestos.

evidence_matrix.csv/json localizan objetivo, activos, frecuencia, horizonte, arquitectura, rol, métricas y páginas o tablas. family_frequencies.csv separa tres estratos: los siete textos cripto, los seis con referencias DL extraíbles y los dos con identificación editorial en el original. La identificación de un congreso no acredita Q1 ni un registro individual en WoS/Scopus.

Los seis valores QLIKE de S9 pertenecen a ganadores clásicos o de árboles; no se asignan a LSTM. Una cabeza MLP dentro de LSTM, una TCN o una CNN genérica cuya dimensionalidad no se informa no suman una referencia MLP/CNN1D simple. El experimento sigue siendo de Bitcoin: no se modifican recetas, preregistros o resultados.

harnet_erratum.json retira la inferencia no sustentada de fuga por scaler.fit_transform: es un escalador local con límites configurados. Esto no certifica ausencia global de fuga. harnet_schema.json documenta la diferencia rv5/rv5_ss. HARNet y DeepVol conservan sus niveles de código de autor e identidad editorial; sus originales siguen pendientes.

journal_update_20261105/ y recovery_20261105/ son cortes históricos. Este addendum es la interpretación documental vigente. La búsqueda acotada y dirigida no estima la prevalencia global de la literatura ni demuestra inexistencia de artículos. Los conteos GitHub son repositorios, páginas y ocurrencias; no son resultados de WoS, Scopus o arXiv.

Se publican metadatos y tablas extraídos, sin redistribuir manuscritos, cuadernos o datos de terceros. Para reconstruir las tablas agregadas: python scripts/build_crypto_literature_extension.py. La consulta original se sustenta en URL, SHA-256 y ubicación específica.
