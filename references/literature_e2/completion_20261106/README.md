# Literatura E2: verificación acotada del material disponible

Esta ampliación del 06/11/2026 (America/Bogota) aplica criterios explícitos a los
materiales ya recibidos. Se preservan los registros y las extracciones anteriores.
No constituye una nueva búsqueda bibliográfica ni un PRISMA global terminado.

Se verifican diez originales arquitectónicos mediante sus extracciones completas
(215 páginas físicas) y se cotejan los diez archivos principales del código de
autores con los hashes fijados. Esos diez estudios son admisibles como fuentes de
mecanismos de pronóstico transferibles, con sus limitaciones de reporte. Esta
admisibilidad acotada no prueba efectividad en Bitcoin, exhaustividad, optimalidad
del Top10, licencia de todo código ni validez de los adaptadores finales D7.

Los 17 registros financieros conservan la identidad de sus resúmenes nativos y
sus rutas explícitas: diez son candidatos de volatilidad para lectura completa,
seis corresponden a precio/retorno/trading y uno necesita aclarar el objetivo.
Una ruta basada en abstract no acredita lectura del cuerpo, arquitectura, ausencia
de fuga, semillas o ajuste comparable. No se afirma doble revisión humana.

La tabla de intervalos tiene 29 modelos/familias. Mantiene los componentes
verificados y deja desconocidos como nulos/vacíos. Los extremos algebraicos no
son valores observados. Se identifican en los registros nativos dos originales
adicionales: MICN (597 citas Scopus) y CryptoMamba-SSM de volatilidad (3); este
último se distingue de CryptoMamba de precio. Identificarlos no acredita sus
cuerpos ni su código, y no sustituye automáticamente integrantes del Top10.

Tres familias de mecanismos se trazan a bibliografías/motivaciones localizadas en
originales completos ya leídos: ViT/Swin desde Crossformer, MLP-Mixer desde
TSMixer y Mamba desde S-Mamba. Se publican títulos, autores, año, páginas y
limitaciones D7. No se afirma lectura nueva de los originales de visión/lenguaje
ni tres consultas nativas ejecutadas.

El rastreo de Binance en los abstracts devuelve ocho candidatos: cinco se
refieren a Binance Coin y tres no bastan para identificar el exchange/fuente de
sus datos. Ninguno se convierte por ello en una referencia Q1 verificada. El
criterio de uso previo de una fuente no exige identidad del snapshot privado.

La recuperación adicional revisó 40 archivos PDF privados (35 documentos únicos)
y verificó por hash siete cuerpos ya pertenecientes a E1, incluido el manuscrito
S2 recuperado de la revisión fijada de autores. Esos siete estudios no son los
siete apoyos aplicados nuevos de E2. No se recuperó el cuerpo de ninguno de estos
siete apoyos ni de los once candidatos financieros prioritarios. Los README de
dos repositorios aplicados identifican artículos y código, pero tienen 230 y
246 bytes: no acreditan lectura del manuscrito ni sus resultados. Dos falsas
pistas de DOI se resolvieron como una cita en otro estudio y una bibliografía
generada por el proyecto. El alcance de esa búsqueda no prueba indisponibilidad
global de los artículos.

## Archivos

- `architectural_originals_eligibility.csv`: diez evaluaciones de admisibilidad acotada.
- `author_code_verification.json`: identidades y cotejos de hashes de código.
- `financial_abstract_routes_17.csv`: rutas de los 17 resúmenes y pendientes concretos.
- `candidate_score_intervals_29.csv` y `score_protocol.json`: intervalos sin ranking ficticio.
- `additional_native_primary_identities.json`: evidencia nativa MICN/CryptoMamba-SSM.
- `origin_mechanism_trace.json`: fuentes de origen identificadas desde los originales leídos.
- `dataset_source_candidates.csv`: pistas de fuente distinguiendo Binance Coin del exchange.
- `manual_inputs_required.json`: insumos externos que no están presentes.
- `inherited_E1_body_reverification.json`: siete identidades heredadas cotejadas, sin duplicarlas como estudios nuevos.
- `applied_fulltext_recovery_status.json`: estado de los siete apoyos E2 y once candidatos financieros, con DOI y pendientes.
- `applied_author_repository_context.json` y `recovery_scope.json`: repositorios fijados y alcance de la recuperación, sin redistribuir cuerpos.
- `closure_checklist.json`: condiciones verificables para cerrar cobertura, elegibilidad, selección y calidad.
- `summary.json` y `provenance.json`: alcance, recuentos y hashes.

## Lo que sigue abierto

Faltan 374 registros Scopus y confirmar la historia/rangos WoS. También se
necesitan búsquedas transversales nativas, cuerpos aplicados pertinentes,
verificación de calidad por año y adjudicación global de las decisiones
provisionales. La proporción de código/datos públicos de toda la revisión no se
estima con el denominador de estos diez originales preseleccionados. NR en una
fuente leída significa no reportado; no se inventan parámetros, tiempos o
semillas para rellenarlo.

## Verificación sin adjuntos privados

Desde la raíz del repositorio:

    python references/literature_e2/completion_20261106/build_completion.py --check

El comando verifica hashes de entradas públicas y derivados espejo sin abrir
abstracts ni datos de mercado. Regenerar los cotejos positivos requiere pasar
explícitamente `--private-records`, `--author-source-root` y `--paper-text-root`
a sus archivos privados conservados; `--recovery-root` verifica los recibos y
hashes de cuerpos/textos de la recuperación adicional. Los insumos privados, PDF y código ajeno
no se redistribuyen. La herramienta no ajusta modelos ni abre prueba de Bitcoin.
