# Preparar F00–F03 desde el Colab ya instalado

Este asistente resuelve el paso de formularios de desarrollo sin editar a mano
cuatro JSON. Requiere haber ejecutado las etapas 1 y 2 del cuaderno
`E3_R3_Temporal_GitHub.ipynb` del commit
`7aa5ccff0b881cbad270129d5163011c4429a774`. Verifica la fuente inmutable
`c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c` y sus 288 archivos antes de operar.
No modifica ese código ni la identidad del experimento.

En Colab, guardar la clave real de Nasdaq Data Link en **Secretos** con el nombre
`NASDAQ_DATA_LINK_API_KEY` y permitir acceso al cuaderno. Si no existe, el cliente
original la solicita con entrada oculta. Nunca introducirla en el chat, en los
JSON, en la URL visible o en GitHub.

El cargador distribuido verifica el SHA256 del asistente y llama
`prepare_development(globals())` en una celda posterior a la etapa 2. El
asistente presenta instrumentos, periodos y fuentes primarias de licencia. Pide
nombre, contacto y función reales cuando debe completar una declaración o un
recibo de derechos, además de la aceptación escrita `ACEPTO DESARROLLO` antes de
guardar esos metadatos o descargar. Sin esa respuesta no escribe ni descarga.

La aceptación comprende el reconocimiento de exposición de D00–D03, la
designación de F00–F03 como desarrollo expuesto, el uso académico no comercial,
las condiciones de atribución/licencia de los derivados y la ausencia de
redistribución raw. No establece una reserva final, no autentica identidad y no
crea una firma criptográfica. Las fechas de la declaración son las reales de su
ejecución. El programa declara asistencia de IA en su implementación y revisión.
Los términos de datos siguen siendo los de las fuentes, no los del asistente.

Se conservan las plantillas originales. No se reescriben declaraciones
completadas, derechos, recibos, manifiestos, particiones ni metadatos de
recuperación existentes. Se verifican hashes, procedencia, las cuatro particiones
y el enlace de recuperación antes de reutilizar una admisión. Un estado parcial
se conserva y detiene la ejecución: no se borra ni se etiqueta como completo.
Ante discrepancias, revisar el motivo impreso y las rutas conservadas antes de
cualquier recuperación. Una descarga parcialmente guardada sin manifiesto no es
reanudable por el colector original; este asistente tampoco afirma que lo sea.

Se consulta primero WIKI para detectar falta de acceso antes de miles de
peticiones BTC. El cliente original evita registrar claves en los recibos y
expone sólo resúmenes de estado. La disponibilidad de las fuentes y el acceso
del propietario no se sustituyen con fixtures ni se dan por comprobados por
estas pruebas de software. No hay sustitución automática de fuentes ni cambio
de calendario, particiones, candidata o reglas por resultados.

Al terminar debe indicar **Cuatro paneles verificados**. Después ejecutar la
etapa 3 original para actualizar `missing` con la comprobación de integridad.
La etapa 4 requiere activación separada y explícita para entrenar. Este asistente
no cambia sus banderas, no entrena, no prepara ni abre test y no escribe en
GitHub. Los archivos de autor y datos permanecen privados en Colab. Los
respaldos de campaña los gestiona el controlador original cuando se prepara y
activa esa campaña.

Las pruebas añadidas usan archivos generados y red simulada. No acreditan
disponibilidad de mercado, aceptación auténtica del autor ni resultados R3.

Para repetirlas fuera de Colab, configurar `E3_HELPER_TEST_SOURCE` con la carpeta
verificada de la fuente A y `E3_HELPER_TEST_PYTHON` con su intérprete aislado;
ejecutar `python -m unittest -v test_prepare_development_owner.py` desde esta
carpeta. La salida de la ejecución entregada queda en
`HELPER_SOFTWARE_TEST_OUTPUT.txt`; sus identidades y precios son fixtures
de software explícitos, no declaraciones o admisiones de mercado reales.
