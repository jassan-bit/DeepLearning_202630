# Ejecución Colab L4 reportada por la persona autora

El 18 de noviembre de 2026 la persona autora compartió la salida de Colab de
la versión `530479e4dbc4bc60be7fb363291a0f3420c5e7ac`. La consola reporta
**17 de 17 perfiles aprobados**, `passed_synthetic_only`, código de salida 0
y verificación global GPU verdadera en una **NVIDIA L4**. El entorno reportado
es Python 3.12.12, Torch 2.8.0+cu126, CUDA 12.6 y driver 580.82.07.

`console_transcript.txt` normaliza la salida pegada en la conversación;
`observation.json` separa esos datos reportados del estado de auditoría.
El ZIP `e3-preflight-receipts-20261118T150022983069Z.zip` todavía no fue
adjuntado. Su SHA-256 y los recibos individuales **no están comprobados**.
El enlace `localhost:8080` corresponde a la sesión de Colab del autor y no
es una URL pública desde la que se haya obtenido el paquete.

## Significado de las marcas GPU

El ejecutor de esa versión creaba `environment.GPU_execution_verified=false`
al inspeccionar el hardware, antes de los modelos, y no actualizaba ese
campo anidado. La marca global se calculaba al finalizar todos los perfiles.
Por tanto, ese `false` inicial no contradice el éxito final reportado.

La corrección activa elimina la marca del bloque de entorno: `cuda_available`
y el inventario GPU describen hardware. La verificación global exige los
17 perfiles aprobados y los 14 recibos neuronales en el dispositivo CUDA
solicitado, con backward y recarga idéntica. Su prueba usa fixtures simulados;
no se presenta como una nueva ejecución GPU.

## Alcance y siguiente revisión

La consola reporta cero ajustes con mercado, cero predicciones de mercado y
cero contrastes nuevos. Sigue bloqueada la campaña real: cero datasets de
tres admitidos, manifiesto/tag previos y tratamiento/presupuesto/potencia
finales pendientes. Este preflight no acredita R1, R3, reproducción humana
independiente o aceptación pública de Render.

Con el ZIP adjunto se podrán comprobar su integridad, el HEAD, los hashes de
fuentes, el entorno y los 17 recibos individuales. Hasta entonces no se
inventan hashes ni se atribuye una auditoría del archivo.

Las copias de código de esta carpeta son instantáneas para revisión. El ejecutor se utiliza desde su ruta activa `experiments/e3_remediacion_20261118/evaluation/colab_preflight.py` en un checkout completo, como hace el cuaderno; las fuentes y el lock conservan esa estructura.
