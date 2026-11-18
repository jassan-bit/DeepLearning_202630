# Recibos Colab guardados directamente en GitHub

El cuaderno `notebooks/e3_remediacion_preflight_colab.ipynb` comprueba las
17 recetas con datos artificiales y guarda al terminar los recibos JSON,
logs, ZIP y un manifiesto de hashes en el repositorio
`jassan-bit/DeepLearning_202630`.

Cada ejecución crea una rama nueva `e3-colab-receipts-<RUN_ID>` y una carpeta
`runs/colab_preflight/<RUN_ID>/`. Incluye `files/` con los JSON/logs,
`receipts.zip` y `storage_manifest.json`. La URL y el commit se imprimen al
confirmar por lectura remota todos los archivos. El guardado conserva el
árbol fuente, usa `base_tree` explícito y crea la referencia de rama sólo al
final. No modifica la rama principal ni registros existentes.

## Preparación en Colab

Seleccionar GPU y ejecutar el cuaderno completo. Para escribir desde Colab
se necesita un token GitHub con **Contents: Read and write**, limitado a
este repositorio, en Secretos con nombre `GITHUB_TOKEN`. Si falta se pide
en un campo oculto; nunca se pega en el código ni en el chat. La consulta
inicial acredita acceso y rol, no el alcance fino de escritura del token.
El token queda en memoria y no se pasa al proceso científico ni al archivo.

El guardado comprueba que el ZIP contiene exactamente los mismos JSON/logs
seleccionados y los mismos bytes. Se rechazan rutas inseguras, enlaces,
credenciales y archivos ajenos al registro. Se conserva la consola completa
del preflight, además de los recibos del entorno y de cada modelo. Los
límites son 1 MiB por archivo de texto, 10 MiB para ZIP y 20 MiB totales.

## Reintentos

Si falla la escritura, mantener la sesión, corregir el secreto y ejecutar
sólo la celda de guardado. El cuaderno reutiliza el token ya introducido y
relee el secreto actualizado. Un guardado idéntico ya existente se reconoce;
un contenido distinto se rechaza sin sobrescribir. Una respuesta perdida
al crear la rama se reconcilia leyendo lo realmente guardado, sin repetir
GPU. El comprobante local de publicación se escribe fuera del directorio
de evidencia para mantener estables el ZIP y el manifiesto.

## Alcance

El ejecutor sigue siendo sintético: no entrena con datos reales ni realiza
la campaña confirmatoria. El ZIP de la pasada L4 anterior no se recuperó;
su consola conserva su alcance de reporte del autor. La primera subida
real desde Colab con este flujo todavía está pendiente. Las comprobaciones
locales usan fixtures y una API simulada; no acreditan una subida real.

La página `/intro.html` del sitio corresponde a E1. E3 está en
`https://jassan-bit.github.io/DeepLearning_202630/entregable3/intro.html`
y el selector de libros en `/entregables.html`. Comparten repositorio.

Las copias de código de esta carpeta son instantáneas para revisión. El
cuaderno ejecuta los módulos desde sus rutas activas en un checkout del
repositorio; los recibos guardan el HEAD utilizado.
