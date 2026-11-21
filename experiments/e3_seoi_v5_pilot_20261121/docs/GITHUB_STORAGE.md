# Guardado V5 en GitHub sin Google Drive

Esta revisión del controlador responde al fallo de `drive.mount` por
`credential propagation was unsuccessful`. El modelo, las seis recetas,
las cinco semillas y los umbrales numéricos del piloto se conservan.
No se ejecutaron entrenamientos al preparar este cambio.

El notebook usa CPU y una carpeta local de Colab. Antes de entrenar verifica
un primer respaldo en una rama de ejecución del repositorio. Después guarda
cada trabajo completo antes de pasar al siguiente y, durante trabajos largos,
checkpoint durables cuando ha transcurrido un minuto entre guardados.
Si GitHub rechaza un guardado, no se inician nuevos ajustes.

## Acceso privado

El token debe permitir `Contents: Read and write` para
`jassan-bit/DeepLearning_202630`. Puede proporcionarse mediante el secreto
Colab `GITHUB_TOKEN` o un campo `getpass`. No se solicita en el chat ni se
escribe en el notebook, argumentos, archivos o recibos.

La clave de respaldo es independiente del token y tiene al menos32caracteres.
Usa una clave aleatoria o una frase larga y conserva una copia privada.
Puede guardarse como `E3_V5_BACKUP_KEY` en secretos Colab. Es necesaria para
reanudar después de perder la VM. Cambiar el token no cambia esa clave.
El controlador rechaza cambiar de clave silenciosamente sobre un respaldo
existente. No se imprime ni se guarda la clave en GitHub.

## Contenido y recuperación

- `status.json`: identidad del caso, fuente, plan, progreso y hashes.
- `public_results.zip`: tablas, curvas, fallos y recibos de la lista permitida.
- `checkpoint.aesgcm`: respaldo cifrado con los últimos checkpoints referidos
  por los punteros de cada trabajo, incluyendo Adam y RNG.

El respaldo privado excluye el NPZ de datos y snapshots antiguos no referidos.
AES-256-GCM autentica contenido y contexto. Scrypt deriva la clave con sal
aleatoria y parámetros fijos. La restauración comprueba autenticidad, hashes,
rutas, tamaños, plan y fuente antes de permitir cargar pesos. Los datos se
descargan de nuevo desde la URL y el hash originales.

El cliente espacia las escrituras, distingue límites temporales de errores de
autenticación y conserva el mismo payload durante un reintento. Publica blobs,
árbol y commit, actualiza la referencia sin forzar y verifica la lectura. Una
sesión local atrasada no puede sobrescribir un commit remoto más reciente.

## Contabilidad de una desconexión

Se recupera el estado que llegó al último respaldo verificado. El trabajo
posterior a ese punto puede perderse. Si la sesión remota estaba abierta,
`remote_restore.json` y `recovery_history.json` conservan permanentemente la
posibilidad de cómputo no observado. Los costes y pasos registrados pasan a
ser cotas inferiores; no se reconstruye una cantidad exacta ni se certifica
un límite histórico real de120s después de esa restauración.

Las métricas conservan su alcance de diagnóstico de desarrollo y los umbrales
numéricos no cambian. La comparabilidad completa del coste queda sin verificar
cuando hubo cómputo perdido. Este control no añade una condición universal
nueva a la guía ni concede acreditación estadística.

Las pruebas del cliente usan cifrado real y GitHub simulado. Su aprobación no
se presenta como un guardado realizado con el token del autor o una ejecución
real de entrenamiento en Colab.
