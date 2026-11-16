# Reproducción portátil con alcance declarado

Este flujo nuevo prepara un paquete autónomo con las fuentes científicas fijadas, las fuentes de autores, los 81 predictores CPU ya ajustados —incluidos los 79 checkpoints binarios—, sus escaladores y calibradores, el OHLCV diario admitido y las matrices auditadas. No depende de un checkout Git, de rutas `/workspace`, de la web publicada ni del diario presupuestario histórico para ejecutarse en otro equipo. Los registros históricos copiados pueden mencionar sus rutas originales como procedencia; el controlador no las resuelve ni las utiliza para adquirir entradas.

La entrada numérica es el **derivado diario y las matrices admitidas**. El paquete conserva el recibo de agregación y los hashes de los veinte ZIP originales; una ejecución de este paquete **no vuelve a descargar ni decodificar esos ZIP**. El calendario de 608 fechas se conoce desde E1 y no se recupera cegamiento prospectivo.

## Preparación del entorno

El paquete preparado contiene `run.py`, `portable_campaign.json` y `requirements_cpu.txt`. Instalar las dependencias requiere inicialmente acceso al índice de paquetes; la ejecución posterior utiliza los archivos locales. CPU es la opción predeterminada. CUDA es opcional para nuevos ajustes, con un entorno compatible declarado por separado; no es requisito para reanalizar o ejecutar los pesos CPU incluidos.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements_cpu.txt
.venv/bin/python run.py --mode check --output ../comprobacion
```

`check` verifica los hashes y registra las versiones y el sistema operativo actual. No ejecuta entrenamiento, inferencia ni pruebas estadísticas. El SO registrado corresponde a la nueva ejecución; no autentica retrospectivamente una imagen A100 que no quedó fijada.

## Tres alcances computacionales distintos

```bash
# Reconstruye la A2 desde los 81 pronósticos guardados, sin forwards ni ajustes.
.venv/bin/python run.py --mode reanalysis --execute --output ../reanalisis --cap-seconds 120

# Ejecuta los 81 predictores incluidos sobre las 608 fechas y compara sus
# pronósticos y A2 con sus propios resultados CPU originales.
.venv/bin/python run.py --mode replay-checkpoints --execute --device cpu --output ../inferencia --cap-seconds 600

# Vuelve a ajustar 78 redes y tres clásicos con las recetas finales fijadas.
# Es una ejecución nueva; estimar antes sus recursos y elegir un límite adecuado.
.venv/bin/python run.py --mode train-fixed --execute --device cpu --output ../nuevos_ajustes --cap-seconds 10000
```

Los límites anteriores son ejemplos para el equipo del profesor, no nuevas asignaciones al presupuesto histórico del proyecto. Para la campaña actual cada prueba científica se ejecuta dentro de la asignación concedida mediante su wrapper presupuestario. El límite local se carga como suma de los tiempos de los workers, sin añadir otra vez el tiempo del controlador. Los fallos y resultados parciales se conservan.

`--primary20000` agrega el análisis registrado de 20.000 remuestreos a cualquiera de los tres modos. Su coste es adicional dentro del mismo límite. El reanálisis y la inferencia exigen igualdad numérica exacta con **la referencia CPU que produjo los pesos incluidos**. Los nuevos ajustes no exigen igualdad con esos pesos ni con A100. La configuración CPU conserva Torch 2.8.0+cpu, FP32, batch de inferencia 1, dos hilos intraop y dos interop, algoritmos deterministas y MKLDNN activado. No promete igualdad entre otras versiones o dispositivos.

Cada etapa escribe un log, un recibo con los hashes de sus salidas, su duración y su estado. `--resume` sólo admite etapas completadas cuyos hashes sigan siendo válidos, con la misma configuración, intérprete, modo y límite. Un fallo parcial exige una salida nueva y una recuperación explícita; no se sobrescribe ni se repite silenciosamente. Las salidas deben quedar fuera de la carpeta inmutable del paquete.

## Estado y requisitos que permanecen separados

La existencia del código o una comprobación de hashes **no acredita una ejecución completa**. El recibo de prueba aislada identifica los modos efectivamente ejecutados y sus discrepancias. No se marca B21 completo sólo por crear este controlador.

El modo `train-fixed` utiliza las API de ajuste supervisado reales y las épocas fijadas; no reutiliza pesos para llamarlos nuevos ajustes. Su implementación requiere todavía una ejecución íntegra nueva para quedar demostrada como flujo portátil de entrenamiento. La repetición CPU anterior cobró **6.853,632183 segundos**; antes de este trabajo quedaban **4.646,109374 segundos** en el presupuesto acumulado. Por ello no se agenda una repetición de todos los ajustes dentro de ese saldo usando el tiempo anterior como si cupiera. No es una proyección garantizada de otro equipo.

Este paquete tampoco repite la búsqueda HPO histórica, todas las perturbaciones, explicaciones y ablaciones, ni genera todas las tablas y figuras del libro. Esas etapas deben integrarse y probarse antes de acreditar ese alcance integral de B21. Las duraciones históricas no conservadas no se recuperan entrenando modelos nuevos. NAS sigue sin aplicar a las arquitecturas definidas.

La nueva edición debe conservar identidad, manifiesto y recibos propios; los ZIP v3 y las descargas históricas permanecen inmutables.
