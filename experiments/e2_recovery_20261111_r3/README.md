# Continuación R3 de la campaña CUDA

Los diagnósticos de solo lectura del Colab conectado muestran `RUN_SEARCH` sin definir, ningún proceso `run_e2_cuda_search.py` y ausencia de la campaña y del resultado pendiente en la ruta local registrada. La comprobación no inspeccionó otras máquinas ni discos. El snapshot remoto `f3c19e8589c3341d4cffb78f948687c64c90ad0c` conserva 1.552 resultados lógicos y un inicio sin resultado: `4c391a70d987e8d418877f621c82a8ad16a0d224a0ebdca637abdea4a5447ea9`.

`authorization.json` registra una decisión concreta de recuperación dentro de la continuación del proyecto solicitada. No representa un entrenamiento ya realizado ni una aprobación humana adicional. El desenlace, duración y pasos del intento anterior siguen desconocidos. El controlador admite una sola repetición física de su configuración y semilla originales; el recibo físico observado debe confirmarse antes de escribir la reconciliación lógica. Un recibo iniciado sin observación bloquea otra repetición automática.

La primera invocación R3 terminó con `private_or_unapproved_text` antes de reclamar un recibo físico: el texto de evidencia incluía una ubicación local que el filtro público rechaza. La corrección describe la ubicación sin publicar la ruta, mantiene el filtro y valida la autorización completa antes de restaurar la campaña. Los controladores se conservan en cachés identificadas por commit para mantener el checkout y los informes anteriores.

## Presupuesto y fuente

- Motor científico A3: `564ba4762f84a800b02fa4c734f5752a169f5223`; registro `78c3c5b8ca042a347aaa80d642f3affd8b5911e8697d8747a2de915d1c3194a9`.
- Plan: `4eae2d2ff9ef6f83a2f9eb0dc5ba647208d695e1665a8e522d812fed67893bda`. No cambian candidatos, semillas, modelos ni orígenes.
- Controles iniciales: 8 horas de sesión, 24 acumuladas, 10 GiB. El diario ya contiene la ampliación a 36 horas: evento 2303, SHA256 `899ab1bc8c440b0e90be19abc38e3f22785aaa87902b8b531752080ad444cf9a`.
- R3 verifica por separado las 24 horas iniciales y las 36 efectivas. No escribe una nueva ampliación ni restablece el presupuesto. El cierre conservador asigna a la sesión anterior su techo sellado de 28.800 s; ese cargo no se presenta como duración GPU observada.

## Ejecución en Colab

El notebook `notebooks/entregable2_recuperacion_r3.ipynb` conserva los pasos de preparación 1–7 y sustituye el ejecutor normal por el controlador R3 fijado a un commit y hashes. Usa una T4 y el secreto GitHub ya configurado en Colab.

En la primera celda, activar `RUN_SEARCH=True` y `RETRY_LOST_ONCE=True`, y ejecutar en orden. La recuperación restaura una caché ausente desde la campaña remota validada; no inicializa una campaña nueva. La sección 8 imprime el PID y conserva un informe único de cada invocación. La sección 9 muestra ese informe, sin confundir un `status.json` antiguo con actividad actual.

Si otro escritor cambió el head, falta una prueba de integridad, hay una ejecución activa o aparece otro inicio sin resultado, el controlador bloquea la continuación. Un cierre por presupuesto o sincronización no declara completo el benchmark. Este flujo no abre la evaluación final ni fija R.

## Validación sin entrenamiento

Las pruebas usan trabajadores simulados y no solicitan GPU, descargan datos de mercado ni alteran la campaña remota. Cubren conservación de registros, presupuesto ampliado, restauración con caché ausente, conflictos del head, recibos de creación única, recuperación de una observación ya guardada, bloqueo tras una segunda interrupción y cierre ante SIGINT.

```bash
E2_ENGINE_ROOT=/ruta/al/motor/A3 \
PYTHONPATH=scripts:/ruta/al/motor/A3/scripts \
python -m unittest discover -s tests -p 'test_e2_*v3*.py' -v
```

Los controladores R3 son externos a los 61 archivos de la implementación científica A3. Su procedencia es la recuperación R2 publicada en `20248e2917746962f92dccbcf89ae1874f6cbad2`, con autorización, versiones y pruebas nuevas para este incidente y el presupuesto ampliado.
