# Colab operativo: ocho etapas y recuperación en GitHub

El notebook publicado fija commit de fuentes, SHA256 del manifiesto y SHA256 del bootstrap. El generador `package/notebook_template.py` produce el notebook y, con `--starter-output`, su primera celda independiente. El artefacto de la entrega sustituye los tres tokens por identidades reales de GitHub. El generador de plantillas no debe ejecutarse directamente como si sus tokens fueran esos pins.

No se entrenó con mercado ni se abrió reserva final durante esta implementación. Las pruebas contienen ajustes mínimos reales sobre números artificiales. Las cuatro admisiones de mercado, su planificación, el dimensionamiento suficiente y la confirmación siguen siendo entradas/experimentos auténticos, no resultados de los tests.

## 1. Instalar y verificar las fuentes

Abrir `E3_R3_Temporal_GitHub.ipynb` desde el commit publicado y ejecutar sus dos primeras celdas de código. La primera comprueba los bytes del bootstrap antes de importarlo, descarga el resto por manifiesto e instala un entorno CPU aislado. Verifica las versiones exactas, ejecuta pytest y `engine.cli check`. No requiere A100 ni modifica Torch del kernel de Colab.

La segunda celda define rutas privadas, casos/ramas distintos de planificación y final, y los adaptadores de comandos. Antes de preparar un caso, fijar `RUN_LABEL` y conservarlo al recuperar la campaña. No cambiarlo para borrar una exposición o un resultado adverso.

Configurar en **Colab Secrets** `E3_R3_GITHUB_TOKEN` con acceso Contents de lectura/escritura al repositorio, y `E3_R3_BACKUP_SECRET`, una clave independiente de al menos 32 caracteres. Si no existen, las operaciones remotas usan entrada privada. Conservar la clave fuera de la sesión; sin ella no se puede restaurar el cifrado. No pegar claves en celdas, commits o conversaciones. Nasdaq usa la credencial distinta `NASDAQ_DATA_LINK_API_KEY`, sólo al adquirir esa fuente. Los comandos muestran progreso inmediatamente: el hijo usa salida sin buffering; el padre combina stdout/stderr y redacta claves literales/codificadas **antes** de imprimirlas o retenerlas en el resultado.

## 2. Registrar evidencia y admitir únicamente desarrollo

Con todos sus flags en `False`, la celda prepara por panel:

- `private/inputs/F0x/owner-development.json`, con identidad/contacto y declaración aún vacíos; no firma por el autor.
- `rights-DRAFT.json`, con procedencia y evidencia primaria ya vinculadas, pero estado `requires_real_actor`.
- `acquisition.json`, con el proveedor/instrumento y rango de desarrollo propuestos.
- `development-split.json`, con bounds cronológicos por **sesiones esperadas del calendario**, sin adquirir precios o construir etiquetas.

La regla prospectiva de desarrollo es **60 % ajuste, 10 % selección, 20 % planificación y 10 % calibración** sobre barras raw. Los cortes son `[0,floor(.60N)-1]`, `[floor(.60N),floor(.70N)-1]`, `[floor(.70N),floor(.90N)-1]` y el resto. Se exige planificación de al menos 80 endpoints tras el calentamiento local. `engine.data` aplica además la purga por soporte de quince cierres; los porcentajes no eliminan esa protección. El archivo registra el algoritmo y el número de sesiones calculado sólo con metadatos; la admisión vincula todo el split por hash. Los conteos reales y exclusiones por splits corporativos deben verificarse al admitir los datos.

Los rangos vienen de `configs/data_panels.json`: BTCUSDT desde 2017-08-17 hasta 2023-12-31; IBM/MSFT/XOM WIKI desde 1996-01-02 hasta 2013-03-01. Son propuestas de **desarrollo expuesto**. Si se modifica el diseño antes de usarlo, conservar su nueva versión y justificación antes de los ajustes. Nunca escoger cortes por pérdidas observadas.

El actor debe completar su formulario con nombre/contacto/rol, fecha UTC real, origen de la declaración y cutoff observado, reconocer D00–D03 expuestos y designar todo ese desarrollo como expuesto. Debe verificar los derechos aplicables, copiar el borrador a `rights.json` y fijar su estado `approved` con identidad y alcance reales. El JSON completado por el actor puede ser su propia evidencia de desarrollo; no se exige una firma/custodia externa para este modo.

Después, **editar los flags dentro de la celda** en este orden y volver a ejecutarla:

1. `RECORD_OWNER_EVIDENCE=True`: emite el recibo inmutable de exposición.
2. `ACQUIRE_MARKET_DEVELOPMENT=True`: activa el colector acotado y verifica fuentes/hashes.
3. `PREPARE_MARKET_DEVELOPMENT=True`: prepara las cuatro particiones de desarrollo.

No activar registro de nuevo si el recibo ya existe: conservar su versión; para una corrección real usar un directorio/versionado nuevo antes de entrenar. Descargar precios no demuestra derechos o reserva. WIKI terminó en 2018: si no hay reserva histórica auténtica, tendrá que sustituirse antes del protocolo final por una fuente capaz de proporcionar reserva adecuada. El notebook no inventa una reserva futura WIKI.

Llamadas reales utilizadas, desde el directorio fuente y con el Python aislado:

```bash
python -m engine.data record-owner-evidence --form private/F01-owner-development.json --evidence private/F01-owner-development.json --output private/F01-exposure-development.json
python -m engine.data collect-development --config private/F01-acquisition.json --rights private/F01-rights.json --exposure private/F01-exposure-development.json --output private/F01-raw-v1 --enable-market-acquisition
python -m engine.data prepare-development --manifest private/F01-raw-v1/raw_manifest.json --split-config private/F01-development-split.json --rights private/F01-rights.json --exposure private/F01-exposure-development.json --output private/F01-development-v1
```

Las rutas efectivas del notebook se generan bajo `WORK/private`, no son las abreviaturas del ejemplo. No hay flag global que active silenciosamente el resto de la campaña.

## 3. Verificar sin entrenar

La celda ejecuta `check`; si existen los cuatro paneles, valida por separado `fit`, `selection`, `planning` y `calibration`. Rechaza `software_fixture` como admisión real y rechaza test en desarrollo. Si faltan admisiones, imprime cuáles, sin crear un caso entrenable. El dictamen metodológico vigente se lee del protocolo/manifiesto publicado, no de un nombre V2 fijo.

## 4. Ejecutar o recuperar planificación

Editar `RUN_PLANNING_TRAINING=True`. La celda prepara el caso si falta y llama:

```python
run_cli(["run", *remote_args(PLANNING_CASE, panels=DEVELOPMENT_MAP), "--activate"], remote=True)
```

Son **340 trabajos**: 320 ajustes neuronales (16×4×5), cuatro ARIMA y 16 trayectorias analíticas. Usa las cinco últimas semillas del banco original; las primeras 4091 quedan disponibles para el prefijo final. Verifica el respaldo inicial antes de optimizar, respalda checkpoints intermedios y cada trabajo, y omite sólo completos verificados. Si falla GitHub, se detienen nuevos ajustes y se conservan fallos/estado local.

Tras completar todos, editar `EXPORT_PLANNING_PREDICTIONS=True` para obtener CSV y recibo completos. Después `RUN_DIMENSIONING=True` ejecuta el planificador contra esas predicciones y los archivos de evidencia; no abre test. El plan utiliza la familia efectiva y calcula cotas de potencia, con semilla de validación distinta de la selección de tamaño. Un resultado `NOT_DIMENSIONED` conserva R/T/B nulos: se revisa su razón concreta, no se etiqueta como un diseño suficiente.

El presupuesto neuronal nominal de planificación es **10 h 40 min** a 120 s por ajuste; ARIMA hasta ocho minutos nominales más. Se excluyen de ese nominal instalación, preprocesamiento, predicción, análisis estadístico y transporte. No es consumo medido ni un límite duro de reloj. Consultar el límite explícito de dimensionamiento en `configs/power_design.json` antes de activar su Monte Carlo.

## 5. Cerrar el protocolo completo

El autor/custodio aporta, por panel, `private/inputs/F0x/reserve-inventory.json`: bytes de un **inventario auténtico e independiente de exposición/reserva**, existente antes del cierre del protocolo, con procedencia y alcance reales. No es el formulario final que contendrá el hash del propio protocolo: eso crearía un ciclo imposible de hashes. Se preserva el archivo sin modificarlo después del cierre. Tampoco es una declaración de reserva firmada por la IA.

La celda prepara `private/authentic-test-spec-DRAFT-<hash>.json`, con un nombre nuevo cuando cambian sus insumos; no sobrescribe borradores anteriores o documentos del autor. Completa automáticamente T si ya existe potencia válida y `source_identity_sha256` desde la **admisión real de desarrollo**, mediante el helper del controlador sobre proveedor/instrumento/calendario. El hash del inventario independiente se llena cuando ese archivo auténtico existe. Con cero admisiones, esos campos permanecen nulos.

El autor completa el borrador vigente con fechas y modo de reserva auténticos y lo copia a `private/authentic-test-spec.json`. No completa esos valores a partir de resultados finales. La celda vuelve a verificar los hashes contra las admisiones y los archivos de inventario existentes antes de cerrar. La validación metodológica debe corresponder exactamente a los tamaños y reglas que selecciona la potencia; un resultado sintético no habilita automáticamente otros tamaños o procesos.

Editar `FREEZE_FINAL_PROTOCOL=True` únicamente con esas entradas. `freeze-protocol` verifica informes, réplicas y cotas ejecutadas, enlaza el SHA de adecuación y los escenarios de estrés prefijados, y escribe un **nuevo** `protocol-final.json`. Una bandera `admitted:true` aislada no satisface el contrato. El protocolo debe quedar publicado y leído de vuelta antes del primer ajuste final; el controlador lo incluye en su barrera GitHub inicial.

La potencia exige ≥0,90 por contraste y conjunta reportada, no una conjunta universalmente ≥0,90. El mínimo de diez semillas no es un cálculo de suficiencia. Las fechas de distintos paneles no necesitan coincidir: la planificación conserva el acoplamiento observado y etiqueta los supuestos/cotas de la conjunta cuando sus calendarios son disjuntos.

## 6. Entrenar y sellar la familia final

Editar `RUN_FINAL_TRAINING=True`. Se prepara un caso con **otra rama/prefijo** y el protocolo final inmutable; se entrena el prefijo R del banco congelado. El número es `64*R+20` trabajos, con `64*R` ajustes neuronales. Para R=10 ilustrativo: 660 trabajos y **21 h 20 min neuronales nominales**, no suficiencia demostrada o gasto medido. El planificador puede exigir R mayor.

Repetir la etapa reanuda incompletos con modelo/Adam/RNG/época/selección y verifica completos. No sustituye el mejor estado por `latest`, ni un fallo por una semilla de reemplazo. Después editar `SEAL_FINAL_TRAINING=True`: verifica todos los seleccionados, emite predicciones de ajuste/selección/calibración y congela calibradores. El sello y sus hashes se guardan y releen remotamente antes de preparar test.

Tras el sello, la celda prepara por panel `final-split-DRAFT.json`, `owner-final-DRAFT.json` y `final-authorization-DRAFT.json`. Usa únicamente el calendario y los metadatos ya congelados; no lee test. Incluye los hashes verificables, `development_bar_index_origin` y `reserve_inventory={path,sha256}`, pero conserva identidad/evidencia vacías y autorización falsa. El actor completa y copia cada borrador al nombre sin `-DRAFT`; el código no concede autorización por él.

## 7. Preparar y consultar una única evaluación

La reserva histórica requiere sello/accesos auténticos; la futura requiere que **todo** soporte local ocurra después del cierre completo. El custodio suministra un manifiesto local `scope="reserved_final"`, bound al protocolo. Incluye **todo el prefijo histórico conocido desde `admission.provenance.bar_index_origin`**, más la extensión reservada, para mantener los offsets causales ARIMA. No descargar un sufijo y volver a numerarlo desde cero. El contrato `bar_index_convention="zero_based_complete_observed_calendar_sessions_v1"` y el origen se conservan; la autorización declara el origen observado de desarrollo y el preparador lo comprueba antes de construir etiquetas.

El borrador del split usa `raw_start=bar_index_origin`, `raw_end=última fecha objetivo`, `target_start/target_end` del protocolo y `raw_bounds.test=[índice_global(primer_target)-14, índice_global(último_target)]`. Calcula esos índices con el calendario completo, sin precios. El conteo efectivo aún debe satisfacer T y las exclusiones prefijadas por soporte/splits; el borrador no acredita integridad de archivos no adquiridos. No descargar toda la reserva primero para después llamarla oculta.

Completar `owner-final.json`, evidencia real del custodio, `final-authorization.json` autorizado y `final-split.json` según [DATA_ADMISSION.md](DATA_ADMISSION.md). El formulario final conserva `exposure_inventory_sha256` del inventario independiente; la autorización `reserve_inventory` enlaza sus bytes existentes. El controlador compara esos hashes y la identidad real de la fuente con el test-spec congelado antes de abrir test. No sustituye inventarios por el hash de un formulario posterior. Activar en orden:

1. `REGISTER_FINAL_EXPOSURE=True`, para su recibo auténtico.
2. `PREPARE_RESERVED_TEST=True`, sólo después del sello de entrenamiento, para un directorio **nuevo** test-only.
3. `OPEN_RESERVED_EVALUATION=True`, para consulta nominal y los escenarios de ruido/faltantes fijados antes de test.
4. `ASSESS_SAVED_RESULTS=True`, para analizar los CSV guardados, persistir `metrics/assessment.json` y `analysis_receipt.json` y **sincronizarlo inmediatamente con GitHub**.
5. `EXPORT_SAVED_RESULTS=True`, para empaquetar derivados/recibos ya guardados fuera del caso.

La evaluación escribe un marcador remoto previo por trabajo/escenario y una cache con resultado. Un resultado completo se reutiliza. Un marcador pendiente sin cache detiene la recuperación automática: no repite una consulta cuyo estado quedó incierto. Assessment no ejecuta forwards, no inventa stress CSV y conserva inferioridad/fallos. `SYNC_SAVED_RESULTS=True` puede sincronizar un análisis ya guardado sin repetirlo; esa ruta no requiere abrir los arrays privados ni pasar el mapa test-only como si fuera desarrollo.

```python
run_cli(["assess", "--case", str(FINAL_CASE), "--protocol", str(FINAL_PROTOCOL)])
run_cli(["sync", *remote_args(FINAL_CASE, final=True)], remote=True)
run_cli(["export", "--case", str(FINAL_CASE), "--output", str(EXPORT_ZIP)])
```

Un CSV o ZIP exportado no prueba por sí solo superioridad, adecuación o aceptación académica. La decisión exige todas las condiciones de la guía y resultados favorables reales; no cambia reglas o candidata tras verlos.

## 8. Comprobar GitHub y recuperar sin Drive

Elegir `TARGET_STAGE="planning"` o `"final"`. `VERIFY_REMOTE_GITHUB=True` lee/autentica el respaldo cifrado real con las credenciales privadas y comprueba su identidad. El estado local es sólo un listado y se distingue explícitamente de esa comprobación.

Para una VM nueva: ejecutar etapas 1–2 sin activaciones, conservar la misma versión/commit, `RUN_LABEL`, rama, prefijo y clave; cambiar `RESTORE_FROM_GITHUB=True`. Se restaura en un **directorio nuevo**, se verifican archivos, y se reacquieren por separado los datos privados con hashes idénticos. Señalar `PLANNING_CASE` o `FINAL_CASE` al destino restaurado y actualizar el mapa de rutas privadas, sin editar el plan para cambiar identidades. El siguiente `run` vuelve a comprobar esas identidades y sólo reanuda lo permitido.

El backup no contiene barras raw ni archivos NPZ de entradas. Conservar los formularios/evidencias privados auténticos y el modo de recuperar exactamente las fuentes. Si una fuente cambió, detenerse: una descarga parecida no reemplaza el hash original. No reutilizar V5 como identidad de esta campaña nueva.

## Verificaciones ejecutadas de este notebook

`tests/test_notebook.py` valida ocho etapas, sintaxis de todas las celdas y programas internos, pins, flags sensibles desactivados, separación de ramas, bootstrap HTTP simulado y ejecución de todas las celdas por defecto. La única preparación de datos de ese recorrido es **calendario/metadatos**, con subproceso real y sin observaciones. Verifica además que la identidad del actor continúa vacía, no se crean admisiones/casos/raw, no se solicitan secretos por defecto y el análisis guardado tiene sincronización. Las pruebas adicionales ejecutan el generador de borradores finales con metadatos artificiales: comprueban inventario independiente, origen completo y rechazo de fuente/inventario alterados, sin cargar etiquetas. Las pruebas del bootstrap, controlador, trabajador y almacenamiento cubren por separado sus rutas reales con fixtures.

La asistencia de IA en implementación, revisión y redacción se declara. Las pruebas de software y las simulaciones metodológicas no se presentan como entrenamiento de mercado, reproducción humana o acreditación de R3.
