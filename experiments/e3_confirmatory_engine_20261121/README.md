# Motor temporal R3: entrenamiento, recuperación y evaluación separada

Esta entrega implementa el flujo experimental de `input-only-overlap-mse-1` y sus **20 rivales**. Incluye entrenamiento real de las 16 redes, cinco métodos deterministas, admisión de datos, planificación, potencia, respaldo cifrado y evaluación con consulta única. Las pruebas ejecutadas usan datos artificiales; **no se adquirieron barras de mercado, no se entrenó con ellas y no se abrió una prueba reservada**.

La identidad de la candidata, su pérdida y su preprocesamiento se conservan. Los 796 trabajos aditivos, los pilotos V3/V4/V5 y sus decisiones adversas permanecen separados. Esta entrega es ingeniería y desarrollo metodológico; no acredita resultados confirmatorios ni aceptación académica de R3.

## Entradas y etapas reales

1. `bootstrap_colab.py` obtiene fuentes por commit y manifiesto SHA256, instala el entorno CPU aislado y ejecuta pruebas con fixtures. No requiere una GPU específica ni cambia Torch del kernel.
2. `engine.data` adquiere y verifica desarrollo **sólo tras activación**. Hay colectores para Binance, Stooq, Alpha Vantage y WIKI; la propuesta vigente usa BTCUSDT, IBM, MSFT y XOM. Cada panel conserva proveedor, derechos, exposición, hashes, calendario y soportes de ventanas. Los formularios requieren datos auténticos del autor.
3. `engine.cli check` y `prepare` comprueban identidades sin entrenar. Un caso de desarrollo no exige circularmente potencia o reserva final.
4. `engine.controller.run_case` ejecuta la familia de planificación con cinco semillas separadas del prefijo final. Respaldar y leer el estado remoto es obligatorio antes del primer ajuste y entre trabajos. `predict-planning` emite CSV y recibos utilizables por el planificador.
5. `engine.power.dimension` utiliza esas predicciones guardadas. Exige potencia ≥0,90 por contraste y reporta la conjunta, sus supuestos y cotas. No inventa R/T/B sin insumos. `freeze-protocol` crea un archivo nuevo y verifica evidencia metodológica y de potencia antes del cierre.
6. El mismo trabajador ejecuta/reanuda la familia final con un mínimo de diez semillas o el tamaño mayor dimensionado. Sólo omite trabajos completos cuya identidad y checkpoint se verifican.
7. `seal-training` guarda estados seleccionados y calibradores **antes** de test. La admisión final vincula la reserva auténtica, el protocolo y ese manifiesto. `evaluate` deja un marcador durable antes de cada consulta: reutiliza caches completas y se detiene si una consulta quedó incierta. `assess` lee únicamente predicciones guardadas y comprueba las seis condiciones de R3; no entrena o consulta modelos.
8. `status`, `restore` y `export` permiten comprobar GitHub, recuperar el caso y exportar los mismos resultados sin volver a consultar test.

El notebook y [COLAB_RUNBOOK.md](docs/COLAB_RUNBOOK.md) proporcionan las llamadas concretas. La instalación/pruebas no activa adquisición, planificación, entrenamiento final o test. Las credenciales llegan por Colab Secrets/entrada privada y stdin; no se guardan en el notebook, argumentos o fuentes. El progreso aparece por trabajo y respaldo, con redacción antes de mostrar o conservar salida.

## Qué se verifica y qué no

- [Modelos y procedencia](docs/MODEL_PROVENANCE.md): implementaciones históricas atribuibles, fuentes primarias y licencias preservadas; verificación de 104 archivos de proveedor.
- [Entrenamiento](docs/TRAINING_AND_RECOVERY.md): mejor checkpoint, Adam/RNG, fases auxiliares, cómputo nominal/medido/incierto y detención ante fallos de guardado.
- [GitHub](docs/GITHUB_STORAGE.md): AES-GCM, fragmentos cifrados, escritura con lease, lectura remota y restauración. Las pruebas Git REST usan un servidor simulado; no se atribuyen a un respaldo de mercado efectivamente publicado. Las predicciones completas permanecen en fragmentos cifrados y en la exportación íntegra. El ZIP público de checkpoints contiene metadatos/recibos acotados e inventario por ruta, hash y tamaño: no intenta meter todos los CSV en un único blob limitado ni selecciona filas por resultado.
- [Datos](docs/DATA_ADMISSION.md): desarrollo expuesto separado de reserva final, purgas por barras observadas y etiquetas causales. WIKI tiene derechos públicos documentados, pero terminó en 2018: requiere una reserva histórica auténtica. Si no existe, esos paneles no pueden convertirse en confirmación por una declaración retrospectiva.
- [Batería temporal](docs/TEMPORAL_BATTERY.md): estimandos, supuestos, planes previos y resultados sintéticos, incluidos los fallos. El informe vigente se conserva en `qa/statistical_validation/`; su alcance no se generaliza automáticamente a mercados o tamaños distintos.
- [Potencia y presupuesto](docs/POWER_AND_BUDGET.md): fechas coincidentes comparten bloques; calendarios disjuntos no tienen covarianzas observadas inventadas. La potencia conjunta del DGP condicionado se distingue de cotas sin identificar ese acoplamiento.
- [Adecuación y robustez](docs/ADEQUACY_AND_ROBUSTNESS.md): intervalos calibrados antes de test, diagnósticos de residuos y sensibilidades guardadas. Los átomos entrenados por MSE no se presentan como una distribución predictiva calibrada.

## Presupuesto revisable

La planificación de cuatro paneles usa **320 ajustes neuronales**, cuatro ARIMA y 16 trayectorias analíticas: 340 trabajos, no 340 ajustes neuronales. A 120 s nominales por ajuste neuronal son **10 h 40 min**; ARIMA aporta hasta ocho minutos nominales más. El tiempo de instalación, selección, predicción, validación estadística y transporte se registra aparte. No son consumo medido ni un límite duro de reloj.

La final con R=10 ilustrativo usa **640 ajustes neuronales**, cuatro ARIMA y 16 trayectorias analíticas: 660 trabajos y **21 h 20 min neuronales nominales**. R=10 no se declara suficiente: debe salir del dimensionamiento. La hipótesis de almacenamiento es 64 MiB por ajuste neuronal para estados de recuperación/selección, más predicciones y versiones cifradas conservadas por Git. El historial remoto puede multiplicar esa cifra. Precio monetario y energía no se inventan.

## Integridad y decisiones

`configs/CANDIDATE_IDENTITY.json` y `configs/FINAL_SEED_BANK.json` conservan sus hashes previos. Planificación usa las últimas cinco semillas; final usa un prefijo independiente de las primeras 4091, nunca reemplazos por rendimiento. El nuevo estimando temporal y las reglas de adecuación se versionan prospectivamente; no convierten calendarios examinados o registros posteriores en preregistros.

Se declara asistencia de IA en implementación, revisión matemática, pruebas y redacción. Las fuentes/licencias de terceros y la autoría del proyecto figuran en el inventario de procedencia. R1/R2 y los experimentos de versiones anteriores no se transfieren a una candidata por cambiar el nombre.
