# Hipótesis y adecuación: resultados ejecutados y desarrollo cripto preparado

Esta entrega mejora el componente del 15 % mediante análisis reales de
predicciones ya guardadas. Conserva `input-only-overlap-mse-1`, los 796 trabajos
aditivos y los pilotos V3/V4/V5, sus resultados adversos y sus decisiones.
No hubo entrenamiento de pronóstico, inferencia, adquisición de mercado ni
apertura de reserva en estos análisis. Las pruebas del software sí usan números
generados y pequeños pasos de optimizador, identificados como fixtures.
Implementación, análisis y revisión asistidos por IA declarados; no hay firma,
identidad autenticada, defensa observada ni reproducción humana inventadas.

El identificador de carpeta usa fecha UTC22; las operaciones se realizaron
durante el21 de noviembre de2026 en Bogotá. Los recibos conservan sus UTC reales.

## Evidencia nueva ejecutada

| Operación | Resultado y alcance |
|---|---|
| Adecuación de la candidata | 1.760 puntos/32 trayectorias/55 fechas expuestas. 128 diagnósticos con Holm128 y64 intervalos puntuales de sesgo HAC. 27 rechazos MZ, incluidos2/5 de input-only; los otros no rechazos no certifican adecuación. |
| IC temporales del funcional primario | Media de RMSE individuales, fechas pareadas y cinco semillas fijas. Frente a global, ΔRMSE=−0,00005821987; IC95 bloque7[−0,00013727537;+0,00002937367] y bloque14[−0,00011955731;+0,00001910637], ambos incluyen cero. |
| MCS secundario | Tmax estándar preservado, ocho recetas, B1999 y bloques7/14, confianza90 %. Ambos retienen siete recetas, incluida input-only; excluyen sólo floor_only. No hay ganador único ni inferencia confirmatoria de media-RMSE/familia21. |
| Método y potencia | Se verificaron los288 archivos de A y sus ledgers, sin nuevas simulaciones de calibración. Se documentaron dos brechas confirmatorias concretas y un contrato ejecutable de entradas; sin planificación real, R/T/B finales, potencia y MDE son nulos. |
| Guía y evaluación | 16 requisitos solapados,25 citas y22 artefactos contrastados; revaloración independiente2,8→3,0 por los nuevos diagnósticos/IC, no por documentos o software. |

Los planes y códigos retrospectivos se fijaron localmente antes de esos cálculos,
pero después de conocer el piloto: **no son preregistros ni confirmación**.
Hubo128 OLS auxiliares y11.996 sorteos temporales para análisis de predicciones,
contabilizados aparte de entrenamientos neuronales y de validación Monte Carlo
de métodos. Los IC son puntuales y condicionales, sin cobertura finita validada
ni incertidumbre poblacional completa de inicializaciones.

Consultar [retrospective](retrospective/README.md), [MCS](mcs/README.md),
[métodos/potencia](methods_power/README.md) y [matriz de guía](guide_audit/README.md).
Los originales V5 se verifican por SHA y se reutilizan desde el paquete hermano
`e3_input_only_candidate_20261121`; no se incluye otra copia del master.

## Dos brechas que no se ocultan con una activación

1. `PRIMARY_ENDPOINT_ALIGNMENT_NOT_ESTABLISHED`: raíz de MSE agregado y media
   de RMSE individuales son funcionales diferentes. El contraejemplo ejecutado
   invierte sus ordenaciones; p del MSE no se convierte en p de media-RMSE.
2. `JOINT_DECISION_VALIDATION_NOT_ESTABLISHED`: las cotas marginales existentes
   no certifican automáticamente la decisión principal más dos de tres paneles.
   La cota conservadora2q≈0,09572 **no es una inflación observada**, sino un
   límite insuficiente para deducir5 % conjunto sólo de esas cotas.

La versión nueva bloquea los comandos finales en los entrypoints CLI y en el
cuaderno antes de credenciales o datos. Editar un bool no habilita confirmación;
los APIs internos mantienen pruebas explícitas de software. No se cambia alfa,
métrica o regla retrospectivamente. Admisión y planificación exploratorias
continúan disponibles con activaciones separadas.

## Fuentes cripto propuestas y adaptador versionado

| Panel | Fuente, instrumento y desarrollo | Barras / fechas planning esperadas |
|---|---|---:|
| F00 | Binance Vision Spot BTCUSDT,2017-08-17..2023-12-31 | 2328 /452 |
| F01 | Binance Vision USD-M BTCUSDT perpetuo,2020..2022 | 1096 /205 |
| F02 | Binance Vision COIN-M BTCUSD_PERP,2021..2022 | 730 /133 |
| F03 | Binance Vision USD-M ETHUSDT perpetuo,2021..2023 | 1095 /205 |

Se propone dominio económico Spot/derivado financiado y tamaño cronológico como
dos ejes revisables; no se cuenta un ticker como otro eje ni volatilidad como
ruido. Los paneles comparten factores, venue y calendarios; la generalización y
las pruebas entre conjuntos conservan sus límites. El historial es desarrollo
expuesto, con resultados anteriores conocidos donde corresponda; no es una
reserva final. La admisión real sigue en **0/4**.

[Decisión de fuentes](data/DATA_SELECTION_DECISION.md),
[calendarios](data/data_panels_v2.json) y [procedencia](data/DATA_EXTENSION_PROVENANCE.json)
conservan la propuesta inicial y el ajuste de F03 por incertidumbre de listing,
antes de adquirir precios nuevos. No se exige Nasdaq. La propuesta de acciones
del motor A permanece histórica y no se sobrescribe.

La ingestión usa meses completos contenidos en desarrollo y archivos diarios
en los bordes:187 ZIP+187 CHECKSUM=374 solicitudes nominales sin reintentos,
frente a10.498 diarias. Es aritmética, no coste/tiempo medido ni disponibilidad
verificada del proveedor. COIN-M usa volumen base del campo7; Spot/USD-M campo5.
Se validan instrumento, mercado, URL, nombre, frecuencia, fechas, checksum y
calendario; se conservan estados parciales y las identidades originales al
recuperar. Los dos defectos iniciales hallados por revisión independiente y su
corrección están documentados; no se relajan hashes ni se renombran datos.

Binance conserva sus términos de datos CC BY-NC-SA4.0 y condiciones específicas,
con evidencia primaria fijada. El MIT del README no licencia los datos.
Uso académico no comercial, atribución y condiciones de derivados requieren la
aceptación auténtica del actor; raw y secretos permanecen privados. Los permisos
de cada upstream de código se conservan, sin una licencia global nueva.

## Ejecución siguiente y presupuesto

[Abrir el cuaderno fijado en Colab](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/b3d73a750da2643ecfc7b2802904fd54c1efd4e8/experiments/e3_hypothesis_adequacy_20261122/colab/E3_Hipotesis_Adecuacion_GitHub.ipynb).
Fuente runtime: `db74afd0763596f71bcbafb4b9150a44a2e83339`; manifest SHA256
`2c7dc052d2a53f212688340b50b65bed5fa367ff16fb5ed0a2474d417696715b`. El cuaderno y el starter verifican bootstrap,
manifest y cada archivo antes de ejecutar. El motor A `c54a2fd...` permanece
intacto; candidata, banco, entrenador, recetas y métodos conservan sus hashes.
El diseño de potencia original se conserva y el overlay activo sólo actualiza
cuatro metadatos de las fuentes, sin modificar los ajustes científicos.

**Primera acción:** ejecutar la instalación y las definiciones del cuaderno;
en la etapa2 activar sólo `PREPARE_CRYPTO_DEVELOPMENT=True`, contestar la
declaración real y ejecutar esa celda. Luego ejecutar etapa3 para verificar las
cuatro admisiones. No requiere Drive, Nasdaq ni GPU específica.

La planificación de etapa4 exige otra activación y escribir `PLANIFICAR 340`:
340 trabajos,320 ajustes neuronales+4 ARIMA+16 trayectorias analíticas,
cinco semillas neuronales preasignadas. El nominal neuronal es10h40,
excluyendo ARIMA, IO, predicción, potencia, transferencias e interrupciones.
No se inició durante esta tarea. Cada checkpoint/trabajo se respalda cifrado
y verifica en GitHub; se conserva la clave independiente para reanudar.

El dimensionamiento necesita CSV/recibos auténticos de21 sistemas×4 paneles,
cinco semillas por red y al menos80 fechas comunes por panel:≥27.200 filas.
Debe estimar potencia≥0,90 por contraste, conjunta reportada, MDE y≥10 semillas
finales o más según el cálculo. El R10/T768/B8191 calibrado no se adopta como
diseño final por estar escrito en un JSON. El cálculo MSE disponible es
condicional; no resuelve las dos brechas del funcional/regla conjunta.

Las Tablas A/B completas necesitan predicciones de ajuste/selección del
checkpoint elegido; cobertura/PIT necesitan cuantiles o ley predictiva propia.
No se reconstruyen de curvas de batches ni se heredan de la aditiva.

## Evaluación trazable

[Revaloración](guide_audit/COMPONENT_REASSESSMENT.json): componente15 %,
2,8→3,0; aporte0,420→0,450; ponderado4,092→4,122(**4,12**). Pesos y otras
diez notas intactos; sustentación4,5 indicada, no observada. Con R1/R2 aceptadas
y R3 no acreditada, el tope mantiene4,00. Nota oficial sin asignar.
No se acredita R3 ni se convierte la retención en MCS en equivalencia.
