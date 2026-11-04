# Primera ejecución de desarrollo DL — 04/11/2026

El usuario autorizó ejecutar MLP, LSTM y CNN1D poco profunda y actualizar el Entregable 1. Este registro sustituye el aplazamiento operativo anterior para este ciclo, sin declarar cierre académico ni consultar la reserva de test.

Se completaron 96 corridas de búsqueda, 15 de piloto y 90 de comparación: 201 corridas con dos ajustes por corrida, 402 en total. El protocolo y los tres scripts se registraron en `3c3c481` antes del primer ajuste; la selección de D/configuraciones precede al piloto y la fijación de R precede a las semillas nuevas. Ese registro es posterior al EDA histórico.

- Entrenamiento estático: muestras admisibles de 2023, con corte interno y embargo D. Sin ajustar pesos o escaladores sobre etiquetas de 2024.
- Búsqueda: enero–junio de 2024, presupuesto idéntico para las tres familias; D común seleccionado de 7 días.
- Piloto y comparación: las mismas 184 fechas de julio–diciembre de 2024. Las 30 semillas nuevas por modelo no convierten ese periodo expuesto en test.
- Referencia operacional seleccionada: MLP. Líder neuronal descriptivo posterior: LSTM. Persistencia supera a las tres redes.

| Modelo | RMSE medio ± DE | IC95% entre semillas | Parámetros |
|---|---:|---:|---:|
| MLP | 0,015990 ± 0,002854 | [0,014924; 0,017056] | 593 |
| LSTM | 0,008903 ± 0,000485 | [0,008722; 0,009085] | 5.025 |
| CNN1D | 0,017587 ± 0,005192 | [0,015648; 0,019525] | 1.057 |
| Persistencia | 0,004822 | Sin intervalo entre semillas | 0 |

Los IC t condicionan al calendario fijo y a las condiciones del diseño. El análisis temporal usa pérdidas cuadradas por fecha para DM y bootstrap circular pareado de fechas (L14, sensibilidad L7/L28, 2.000 réplicas), condicionado a modelos almacenados, sin reentrenar. Los diagnósticos de residuos corresponden a la semilla 1001. Los intervalos marginales no se vuelven simultáneos por ajustar p-valores. MLP frente a CNN1D no se distingue entre semillas con Holm (p≈0,549), aunque la comparación temporal favorece MLP: no son el mismo alcance de incertidumbre.

El piloto estimó R=1.212/172/724 para MLP/LSTM/CNN1D, respectivamente, para detectar contra cero el efecto pequeño planificado. Se ejecutó el máximo registrado R=30; no se acredita potencia 0,80 para esa pregunta. La prueba de superar el 5% es distinta del cálculo de detección contra cero. No se fijó un margen de equivalencia o de mejora de un modelo avanzado frente a una red.

La verificación de artefactos comprobó 1.311 hashes por corrida, 36.792 predicciones y 7.248 filas de curvas. Cargar los checkpoints de las tres familias para la semilla 1001 reprodujo sus 184 predicciones exactamente. El paquete versionado conserva 1.512 archivos y 105 checkpoints; la búsqueda no guarda pesos. Las 25 pruebas de datos/modelos/estadística pasaron. Ninguna comprobación abrió raw test ni entrenó modelos adicionales.

Los cinco estudios del corpus y sus 81 valores de autores no cambian. MLP no lineal independiente y CNN1D simple siguen con cero apoyos aplicados verificados en ese corpus. La evidencia de desarrollo no cierra esta laguna ni el protocolo de test supervisado.

El [registro JSON](revision_dl_development_20261104.json) contiene fuentes y hashes; las auditorías separadas de [artefactos](revision_dl_artifacts_20261104.json) y [análisis](revision_dl_analysis_20261104.json) delimitan sus comprobaciones. Código y reproducción: `BUILD.md` y `results/dl_benchmark_20261104/README.md`.
