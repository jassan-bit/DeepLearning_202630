# Dashboard: cierre editorial de la evidencia existente

Actualización limitada a `app.py`, `pages/contexto.py`, `pages/modelos.py`, `MODEL_CARD.md` y `DATA_STATEMENT.md`. Los cinco archivos originales se cotejaron por hash de blob Git contra el commit publicado `29f9a26f1fd05484c6581ea6538ad7cb5a4781cb`; `base_verification.json` conserva el cotejo.

Los archivos listos para ensamblar están bajo `edited/`, con sus rutas de repositorio. `dashboard_editorial.patch` expresa el cambio completo. `original/` y `validation/` son material de trabajo local y no deben publicarse como una nueva copia del proyecto.

El catálogo conserva 17 sistemas, cinco ONNX y doce resultados precalculados, sus pesos, calibradores, metadatos y calendarios. El pie conserva el perfil histórico del 18 de noviembre; las fichas identifican el 19 de noviembre de 2026, America/Bogota, como actualización documental. La auditoría posterior de 796 trabajos/19 sistemas/cuatro conjuntos y el piloto V3 de 30 ajustes se enlazan como estudios separados. No se promueve una variante de V3 ni se cambia el modelo servido.

La variante completa `atom_crps` quedó 2,70 % peor en RMSE media que `atom_mse_only`, y no pasó la heurística de desarrollo. El enlace al veredicto queda fijado al commit de resultados `8eaa581bc34b615a8507c9e384a3e4be29084720`. Los 48 contrastes históricos conservan su ámbito y no se transfieren a otros pesos, conjuntos o variantes.

## Verificación

`test_results.log`: tres pruebas existentes de aplicación/router/healthcheck/contrato de carga inválida aprobaron. Se ejecutaron en una copia aislada del runtime; no se invocó entrenamiento ni predicción de mercado.

`verification.json`: se renderizaron ambos layouts mientras la creación de `onnxruntime.InferenceSession` estaba bloqueada, se verificó igualdad AST de 21 callbacks/helpers y se cotejaron 27 archivos de artefactos/bundle byte por byte. Las funciones de inferencia y todas las cifras científicas originales permanecen sin modificaciones.

En el repositorio ensamblado, la comprobación existente se reproduce con:

```bash
python -m unittest discover -s tests -p 'test_e3_dashboard_app.py' -v
```

Este recibo de comprobación local no constituye aceptación pública en Render. El propietario conserva la intervención de despliegue manual y el cliente de humo debe comprobar su versión pública efectiva.

Asistencia de IA declarada para revisión y edición; no se atribuye esta tarea a una reproducción humana independiente ni a una defensa del estudiante.
