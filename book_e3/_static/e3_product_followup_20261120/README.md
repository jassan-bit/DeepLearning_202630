# Perfil histórico suplementario de catorce predictores

El perfil `e3-historical14-supplement-20261120-v1` añade una vista `/perfil-14` y una API separada a la aplicación original. El `Dockerfile` predeterminado, `app.py`, las páginas congeladas, los pesos, sus metadatos y los cuantiles no se modifican. Las tres vistas originales mantienen sus cinco predictores. El perfil suplementario ofrece los catorce representantes neuronales históricos —trece E2 y la aditiva E3—; V3 queda deshabilitada.

Los nueve ONNX ya publicados se ensamblan durante la construcción desde sus partes verificadas, con el constructor existente. No se vuelven a exportar. El perfil vincula cada binario, metadato y calibrador por su hash; preserva la semilla representativa y la transformación original. Todos los intervalos se inhiben para objetivos anteriores o coincidentes con `2024-12-30`. Se identifica explícitamente el calibrador posterior de E3 y el original de E2. No se hereda el calibrador de la aditiva para V3.

Los cuatro metadatos estándar no contenían descriptor OOD ni referencia de oclusión y los cinco temporales carecían de referencia raw. El adaptador reutiliza el descriptor congelado E2 de ajuste después de comprobar igualdad exacta de media, desviación, número y fin de ajuste y escala de etiqueta. La revisión independiente automática verificó también igualdad con la cohorte original de 655 ventanas; no usa calibración ni prueba para escoger la referencia. Los archivos originales no se reescriben.

## Comprobaciones ejecutadas

- `contract_review.json`: identidad y contrato de los nueve aceptados, calibradores y descriptor de ajuste.
- `container_checks_initial.json`: construcción limpia con Python fijado, 40 ruedas bloqueadas, red deshabilitada y ONNX aceptados. El nuevo adaptador reprodujo 5.949 salidas históricas: error máximo `3.099255844185578e-7` sigma7, frente a tolerancia `1e-6`. Verificó catorce contratos API, intervalos y guardas temporales, OOD y errores. RSS máximo del proceso de prueba: `133.9375 MiB`.
- `local_acceptance.json`: ejecución HTTP y Chromium reales del código final, catorce predictores, catorce guardas temporales, catorce entradas OOD, diez errores seguros, tres rutas originales y callback EDA. Error máximo en el caso fijo: `1.5144635804112405e-8` sigma7. Se conserva el recibo inicial anterior a la aclaración del tipo de calibrador; no se usa como evidencia del texto final.
- `original_five_client.json`: el cliente de cinco modelos existente pasó sin cambios sobre el contenedor suplementario local, incluidas sus explicaciones e intervalo E3.
- `local_concurrency.json`: catorce peticiones bajo cuatro clientes simultáneos, atendidas por un trabajador síncrono; ninguna mezcla de identidades o salidas.
- `resource_envelope.json`: cgroup máximo observado `198.5 MiB` con límite `512 MiB`, un CPU, un trabajador y un hilo, UID 10001, raíz de sólo lectura y capacidades retiradas. Son dos sesiones ORT como máximo: una por la vista original y otra por la suplementaria. No se declara una única sesión para toda la aplicación ni RAM incremental de ORT.
- `screenshots/`: capturas de la vista original, suplemento en escritorio y móvil.

Estas ejecuciones son comprobaciones de ingeniería sobre fechas ya guardadas y entradas adversas explícitas. Hubo cero ajustes, recalibraciones, reexportaciones, nuevas fechas científicas o uso de GPU. La prueba de concurrencia local no certifica capacidad o disponibilidad de Render.

## Identidad y límites de la ejecución local

`execution_context.json` registra los hashes del código. La identidad `APP_SOURCE_COMMIT=7db804...` se inyectó únicamente como **identidad local del baseline** para verificar la guarda del cliente. El código suplementario era un overlay nuevo y no se afirma que estuviera en ese commit o desplegado en Render.

La primera imagen de producción se construyó exactamente con `Dockerfile.profile14` y pasó las pruebas numéricas. Una modificación posterior aclaró el tipo de intervalo mostrado. La falta de espacio del host impidió reconstruir localmente esa imagen final; el código final se montó en sólo lectura y pasó nuevamente el navegador/API. El workflow `check-e3-profile14.yml` construye el repositorio completo y ambos clientes contra el SHA real del commit publicado; su resultado debe comprobarse antes de desplegar. `engineering_attempts.json` conserva los fallos de DNS, permisos y espacio con sus resoluciones; no se ocultan ni se cuentan como entrenamientos.

## Despliegue mínimo preparado

1. Verificar que el workflow **E3 optional historical14 profile, CPU only** del commit que publique este paquete terminó correctamente. Si falla, conservar su recibo y corregir la causa antes del despliegue.
2. En el servicio existente de Render, seleccionar ese commit revisado, cambiar únicamente **Dockerfile Path** a `./Dockerfile.profile14` y mantener contexto raíz, `/healthz`, auto-deploy deshabilitado y un trabajador/hilo. El Blueprint optativo es `render.profile14.yaml`. No crear otro servicio si se pretende actualizar el existente.
3. Ejecutar Manual Deploy del commit concreto. No introducir manualmente un SHA ajeno en el servicio: `/healthz` debe informar `source_commit` a partir de `RENDER_GIT_COMMIT` y anunciar `profile_predictors:14`, `original_UI_predictors:5`, `V3_enabled:false`.
4. Repetir los dos clientes con el SHA realmente informado y conservar recibos y capturas. El libro contiene el commit concreto que corresponda a la publicación final; este documento evita anticipar su identificador antes de crearlo.

```bash
python experiments/e3_cierre_integral_20261118/acceptance/smoke_render_v2.py \
  --url https://deeplearning-e3-dashboard.onrender.com \
  --deployment-commit SHA_REAL_DEL_DESPLIEGUE \
  --output /tmp/render-five.json --screenshots-dir /tmp/render-five-captures

python experiments/e3_product_followup_20261120/smoke_profile14.py \
  --url https://deeplearning-e3-dashboard.onrender.com \
  --expected-sha SHA_REAL_DEL_DESPLIEGUE \
  --output /tmp/render-fourteen.json --screenshots-dir /tmp/render-fourteen-captures
```

La CLI exige un SHA completo; se detiene antes de predecir si la identidad falta o no coincide. En CI utiliza el Chromium fijado de Playwright; `--chromium-path` es opcional para un navegador de desarrollo. Una publicación del libro no sustituye este despliegue ni la aceptación pública. Si sólo se desea conservar los cinco predictores originales, seguir usando `Dockerfile` y `app:server`; el suplemento es optativo.

## Clásicos y ámbito científico

Persistence necesita los ocho cierres hasta t−1 para reconstruir la última sigma7 conocida; siete barras sólo proporcionan seis retornos. SeasonalNaive necesita la etiqueta de lag siete: catorce barras si recibe historia diaria continua. ARIMA necesita su estado original, con parámetros congelados, asimilado secuencialmente hasta t−1. No se aproximan con el contrato de siete barras ni se vuelven a ajustar. Sus resultados históricos siguen disponibles en el catálogo de diecisiete sistemas.

Este trabajo no modifica el dictamen de R3, no valida V3, no convierte una exportación en novedad ni acredita reproducción humana. Se declara asistencia de IA para implementación, revisión y pruebas automáticas.
