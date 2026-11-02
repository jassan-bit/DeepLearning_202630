# Decisiones pendientes y control de alcance

## Antes de cerrar la Sección 1

1. **Configuración confirmada:** modelado diario, horizonte principal de un día y entradas de 7/14/21/28 días; brutos de un minuto.
2. **Objetivos explorados:** volatilidad de retornos diarios con ventanas de 7/14/21/28 días y divisor poblacional. Fijar las configuraciones que se entrenarán y el protocolo para compararlas; no se eligió una ganadora por test.
3. **Covariables:** OHLCV diario auditado. Selección final y transformaciones pendientes; noticias no adquiridas ni incorporadas.
4. **Adquisición y EDA:** 75 ZIP verificados; EDA ejecutado exclusivamente en desarrollo 2023–2024. Conservar los brutos y manifiestos y materializar muestras supervisadas.
5. **Ventanas y validación:** folds de validación2024 implementados con `timeseries-cv`; mantener exclusiones y disponibilidad por fechas. Entrenamiento y test supervisado final pendientes.
6. **Preregistro:** congelar métricas y política de ceros para MAPE, rezago MASE, margen $\delta$, bloques inferenciales y test supervisado antes de evaluar modelos.
7. **Horizontes adicionales:** el principal confirmado es un día; cualquier salida adicional requiere declaración explícita.

## Más adelante, dentro del Entregable 1

El EDA de calidad, precios y retornos de desarrollo está ejecutado y documentado en la sección 2. Siguen **pendientes** el cierre de configuraciones de la variable objetivo, la búsqueda y síntesis de literatura, la selección de exactamente tres modelos simples, el piloto de al menos cinco semillas por modelo, el cálculo de potencia y número de corridas definitivas, y el entrenamiento, comparación e inferencia.

La guía contiene requisitos detallados adicionales. Esta lista no los reemplaza ni afirma cumplirlos. No se estima tiempo de entrenamiento sin datos, arquitectura, hardware y piloto medido.

## Registro de decisiones

| Decisión | Estado al 2026-11-02 |
|---|---|
| Activo | Confirmado: Bitcoin |
| Entregable inmediato | Entregable 1: secciones 1–7 documentadas como avance; EDA y folds ejecutados, modelos y benchmark pendientes |
| Repositorio | Confirmado: DeepLearning_202630, público por decisión del usuario |
| Uso de cierre para volatilidad | Confirmado |
| Pronóstico y objetivo | Diario, un día adelante; cuatro ventanas de volatilidad exploradas; configuraciones finales del benchmark pendientes |
| Calendario | Historia 2023; validación 2024; ajuste final elegible 2023–2024; test 01/01/2025–31/08/2026, reportando cada año por separado |
| Modelos finales | No seleccionados |
| Extensión a otras criptomonedas | No autorizada en el alcance actual |
| Datos, EDA y entrenamiento | Descarga oficial verificada; EDA de desarrollo ejecutado; objetivo definitivo y entrenamiento pendientes |
| Sitio público / GitHub Pages | HTML publicado en `gh-pages`; el usuario confirmó la activación de Pages. Verificación HTTP externa bloqueada por la red del entorno |
