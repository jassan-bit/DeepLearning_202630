# 1 Introducción

Este proyecto pronostica la volatilidad diaria de Bitcoin, BTCUSDT de Binance Spot. Al inicio del día $t$ utiliza las siete barras OHLCV anteriores para estimar la desviación poblacional de siete retornos logarítmicos, sin anualizar y con horizonte de un día:

$$
r_t=\log(C_t/C_{t-1}),\qquad
\bar r_t=\frac17\sum_{j=0}^{6}r_{t-j},\qquad
\sigma_7(t)=\sqrt{\frac17\sum_{j=0}^{6}(r_{t-j}-\bar r_t)^2}.
$$

La etiqueta requiere ocho cierres y contiene el retorno del día objetivo, aún desconocido al emitir el pronóstico. Las entradas son las siete barras de $[t-7,t)$, ordenadas de antigua a reciente, con cinco canales. El [contrato de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/17_contrato_sincronizado.html) define `relative_window`, escaladores de entrenamiento, disponibilidad y máscaras.

## 1.1 Antecedentes y motivación

E1 estableció las referencias neuronales. E2 ejecutó diez candidatos, tres referencias neuronales y tres comparadores estadísticos: **16 sistemas sobre 608 fechas conocidas**, con seis semillas por red. El LSTM-autoencoder obtuvo la menor RMSE media observada, **0,00380343**. Las veinte comparaciones primarias no demostraron la mejora práctica registrada tras Holm; las ampliaciones posteriores conservan esa conclusión.

E3 debe convertir los artefactos en un producto predictivo y desarrollar una contribución neuronal defendible. La combinación CNN-LSTM-atención no acredita por sí sola originalidad. El candidato actual incorpora penalización de discordancia temporal y saturación radial; su novedad y su ventaja predictiva siguen por comprobar.

## 1.2 Objetivos

1. Resolver y documentar las correcciones aplicables de E2, con trazabilidad e impacto en resultados.
2. Seleccionar configuraciones desde validación y preparar artefactos CPU con metadatos y paridad verificada.
3. Desarrollar el candidato desde su formulación, demostrar una propiedad y comprobar su implementación.
4. Evaluarlo con comparación justa, protocolo previo y evidencia adicional independiente.
5. Integrar resultados en Dash Plotly, Render y un informe reproducible.

## 1.3 Avances y límites actuales

Se cuenta con un inventario de 81 predictores, comprobación de 167 metadatos, selección documentada de configuraciones y tres exportaciones TorchScript sobre calibración histórica. El candidato tiene código, una demostración de estabilidad de su cabeza latente y **35 pruebas sobre datos artificiales**.

Estos avances no equivalen a una auditoría integral cerrada, un dashboard terminado ni superioridad del candidato. **R1, la defensa académica de R2 y R3 siguen abiertos.** Los periodos ya evaluados en E1/E2 continúan siendo conocidos; repetirlos no recupera una confirmación independiente.

## 1.4 Organización del libro

Las [secciones 2](02_estandar_profundidad.md) y [3](03_panorama_general.md) fijan el estándar y las dependencias. Las [secciones 4](04_correcciones_e2.md), [5](05_configuraciones_exportacion.md) y [6](06_modelo_original.md) reúnen los bloques A, B y C. La [sección 7](07_diseno_riguroso.md) organiza hipótesis, disciplina de diseño, teoría y ablaciones. Los [anexos](concordancia_secciones.md) conservan las páginas y evidencias anteriores.

El dashboard de Render aún no tiene URL pública. La publicación de este Jupyter Book no se presenta como ese dashboard. La asistencia de IA en código, teoría, revisión focal y documentación está declarada; el autor deberá comprender y defender la entrega.
