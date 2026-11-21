# Admisión prospectiva: cuatro propuestas concretas, todavía no admitidas

Preparado con asistencia de IA el 20 de noviembre de 2026 en Bogotá
(recibos en UTC del día 21). Fuente del proyecto:
`f5c960eefedb5fc49b313dd795b1ec84e784bd78`.
La receta V3 actual conserva **NO-GO**; no se adquieren observaciones de mercado,
entrenan modelos ni ejecutan predicciones en este paquete.

## Avance concreto

El registro anterior sólo describía cuatro clases con instrumentos y URLs
vacíos. `concrete_panels.DRAFT.json` identifica estos candidatos:

| Panel | Instrumento concreto y fuente propuesta | Naturaleza y límite |
|---|---|---|
| F00 principal | BTCUSDT Spot, archivos oficiales Binance Vision, intervalo 1d | Criptoactivos, barras diarias UTC y volumen BTC; mismo subyacente que D00, pero sólo una eventual ventana futura podría aportar observaciones nuevas. |
| F01 adicional | SPY, `TIME_SERIES_DAILY` de Alpha Vantage | Participaciones de un fondo de renta variable estadounidense; fuente de identidad: State Street. |
| F02 adicional | USO, `TIME_SERIES_DAILY` de Alpha Vantage | Participaciones de un fondo con exposición a futuros de petróleo; no se identifica como petróleo spot ni como un futuro empalmado por el investigador. Fuente: USCF. |
| F03 adicional | TLT, `TIME_SERIES_DAILY` de Alpha Vantage | Participaciones de un fondo de bonos del Tesoro de largo vencimiento; fuente de identidad: iShares. Cotiza en Nasdaq, no se sustituye su calendario por el de NYSE. |

Son propuestas elegidas por el tipo de exposición y acceso documentado, sin
ver desempeño de la candidata. No se consultó un endpoint de velas, cotizaciones
o predicciones. La documentación de proveedores contiene algunos ejemplos
ilustrativos, que se ignoraron y no se usan como observaciones científicas.

Los tres fondos comparten intermediación, moneda, infraestructura de proveedor,
gran parte del calendario estadounidense y factores macroeconómicos. La
diversidad de exposiciones no demuestra independencia económica o estadística.
No se seleccionaron por volatilidad, correlación o resultados calculados.

## Qué exige la guía y qué queda bloqueado

La sección 8.3, página 23 del PDF cuyo hash consta en
`guide_requirements.json`, requiere la misma tarea/métrica y al menos dos ejes
frente al principal: tamaño, dominio, ruido, desbalance o dimensionalidad.
**Ticker, proveedor y reloj no se cuentan como ejes adicionales independientes.**

El dominio se respalda en los emisores. El segundo eje propuesto es tamaño:
la misma ventana de calendario contiene 366 fechas UTC y 253 sesiones NYSE
programadas según el calendario oficial 2027–2028. Es aritmética de metadatos,
no 253 filas de precios ya obtenidas. La diversidad por tamaño sigue pendiente
de equivalencia de tarea, política T definida antes de resultados, calendario
del mercado concreto y comprobación de completitud. No se recortan filas para
conseguir una casilla. Ruido, desbalance y dimensionalidad no se declaran nuevos
sin evidencia.

El cálculo NYSE no se atribuye a Nasdaq: la página Nasdaq observada sólo
mostró 2026; el registro TLT mantiene su conteo futuro sin verificar. Los
calendarios programados tampoco anticipan cierres extraordinarios futuros.

## La dificultad del horizonte es real

Se propone conservar la fórmula de desviación poblacional de siete retornos
logarítmicos RAW (`ddof=0`, sin anualizar), con ocho cierres de soporte y una
observación futura. Eso **no convierte** una sesión en 24 horas:

- BTC usa barras UTC de calendario, incluyendo fines de semana.
- Los fondos tienen sesiones, festivos, cierres adelantados y horario de verano.
  Un siguiente cierre puede quedar entre uno y cuatro días de calendario después
  del anterior, incluso en el calendario programado normal.

El protocolo futuro debe aceptar y justificar explícitamente ese alcance o
elegir otra tarea/fuentes. El guard histórico de días UTC permanece intacto.
No se rellenan fines de semana con barras sintéticas ni se usa volumen ficticio.
Una aceptación declarada sin responsable, criterio y justificación no basta.

## Ventanas concretas de trabajo: DRAFT

| Uso propuesto | Soporte RAW | Objetivos |
|---|---|---|
| Planificación | Desde 2026-12-01 | 2026-12-14 a 2027-02-26 |
| Posible confirmación | Desde 2027-03-01 | 2027-03-15 a 2028-03-14 |

No son un preregistro, no fijan R/T suficientes y sus datos aún no existen como
archivos comprobados aquí. El congelamiento real debe preceder también al
soporte RAW confirmatorio, no sólo al primer objetivo puntuado. Si ciencia,
acceso o potencia no están preparados a tiempo, se registra una ventana nueva
antes de observarla; no se cambia la etiqueta de un calendario ya expuesto.

Se requieren al menos siete endpoints purgados en cada frontera y comprobar los
soportes de ocho cierres. La separación de las fechas propuestas no sustituye
la verificación de esa geometría en los archivos reales. D00–D03 conservan su
exposición previa y no se renombran como F00–F03.

## Procedencia, disponibilidad y derechos

`source_observations.json` conserva consultas documentales y límites por fuente.
Se abrieron documentos primarios y metadatos de licencias/calendarios, sin
aceptar cuentas ni usar una clave autenticada. El intento único no autenticado
de `exchangeInfo` de BTCUSDT falló en el proxy CONNECT con 403: el origen no fue
alcanzado, lo que no prueba ausencia del símbolo o caída del proveedor.

La documentación Alpha Vantage describe OHLCV diario sin ajustar y un buscador
de símbolos ETF. No se verificaron respuestas de SPY/USO/TLT ni disponibilidad
de plan/cuenta, historial completo o demora de publicación. Los términos leídos
limitan el acceso privado individual/no comercial y conservan excepciones;
no se infiere autorización institucional ni redistribución pública. La condición
de repositorio público de la guía y el acceso autónomo del reproductor necesitan
una decisión concreta. No se pide una clave ni se incluye una en el registro.

Binance publica términos específicos de dataset v1.0 de 2026-08-26: CC BY-NC-SA
4.0 y condiciones adicionales. Su archivo exacto tiene 9807 bytes, SHA256
`99c4921dbe023cd68c1a447775707d621453cc78aec6ac0bfe2e6c06a5114ee0`,
ligado al blob Git `d47bf55acfd83456d6fe57b31f400b42bda83c19`.
Un `LICENSE` raíz consultado devolvió 404; ese hecho no anula los términos del
dataset encontrados después. La licencia propia del suplemento DOI no se
extiende a estos proveedores.

Un archivo mensual recuperado después de su período no prueba por sí solo
qué entradas estaban publicadas en cada origen. Se necesitan registros reales
de publicación, recuperación y disponibilidad antes de la predicción, y hashes
de las versiones efectivamente usadas. Esos campos permanecen vacíos.

## Comando ejecutable, sólo metadatos

Desde este directorio, con Python estándar:

```bash
python admission_validator.py \
  --registry concrete_panels.DRAFT.json \
  --declaration AUTHOR_ACCESS_DECLARATION.template.json \
  --output /tmp/e3-data-admission-new-case.json
```

El caso actual devuelve salida **2**, con estado bloqueado, cuatro propuestas y
cero datasets admitidos. No lee matrices o precios, no hace solicitudes de red,
no valida firmas ni declara autorización legal. Incluso una fixture completa
sólo resulta lista para revisión externa de metadatos; nunca habilita confirmar.
El recibo de salida debe ser nuevo y no se sobrescribe.

`AUTHOR_ACCESS_DECLARATION.template.json` está sin nombre, firma o hechos de
acceso supuestos. El autor o custodio real debe declarar su relación, exposición
y decisiones antes de producir resultados. El software no los autentica.

`calendar_design.py` comprueba fechas programadas y duraciones de calendario,
no velas. `build_registry.py --output-root /ruta/nueva-vacia` sólo construye
otra propuesta DRAFT fechada y rechaza un destino con contenido; no debe usarse
para reescribir la presente versión congelada.

## Fuentes primarias verificadas

- [Alpha Vantage: capacidad de serie diaria y búsqueda de símbolos](https://www.alphavantage.co/documentation/#daily).
- [Alpha Vantage: términos](https://www.alphavantage.co/terms_of_service/).
- [Binance: archivo público y esquema](https://github.com/binance/binance-public-data/blob/master/README.md).
- [Binance: términos específicos del dataset](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md).
- [State Street: SPY](https://www.ssga.com/us/en/institutional/etfs/state-street-spdr-sp-500-etf-trust-spy).
- [USCF: exposición de USO](https://www.uscfinvestments.com/disclosures).
- [iShares: prospecto resumen TLT](https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-20-plus-year-treasury-bond-etf-2-28.pdf).
- [NYSE: calendario 2026–2028](https://www.nyse.com/trade/hours-calendars).
- [Nasdaq: calendario observado 2026](https://www.nasdaqtrader.com/Trader.aspx?id=Calendar).

La clasificación y los scripts son preparación auditable. La confirmación, la
potencia final, el uso de los archivos reales y R3 mantienen puertas propias.
