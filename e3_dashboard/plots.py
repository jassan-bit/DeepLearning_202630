"""Figures for the E3 service; saved observations only, with explicit supports.

The module does not load model weights, fit a predictor or import training code.
Time-dependence diagnostics use one contiguous daily segment, never a sequence
created by joining the purges between cohorts or gaps introduced by filters.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import math

import numpy as np
import plotly.graph_objects as go


NAVY, TEAL, ORANGE, MUTED = "#12354A", "#0F7F87", "#E69138", "#667788"
PARTITIONS = {
    "training": ("Entrenamiento", NAVY),
    "calibration": ("Calibración E2", TEAL),
    "evaluation": ("Prueba histórica", ORANGE),
}
FIELDS = {
    "target": ("σ₇ objetivo", "σ₇"),
    "last_open": ("Última apertura conocida", "USDT"),
    "last_high": ("Último máximo conocido", "USDT"),
    "last_low": ("Último mínimo conocido", "USDT"),
    "last_close": ("Último cierre conocido", "USDT"),
    "last_volume": ("Último volumen conocido", "BTC"),
    "known_return6": ("Último retorno logarítmico conocido", "Retorno logarítmico"),
    "known_floor": ("Piso observable de σ₇", "σ₇"),
}


def _base(title, *, x_title=None, y_title=None):
    figure = go.Figure()
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 16}},
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": NAVY, "size": 12},
        margin={"l": 55, "r": 20, "t": 62, "b": 48}, autosize=True,
        hovermode="x unified", legend={"orientation": "h", "y": 1.02, "x": 0},
        xaxis_title=x_title, yaxis_title=y_title,
    )
    figure.update_xaxes(gridcolor="rgba(18,53,74,0.08)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(18,53,74,0.08)", zeroline=False)
    return figure


def figure_empty(message="No hay observaciones con estos filtros."):
    figure = _base("")
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font={"color": MUTED, "size": 14})
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def _values(rows, field):
    try:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return values if values.ndim == 1 and np.isfinite(values).all() else None


def _broken_series(rows, field, *, date_field="target_date"):
    """Insert a visible break instead of interpolating across unobserved dates."""
    xs, ys, previous = [], [], None
    for row in rows:
        current = date.fromisoformat(row[date_field])
        if previous is not None and (current - previous).days != 1:
            xs.append(None)
            ys.append(None)
        xs.append(current.isoformat())
        ys.append(float(row[field]))
        previous = current
    return xs, ys


def target_series(rows):
    if not rows or _values(rows, "target") is None:
        return figure_empty()
    figure = _base(f"¿Cómo cambia la volatilidad? · {len(rows):,} fechas",
                   x_title="Fecha objetivo (UTC)", y_title="σ₇ realizado")
    try:
        for partition, (label, color) in PARTITIONS.items():
            selected = [row for row in rows if row["partition"] == partition]
            if selected:
                x, y = _broken_series(selected, "target")
                figure.add_scatter(x=x, y=y, mode="lines+markers", name=label,
                                   marker={"size": 3}, line={"color": color, "width": 1.6},
                                   connectgaps=False, hovertemplate="σ₇=%{y:.5f}<extra></extra>")
    except (ValueError, KeyError):
        return figure_empty("Las fechas no permiten dibujar una serie temporal válida.")
    return figure


def target_histogram(rows):
    values = _values(rows, "target")
    if values is None or len(values) == 0:
        return figure_empty()
    figure = _base("¿Qué valores de volatilidad son frecuentes?",
                   x_title="σ₇ realizado", y_title="Número de fechas")
    figure.add_histogram(x=values.tolist(), nbinsx=min(36, max(5, int(math.sqrt(len(values))))),
                         marker_color=TEAL, opacity=0.88, name="Selección",
                         hovertemplate="σ₇=%{x:.5f}<br>Fechas=%{y}<extra></extra>")
    return figure


def feature_explorer(rows, field="last_close"):
    if field not in FIELDS:
        return figure_empty("Seleccione una variable del explorador.")
    values = _values(rows, field)
    if not rows or values is None:
        return figure_empty()
    label, unit = FIELDS[field]
    date_field = "target_date" if field == "target" else "feature_date"
    figure = _base(label, x_title="Fecha objetivo (UTC)" if field == "target" else
                   "Fecha de la última observación disponible (UTC, t−1)", y_title=unit)
    try:
        x, y = _broken_series(rows, field, date_field=date_field)
    except (ValueError, KeyError):
        return figure_empty("Falta una fecha válida de disponibilidad de la variable.")
    figure.add_scatter(x=x, y=y, mode="lines+markers", marker={"size": 3},
                       line={"color": TEAL, "width": 1.8}, name=label,
                       connectgaps=False, hovertemplate="%{y:.5g}<extra></extra>")
    return figure


def partition_comparison(rows):
    if not rows or _values(rows, "target") is None:
        return figure_empty()
    figure = _base("¿Cómo difiere el objetivo entre particiones?",
                   x_title="Partición", y_title="σ₇ realizado")
    for partition, (label, color) in PARTITIONS.items():
        values = [float(row["target"]) for row in rows if row["partition"] == partition]
        if values:
            figure.add_box(y=values, name=f"{label} · n={len(values)}", marker_color=color,
                           boxpoints=False, line={"width": 1.6})
    figure.update_layout(showlegend=False, hovermode="closest")
    return figure


def quality_summary(rows):
    """Describe this selected view, not a new certification of the full dataset."""
    valid_dates, invalid_dates, nonfinite, invalid_bars = [], 0, 0, 0
    features = []
    for row in rows:
        try:
            valid_dates.append(date.fromisoformat(row["target_date"]))
        except (ValueError, TypeError, KeyError):
            invalid_dates += 1
        for key in ("target", "last_open", "last_high", "last_low", "last_close", "last_volume"):
            try:
                if not math.isfinite(float(row[key])):
                    nonfinite += 1
            except (ValueError, TypeError, KeyError, OverflowError):
                nonfinite += 1
        try:
            op, high, low, close, volume = [float(row[key]) for key in
                                           ("last_open", "last_high", "last_low", "last_close", "last_volume")]
            values = (op, high, low, close, volume)
            features.append(values)
            if not (all(math.isfinite(v) for v in values) and min(op, high, low, close) > 0
                    and volume >= 0 and low <= min(op, close) <= max(op, close) <= high):
                invalid_bars += 1
        except (ValueError, TypeError, KeyError, OverflowError):
            invalid_bars += 1
    unique = sorted(set(valid_dates))
    gaps = [(second - first).days - 1 for first, second in zip(unique, unique[1:])
            if (second - first).days > 1]
    duplicates = len(valid_dates) - len(unique)
    feature_duplicates = sum(count - 1 for count in Counter(features).values())
    insight = (
        f"La selección contiene {len(rows):,} fechas: {duplicates} fechas duplicadas, "
        f"{invalid_dates} fechas inválidas y {nonfinite} campos no finitos o ausentes. "
        f"Hay {len(gaps)} saltos del calendario seleccionado. Los filtros, las purgas y "
        "los límites entre cohortes pueden producir esos saltos; no se atribuyen por sí solos a pérdida de datos."
    )
    return {"n": len(rows), "duplicate_dates": duplicates, "invalid_dates": invalid_dates,
            "nonfinite_values": nonfinite, "invalid_last_bars": invalid_bars,
            "calendar_breaks": len(gaps), "gap_days_in_selected_calendar": sum(gaps),
            "exact_last_feature_duplicates": feature_duplicates,
            "near_duplicates": None, "insight": insight}


def _longest_daily_segment(rows):
    if not rows:
        return [], "No hay fechas seleccionadas."
    if _values(rows, "target") is None:
        return [], "Hay valores no finitos o ausentes; no se calcula dependencia."
    longest, current, previous = [], [], None
    for row in rows:
        try:
            day = date.fromisoformat(row["target_date"])
        except (ValueError, TypeError, KeyError):
            return [], "Hay fechas inválidas; no se calcula dependencia."
        if previous is not None and day <= previous:
            return [], "Las fechas deben estar ordenadas y no repetidas para calcular dependencia."
        if previous is None or (day - previous).days == 1:
            current.append(row)
        else:
            if len(current) > len(longest):
                longest = current
            current = [row]
        previous = day
    if len(current) > len(longest):
        longest = current
    support = (f"Segmento diario continuo: n={len(longest)}, "
               f"{longest[0]['target_date']}–{longest[-1]['target_date']}; "
               f"{len(rows)} fechas en la selección total. No se unen saltos o purgas.")
    return longest, support


def _acf(values, max_lag):
    if len(values) == 0:
        return None
    spread = float(np.ptp(values))
    numerical_scale = max(1.0, float(np.max(np.abs(values))))
    if spread <= 8 * np.finfo(np.float64).eps * numerical_scale:
        return None
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    if not math.isfinite(denominator) or denominator <= 0:
        return None
    return np.asarray([1.0] + [float(np.dot(centered[:-lag], centered[lag:])) / denominator
                              for lag in range(1, max_lag + 1)], dtype=np.float64)


def _pacf(correlations):
    """Durbin–Levinson recursion using the same biased ACF estimate."""
    result = np.ones(len(correlations), dtype=np.float64)
    previous = np.empty(0, dtype=np.float64)
    variance = 1.0
    for lag in range(1, len(correlations)):
        if variance <= 1e-12 or not math.isfinite(variance):
            return None
        coefficient = (correlations[lag] - float(np.dot(previous, correlations[1:lag][::-1]))) / variance
        if not math.isfinite(coefficient) or abs(coefficient) > 1 + 1e-9:
            return None
        updated = np.empty(lag, dtype=np.float64)
        if lag > 1:
            updated[:-1] = previous - coefficient * previous[::-1]
        updated[-1] = coefficient
        result[lag] = coefficient
        variance *= 1 - coefficient * coefficient
        previous = updated
    return result


def _correlation_figure(values, n, title):
    figure = _base(title, x_title="Retardo (días)", y_title="Correlación")
    lags = list(range(1, len(values)))
    reference = 1.96 / math.sqrt(n)
    figure.add_hrect(y0=-reference, y1=reference, line_width=0,
                     fillcolor="rgba(18,53,74,0.08)", annotation_text="Referencia ±1,96/√n",
                     annotation_position="top right")
    figure.add_bar(x=lags, y=values[1:].tolist(), marker_color=TEAL, name="Correlación",
                   hovertemplate="Retardo %{x}: %{y:.3f}<extra></extra>")
    figure.update_yaxes(range=[-1.05, 1.05])
    figure.update_layout(showlegend=False, hovermode="closest")
    return figure


def dependency_panel(rows):
    segment, support = _longest_daily_segment(rows)
    if len(segment) < 8:
        message = "Se necesitan al menos ocho fechas diarias consecutivas para estos diagnósticos."
        return figure_empty(message), figure_empty(message), f"{support} {message}"
    values = _values(segment, "target")
    max_lag = min(30, (len(values) - 1) // 4)
    correlations = _acf(values, max_lag)
    if correlations is None:
        message = "La serie es constante a precisión numérica; ACF, PACF y ADF no están definidas."
        return figure_empty(message), figure_empty(message), f"{support} {message}"
    partial = _pacf(correlations)
    acf = _correlation_figure(correlations, len(values), "Dependencia de σ₇: ACF")
    pacf = (_correlation_figure(partial, len(values), "Dependencia parcial de σ₇: PACF")
            if partial is not None else figure_empty("La PACF no es estable en este segmento."))
    diagnostic = "ADF requiere al menos 24 fechas continuas en esta vista."
    if len(values) >= 24:
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(values, maxlag=min(12, len(values) // 4 - 2), regression="c", autolag="AIC")
            p = float(result[1])
            p_label = "<0,0001" if p < 0.0001 else f"{p:.4f}".replace(".", ",")
            diagnostic = (f"ADF con constante y retardo por AIC: p={p_label}; "
                          f"{int(result[3])} observaciones efectivas. "
                          + ("Hay evidencia contra la hipótesis de raíz unitaria en este segmento. " if p < 0.05
                             else "No se rechaza la hipótesis de raíz unitaria al 5 % en este segmento. "))
        except ImportError:
            diagnostic = "ADF pendiente: el componente estadístico no está disponible en este entorno."
        except (ValueError, np.linalg.LinAlgError):
            diagnostic = "ADF no estimable de forma fiable en este segmento; no se informa un p-valor."
    return acf, pacf, (f"{support} {diagnostic} Las bandas ACF/PACF son una referencia aproximada de ruido blanco; "
                       "el solapamiento del objetivo y los cambios de régimen limitan esa interpretación. "
                       "Un diagnóstico ADF no certifica estacionariedad global.")


def _paired(dates, targets, predictions):
    if not dates or len(dates) != len(targets) or len(targets) != len(predictions):
        return None
    try:
        observed, predicted = np.asarray(targets, dtype=float), np.asarray(predictions, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if observed.ndim != 1 or predicted.ndim != 1 or not (np.isfinite(observed).all() and np.isfinite(predicted).all()):
        return None
    return observed, predicted


def forecast_figure(dates, targets, predictions, q=None):
    pair = _paired(dates, targets, predictions)
    if pair is None:
        return figure_empty("No hay una trayectoria compatible para esta selección.")
    observed, predicted = pair
    figure = _base("Realizado y pronóstico de la trayectoria seleccionada",
                   x_title="Fecha objetivo (UTC)", y_title="σ₇ en unidades originales")
    if q is not None and math.isfinite(float(q)) and float(q) >= 0:
        lower, upper = np.maximum(0, predicted - float(q)), predicted + float(q)
        if (lower <= upper).all():
            figure.add_scatter(x=dates, y=lower.tolist(), mode="lines", line={"width": 0},
                               showlegend=False, hoverinfo="skip")
            figure.add_scatter(x=dates, y=upper.tolist(), mode="lines", line={"width": 0},
                               fill="tonexty", fillcolor="rgba(15,127,135,0.12)",
                               name="Intervalo heredado nominal 90 %", hoverinfo="skip")
    figure.add_scatter(x=dates, y=observed.tolist(), mode="lines", line={"color": NAVY, "width": 1.6},
                       name="Realizado", hovertemplate="σ₇=%{y:.5f}<extra></extra>")
    figure.add_scatter(x=dates, y=predicted.tolist(), mode="lines", line={"color": TEAL, "width": 1.6},
                       name="Pronóstico", hovertemplate="σ̂₇=%{y:.5f}<extra></extra>")
    return figure


def residual_figure(dates, targets, predictions):
    pair = _paired(dates, targets, predictions)
    if pair is None:
        return figure_empty()
    observed, predicted = pair
    figure = _base("¿Cuándo se equivoca la trayectoria seleccionada?",
                   x_title="Fecha objetivo (UTC)", y_title="Residuo: realizado − pronóstico")
    figure.add_scatter(x=dates, y=(observed - predicted).tolist(), mode="lines",
                       line={"color": ORANGE, "width": 1.5}, name="Residuo")
    figure.add_hline(y=0, line_color=NAVY, line_width=1, line_dash="dot")
    return figure


def residual_acf(dates, targets, predictions):
    pair = _paired(dates, targets, predictions)
    if pair is None:
        return figure_empty()
    observed, predicted = pair
    rows = [{"target_date": str(day), "target": float(value)}
            for day, value in zip(dates, observed - predicted)]
    segment, _ = _longest_daily_segment(rows)
    if len(segment) < 8:
        return figure_empty("Se necesitan al menos ocho residuos de días consecutivos.")
    values = _values(segment, "target")
    correlations = _acf(values, min(30, (len(values) - 1) // 4))
    if correlations is None:
        return figure_empty("Los residuos son constantes a precisión numérica; su ACF no está definida.")
    return _correlation_figure(correlations, len(values), f"ACF de residuos · segmento n={len(values)}")


def scatter_actual_forecast(dates, targets, predictions):
    pair = _paired(dates, targets, predictions)
    if pair is None:
        return figure_empty()
    observed, predicted = pair
    figure = _base("¿Se alinean las predicciones con lo observado?",
                   x_title="σ₇ realizado", y_title="σ₇ pronosticado")
    figure.add_scatter(x=observed.tolist(), y=predicted.tolist(), mode="markers",
                       marker={"color": TEAL, "size": 5, "opacity": 0.55}, name="Fechas",
                       text=list(dates), hovertemplate="%{text}<br>Real: %{x:.5f}<br>Pronóstico: %{y:.5f}<extra></extra>")
    low, high = float(min(observed.min(), predicted.min())), float(max(observed.max(), predicted.max()))
    if low == high:
        low, high = low - 1e-6, high + 1e-6
    figure.add_scatter(x=[low, high], y=[low, high], mode="lines", line={"color": ORANGE, "dash": "dot"},
                       name="Coincidencia exacta", hoverinfo="skip")
    figure.update_layout(hovermode="closest")
    return figure


def training_curve(record):
    """Show saved epoch history with different metrics on explicit separate axes.

    The training loss was recorded during optimizer updates; it is not a final
    evaluation risk. Its normalized MSE cannot be subtracted from validation
    RMSE to infer a train/validation adequacy gap.
    """
    if not isinstance(record, dict) or not isinstance(record.get("history"), list) or not record["history"]:
        return figure_empty("No hay una historia de épocas verificada para esta selección.")
    history = record["history"]
    try:
        epochs = [row["epoch"] for row in history]
        if not all(type(epoch) is int and epoch > 0 for epoch in epochs):
            raise ValueError()
        if epochs != sorted(set(epochs)):
            raise ValueError()
        training = [float(row["training_loss_normalized"]) for row in history]
        validation = [float(row["validation_RMSE_original"]) for row in history]
        if not all(math.isfinite(value) and value >= 0 for value in (*training, *validation)):
            raise ValueError()
    except (KeyError, TypeError, ValueError, OverflowError):
        return figure_empty("La historia guardada no permite una curva de épocas válida.")
    seed = record.get("seed", "registrada")
    figure = _base(f"Curva de épocas registrada · semilla {seed}",
                   x_title="Época", y_title="MSE normalizado durante ajuste")
    figure.add_scatter(x=epochs, y=training, mode="lines+markers",
                       name="Ajuste: MSE normalizado", line={"color": TEAL, "width": 1.8},
                       marker={"size": 4},
                       hovertemplate="Época %{x}<br>MSE normalizado=%{y:.6g}<extra></extra>")
    figure.add_scatter(x=epochs, y=validation, mode="lines+markers", yaxis="y2",
                       name="Validación: RMSE en σ₇", line={"color": ORANGE, "width": 1.8},
                       marker={"size": 4},
                       hovertemplate="Época %{x}<br>RMSE validación=%{y:.6f} σ₇<extra></extra>")
    figure.update_layout(
        yaxis2={"title": "RMSE de validación (σ₇ originales)", "overlaying": "y", "side": "right",
                "showgrid": False, "zeroline": False},
        margin={"l": 70, "r": 85, "t": 65, "b": 55},
    )
    best = record.get("best_epoch")
    if type(best) is int and best in epochs:
        figure.add_vline(x=best, line_width=1.3, line_dash="dot", line_color=NAVY,
                         annotation_text=f"Mejor época: {best}", annotation_position="top right")
    return figure
