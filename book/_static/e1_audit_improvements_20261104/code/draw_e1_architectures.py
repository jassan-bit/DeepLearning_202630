"""Draw the implemented baseline architectures as standalone, data-free SVGs."""
from pathlib import Path
import textwrap
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "book/_static/e1_architectures_20261104"


def text(x, y, value, size=19, weight="normal", color="#17324d", anchor="middle"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="Arial, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}">{escape(value)}</text>')


def draw(filename, title, boxes, parameters, detail):
    width, height = 1040, 565
    margin, gap, columns = 35, 40, 3
    box_width = (width-2*margin-gap*(columns-1))/columns
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
             '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0 0 L10 4 L0 8Z" fill="#476783"/></marker></defs>',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             text(35, 37, title, 24, "bold", anchor="start"),
             text(35, 72, "Entrada causal: OHLCV observado de d−D a d−1; D = 7, 14, 21 o 28 días.", 19, anchor="start"),
             text(35, 99, "Objetivo fijo: volatilidad de siete retornos diarios σ₇. Horizonte: h = 1 día.", 19, anchor="start")]
    for i, lines in enumerate(boxes):
        row, position = divmod(i, columns)
        column = position if row == 0 else columns-1-position
        x = margin+column*(box_width+gap)
        y = 136+row*160
        fill = "#e8f0f8" if i != len(boxes)-1 else "#e7f4ec"
        border = "#38678f" if i != len(boxes)-1 else "#26734b"
        parts.append(f'<rect x="{x}" y="{y}" width="{box_width}" height="118" rx="10" fill="{fill}" stroke="{border}" stroke-width="2"/>')
        first = y+38-(len(lines)-2)*9
        for j, line in enumerate(lines):
            if j == 0:
                line = f"{i+1}. {line}"
            parts.append(text(x+box_width/2, first+j*27, line, 20,
                              "bold" if j==0 else "normal"))
        if i < len(boxes)-1:
            center = y+59
            if i == columns-1:
                edge = x+box_width
                parts.append(f'<path d="M{edge+3} {center} H1023 V{center+160} H{edge+5}" fill="none" stroke="#476783" stroke-width="2" marker-end="url(#arrow)"/>')
            elif row == 0:
                start, end = x+box_width+3, x+box_width+gap-5
                parts.append(f'<line x1="{start}" y1="{center}" x2="{end}" y2="{center}" stroke="#476783" stroke-width="2" marker-end="url(#arrow)"/>')
            else:
                parts.append(f'<line x1="{x-3}" y1="{center}" x2="{x-gap+5}" y2="{center}" stroke="#476783" stroke-width="2" marker-end="url(#arrow)"/>')
    for i, line in enumerate(textwrap.wrap(parameters, width=98)):
        parts.append(text(35, 454+i*27, line, 19, "bold", anchor="start"))
    for i, line in enumerate(textwrap.wrap(detail, width=110)):
        parts.append(text(35, 519+i*24, line, 17, anchor="start"))
    parts.append('</svg>')
    (DESTINATION/filename).write_text("\n".join(parts)+"\n")


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    draw("mlp.svg", "MLP implementada: una capa oculta",
         [("Ventana", "N × D × 5"), ("Flatten", "N × 5D"),
          ("Capa oculta", "Linear(5D, 16)", "ReLU · Dropout 0,1"),
          ("Salida", "Linear(16, 1)", "Softplus"), ("Pronóstico", "σ̂₇(d) ≥ 0")],
         "Parámetros: 80D + 33 → 593 / 1.153 / 1.713 / 2.273 para D = 7 / 14 / 21 / 28.",
         "El orden de las posiciones se conserva al aplanar. La estandarización y la escala del objetivo se ajustan solo en train.")
    draw("lstm.svg", "LSTM implementada: una capa unidireccional",
         [("Ventana", "N × D × 5"), ("LSTM", "5 entradas · H = 32", "Una capa; sin bidirección"),
          ("Último estado", "h_D: N × 32", "Dropout 0,1"),
          ("Salida", "Linear(32, 1)", "Softplus"), ("Pronóstico", "σ̂₇(d) ≥ 0")],
         "Parámetros: 4H(5 + H + 2) + H + 1 = 5.025, con H = 32; constantes para los cuatro D.",
         "PyTorch usa sesgos de entrada y recurrentes en cada puerta. No hay dropout entre capas porque solo se utiliza una.")
    draw("cnn1d.svg", "CNN1D implementada: dos convoluciones causales poco profundas",
         [("Ventana", "N × D × 5", "Transpose: N × 5 × D"),
          ("Conv1D causal", "5 → 16 · k = 3", "ReLU · Dropout 0,1"),
          ("Conv1D causal", "16 → 16 · k = 3", "ReLU · Dropout 0,1"),
          ("Media temporal", "Sobre las D posiciones", "N × 16"),
          ("Salida", "Linear(16, 1)", "Softplus"), ("Pronóstico", "σ̂₇(d) ≥ 0")],
         "Parámetros: (16·5·3 + 16) + (16·16·3 + 16) + (16 + 1) = 1.057, constantes para los cuatro D.",
         "Padding solo a la izquierda; campo receptivo local de cinco posiciones. La media usa toda la ventana pasada observada.")
    print("Generated three data-free, standalone SVG architecture diagrams.")


if __name__ == "__main__":
    main()
