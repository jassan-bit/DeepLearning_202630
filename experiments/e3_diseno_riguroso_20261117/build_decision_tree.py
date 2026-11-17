"""Render the guide's conditional R1/R2/R3 decision tree; no grading logic."""

from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring


def build() -> bytes:
    svg = Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "viewBox": "0 0 780 700",
        "role": "img", "aria-labelledby": "title description",
    })
    SubElement(svg, "title", {"id": "title"}).text = "Regla condicional de topes R1, R2 y R3"
    SubElement(svg, "desc", {"id": "description"}).text = (
        "Si falla R1 o R2, el tope es 3,0. Si R1 y R2 se acreditan y falla R3, "
        "el tope es 4,0. Si las tres se acreditan, el tope es 5,0, sujeto a la "
        "propiedad formal y los demás requisitos. No asigna nota al proyecto."
    )
    defs = SubElement(svg, "defs")
    for name, color in (("blue", "#165b91"), ("red", "#a62525"), ("green", "#176840")):
        marker = SubElement(defs, "marker", {
            "id": name, "viewBox": "0 0 10 10", "refX": "9", "refY": "5",
            "markerWidth": "7", "markerHeight": "7", "orient": "auto-start-reverse",
        })
        SubElement(marker, "path", {"d": "M 0 0 L 10 5 L 0 10 z", "fill": color})

    def label(x, y, lines, *, color="#172b3a", size=17, bold=False):
        text = SubElement(svg, "text", {
            "x": str(x), "y": str(y), "fill": color, "font-family": "Arial, sans-serif",
            "font-size": str(size), "text-anchor": "middle",
            "font-weight": "bold" if bold else "normal",
        })
        for i, line in enumerate(lines):
            SubElement(text, "tspan", {"x": str(x), "dy": "0" if i == 0 else "23"}).text = line

    def box(x, y, width, height, fill, stroke):
        SubElement(svg, "rect", {
            "x": str(x), "y": str(y), "width": str(width), "height": str(height),
            "rx": "8", "fill": fill, "stroke": stroke, "stroke-width": "1.8",
        })

    def arrow(path, color, marker):
        SubElement(svg, "path", {
            "d": path, "fill": "none", "stroke": color, "stroke-width": "1.8",
            "marker-end": f"url(#{marker})",
        })

    questions = [
        (30, ["R1. ¿Novedad para el mismo problema,", "con diferencia sustantiva de mecanismo,", "pérdida o entrenamiento y atribución", "de la fuente cuando corresponda?"]),
        (210, ["R2. ¿El núcleo fue implementado", "desde su formulación y puede ser", "explicado y modificado por el autor,", "con procedencia declarada?"]),
        (390, ["R3. ¿Supera a cada rival de la familia", "completa con significancia en las", "pruebas exigidas y bajo el resto", "del protocolo de evaluación?"]),
    ]
    for y, lines in questions:
        box(25, y, 420, 120, "#edf5fd", "#165b91")
        label(235, y + 30, lines)
        arrow(f"M 445 {y + 60} L 555 {y + 60}", "#a62525", "red")
        label(497, y + 47, ["No"], color="#a62525", size=16, bold=True)
    for y in (150, 330):
        arrow(f"M 235 {y} L 235 {y + 60}", "#165b91", "blue")
        label(258, y + 34, ["Sí"], color="#165b91", size=16, bold=True)

    for y, grade, lines, fill, stroke in (
        (30, "Tope 3,0", ["R1 no acreditada", "por incumplimiento."], "#fceeee", "#a62525"),
        (210, "Tope 3,0", ["R2 no acreditada", "por incumplimiento."], "#fceeee", "#a62525"),
        (390, "Tope 4,0", ["R1 y R2 acreditadas;", "R3 no satisfecha."], "#fff2df", "#a65b0c"),
    ):
        box(560, y, 195, 120, fill, stroke)
        label(657.5, y + 34, [grade], color=stroke, size=25, bold=True)
        label(657.5, y + 66, lines, size=16)

    arrow("M 235 510 L 235 548 L 305 548 L 305 575", "#176840", "green")
    label(258, 538, ["Sí"], color="#176840", size=16, bold=True)
    box(165, 580, 280, 92, "#e8f4ec", "#176840")
    label(305, 613, ["Tope 5,0"], color="#176840", size=25, bold=True)
    label(305, 643, ["R1, R2 y R3 acreditadas."], size=16)
    return b'<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(svg, encoding="utf-8") + b"\n"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    output = root / "book_e3/_static/e3_diseno_riguroso_20261117/decision_tree.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(build())
    print(output.relative_to(root))
