"""Solución reproducible del ejemplo de clase de seis pesos y diez nodos.

Datos del PDF aportado: y1=-1, y2=2, y=1, alpha=0.01 y pesos iniciales
(-0.33, -0.33, 0.57, -0.01, 0.07, 0.82). Dos neuronas ocultas sigmoideas,
sin sesgos y con salida lineal. La pérdida es (y-y_pred)**2, sin factor 1/2.

Python estándar basta para calcular y exportar tablas. --figuras requiere
matplotlib. Los cálculos conservan toda la precisión de float; las tablas
impresas se redondean a seis decimales.
"""

import argparse
import csv
import json
from math import exp, isfinite
from pathlib import Path


Y1, Y2, Y = -1.0, 2.0, 1.0
ALPHA = 0.01
PESOS_INICIALES = (-0.33, -0.33, 0.57, -0.01, 0.07, 0.82)


def evaluar(pesos, t=0):
    """Calcula p1..p10, salida, pérdida y gradiente en un mismo estado."""
    if len(pesos) != 6 or not all(isfinite(v) for v in pesos):
        raise ValueError("Se requieren seis pesos finitos.")
    w1, w2, w3, w4, w5, w6 = pesos
    p1, p2, p3, p4 = Y1 * w1, Y2 * w2, Y1 * w3, Y2 * w4
    p5, p6 = p1 + p2, p3 + p4
    p7, p8 = 1.0 / (1.0 + exp(-p5)), 1.0 / (1.0 + exp(-p6))
    p9, p10 = p7 * w5, p8 * w6
    prediccion = p9 + p10
    perdida = (Y - prediccion) ** 2
    delta = 2.0 * (prediccion - Y)
    gradiente = (
        delta * w5 * p7 * (1.0 - p7) * Y1,
        delta * w5 * p7 * (1.0 - p7) * Y2,
        delta * w6 * p8 * (1.0 - p8) * Y1,
        delta * w6 * p8 * (1.0 - p8) * Y2,
        delta * p7,
        delta * p8,
    )
    if not all(isfinite(v) for v in (prediccion, perdida, *gradiente)):
        raise ValueError("La evaluación produjo un valor no finito.")
    return {"t": t, "pesos": tuple(pesos),
            "p": (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10),
            "y_pred": prediccion, "L": perdida, "gradiente": gradiente}


def resolver(tolerancia=1e-6, max_actualizaciones=100000):
    """Registra cinco pasos y el primer cambio de pérdida bajo la tolerancia."""
    if not isfinite(tolerancia) or tolerancia <= 0 or max_actualizaciones < 5:
        raise ValueError("Use una tolerancia positiva y al menos cinco actualizaciones.")
    estados = [evaluar(PESOS_INICIALES)]
    primer_cumplimiento = None
    for t in range(1, max_actualizaciones + 1):
        anterior = estados[-1]
        # Todos los gradientes proceden del mismo estado anterior.
        nuevos_pesos = tuple(w - ALPHA * g for w, g in
                            zip(anterior["pesos"], anterior["gradiente"]))
        actual = evaluar(nuevos_pesos, t)
        estados.append(actual)
        if abs(actual["L"] - anterior["L"]) < tolerancia and primer_cumplimiento is None:
            primer_cumplimiento = t
        if primer_cumplimiento is not None and t >= 5:
            break
    if primer_cumplimiento is None:
        raise RuntimeError("No se alcanzó la tolerancia dentro del máximo indicado.")
    t = primer_cumplimiento
    cambios_previos = [abs(estados[k]["L"] - estados[k - 1]["L"])
                       for k in range(1, t)]
    informe = {
        "datos": {"y1": Y1, "y2": Y2, "y": Y, "alpha": ALPHA,
                  "pesos_iniciales": PESOS_INICIALES, "perdida": "(y-y_pred)^2"},
        "criterio": "abs(L(t)-L(t-1)) < tolerancia",
        "tolerancia": tolerancia,
        "primera_actualizacion_que_cumple": t,
        "actualizaciones_adicionales_despues_de_5": max(0, t - 5),
        "L_anterior": estados[t - 1]["L"], "L_actual": estados[t]["L"],
        "cambio": abs(estados[t]["L"] - estados[t - 1]["L"]),
        "cambio_previo": cambios_previos[-1] if cambios_previos else None,
        "actualizaciones_previas_comprobadas": len(cambios_previos),
        "ninguna_anterior_cumple": all(cambio >= tolerancia for cambio in cambios_previos),
        "L_inicial": estados[0]["L"], "L_despues_de_5": estados[5]["L"],
        "reduccion_absoluta_en_5": estados[0]["L"] - estados[5]["L"],
        "reduccion_porcentual_en_5": 100.0 * (1.0 - estados[5]["L"] / estados[0]["L"]),
    }
    return estados, informe


def mostrar_tabla(estados):
    print("t | omega1 | omega2 | omega3 | omega4 | omega5 | omega6 | y_pred | L")
    for fila in estados[:6]:
        valores = (*fila["pesos"], fila["y_pred"], fila["L"])
        print(str(fila["t"]) + " | " + " | ".join(f"{v:.6f}" for v in valores))


def exportar(estados, informe, directorio):
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    tablas = (
        ("mlp_cinco_iteraciones_v1.csv",
         ["t", *[f"omega{j}" for j in range(1, 7)], "y_pred", "L"],
         [(r["t"], *r["pesos"], r["y_pred"], r["L"]) for r in estados[:6]]),
        ("mlp_nodos_v1.csv", ["t", *[f"p{j}" for j in range(1, 11)]],
         [(r["t"], *r["p"]) for r in estados[:6]]),
        ("mlp_gradientes_v1.csv", ["t", *[f"dL_domega{j}" for j in range(1, 7)]],
         [(r["t"], *r["gradiente"]) for r in estados[:5]]),
    )
    for nombre, encabezado, filas in tablas:
        with (directorio / nombre).open("w", encoding="utf-8", newline="") as archivo:
            escritor = csv.writer(archivo, lineterminator="\n")
            escritor.writerow(encabezado)
            escritor.writerows(filas)
    (directorio / "mlp_tolerancia_v1.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def graficar(estados, directorio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.hashsalt"] = "mlp-ejemplo-clase-v1"
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    filas = estados[:6]
    fig, ax = plt.subplots(figsize=(7.6, 4.3), layout="constrained")
    ax.plot([r["t"] for r in filas], [r["L"] for r in filas], color="#126A9E",
            linewidth=2.4, marker="o", markersize=6)
    ax.set(title="Descenso de la pérdida del MLP", xlabel="Actualizaciones t",
           ylabel="Error cuadrático L", xlim=(-0.25, 5.25), ylim=(0.409, 0.469))
    ax.set_xticks(range(6))
    ax.grid(axis="both", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(f"L(0) = {filas[0]['L']:.6f}", xy=(0, filas[0]["L"]),
                xytext=(0.2, 0.462), fontsize=10, color="#16334A")
    ax.annotate(f"L(5) = {filas[5]['L']:.6f}", xy=(5, filas[5]["L"]),
                xytext=(3.55, 0.412), fontsize=10, color="#16334A")
    fig.savefig(directorio / "mlp_perdida_cinco_iteraciones_v1.svg", metadata={"Date": None})
    fig.savefig(directorio / "mlp_perdida_cinco_iteraciones_v1.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exportar", type=Path, help="Directorio para CSV y JSON reproducibles")
    parser.add_argument("--figuras", type=Path, help="Directorio para gráfica SVG y PNG (matplotlib)")
    args = parser.parse_args()
    estados, informe = resolver()
    mostrar_tabla(estados)
    print("\nPrimera actualización bajo la tolerancia:", informe["primera_actualizacion_que_cumple"])
    print("Actualizaciones adicionales después de cinco:", informe["actualizaciones_adicionales_despues_de_5"])
    print("Cambio de pérdida:", informe["cambio"])
    if args.exportar is not None:
        exportar(estados, informe, args.exportar)
    if args.figuras is not None:
        graficar(estados, args.figuras)


if __name__ == "__main__":
    main()
