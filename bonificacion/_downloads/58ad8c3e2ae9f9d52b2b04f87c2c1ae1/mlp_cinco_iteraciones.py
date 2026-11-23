"""Cinco pasos del ejercicio MLP sin suponer una arquitectura no proporcionada.

La función propagar debe tener firma propagar(pesos) y devolver:
    y_pred, d_y_pred_d_pesos, valores_p
d_y_pred_d_pesos contiene las seis derivadas de la predicción, NO de la pérdida.
valores_p contiene p1,...,p10 del grafo original, en ese orden.
"""

from math import isfinite


def cinco_iteraciones(propagar, pesos_iniciales, y, alpha=0.01):
    """Devuelve seis filas (t=0,...,5) coherentes y cinco gradientes."""
    pesos = [float(valor) for valor in pesos_iniciales]
    y = float(y)
    alpha = float(alpha)
    if len(pesos) != 6:
        raise ValueError("Se requieren los seis pesos iniciales del ejemplo.")
    if not all(isfinite(v) for v in pesos + [y, alpha]) or alpha <= 0:
        raise ValueError("Los datos deben ser finitos y alpha debe ser positivo.")

    filas = []
    gradientes = []
    for t in range(6):
        prediccion, derivadas, valores_p = propagar(tuple(pesos))
        prediccion = float(prediccion)
        derivadas = [float(v) for v in derivadas]
        valores_p = [float(v) for v in valores_p]
        if len(derivadas) != 6 or len(valores_p) != 10:
            raise ValueError("propagar debe devolver seis derivadas y diez valores p.")
        if not all(isfinite(v) for v in [prediccion] + derivadas + valores_p):
            raise ValueError("La propagación produjo un valor no finito.")
        perdida = (y - prediccion) ** 2
        if not isfinite(perdida):
            raise ValueError("La pérdida dejó de ser finita.")
        filas.append({
            "t": t,
            "pesos": tuple(pesos),
            "y_pred": prediccion,
            "L": perdida,
            "p": tuple(valores_p),
        })
        if t < 5:
            gradiente = [2.0 * (prediccion - y) * d for d in derivadas]
            gradientes.append(tuple(gradiente))
            pesos = [w - alpha * g for w, g in zip(pesos, gradiente)]
            if not all(isfinite(v) for v in pesos):
                raise ValueError("La actualización produjo un peso no finito.")
    return filas, gradientes


def mostrar_tabla(filas):
    """Muestra seis decimales; el cálculo conserva su precisión interna."""
    print("t | omega1 | omega2 | omega3 | omega4 | omega5 | omega6 | y_pred | L")
    for fila in filas:
        valores = (*fila["pesos"], fila["y_pred"], fila["L"])
        print(str(fila["t"]) + " | " + " | ".join(f"{v:.6f}" for v in valores))


if __name__ == "__main__":
    print("Función preparada: cinco_iteraciones(propagar, pesos_iniciales, y).")
    print("Faltan las ecuaciones de p1,...,p10 y los datos del ejemplo de clase.")
