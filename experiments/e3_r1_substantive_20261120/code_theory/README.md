# Auditoría de código y teoría para el cierre de R1

Este paquete documenta una diferencia programada de estructura y entrenamiento
de V3, su correspondencia con las ecuaciones y sus límites. No emite por sí
solo acreditación de novedad para el problema; corresponde al contraste de
antecedentes y al dictamen académico independiente.

Se usó IA para lectura estática, comparación algebraica y redacción. No se
importaron módulos de modelos, no se entrenó, no se ejecutó inferencia, no se
abrieron precios y no se recalculó el piloto.

- `theory_audit.md`: formulación, controles, propiedades, supuestos y objeciones.
- `contribution_wording.md`: declaración concreta y afirmaciones que deben evitarse.
- `mechanism_audit.json`: identidad, hashes, representación y alcance.
- `mechanism_crosswalk.csv`: catorce elementos vinculados al código.
- `counterarguments.json`: siete objeciones con respuesta y límites.
- `frozen_sources/`: dos fuentes propias leídas del commit efectivo del piloto,
  conservadas sin modificación y vinculadas a sus hashes.
- `verify_static_audit.py` y `static_verification.json`: comprobación sólo de
  texto, JSON y aritmética exacta; no importan Torch.

Desde un checkout del repositorio con el paquete instalado:

```bash
python RUTA_DEL_PAQUETE/verify_static_audit.py --repo-root .
```

La verificación comprueba diez insumos, la firma causal de `forward`, los
conteos estáticos de parámetros, la identidad de sigma7 poblacional como
polinomio de 28 coeficientes racionales y la coherencia de los cien casos
artificiales ya guardados. No genera otras leyes ni recupera independencia
confirmatoria. El resultado adverso del piloto y sus reglas siguen intactos.
