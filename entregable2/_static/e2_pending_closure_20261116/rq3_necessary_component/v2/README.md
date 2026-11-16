# RQ3: respuesta condicionada por el componente necesario de coste

Esta es una deducción exploratoria posterior al benchmark y a los resultados CPU
ya conocidos; no recupera un prerregistro histórico.

Joint improvement is not established in the measured CPU environment: its necessary reduction in forward cost is not supported.
Los diez p de coste existentes son exactamente 1; los diez límites inferiores son 1.
Para cualquier p predictivo CPU válido u, max(u,p_coste)≥p_coste. Si ese límite
es 1, una IUT compatible con ESE componente CPU no podría rechazar, tampoco con
Holm10. Si el límite es menor, se conserva su valor y la conclusión que permite.

El CSV publica el límite inferior, no un p de IUT estimado. El componente
predictivo permanece ausente; el JSON lo conserva como null. No se usa precisión
GPU, no se calculan nuevos contrastes y no se ejecuta la IUT original714.
La deducción sólo aplica a las hipótesis, referencia GRU32, pesos, entradas de
desarrollo y mediciones CPU FP32/batch1 de las fuentes enlazadas por hash.
No establece equivalencia ni imposibilidad de mejora en otros entornos.

`necessary_component_proof.json` contiene la identidad, diseños y procedencia;
`necessary_component_bounds_10.csv` conserva las diez cotas y sus estados.
Cerrar esta respuesta condicionada no equivale a completar la ejecución integral
del método original de RQ3. Repetición desde el repositorio con Python estándar:

```sh
python scripts/e2_rq3_necessary_component.py --output /tmp/e2-rq3-necessary-repeat
```
