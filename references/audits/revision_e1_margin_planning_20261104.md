# Auditoría independiente de escenarios de margen y potencia

Estado: **PASS**.

No project planning/math/training/analysis helper imported; minimum R enumerated directly; MDE independently bisected in original units.

Se verificaron 36 escenarios de varianza pareada, R por aproximación normal, mínimo entero R de t no central y MDE para R=10; además de cuatro umbrales operativos. Ambos extremos de cada mínimo R se comprobaron.

La referencia sigue siendo MLP/D7 elegida con CV2024. No se seleccionó el baseline con test. α=0,025 bilateral y potencia objetivo 0,80 son supuestos de planificación para dos comparaciones futuras.

R mínimo condicionado a los escenarios: 3–1007. Estos extremos no son una recomendación única de tamaño muestral.

Los resultados no acreditan potencia alcanzada, un piloto prospectivo ni un margen financiero validado. El margen aplicado sigue pendiente de contexto de decisión/costo. Detectar delta frente a cero no demuestra exceder una frontera delta.

Los cuatro objetivos neuronales ilustrativos conservan mayor RMSE que la referencia CV de persistencia. El salto necesario para igualar esa referencia es contexto descriptivo, no un margen aplicado o una garantía de utilidad.

No se leyeron archivos raw, pesos o vectores de predicciones, ni se entrenó o puntuó un modelo.

## Límites

- Future-model dispersion and correlation are hypothetical, not measured from an advanced model.
- Normal paired differences and independent seed pairs are planning assumptions conditional on the historical calendar.
- These tables do not measure achieved power and do not replace a prospective pilot.
- A finite-sample noncentral-t calculation is exact only within its assumed normal-difference model.
- The applied utility margin remains open until the user/application supplies a defensible decision/cost context.
- The baseline comes from frozen CV2024 selection; no test-driven baseline selection or new model scoring occurred.

## Fuentes y hashes

- `experiments/e1_audit_improvements_20261104/protocol.json`: `e6319edf4c320707d51fc987b59f62967538784718ffe369d5a0e5b468576442`.
- `experiments/filipe_multihistory_20261104/freeze.json`: `7c789040819598a4339b60c7be56e104b82ef94dbba6b70ae904128387f47837`.
- `scripts/plan_e1_advanced_margin.py`: `3340841ede12fec9eb1a9d44592a1a93f5c908745bcb498c6a84d378202fb5c1`.
- `scripts/verify_e1_margin_planning.py`: `c656c216264a29050123a4866038aa16a9492097cdfa9db15bfc92c196293d71`.
- `results/e1_audit_improvements_20261104/planning/summary.json`: `42b047dd99dc48751cf3e7ae4d6f159c2ae35372c40726919fe784b8bb82f8f5`.
- `results/e1_audit_improvements_20261104/planning/registration.json`: `f9c52a66b9c3348d60c12320b86f0d9a6be2941d83cde17803ab02cc56072b2b`.
- `results/e1_audit_improvements_20261104/planning/README.md`: `08dfdbd37c8669caf6e5682dc5122642ab22436173f8fbae7c3e1cc10ead8b03`.
- `results/e1_audit_improvements_20261104/planning/advanced_margin_scenarios.csv`: `119175ed78e06016aa94cfb981dbe4bc28cb53b99dcbd1c8cbb6c7ca05c1e0f1`.
- `results/e1_audit_improvements_20261104/planning/operational_threshold_scenarios.csv`: `0a419197c47f4b356b4682ff23fa992d7978f1e401ec24f55158ada9265848e6`.
