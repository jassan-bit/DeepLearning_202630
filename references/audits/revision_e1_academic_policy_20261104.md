# Auditoría de política académica y planificación conjunta

Estado: **PASS**.

No project math/planning helper imported; both component minimum integer R values independently enumerated using upper-tail noncentral-t probabilities.

Se verificaron 27 escenarios, dos varianzas y dos mínimos enteros R por escenario. Se usó una única media avanzada por fila frente a ambas fronteras del 5 %; la persistencia es determinista respecto a las semillas.

El estimando es la mejora de la razón de medias de RMSE por semilla: 1−E[m_A]/E[m_ref]. Con E[m_ref]>0, la condición E[0,95m_ref−m_A]>0 representa superar el 5 %. No es la media de las razones por semilla, que puede diferir para una referencia neuronal aleatoria.

Cada contraste es unilateral con α=0,025 y potencia marginal objetivo0,90. El mínimo conjunto es max(5,R_neuronal,R_persistencia). La cota de unión da una potencia conjunta al menos0,80 bajo los supuestos, sin asumir independencia entre contrastes.

R conjunto ilustrativo: 5–229; no es una recomendación única ni potencia alcanzada.

La política se fechó después de observar el test actual y antes del nuevo cálculo de planificación. El protocolo global original conserva su SHA-256. MLP/D7 se toma de la selección congelada por CV, sin reselección por test.

Conviene explicitar esta definición de razón de medias en el libro o una aclaración fechada separada, conservando los hashes de la política ya calculada.

El 5 % queda definido como convención operativa para un benchmark académico futuro; no acredita utilidad financiera ni se presenta como margen previo al estudio terminado. El piloto prospectivo y otra evaluación independiente permanecen pendientes.

## Límites

- A future academic convention is now defined; this does not turn it into a margin registered before the already-inspected test.
- Dispersion, correlation and advanced-model alternatives remain illustrative; no advanced model was trained or measured.
- R and the union-bound power refer to the assumed normal paired-seed model conditional on a fixed calendar, not achieved market generalization power.
- The 5% boundary is an academic operational choice, not a validated economic or financial decision threshold.
- A prospective pilot and a new independently evaluated cohort are still required before claiming prospective design sufficiency.
- No new test period is assigned and the current baseline/epochs/checkpoints remain unchanged.

## Fuentes y hashes

- `scripts/verify_e1_academic_policy.py`: `5b6b68ec5e25d34b80e9e4d777cc1d94809319958c71ae749a7436777d966471`.
- `experiments/e1_audit_improvements_20261104/academic_margin_policy.json`: `765a845cbc59c115787d1279b6456e2f45a69bd456cd16ee81a02f1b6216ace6`.
- `experiments/e1_audit_improvements_20261104/protocol.json`: `e6319edf4c320707d51fc987b59f62967538784718ffe369d5a0e5b468576442`.
- `experiments/filipe_multihistory_20261104/freeze.json`: `7c789040819598a4339b60c7be56e104b82ef94dbba6b70ae904128387f47837`.
- `scripts/plan_e1_academic_benchmark.py`: `4830172a4f6032e95f1d86fb37be944ba76df8a28d32f1193eb2b1dda2d0c1c9`.
- `results/e1_audit_improvements_20261104/planning/academic_joint_power_scenarios.csv`: `54557282f012d5a256954856f8eb39175b752b886f58728d9870999714027aaa`.
- `results/e1_audit_improvements_20261104/planning/academic_planning_summary.json`: `e0729334713fab3d00a4e6b7995d56f50e5bec051ce83285b30859c1501cd604`.
- `results/e1_audit_improvements_20261104/planning/academic_margin_README.md`: `18dee5cab6115a7d16008b94a087378f1797f5fe06bec5061d238a505c44e40c`.
- `results/e1_audit_improvements_20261104/planning/summary.json`: `42b047dd99dc48751cf3e7ae4d6f159c2ae35372c40726919fe784b8bb82f8f5`.
- `results/filipe_multihistory_20261104/test/raw_access_started.json`: `cb07e3ea72184c1b3ca0e6e4d825e08bf8e0a5c741f6d17e561acf5577edbe08`.
- `results/filipe_multihistory_20261104/test/execution_summary.json`: `850b4ef78efd8ac360aa6bd41078351f0bacfcd7cae6d1e80a94517f9a0b7049`.
