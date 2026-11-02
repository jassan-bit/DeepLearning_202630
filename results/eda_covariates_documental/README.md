# EDA descriptivo de regresión temporal

Se reutilizan únicamente los CSV de desarrollo 2023–2024 ya existentes. No se ajusta ningún modelo, no se ejecutan contrastes de hipótesis y no se abren archivos de prueba. Se conserva la exclusión estricta del día incompleto y la cuadrícula diaria.

`lagged_correlations_train.csv` contiene Pearson/Spearman para cinco covariables con rezagos causales de 1 a 28 días frente a sigma7, solo en 2023. `rolling_60d_correlations_development.csv` y sus conteos describen ventanas móviles de 60 días con al menos 50 pares de desarrollo. `relationship_profiles_2023_2024.csv` no constituye un contraste formal de deriva. Los niveles de precios pueden producir correlaciones espurias.

Reproducir desde el repositorio: `/workspace/.venvs/deeplearning-eda/bin/python scripts/eda_covariates_documental.py`. Los hashes, controles de alineación y límites se conservan en `summary.json`. Los diagnósticos de residuos de una referencia requieren ajuste de modelo y permanecen para E2 según el alcance solicitado.
