# Implementación de recetas: corte sintético 2026-11-09

Once recetas con dimensiones completas; un paso supervisado por modelo y uno en cada fase de preentrenamiento SAE. No hay entrenamiento de mercado, HPO/NAS ni resultado de superioridad. Los tiempos y pérdidas son diagnósticos de esta ejecución artificial CPU, no presupuestos ni resultados Bitcoin.

Los cuatro cuerpos nuevos están separados de los prototipos históricos. Se reutilizan seis adaptadores y el híbrido sin modificar sus scripts. La verificación de 70 archivos corresponde sólo al subconjunto declarado de cuatro adaptadores externos, no a repositorios completos ni a los cuerpos propios. Los pesos de los autores no se cargan.

Pruebas: 14 cuerpos, 6 catálogo y 19 controles, todas ejecutadas sin omisiones en sus entornos correspondientes. El catálogo requiere E2_CURRENT_AUTHOR_SOURCES apuntando al árbol verificado. Los controles requieren statsmodels 0.14.5; sin esa dependencia, once pruebas de integración se omiten y no deben contarse como aprobadas.

Reproducir con Python 3.12.14 y PyTorch 2.15.0+cpu: `python scripts/smoke_e2_recipe_catalogue.py --source-root <fuentes_verificadas> --output <archivo_nuevo_synthetic.json>`. Los datos artificiales se generan dentro del flujo; no acepta archivo de mercado. Una salida existente se rechaza.

Pruebas de controles: `python tests/test_e2_temporal_controls.py` en entorno separado con statsmodels 0.14.5, NumPy 2.3.5, pandas 2.2.3 y SciPy 1.17.0. Se verifica también un caso de no convergencia: falla sin rescate. No se ejecuta una búsqueda de los 18 órdenes ni se certifican las cohortes reales.

Integridad: `python scripts/publish_e2_recipe_implementation.py --check`. El manifiesto cubre exclusivamente los tres artefactos públicos de esta carpeta y su espejo. El recibo liga los hashes de implementación; no reemplaza un preregistro ni certifica las pruebas no ejecutadas.
