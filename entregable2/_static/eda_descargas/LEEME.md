# EDA del Entregable 2: código, tablas y figuras

Este paquete permite revisar los diagnósticos y sus evidencias. Se utiliza
junto con el checkout del repositorio que publica este archivo:
https://github.com/jassan-bit/DeepLearning_202630

Los SHA-256 de MANIFIESTO.json identifican cada archivo incluido. Para ejecutar,
extraer conservando directorios sobre una copia del repositorio y crear el
entorno Python 3.12 con requirements-eda.lock.txt. Desde la raíz:

    python scripts/eda_e2_decisions.py

La entrada de desarrollo está versionada en results/eda_development del repo.
Los índices forward y los materiales heredados se conservan allí también.
La reconstrucción del perfil de la reserva ya conocida se explica en
book_e2/_static/eda_known_shift/README.md; admite caché verificado o los ZIP
oficiales, sin entrenar ni volver a puntuar modelos.

El contenido corresponde a EDA: Ridge es una referencia diagnóstica mensual;
el benchmark de diez modelos nuevos más las cinco referencias de E2 (mínimo
quince configuraciones) aún está pendiente. El perfil del test es
retrospectivo y no constituye confirmación nueva. Los resultados adversos y
las limitaciones de ruido, ruta y sellado se conservan en el capítulo.

Se incluyen agregados y derivados diagnósticos de desarrollo; los originales
de mercado y el caché de precios de test se obtienen/conservan por separado.
Fuente de datos: Binance Vision / BTCUSDT. Procesamiento: Jassan Arteta.
Los derivados mantienen atribución CC BY-NC-SA 4.0 y condiciones de la fuente,
documentadas en la Data Card. La licencia de datos se distingue de la de código.
