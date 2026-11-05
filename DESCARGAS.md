# Descargas del Entregable 1

[Descargar directamente desde la unidad 8 del Jupyter Book](https://jassan-bit.github.io/DeepLearning_202630/sections/08_entregables.html#descargar-entregable-1).

[Descargar los siete entregables juntos (ZIP)](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/entregable1_completo.zip) · [Ver carpeta de descargas](https://github.com/jassan-bit/DeepLearning_202630/tree/entregables)

Estos archivos se descargan directamente desde GitHub y se utilizan fuera del Jupyter Book.

| Ítem | Archivo | Contenido |
|---|---|---|
| 8.1 | [Informe HTML](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_1_informe_html.zip) | Informe navegable, figuras y archivos públicos de respaldo |
| 8.2 | [Código](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_2_codigo.zip) | Scripts EDA, MLP/LSTM/CNN1D, análisis estadístico, pruebas, dependencias y `BUILD.md` |
| 8.3 | [Obtención del dataset](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_3_dataset_instrucciones.zip) | Instrucciones exactas, descargador de Binance y manifiestos de adquisición |
| 8.4 | [Literatura y flujo](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_4_literatura.zip) | Tabla de síntesis, extracción, evaluación de calidad y diagrama de búsqueda |
| 8.5 | [Benchmark e inferencia](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_5_benchmark.zip) | Métricas agregadas de CV/test y suplemento estadístico con sus límites |
| 8.6 | [Configuraciones](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_6_configuraciones.zip) | Dependencias, semillas y catálogos de 1.440 segmentos CV y 120 ajustes finales |
| 8.7 | [Declaración](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_7_declaracion.zip) | Métrica, margen, unidad de independencia, partición y cronología de decisiones |

Descomprime el archivo antes de abrirlo y consulta su `LEEME.md`. En el informe, abre `informe/intro.html`. Si el navegador restringe los recursos locales, desde la carpeta extraída ejecuta:

```bash
python -m http.server 8000 --directory informe
```

Después abre <http://localhost:8000>. Las fórmulas MathJax y las fuentes enlazadas pueden necesitar internet.

El código incluye los ejecutores y el manual para preparar sus entornos. La verificación de resultados individuales originales requiere el paquete privado de evidencia descrito en `BUILD.md`. El paquete público de datos entrega la ruta exacta y el descargador para obtener los ZIP brutos de Binance; esos ZIP no forman parte del paquete privado. Las predicciones individuales y los pesos se conservan en la entrega privada. Las tablas y figuras públicas no sustituyen esos insumos.

La [versión y el inventario de los paquetes](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/catalogo.json) quedan registrados en `catalogo.json`. Para comprobar las descargas, guarda [SHA256SUMS.txt](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/SHA256SUMS.txt) junto a los ZIP y ejecuta:

```bash
sha256sum --ignore-missing -c SHA256SUMS.txt
```

La revisión incorpora el texto completo de Miura, Pichl y Kaizoji (2019), que evalúa MLP y CNN independientes para RV de Bitcoin. La síntesis vigente de seis estudios y las tablas de extracción están en `book/_static/literature_e1/recovery_20261105/` dentro del ZIP de literatura. La CNN del artículo no declara dimensionalidad ni todos sus hiperparámetros; las redes del proyecto son adaptaciones propias, no réplicas exactas. La revisión es posterior al benchmark y no cambia sus resultados. El criterio académico avanzado conserva su definición posterior al test y su alcance futuro.
