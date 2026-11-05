# Descargas del Entregable 1 · GRU–LSTM–TCN

[Descargar directamente desde la unidad8 del Jupyter Book](https://jassan-bit.github.io/DeepLearning_202630/sections/08_entregables.html#descargar-entregable-1).

[Descargar los siete productos juntos](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/entregable1_completo.zip) · [Carpeta de descargas](https://github.com/jassan-bit/DeepLearning_202630/tree/entregables)

| Ítem | Archivo | Contenido vigente |
|---|---|---|
| 8.1 | [Informe HTML](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_1_informe_html.zip) | Libro navegable de GRU/LSTM/TCN, figuras y agregados |
| 8.2 | [Código](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_2_codigo.zip) | EDA, API actual, ejecutor portable, diagnósticos, pruebas, locks y BUILD.md |
| 8.3 | [Obtención del dataset](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_3_dataset_instrucciones.zip) | Instrucciones exactas, descargador oficial y manifiestos |
| 8.4 | [Literatura y flujo](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_4_literatura.zip) | Siete estudios cripto,40 referencias y selección GRU/LSTM/TCN |
| 8.5 | [Benchmark e inferencia](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_5_benchmark.zip) | Cuatro entradas, métricas actuales, pares del trío y diagnósticos exploratorios |
| 8.6 | [Configuraciones](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_6_configuraciones.zip) | Semillas, locks,1.440 segmentosCV y120 registros finales; reuso identificado |
| 8.7 | [Declaración](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/8_7_declaracion.zip) | Métrica, margen futuro, independencia, partición y cronología explícita |

Abrir los ZIP descomprimidos y leer LEEME.md. El HTML se inicia en informe/intro.html; si el navegador restringe recursos, ejecutar `python -m http.server 8000 --directory informe` y abrir `http://localhost:8000`. MathJax y los enlaces externos pueden necesitar internet.

Los paquetes públicos contienen código, metadatos, resultados agregados y figuras; la ilustración temporal usa una semilla fija4101. No redistribuyen los datos brutos, los vectores tabulares de predicciones individuales, los escaladores o los pesos conservados para auditoría. El paquete8.3 describe la obtención de los archivos oficiales. BUILD.md distingue la verificación pública, la evidencia completa y una reproducción en una salida nueva.

El adaptador público se incorpora después de la ejecución observada; no se le atribuye su hash retroactivamente. La evidencia publicada conserva120 trayectoriasCV y un ajuste final de80 redes nuevas más40 LSTM idénticas reutilizadas. Una reproducción fresca portable ajusta sus propios120 modelos finales.

El test histórico ya había sido observado. Los contrastes dentro de la terna son posteriores al test y exploratorios. El piloto del procedimiento actual sigue pendiente; las diez semillas no son un tamaño derivado de potencia 0,80. TCN conserva respaldo menos recurrente. El margen académico futuro se declara después del test y no implica utilidad financiera.

[Catálogo de versión](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/catalogo.json) · [SHA256SUMS.txt](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/SHA256SUMS.txt).

```bash
sha256sum --ignore-missing -c SHA256SUMS.txt
```
