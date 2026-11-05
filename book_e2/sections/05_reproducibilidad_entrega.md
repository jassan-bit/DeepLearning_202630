# 5. Reproducibilidad y entrega

Este libro se construye desde `book_e2/`, con `_config.yml` e `_toc.yml` propios. Comparte el repositorio y el entorno editorial con E1/E3, pero su compilación no importa las carpetas de esos libros ni ejecuta sus experimentos.

Desde la raíz del repositorio, después de instalar `requirements-book.lock.txt` en un entorno Python 3.12:

```bash
.venv/bin/jupyter-book build book_e2 --all --warningiserror --keep-going
.venv/bin/python scripts/check_book_links.py --book-dir book_e2
```

El workflow común publica este HTML en `/entregable2/`, junto con los otros libros en sus rutas. La construcción del informe no entrena redes ni abre datos de mercado.

El código experimental, dependencias de entrenamiento, registros, resultados y materiales de entrega específicos se incorporarán después de definir la propuesta y conocer la guía oficial. El estado inicial del sitio no significa que esas evidencias ya existan.
