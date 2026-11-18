# Python 3.12.14 slim, resolved and tested for the dashboard build.
FROM python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt requirements-dashboard.lock.txt ./
ARG DASHBOARD_OFFLINE=0
RUN --mount=type=bind,source=deployment/wheels,target=/tmp/wheels,readonly \
    if [ "$DASHBOARD_OFFLINE" = "1" ]; then \
      python -m pip install --no-cache-dir --no-index --find-links=/tmp/wheels -r requirements.txt; \
    else \
      python -m pip install --no-cache-dir -r requirements.txt; \
    fi && python -m pip check
COPY --chown=10001:10001 app.py MODEL_CARD.md DATA_STATEMENT.md ./
COPY --chown=10001:10001 pages ./pages
COPY --chown=10001:10001 e3_dashboard ./e3_dashboard
COPY --chown=10001:10001 assets ./assets
COPY --chown=10001:10001 data/e3_dashboard_v1 ./data/e3_dashboard_v1
COPY --chown=10001:10001 data/e3_dashboard_training_v1 ./data/e3_dashboard_training_v1
COPY --chown=10001:10001 artifacts/e3_dashboard_v1 ./artifacts/e3_dashboard_v1
USER 10001:10001
EXPOSE 8050
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8050')+'/healthz',timeout=4).read()"
CMD ["sh", "-c", "exec gunicorn --bind \"0.0.0.0:${PORT:-8050}\" --workers 1 --threads 1 --timeout 120 app:server"]
