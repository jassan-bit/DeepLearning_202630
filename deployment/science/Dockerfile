# Scientific aggregate replay, separate from the ONNX web-service image.
FROM python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/e3-scientific-matplotlib
WORKDIR /analysis
COPY deployment/science/requirements.lock.txt /analysis/deployment/science/requirements.lock.txt
ARG SCIENCE_OFFLINE=0
RUN --mount=type=bind,source=deployment/science/offline_packages,target=/tmp/packages,readonly \
    if [ "$SCIENCE_OFFLINE" = "1" ]; then \
      python -m pip install --no-cache-dir --no-index --find-links=/tmp/packages -r deployment/science/requirements.lock.txt; \
    else \
      python -m pip install --no-cache-dir -r deployment/science/requirements.lock.txt; \
    fi && python -m pip check
COPY experiments/e3_estadistica_posterior_20261117 /analysis/experiments/e3_estadistica_posterior_20261117
COPY experiments/e3_revision_bloque_d_20261117/metrics_supplement.py /analysis/experiments/e3_revision_bloque_d_20261117/metrics_supplement.py
COPY experiments/e3_remediacion_20261118/reproduction /analysis/experiments/e3_remediacion_20261118/reproduction
COPY book_e3/_static/e3_estadistica_posterior_20261117 /analysis/book_e3/_static/e3_estadistica_posterior_20261117
COPY book_e3/_static/e3_revision_bloque_d_20261117 /analysis/book_e3/_static/e3_revision_bloque_d_20261117
RUN chmod -R a+rX /analysis
USER 10001:10001
ENTRYPOINT ["python", "experiments/e3_remediacion_20261118/reproduction/reproduce.py"]
CMD ["run", "--data", "/inputs", "--output", "/output"]
