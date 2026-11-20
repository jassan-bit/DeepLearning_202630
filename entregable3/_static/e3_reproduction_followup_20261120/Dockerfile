FROM python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY requirements-runtime.lock /opt/replay/requirements-runtime.lock
COPY wheelhouse /opt/wheelhouse
RUN python -m pip install --no-cache-dir --no-index --find-links=/opt/wheelhouse --only-binary=:all: --no-deps --require-hashes -r /opt/replay/requirements-runtime.lock && rm -rf /opt/wheelhouse
WORKDIR /payload
ENTRYPOINT ["python", "/payload/replay_accepted_v3.py"]
