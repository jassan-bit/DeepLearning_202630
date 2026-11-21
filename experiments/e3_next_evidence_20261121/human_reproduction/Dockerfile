FROM python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 MPLBACKEND=Agg HOME=/tmp
COPY requirements-book.lock.txt /opt/e3/requirements-book.lock.txt
RUN python -m pip install --no-cache-dir --no-deps --report /opt/e3/pip-install-report.json -r /opt/e3/requirements-book.lock.txt && python -m pip check
WORKDIR /case
ENTRYPOINT ["python", "/case/kit/run_book_case.py", "--case", "/case"]
