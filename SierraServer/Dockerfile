FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv/sierra

RUN useradd --create-home --uid 10001 sierra
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install '.[dev]'

COPY . .
RUN mkdir -p /srv/sierra/runtime && chown -R sierra:sierra /srv/sierra
USER sierra

EXPOSE 8080
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1"]
