---
title: Deploy to Production
description: Package ContextAgent pipelines with durable storage, workers, and monitoring.
---

# Deploy to production

This guide outlines a reference deployment for ContextAgent with a queue-backed worker fleet and persistent context stores.

## Overview

!!! abstract
    1. Containerize the app with uv-managed dependencies.
    2. Provision managed Redis/Postgres for context + artifact storage.
    3. Run workers behind your orchestration layer (Celery, Temporal, Argo).
    4. Stream traces + metrics to your observability stack.

## 1. Package the service

```dockerfile title="Dockerfile"
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
CMD ["uv", "run", "pipelines.worker", "--config", "pipelines/configs/research.yaml"]
```

## 2. Configure persistent stores

```yaml title="pipelines/configs/research.yaml"
context_store:
  provider: redis
  url: ${REDIS_URL}
artifact_store:
  provider: s3
  bucket: ${S3_BUCKET}
  prefix: contextagent/artifacts
```

## 3. Run worker processes

```bash
uv run python -m pipelines.worker \
  --queue research \
  --concurrency 4
```

Scale horizontally by launching additional workers subscribed to the same queue.

## 4. Expose an API gateway

```python
from fastapi import FastAPI
from pipelines.research import ResearchPipeline, ResearchQuery

app = FastAPI()
pipeline = ResearchPipeline("pipelines/configs/research.yaml")

@app.post("/research")
async def run_pipeline(query: ResearchQuery):
    return await pipeline.run(query)
```

Deploy behind a managed ingress (Cloud Run, ECS, Kubernetes) with TLS enabled.

## 5. Wire observability

- Export traces to OTLP (Honeycomb, Datadog, Grafana Tempo).
- Push logs to your centralized logging stack.
- Emit run metrics (latency, retry counts, success ratios).

```bash
uv run python -m pipelines.worker \
  --with-telemetry otlp://otel-collector:4317
```

## 6. Establish release cadence

| Step | Action |
| ---- | ------ |
| 1 | Branch from `main` for each docs + pipeline update |
| 2 | Record prompt changes in the [changelog](../changelog.md) |
| 3 | Tag releases with `vX.Y.Z` and publish artifacts |

!!! tip
    Automate builds + docs deploys with GitHub Actions so the documentation stays in sync with your latest release.

Return to the [Quickstart](../quickstart.md) if you need a refresher.
