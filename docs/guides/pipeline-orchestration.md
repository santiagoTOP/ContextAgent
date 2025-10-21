---
title: Orchestrate Pipelines
description: Coordinate multi-agent pipelines with tracing, retries, and branching logic.
---

# Orchestrate pipelines

ContextAgent pipelines mirror PyTorch modules: inherit `BasePipeline`, define `run`, and wire up steps through the context core. This guide shows how to build a branching pipeline with retries and observability.

## 1. Scaffold the pipeline

```python
from pipelines.base import BasePipeline
from pydantic import BaseModel

class ResearchQuery(BaseModel):
    topic: str
    tone: str = "neutral"

class ResearchPipeline(BasePipeline[ResearchQuery]):
    async def run(self, query: ResearchQuery):
        with self.tracer.span("research.run", attributes=query.dict()):
            plan = await self.context.agents["planner"](query.topic)
            findings = await self.collect_sources(plan)
            draft = await self.context.agents["writer"](
                plan=plan,
                findings=findings,
                tone=query.tone,
            )
            return await self.review(draft)
```

## 2. Fan out work

```python
    async def collect_sources(self, plan):
        tasks = [
            self.context.agents["researcher"](task)
            for task in plan.tasks
        ]
        return await self.concurrent(tasks, label="collect_sources")
```

## 3. Handle retries

```python
    async def review(self, draft):
        for attempt in self.retry(max_attempts=3, name="review"):
            with attempt:
                critique = await self.context.agents["reviewer"](draft=draft)
                if critique.score >= 0.8:
                    return critique.final_report
                attempt.need_retry(reason="Quality score too low")
```

## 4. Emit artifacts

```python
        artifact = self.artifacts.save_markdown("report.md", critique.final_report)
        self.artifacts.link("latest-report", artifact)
        return artifact
```

## 5. Route branches

```python
        if critique.requires_followup:
            return await self.context.agents["researcher"](
                critique.followup_prompt,
            )
        return critique.final_report
```

## 6. Configure tracing + telemetry

```yaml title="pipelines/configs/research.yaml"
telemetry:
  exporter: otlp
  endpoint: https://telemetry.contextagent.ai
  sampling_rate: 0.1
retries:
  default:
    max_attempts: 3
    backoff: exponential
```

## 7. Run the pipeline

```bash
uv run python -m pipelines.research --topic "GenAI evaluation frameworks"
```

## Observability checklist

- ✅ Spans show each agent + tool invocation.
- ✅ Artifacts tracked per run and accessible via pipeline manager UI.
- ✅ Retries labeled with cause + resolution.

!!! tip
    Use the `pipelines.manager` module to schedule recurring jobs or connect to your orchestration platform (Airflow, Temporal, Dagster).

Next: learn how to [deploy pipelines to production](deployment.md).
