---
title: Tutorial – Build a Support Assistant
description: Create a triage + resolution assistant that classifies tickets and drafts responses.
---

# Tutorial: Build a support assistant

!!! info "What you'll build"
    A two-agent pipeline that triages support tickets, queries knowledge bases, and drafts helpful responses in under 30 minutes.

## Prerequisites

- Completed the [Quickstart](../quickstart.md)
- Knowledge base stored in `data/support_articles/`
- Provider keys configured in `.env`

## Step 1 — Create the pipeline config

```yaml title="pipelines/configs/support.yaml"
agents:
  triage:
    profile: support/triage@v1
  responder:
    profile: support/responder@v1
tools:
  knowledge_base:
    path: contextagent.tools.notebook.VectorSearchTool
    scopes: [support]
  escalation:
    path: contextagent.tools.support.EscalationTool
pipelines:
  support_assistant:
    entrypoint: pipelines.support.SupportPipeline
```

## Step 2 — Author agent profiles

```yaml title="contextagent/profiles/support/triage.yaml"
prompt: |
  You are an experienced support triage specialist...
state:
  severity_threshold: 0.8
tools:
  allow: [knowledge_base]
```

```yaml title="contextagent/profiles/support/responder.yaml"
prompt: |
  You craft empathetic, helpful, and concise replies...
tools:
  allow: [knowledge_base, escalation]
```

## Step 3 — Implement the pipeline

```python
from pipelines.base import BasePipeline
from pydantic import BaseModel

class SupportQuery(BaseModel):
    ticket_id: str
    body: str

class SupportPipeline(BasePipeline[SupportQuery]):
    async def run(self, query: SupportQuery):
        triage = await self.context.agents["triage"](body=query.body)
        if triage.escalate:
            return await self.context.tools.escalation(ticket=query.ticket_id, reason=triage.reason)

        docs = await self.context.tools.knowledge_base(
            question=query.body,
            limit=5,
        )
        reply = await self.context.agents["responder"](
            ticket=query.body,
            docs=docs,
        )
        return reply
```

## Step 4 — Add observability

```python
        with self.tracer.span("support.reply", attributes={"ticket": query.ticket_id}):
            ...
```

!!! tip
    Capture every generated reply as an artifact: `self.artifacts.save_text("replies/{query.ticket_id}.md", reply)`

## Step 5 — Run the tutorial

```bash
uv run python -m pipelines.support --ticket-id 1234 --path data/tickets/1234.json
```

## Checkpoint

- ✅ Triage agent labels priority and next action.
- ✅ Responder agent drafts a reply referencing the knowledge base.
- ✅ Escalations route to the escalation tool with metadata.

Next up: polish the experience with [Pipeline Orchestration](../guides/pipeline-orchestration.md).
