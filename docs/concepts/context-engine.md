---
title: Context Engine
description: Centralized context primitives that keep multi-agent runs in sync.
---

# Context engine

The context engine is the authoritative source of truth for prompts, state, and artifacts. It ensures that every agent sees the same world, no matter which pipeline invokes it.

## Primitives

- **Profiles** — typed templates containing reusable system + assistant prompts.
- **State** — mutable key-value store scoped by run, agent, or tool.
- **Artifacts** — rich payloads (files, embeddings, structured JSON) referenced by agents downstream.
- **Events** — timestamped records emitted during a run for tracing and analytics.

```python
from contextagent.context import Context

context = Context(["profiles", "state"])

context.set_state("analysis.notes", "Focus on customer sentiment and product gaps.")
profile = context.get_profile("data_science/analyst")
context.log_event(kind="checkpoint", data={"step": "analysis"})
```

## Access patterns

| Operation | Method | Notes |
| --------- | ------ | ----- |
| Read profile | `context.get_profile(name)` | Returns structured config objects |
| Update state | `context.set_state(path, value)` | Path-based addressing with dot notation |
| Fetch artifacts | `context.get_artifact(name)` | Supports lazy loading for large files |

Use state scopes to keep data isolated:

- `run.*` for orchestration-wide metadata.
- `agent.*` for specific agent instructions or scratchpads.
- `tool.*` for tool outputs and caches.

## Versioning

Profiles and templates can be versioned individually. Example directory layout:

```
contextagent/
  profiles/
    data_science/
      analyst.yaml
      analyst@v2.yaml
    support/
      triage.yaml
```

Set the active revision in pipeline configs:

```yaml
agents:
  analyst:
    profile: data_science/analyst@v2
```

## Persistence

- **In-memory** for local development.
- **Redis / Postgres adapters** for distributed runs.
- **Artifact stores** (S3, GCS, local disk) configured per environment.

!!! note "Custom adapters"
    Implement `ContextStore` and `ArtifactStore` interfaces to plug in your own persistence layer.

With the context engine in place, explore [tooling integrations](tooling.md) to extend agent capabilities.
