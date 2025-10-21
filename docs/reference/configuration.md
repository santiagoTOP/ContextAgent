---
title: Configuration Reference
description: Schema for pipeline, agent, and telemetry configuration files.
---

# Configuration reference

ContextAgent uses YAML for pipeline definitions. This reference outlines the core sections, defaults, and accepted values.

## Top-level schema

```yaml
version: 1
defaults:
  model: gpt-4.1-mini
  temperature: 0.3
context_store:
  provider: memory
artifact_store:
  provider: disk
  path: outputs/artifacts
agents:
  planner:
    profile: research/planner@v1
pipelines:
  research:
    entrypoint: pipelines.research.ResearchPipeline
```

## `agents`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `profile` | `str` | Profile bundle to hydrate prompt + defaults |
| `model` | `str` | Override the model for this agent |
| `planner` | `bool` | Enable planning phase before execution |
| `tools.allow` | `list[str]` | Tool scopes the agent can call |

Example:

```yaml
agents:
  writer:
    profile: research/writer@v2
    model: claude-3-sonnet
    planner: false
    tools:
      allow: [github_insights, web_search]
```

## `tools`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `path` | `str` | Python import path to the tool class |
| `timeout` | `int` | Seconds before the tool call aborts |
| `scopes` | `list[str]` | Access scopes required by agents |
| `secrets` | `list[str]` | Environment variables that must be set |

## `pipelines`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `entrypoint` | `str` | Import path to the pipeline class |
| `inputs` | `dict` | Pydantic schema metadata for inputs |
| `schedule` | `str` | Optional cron expression for recurring runs |

## Observability

```yaml
telemetry:
  exporter: otlp
  endpoint: ${OTLP_ENDPOINT}
logging:
  level: info
redactions:
  fields:
    - api_key
    - password
```

!!! note
    Observability settings cascade to both agents and tools. Override per-component if you need granular control.

## Environments

Use environment overlays to manage staging vs production:

```yaml
environments:
  staging:
    defaults:
      model: gpt-4o-mini
  production:
    defaults:
      model: gpt-4.1
    context_store:
      provider: redis
      url: ${REDIS_URL}
```

### Selecting an environment

```bash
uv run contextagent run \
  --config pipelines/configs/research.yaml \
  --env production
```

## Validation

Validate configs with the built-in checker:

```bash
uv run contextagent config validate --config pipelines/configs/research.yaml
```
