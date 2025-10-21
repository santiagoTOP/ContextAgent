---
title: Tooling & Integrations
description: How ContextAgent exposes external capabilities to your agents.
---

# Tooling & integrations

Tools let agents interact with external systems such as web search, code execution, or internal APIs. ContextAgent treats every tool as a structured contract with policies, schemas, and runtime adapters.

## Tool anatomy

```python
from contextagent.tools import Tool
from pydantic import BaseModel

class ResearchInput(BaseModel):
    query: str
    depth: int = 5

class WebSearchTool(Tool[ResearchInput, str]):
    name = "web_search"
    description = "Search the web and aggregate findings."

    async def run(self, payload: ResearchInput) -> str:
        results = await self.client.search(payload.query, depth=payload.depth)
        return self.summarize(results)
```

- **Schema-first** — Inputs/outputs are typed with Pydantic models so the LLM receives clear affordances.
- **Async by default** — Tools run concurrently with pipeline steps.
- **Policies** — Tools declare scopes and rate limits so agents only call approved capabilities.

## Registration

Declare tools in pipeline configs:

```yaml
tools:
  web_search:
    path: contextagent.tools.web.WebSearchTool
    timeout: 30
    scopes: [research]
```

Agents automatically receive the tool definition if their profile lists the appropriate scopes.

## Built-in tools

| Module | What it does |
| ------ | ------------ |
| `contextagent.tools.web` | Web search + extraction |
| `contextagent.tools.github` | Repo cloning, issues, PR summarization |
| `contextagent.tools.notebook` | Sandbox python execution & plotting |
| `contextagent.tools.mcp` | Connect to Model Context Protocol servers |

## Creating custom tools

1. Subclass `Tool` with your payload models.
2. Implement `run` for the async execution.
3. Register the tool in YAML with scopes.
4. Reference the scope in an agent profile.

!!! tip
    Tools can emit artifacts via `self.artifacts.save(...)`. The context engine automatically shares them with downstream agents.

## Observability

- Tool invocations are traced as child spans.
- Inputs/outputs are redacted according to your `redactions` config.
- Failures bubble up to the pipeline runtime for retries or fallbacks.

Next, jump to the [guides](../guides/custom-tool.md) for end-to-end tutorials.
