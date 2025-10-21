---
title: Quickstart
description: Install ContextAgent, configure keys, and launch your first multi-agent pipeline.
---

# Quickstart

!!! tip "Prerequisites"
    - Python 3.10+
    - [uv](https://docs.astral.sh/uv/) package manager
    - API keys for at least one LLM provider (OpenAI, Anthropic, Google, etc.)

## 1. Install ContextAgent

=== "Clone + Develop"
    ```bash
    git clone https://github.com/context-machine-lab/contextagent.git
    cd contextagent
    uv sync
    ```

=== "PyPI"
    ```bash
    pip install contextagent
    ```

## 2. Configure providers

Copy the template `.env` and add your keys.

```bash
cp .env.example .env
```

Update the following fields at a minimum:

| Variable | Description |
| -------- | ----------- |
| `OPENAI_API_KEY` | Required for OpenAI models |
| `ANTHROPIC_API_KEY` | Needed for Claude agents |
| `GOOGLE_API_KEY` | Enables Gemini support |

!!! warning
    Do not commit `.env`. Keep your provider credentials private.

## 3. Run an example pipeline

```bash
uv run python -m examples.web_researcher \
  --topic "Latest transformer architecture techniques"
```

The pipeline spins up researcher + writer agents, fetches web context, and delivers a concise report.

## 4. Explore the pipeline manager UI

```bash
uv run python frontend/app.py --host localhost --port 9090 --debug
```

Visit `http://localhost:9090` to trigger pipelines, stream logs, and monitor artifacts.

## 5. Build your own agent

Create a class that inherits from `BasePipeline` and implements `run`.

```python
from pipelines.base import BasePipeline

class SupportPipeline(BasePipeline):
    def __init__(self, config_path: str):
        super().__init__(config_path)

    async def run(self, query):
        ticket = await self.context.agents["triage"](query)
        summary = await self.context.agents["writer"](ticket)
        return summary
```

Next, point the pipeline at a YAML config that defines the agents, tools, and prompts you need. Continue in the [Custom Tool guide](guides/custom-tool.md).

## Where to go next

- Understand how the [context engine works](concepts/context-engine.md).
- Learn to [orchestrate production pipelines](guides/pipeline-orchestration.md).
- Dive into the [Python API reference](reference/python.md).
