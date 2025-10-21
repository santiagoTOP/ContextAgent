---
title: CLI Reference
description: Commands for managing ContextAgent pipelines and tools from the terminal.
---

# CLI reference

Use the CLI to run pipelines, inspect context state, and manage artifacts.

## Usage

```bash
uv run contextagent --help
```

```
Usage: contextagent [OPTIONS] COMMAND [ARGS]...

  ContextAgent command line interface.

Options:
  --config PATH   Path to pipeline configuration.
  --log-level TEXT  Override log level (debug, info, warning, error).
  --help          Show this message and exit.
```

## Commands

### `run`

```bash
uv run contextagent run --pipeline web_researcher --query "Transformer news"
```

| Option | Description |
| ------ | ----------- |
| `--pipeline` | Name of the pipeline defined in your config |
| `--query` | Raw text query to seed the pipeline |
| `--input` | Path to JSON/YAML file containing structured inputs |
| `--stream` | Stream tokens to stdout |

### `list`

```bash
uv run contextagent list pipelines
```

Lists available pipelines, agents, profiles, or tools.

### `artifacts`

```bash
uv run contextagent artifacts show --run-id RUN123
```

- `list` — enumerate artifacts for a run.
- `show` — preview structured artifacts.
- `download` — save artifacts to disk.

### `context`

```bash
uv run contextagent context dump --run-id RUN123 --scope agent.writer
```

Inspect state snapshots for debugging.

### `serve`

```bash
uv run contextagent serve --host 0.0.0.0 --port 9090
```

Runs the lightweight web UI for monitoring pipelines.

!!! tip
    Use `--config pipelines/configs/<name>.yaml` to target non-default setups.
