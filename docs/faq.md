---
title: FAQ
description: Answers to common questions about ContextAgent.
---

# FAQ

??? question "What providers are supported?"
    ContextAgent ships adapters for OpenAI, Anthropic, Google, DeepSeek, and Azure OpenAI. Implement `LLMClient` to add more.

??? question "Does ContextAgent store data?"
    By default artifacts and state live in memory. Configure disk, S3, or database adapters to persist data between runs.

??? question "Can I use my own prompts?"
    Yes. Author profiles in `contextagent/profiles/` and reference them in pipeline configs. Version prompts with the `@vN` suffix.

??? question "Where are logs stored?"
    Logs stream to stdout. Enable telemetry exporters to forward traces + logs to your observability stack.

??? question "How do I deploy to production?"
    Follow the [deployment guide](guides/deployment.md) to containerize workers, configure stores, and expose an API.
