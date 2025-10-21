---
title: Build a Custom Tool
description: Expose your own capability to ContextAgent with typed contracts and policies.
---

# Build a custom tool

This guide walks through creating a GitHub intelligence tool that agents can call during research pipelines.

!!! info "Goal"
    Build a `github_insights` tool that fetches repo statistics and summarizes issues using GitHub's API.

## 1. Define payload schemas

```python
from pydantic import BaseModel

class GitHubInsightsInput(BaseModel):
    owner: str
    repo: str
    include_issues: bool = True

class GitHubInsightsOutput(BaseModel):
    stars: int
    forks: int
    open_issues: int
    summary: str
```

## 2. Implement the tool

```python
from contextagent.tools import Tool

class GitHubInsightsTool(Tool[GitHubInsightsInput, GitHubInsightsOutput]):
    name = "github_insights"
    description = "Summarize GitHub activity for a repository."

    async def run(self, payload: GitHubInsightsInput) -> GitHubInsightsOutput:
        repo = await self.client.get_repo(payload.owner, payload.repo)
        issues = []
        if payload.include_issues:
            issues = await self.client.list_issues(payload.owner, payload.repo, limit=10)

        summary = await self.context.agents["writer"](
            f"Summarize repo activity for {payload.owner}/{payload.repo}",
            issues=issues,
        )

        return GitHubInsightsOutput(
            stars=repo.stargazers_count,
            forks=repo.forks_count,
            open_issues=repo.open_issues_count,
            summary=summary,
        )
```

!!! tip
    Use existing agents (like `writer`) inside tools when you need high-quality language output.

## 3. Register the tool

```yaml title="pipelines/configs/research.yaml"
tools:
  github_insights:
    path: contextagent.tools.github.GitHubInsightsTool
    timeout: 45
    scopes: [research, github]
    secrets:
      - GITHUB_TOKEN
```

## 4. Grant access to agents

```yaml title="contextagent/profiles/researcher.yaml"
tools:
  allow: [github_insights, web_search]
```

## 5. Invoke from a pipeline

```python
from pipelines.base import BasePipeline

class RepoResearchPipeline(BasePipeline):
    async def run(self, query):
        repo = await self.context.agents["triage"](query)
        insights = await self.context.tools.github_insights(
            owner=repo.owner,
            repo=repo.name,
        )
        report = await self.context.agents["writer"](
            "Summarize the repository health.",
            insights=insights,
        )
        return report
```

## 6. Test locally

```bash
uv run python -m pipelines.repo_research \
  --owner context-machine-lab \
  --repo contextagent
```

!!! question "Troubleshooting"
    - **Permission denied**: ensure `GITHUB_TOKEN` is exported in your environment.
    - **Tool not visible**: verify the agent profile includes the proper scope.

Continue to [Pipeline Orchestration](pipeline-orchestration.md) to scale this tool into production.
