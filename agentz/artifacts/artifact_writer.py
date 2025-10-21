"""ArtifactWriter persists run data to markdown and HTML files."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import JsonLexer, PythonLexer, get_lexer_by_name

if TYPE_CHECKING:
    from agentz.artifacts.reporter import AgentStepRecord, PanelRecord


def _utc_timestamp() -> str:
    """Return current UTC timestamp with second precision."""
    return datetime.utcnow().replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def _json_default(obj: Any) -> Any:
    """Fallback JSON serialiser for arbitrary objects."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _markdown_to_html(text: str) -> str:
    """Convert markdown text to HTML with syntax highlighting."""
    extensions = [
        "fenced_code",
        "tables",
        "nl2br",
        "sane_lists",
    ]
    html = markdown.markdown(text, extensions=extensions)
    return html


def _highlight_json(json_str: str) -> str:
    """Apply syntax highlighting to JSON string."""
    formatter = HtmlFormatter(style="friendly", noclasses=False)
    highlighted = highlight(json_str, JsonLexer(), formatter)
    return highlighted


def _get_pygments_css() -> str:
    """Get CSS for Pygments syntax highlighting."""
    formatter = HtmlFormatter(style="friendly", noclasses=False)
    return ""


class ArtifactWriter:
    """Collects run data and persists it as markdown and HTML artifacts."""

    def __init__(
        self,
        *,
        base_dir: Path,
        pipeline_slug: str,
        workflow_name: str,
        experiment_id: str,
    ) -> None:
        self.base_dir = base_dir
        self.pipeline_slug = pipeline_slug
        self.workflow_name = workflow_name
        self.experiment_id = experiment_id

        self.run_dir = base_dir / pipeline_slug / experiment_id
        self.terminal_md_path = self.run_dir / "terminal_log.md"
        self.terminal_html_path = self.run_dir / "terminal_log.html"
        self.final_report_md_path = self.run_dir / "final_report.md"
        self.final_report_html_path = self.run_dir / "final_report.html"

        self._panels: List[PanelRecord] = []
        self._agent_steps: Dict[str, AgentStepRecord] = {}
        self._groups: Dict[str, Dict[str, Any]] = {}
        self._iterations: Dict[str, Dict[str, Any]] = {}
        self._final_result: Optional[Any] = None

        self._start_time: Optional[float] = None
        self._started_at_iso: Optional[str] = None
        self._finished_at_iso: Optional[str] = None

    # ------------------------------------------------------------------ basics

    def start(self, config: Any) -> None:  # noqa: ARG002 - config reserved for future use
        """Prepare filesystem layout and capture start metadata."""
        if self._start_time is not None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()
        self._started_at_iso = _utc_timestamp()

    def set_final_result(self, result: Any) -> None:
        """Store pipeline result for later persistence."""
        self._final_result = result

    # ----------------------------------------------------------------- logging

    def record_status_update(
        self,
        *,
        item_id: str,
        content: str,
        is_done: bool,
        title: Optional[str],
        border_style: Optional[str],
        group_id: Optional[str],
    ) -> None:  # noqa: D401 - keeps signature compatibility
        """Currently unused; maintained for interface compatibility."""
        # Intentionally no-op for the simplified reporter.
        return None

    def record_group_start(
        self,
        *,
        group_id: str,
        title: Optional[str],
        border_style: Optional[str],
        iteration: Optional[int] = None,
    ) -> None:
        """Record the start of an iteration/group."""
        timestamp = _utc_timestamp()
        payload = {
            "group_id": group_id,
            "title": title,
            "border_style": border_style,
            "iteration": iteration,
            "started_at": timestamp,
        }
        self._groups[group_id] = payload
        if iteration is not None:
            iter_key = f"iter-{iteration}"
            self._iterations.setdefault(
                iter_key,
                {
                    "iteration": iteration,
                    "title": title or f"Iteration {iteration}",
                    "started_at": timestamp,
                    "finished_at": None,
                    "panels": [],
                    "agent_steps": [],
                },
            )

    def record_group_end(
        self,
        *,
        group_id: str,
        is_done: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """Record the end of an iteration/group."""
        timestamp = _utc_timestamp()
        group_meta = self._groups.get(group_id)
        if not group_meta:
            return
        group_meta.update(
            {
                "title": title or group_meta.get("title"),
                "is_done": is_done,
                "finished_at": timestamp,
            }
        )
        iteration = group_meta.get("iteration")
        if iteration is not None:
            iter_key = f"iter-{iteration}"
            iteration_meta = self._iterations.setdefault(
                iter_key,
                {
                    "iteration": iteration,
                    "title": title or f"Iteration {iteration}",
                    "panels": [],
                    "agent_steps": [],
                },
            )
            iteration_meta["finished_at"] = timestamp

    def record_agent_step_start(
        self,
        *,
        step_id: str,
        agent_name: str,
        span_name: str,
        iteration: Optional[int],
        group_id: Optional[str],
        printer_title: Optional[str],
    ) -> None:
        """Capture metadata when an agent step begins."""
        from agentz.artifacts.reporter import AgentStepRecord
        
        record = AgentStepRecord(
            agent_name=agent_name,
            span_name=span_name,
            iteration=iteration,
            group_id=group_id,
            started_at=_utc_timestamp(),
        )
        self._agent_steps[step_id] = record
        if iteration is not None:
            iter_key = f"iter-{iteration}"
            iteration_meta = self._iterations.setdefault(
                iter_key,
                {
                    "iteration": iteration,
                    "title": printer_title or f"Iteration {iteration}",
                    "panels": [],
                    "agent_steps": [],
                },
            )
            iteration_meta["agent_steps"].append(record)

    def record_agent_step_end(
        self,
        *,
        step_id: str,
        status: str,
        duration_seconds: float,
        error: Optional[str] = None,
    ) -> None:
        """Update agent step telemetry on completion."""
        timestamp = _utc_timestamp()
        record = self._agent_steps.get(step_id)
        if record:
            record.finished_at = timestamp
            record.duration_seconds = round(duration_seconds, 3)
            record.status = status
            record.error = error

    def record_panel(
        self,
        *,
        title: str,
        content: str,
        border_style: Optional[str],
        iteration: Optional[int],
        group_id: Optional[str],
    ) -> None:
        """Persist panel meta for terminal & HTML artefacts."""
        from agentz.artifacts.reporter import PanelRecord
        
        record = PanelRecord(
            title=title,
            content=content,
            border_style=border_style,
            iteration=iteration,
            group_id=group_id,
            recorded_at=_utc_timestamp(),
        )
        self._panels.append(record)
        if iteration is not None:
            iter_key = f"iter-{iteration}"
            iteration_meta = self._iterations.setdefault(
                iter_key,
                {
                    "iteration": iteration,
                    "title": f"Iteration {iteration}",
                    "panels": [],
                    "agent_steps": [],
                },
            )
            iteration_meta["panels"].append(record)

    # ------------------------------------------------------------- finalisation

    def finalize(self) -> None:
        """Persist markdown + HTML artefacts."""
        if self._start_time is None or self._finished_at_iso is not None:
            return
        self._finished_at_iso = _utc_timestamp()
        duration = round(time.time() - self._start_time, 3)

        terminal_sections = self._build_terminal_sections()
        terminal_md = self._render_terminal_markdown(duration, terminal_sections)
        terminal_html = self._render_terminal_html(duration, terminal_sections)

        self.terminal_md_path.write_text(terminal_md, encoding="utf-8")
        self.terminal_html_path.write_text(terminal_html, encoding="utf-8")

        final_md, final_html = self._render_final_report()
        self.final_report_md_path.write_text(final_md, encoding="utf-8")
        self.final_report_html_path.write_text(final_html, encoding="utf-8")

    def _build_terminal_sections(self) -> List[Dict[str, Any]]:
        """Collect ordered sections for terminal artefacts."""
        sections: List[Dict[str, Any]] = []

        # Iteration/scoped panels
        for iter_key, meta in sorted(
            self._iterations.items(),
            key=lambda item: item[1].get("iteration", 0),
        ):
            sections.append(
                {
                    "title": meta.get("title") or iter_key,
                    "started_at": meta.get("started_at"),
                    "finished_at": meta.get("finished_at"),
                    "panels": meta.get("panels", []),
                    "agent_steps": meta.get("agent_steps", []),
                }
            )

        # Global panels (no iteration)
        global_panels = [
            record
            for record in self._panels
            if record.iteration is None
        ]
        if global_panels:
            sections.append(
                {
                    "title": "General",
                    "started_at": None,
                    "finished_at": None,
                    "panels": global_panels,
                    "agent_steps": [],
                }
            )

        return sections

    def _render_terminal_markdown(
        self,
        duration: float,
        sections: List[Dict[str, Any]],
    ) -> str:
        """Render the terminal log as Markdown."""
        lines: List[str] = []
        lines.append(f"# Terminal Log · {self.workflow_name}")
        lines.append("")
        lines.append(f"- **Experiment ID:** `{self.experiment_id}`")
        lines.append(f"- **Started:** {self._started_at_iso or '–'}")
        lines.append(f"- **Finished:** {self._finished_at_iso or '–'}")
        lines.append(f"- **Duration:** {duration} seconds")
        lines.append("")

        if not sections:
            lines.append("_No panels recorded during this run._")
            lines.append("")
            return "\n".join(lines)

        for section in sections:
            lines.append(f"## {section['title']}")
            span = ""
            if section.get("started_at") or section.get("finished_at"):
                span = f"{section.get('started_at', '–')} → {section.get('finished_at', '–')}"
            if span:
                lines.append(f"*Time:* {span}")
            lines.append("")

            agent_steps: List[AgentStepRecord] = section.get("agent_steps", [])
            if agent_steps:
                lines.append("### Agent Steps")
                for step in agent_steps:
                    duration_txt = (
                        f"{step.duration_seconds}s"
                        if step.duration_seconds is not None
                        else "pending"
                    )
                    status = step.status
                    error = f" · Error: {step.error}" if step.error else ""
                    lines.append(
                        f"- **{step.agent_name}** · {step.span_name} "
                        f"({duration_txt}) · {status}{error}"
                    )
                lines.append("")

            panels: List[PanelRecord] = section.get("panels", [])
            for panel in panels:
                panel_title = panel.title or "Panel"
                lines.append(f"### {panel_title}")
                lines.append("")
                lines.append("```")
                lines.append(panel.content.rstrip())
                lines.append("```")
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _render_terminal_html(
        self,
        duration: float,
        sections: List[Dict[str, Any]],
    ) -> str:
        """Render the terminal log as standalone HTML with proper markdown rendering."""
        body_sections: List[str] = []

        for section in sections:
            panels_html: List[str] = []
            for panel in section.get("panels", []):
                # Convert markdown to HTML
                content_html = _markdown_to_html(panel.content)
                panel_html = f"""
        <article class="panel">
          <h3>{panel.title or "Panel"}</h3>
          <div class="panel-content">
            {content_html}
          </div>
        </article>
        """.strip()
                panels_html.append(panel_html)

            agent_html: List[str] = []
            for step in section.get("agent_steps", []):
                info_dict = {
                    "agent": step.agent_name,
                    "span": step.span_name,
                    "status": step.status,
                    "duration_seconds": step.duration_seconds,
                    "error": step.error,
                }
                info_json = json.dumps(info_dict, default=_json_default, indent=2)
                highlighted_json = _highlight_json(info_json)
                agent_html.append(f'<li class="agent-step">{highlighted_json}</li>')

            timeframe = ""
            if section.get("started_at") or section.get("finished_at"):
                timeframe = (
                    f"<p class=\"time\">{section.get('started_at', '–')} → "
                    f"{section.get('finished_at', '–')}</p>"
                )

            agents_block = ""
            if agent_html:
                agents_block = '<ul class="agents">' + "".join(agent_html) + "</ul>"

            panels_block = "".join(panels_html)

            block = (
                f"\n      <section class=\"section\">\n"
                f"        <h2>{section['title']}</h2>\n"
                f"        {timeframe}\n"
                f"        {agents_block}\n"
                f"        {panels_block}\n"
                "      </section>\n      "
            ).strip()
            body_sections.append(block)

        sections_html = "\n".join(body_sections) if body_sections else "<p>No panels recorded.</p>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Terminal Log · {self.workflow_name}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.6;
      color: #24292e;
      background: #fff;
      padding: 20px;
    }}

    .container {{
      max-width: 900px;
      margin: 0 auto;
    }}

    header {{
      margin-bottom: 32px;
      padding-bottom: 16px;
      border-bottom: 1px solid #e1e4e8;
    }}

    h1 {{
      font-size: 2em;
      font-weight: 600;
      margin-bottom: 12px;
    }}

    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      font-size: 0.9em;
      color: #586069;
    }}

    .meta div {{
      display: flex;
      gap: 8px;
    }}

    .meta strong {{
      color: #24292e;
    }}

    main {{
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}

    .section {{
      border: 1px solid #e1e4e8;
      border-radius: 6px;
      padding: 16px;
      background: #f6f8fa;
    }}

    .section h2 {{
      font-size: 1.5em;
      font-weight: 600;
      margin-bottom: 12px;
      color: #24292e;
    }}

    .section .time {{
      color: #6f42c1;
      font-size: 0.85em;
      margin-bottom: 12px;
      font-weight: 500;
    }}

    .agents {{
      list-style: none;
      margin-bottom: 16px;
    }}

    .agent-step {{
      margin-bottom: 12px;
      padding: 8px 12px;
      background: white;
      border-left: 3px solid #6f42c1;
      border-radius: 4px;
    }}

    .agent-step pre {{
      margin: 0;
      overflow-x: auto;
      font-size: 0.9em;
    }}

    .panel {{
      margin-top: 12px;
      padding: 12px;
      background: white;
      border: 1px solid #e1e4e8;
      border-radius: 6px;
    }}

    .panel h3 {{
      font-size: 1.1em;
      font-weight: 600;
      margin-bottom: 12px;
      color: #24292e;
    }}

    .panel-content {{
      font-size: 0.95em;
    }}

    .panel-content h1 {{
      font-size: 1.8em;
      margin-top: 24px;
      margin-bottom: 12px;
    }}

    .panel-content h2 {{
      font-size: 1.5em;
      margin-top: 20px;
      margin-bottom: 10px;
    }}

    .panel-content h3 {{
      font-size: 1.2em;
      margin-top: 16px;
      margin-bottom: 8px;
    }}

    .panel-content h4, .panel-content h5, .panel-content h6 {{
      margin-top: 12px;
      margin-bottom: 6px;
    }}

    .panel-content p {{
      margin-bottom: 12px;
    }}

    .panel-content ul, .panel-content ol {{
      margin-left: 24px;
      margin-bottom: 12px;
    }}

    .panel-content li {{
      margin-bottom: 6px;
    }}

    .panel-content code {{
      background: #f6f8fa;
      border-radius: 3px;
      padding: 2px 6px;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 0.9em;
      color: #24292e;
    }}

    .panel-content pre {{
      background: #f6f8fa;
      border: 1px solid #e1e4e8;
      border-radius: 6px;
      padding: 12px;
      overflow-x: auto;
      margin-bottom: 12px;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 0.85em;
    }}

    .panel-content pre code {{
      background: transparent;
      padding: 0;
      color: inherit;
    }}

    .panel-content blockquote {{
      border-left: 4px solid #dfe2e5;
      padding: 0 12px;
      margin: 0 0 12px 0;
      color: #6a737d;
    }}

    .panel-content table {{
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 12px;
    }}

    .panel-content table th, .panel-content table td {{
      border: 1px solid #dfe2e5;
      padding: 8px 12px;
      text-align: left;
    }}

    .panel-content table th {{
      background: #f6f8fa;
      font-weight: 600;
    }}

    .panel-content strong {{
      font-weight: 600;
      color: #24292e;
    }}

    .panel-content em {{
      font-style: italic;
    }}

    /* Pygments syntax highlighting */
    .c {{ color: #6a737d; }}
    .err {{ color: #d73a49; }}
    .k {{ color: #d73a49; }}
    .n {{ color: #24292e; }}
    .s {{ color: #032f62; }}
    .o {{ color: #d73a49; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Terminal Log · {self.workflow_name}</h1>
      <div class="meta">
        <div><strong>Experiment ID:</strong> {self.experiment_id}</div>
        <div><strong>Started:</strong> {self._started_at_iso or "–"}</div>
        <div><strong>Finished:</strong> {self._finished_at_iso or "–"}</div>
        <div><strong>Duration:</strong> {duration} seconds</div>
      </div>
    </header>
    <main>
      {sections_html}
    </main>
  </div>
</body>
</html>
"""

    def _render_final_report(self) -> tuple[str, str]:
        """Render final report markdown + HTML."""
        if isinstance(self._final_result, str):
            body_md = self._final_result.rstrip()
        elif self._final_result is not None:
            body_md = json.dumps(self._final_result, indent=2, default=_json_default)
        else:
            body_md = "No final report generated."

        markdown_content = f"# Final Report · {self.workflow_name}\n\n{body_md}\n"

        # Convert markdown to HTML
        body_html = _markdown_to_html(body_md)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Final Report · {self.workflow_name}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.6;
      color: #24292e;
      background: #fff;
      padding: 20px;
    }}

    .container {{
      max-width: 900px;
      margin: 0 auto;
    }}

    header {{
      margin-bottom: 32px;
      padding-bottom: 16px;
      border-bottom: 1px solid #e1e4e8;
    }}

    h1 {{
      font-size: 2em;
      font-weight: 600;
      margin-bottom: 12px;
    }}

    .content {{
      background: white;
      padding: 20px;
    }}

    .content h1 {{
      font-size: 2em;
      font-weight: 600;
      margin-top: 24px;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid #e1e4e8;
    }}

    .content h2 {{
      font-size: 1.5em;
      font-weight: 600;
      margin-top: 24px;
      margin-bottom: 12px;
    }}

    .content h3 {{
      font-size: 1.25em;
      font-weight: 600;
      margin-top: 16px;
      margin-bottom: 10px;
    }}

    .content h4, .content h5, .content h6 {{
      font-weight: 600;
      margin-top: 12px;
      margin-bottom: 8px;
    }}

    .content p {{
      margin-bottom: 12px;
    }}

    .content ul, .content ol {{
      margin-left: 24px;
      margin-bottom: 12px;
    }}

    .content li {{
      margin-bottom: 6px;
    }}

    .content code {{
      background: #f6f8fa;
      border-radius: 3px;
      padding: 2px 6px;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 0.9em;
      color: #24292e;
    }}

    .content pre {{
      background: #f6f8fa;
      border: 1px solid #e1e4e8;
      border-radius: 6px;
      padding: 16px;
      overflow-x: auto;
      margin-bottom: 12px;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 0.85em;
      line-height: 1.45;
    }}

    .content pre code {{
      background: transparent;
      padding: 0;
      color: inherit;
    }}

    .content blockquote {{
      border-left: 4px solid #dfe2e5;
      padding: 0 12px;
      margin: 0 0 12px 0;
      color: #6a737d;
    }}

    .content table {{
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 12px;
    }}

    .content table th, .content table td {{
      border: 1px solid #dfe2e5;
      padding: 8px 12px;
      text-align: left;
    }}

    .content table th {{
      background: #f6f8fa;
      font-weight: 600;
    }}

    .content strong {{
      font-weight: 600;
      color: #24292e;
    }}

    .content em {{
      font-style: italic;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Final Report · {self.workflow_name}</h1>
    </header>
    <div class="content">
      {body_html}
    </div>
  </div>
</body>
</html>
"""
        return markdown_content, html_content

    # ------------------------------------------------------------------ helpers

    def ensure_started(self) -> None:
        """Raise if reporter not initialised."""
        if self._start_time is None:
            raise RuntimeError("ArtifactWriter.start must be called before logging events.")
