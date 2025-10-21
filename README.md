<div align="center">

# AgentZ: Agent from Zero

**A Context-Central Multi-Agent System Platform**

</div>

AgentZ is a context-central multi-agent systems framework. AgentZ focuses on efficiently managing the context of each agent, binds all agents through centralized context engineering. Context-central design philosophy significantly improves the reusage of key components and eases the development and maintenance of scaled multi-agent system.

## Features

- **🎯 Context-Central Architecture** - All agents and pipelines are defined based on context operations
- **🔄 Component Reusability** - Unified context design enables easy reuse of agents, tools, and flows
- **📚 Declarative Flows** - Define complex multi-agent workflows through structured, declarative specifications
- **🛠️ Stateful Execution** - Persistent conversation state tracks all agent interactions and tool results
- **🧠 Structured IO Contracts** - Type-safe communication between agents via Pydantic models
- **⚙️ Scalable Development** - Simplified maintenance and extension of multi-agent systems

## Demo

## News
- **[2025-10]** AgentZ is released now! 

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable package management.

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for more options.

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/context-machine-lab/agentz.git
cd agentz

# Sync dependencies
uv sync
```

## Quick Start

```python
from pipelines.data_scientist import DataScientistPipeline, DataScienceQuery

pipe = DataScientistPipeline("pipelines/configs/data_science.yaml")

query = DataScienceQuery(
    prompt="Analyze the dataset and build a predictive model",
    data_path="data/banana_quality.csv"
)

pipe.run_sync(query)
```

### Web UI (Pipeline Manager)

Run the lightweight Flask web UI to submit and monitor pipelines with live logs:

```bash
uv run python frontend/app.py --host localhost --port 9090 --debug
```

Then open `http://localhost:9090` in your browser. The UI streams live status and panels from the running pipeline and lets you stop active runs.

## Steps to Build Your Own System

#### Step 1 - New Agent (Optional)

Create new agents by writing agent profiles in `agentz/profiles`, or use built-in agents:

* Observe agent - reflect on the process, update knowledge gaps
* Evaluate agent - evaluate the current state, form up next steps
* Routing agent - call tool agents automatically
* Writer agent - generate the final report
* ......

#### Step 2 - Custom Pipeline

Create a new pipeline in `pipelines`. Allocate agents in `__init__` and define workflow in `run`. Context management is fully automated!

#### Step 3 - Execution

Execute your pipeline via creating a module like

```python
from pipelines.your_pipeline import YourPipeline, YourQuery

pipe = YourPipeline(...)

query = DataScienceQuery(
    prompt=...
)

pipe.run_sync(query)
```

For more implementation details, please refer to out [Documentation](https://deepwiki.com/context-machine-lab/agentz).

## Architecture

AgentZ is organised around a **central conversation state** and a set of declarative
flow specifications that describe how agents collaborate. The main
components you will interact with are:

- **`pipelines/`** – High level orchestration that wires agents together.
- **`agentz/agents/`** – Capability definitions for manager agents and tool agents.
- **`agentz/flow/`** – Flow primitives (`FlowRunner`, `FlowNode`, `IterationFlow`) that
  execute declarative pipelines.
- **`agentz/memory/`** – Structured state management (`ConversationState`,
  `ToolExecutionResult`, global memory helpers).
- **`examples/`** – Example scripts showing end-to-end usage.

```
agentz/
├── pipelines/
│   ├── base.py               # Base pipeline with config management & helpers
│   ├── flow_runner.py        # Declarative flow executor utilities
│   └── data_scientist.py     # Reference research pipeline
├── agentz/
│   ├── agents/
│   │   ├── manager_agents/   # Observe, evaluate, routing, writer agents
│   │   └── tool_agents/      # Specialised tool executors
│   ├── flow/                 # Flow node definitions and runtime objects
│   ├── memory/               # Conversation state & persistence utilities
│   ├── llm/                  # LLM adapters and setup helpers
│   └── tools/                # Built-in tools
└── examples/
    └── data_science.py       # Example workflows
```


## Benchmarks

AgentZ's context-central design has been validated on multiple research benchmarks:

- **Data Science Tasks**: Efficient context sharing enables streamlined automated ML pipelines
- **Complex Reasoning**: Centralized state tracking improves multi-step reasoning coordination
- **Deep Research**: Search based complex reasoning and report generation

*Detailed benchmark results and comparisons coming soon.*

<!-- ## Roadmap

- [x] Persistence Process - Stateful agent workflows
- [x] Experience Learning - Memory-based reasoning
- [x] Tool Design - Dynamic tool creation
- [ ] Workflow RAG - Retrieval-augmented generation for complex workflows
- [ ] MCPs - Model Context Protocol support for enhanced agent communication -->

## Documentation

More details are available at 📖[Documentation](https://deepwiki.com/context-machine-lab/agentz).

## Citation

If you use AgentZ in your research, please cite:

```bibtex
@software{agentz2025,
  title={AgentZ: A Context-Central Multi-Agent Systems Framework},
  author={Zhimeng Guo, Hangfan Zhang, Siyuan Xu, Huaisheng Zhu, Teng Xiao, Minhao Cheng},
  year={2025},
  url={https://https://github.com/TimeLovercc/agentz}
}
```

## Contributing

We welcome contributions! AgentZ is designed to be a community resource for multi-agent research. Please open an issue or submit a pull request.


## Acknowledgements

AgentZ's context-central design is inspired by the multi-agent systems research community and best practices in distributed state management. We thank the developers of LLM frameworks and orchestration tools that informed this architecture.

## Contributors

<a href="https://github.com/context-machine-lab/agentz/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=context-machine-lab/agentz&max=999&columns=12&anon=1" />
</a>

---

<div align="center">

**AgentZ**: Building intelligent agents from zero to hero 🚀

</div>
