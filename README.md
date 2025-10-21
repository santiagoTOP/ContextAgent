<div align="center">

# ContextAgent

**A Context-Central Multi-Agent Framework**

[![Notion Blog](https://img.shields.io/badge/Notion_Blog-000000?style=for-the-badge&logo=notion&logoColor=white)](https://www.notion.so/zhimengg/Agent-Z-27f111ca2fa080a28de4d76c49f0b08d?source=copy_link)
[![Documentation](https://img.shields.io/badge/Documentation-007ACC?style=for-the-badge&logo=markdown&logoColor=white)](https://context-machine-lab.github.io/ContextAgent/)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-582C83?style=for-the-badge&logo=wikipedia&logoColor=white)](https://deepwiki.com/context-machine-lab/contextagent)
[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white)](./assets/wechat.jpg)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/74my3Wkn)


</div>

ContextAgent is a lightweight, context-central multi-agent systems framework designed for easy context engineering. It focuses on efficiently managing the context of each agent and binds all agents through simplified, centralized context operations. Unlike traditional multi-agent frameworks, ContextAgent treats agents simply as LLMs with different contexts, eliminating unnecessary complexity. Built with a PyTorch-like API, developers can create sophisticated multi-agent systems with minimal code.


## 🌟 Features

- **📋 Context = Template + State**: Dynamic context management based on [Anthropic's blog](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).
- **🔀 Decoupled Agent Design**: Agent = LLM + Context. All agents are just LLMs with different contexts.
- **🎨 PyTorch-Like Pipeline API**: Inherit `BasePipeline`, define async `run()`, use `@autotracing` for tracing.
- **🌐 Multi-LLM Support**: Works with OpenAI, Claude, Gemini, DeepSeek, and more.
- **🧩 Modular Architecture**: Built on OpenAI Agents SDK with clear separation: context, agents, pipeline.
- **⚡ Easy to Use & Customize**: Reuse pipelines with just a query; create new ones with familiar patterns.


## 📢 News
- **[2025-10]** ContextAgent v0.1.1 is released now!


## 🎬 Demo

**Data Science Pipeline**

![Data Science Demo](assets/DS.gif)

**Web Research Pipeline**

![Web Research Demo](assets/WEB.gif)




## 📦 Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable package management.

#### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for more options.

#### Setup Environment

```bash
# Clone the repository
git clone https://github.com/context-machine-lab/contextagent.git
cd contextagent

# Sync dependencies
uv sync
```

##### Configure API Keys

ContextAgent requires API keys for LLM providers. Set up your environment in `.env` file:

```bash
# Copy the example environment file
cp .env.example .env
# Edit .env and add your API keys
```
See [.env.example](.env.example) for complete configuration options.

#### Alternative: Install from PyPI

You can also install ContextAgent directly from PyPI without cloning the repository:

```bash
pip install contextagent
```

Note: You'll still need to configure API keys in a `.env` file for LLM providers.


## 🚀 Quick Start

#### Run Built-in Examples

Try out ContextAgent with pre-configured example pipelines:

**Data Science Pipeline** - Automated ML pipeline for data analysis and model building:
```bash
uv run python -m examples.data_science
```

**Web Research Pipeline** - Search-based research with information extraction:
```bash
uv run python -m examples.web_researcher
```

#### Basic API Pattern

Here's how to use ContextAgent in your own code:

```python
from pipelines.data_scientist import DataScientistPipeline, DataScienceQuery

# Initialize pipeline with config
pipe = DataScientistPipeline("pipelines/configs/data_science.yaml")

# Create a query
query = DataScienceQuery(
    prompt="Analyze the dataset and build a predictive model",
    data_path="data/banana_quality.csv"
)

# Execute
pipe.run_sync(query)
```

#### Web UI (Pipeline Manager)

Run the lightweight Flask web UI to submit and monitor pipelines with live logs:

```bash
uv run python frontend/app.py --host localhost --port 9090 --debug
```

Then open `http://localhost:9090` in your browser. The UI streams live status and panels from the running pipeline and lets you stop active runs.

## Steps to Build Your Own System

## 🛠️ Steps to Build Your Own System

ContextAgent uses a **PyTorch-like API** for building multi-agent systems. Follow these steps to create your own pipeline:

##### Step 1 - Define Pipeline Class

Inherit from `BasePipeline` and call `super().__init__(config)`:

```python
from pipelines.base import BasePipeline
from pydantic import BaseModel

class YourPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__(config)
        # Your initialization here
```

##### Step 2 - Create Context and Bind Agents

Create a centralized `Context`, get the LLM, and bind agents:

```python
from contextagent.agent import ContextAgent
from contextagent.context import Context

class YourPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__(config)

        self.context = Context(["profiles", "states"])
        llm = self.config.llm.main_model

        # Manager agent example
        self.routing_agent = ContextAgent(self.context, profile="routing", llm=llm)

        # Tool agents example
        self.tool_agents = {
            "data_loader": ContextAgent(self.context, profile="data_loader", llm=llm),
            "analyzer": ContextAgent(self.context, profile="analyzer", llm=llm),
            # ... add more agents
        }
        self.context.state.register_tool_agents(self.tool_agents)
```

##### Step 3 - Define Async Run with @autotracing

Define your workflow in an async `run()` method:

```python
import asyncio
from pipelines.base import autotracing

class YourPipeline(BasePipeline):
    @autotracing()
    async def run(self, query: YourQuery):
        self.context.state.set_query(query)

        while self.iteration < self.max_iterations:
            self.iterate()

            # Call agents directly
            routing_result = await self.routing_agent(query)
```

##### Step 4 - Define Query Model and Execute

Create a Pydantic model and run your pipeline:

```python
class YourQuery(BaseModel):
    prompt: str
    # Add your custom fields

# Execute
pipe = YourPipeline("pipelines/configs/your_config.yaml")
query = YourQuery(prompt="Your task here")
result = pipe.run_sync(query)
```

##### Full Example Reference

See complete implementations in:
- **[examples/data_science.py](examples/data_science.py)** - Basic pipeline usage
- **[pipelines/data_scientist.py](pipelines/data_scientist.py)** - Full pipeline implementation reference
- **[Docs Portal](https://context-machine-lab.github.io/ContextAgent/)** - Tutorials, reference, and guides


## 🏗️ Architecture

ContextAgent is organized around a **central conversation state** and a profile-driven agent system. All agents are coordinated through a unified `Context` that manages iteration state and shared information.

#### Core Components:

- **`pipelines/`** – Workflow orchestration and configuration management
- **`contextagent/agent/`** – ContextAgent implementation with context awareness and execution tracking
- **`contextagent/context/`** – Centralized conversation state and coordination
- **`contextagent/profiles/`** – Agent profiles defining capabilities (manager, data, web, code, etc.)
- **`contextagent/tools/`** – Tool implementations for data processing, web operations, and code execution
- **`examples/`** – Example pipelines demonstrating usage
- **`frontend/`** – Web UI for pipeline management and monitoring

#### Project Structure:

```
contextagent/
├── pipelines/          # Workflow orchestration
├── contextagent/
│   ├── agent/          # ContextAgent implementation
│   ├── context/        # Conversation state management
│   ├── profiles/       # Agent profiles (manager, data, web, code)
│   ├── tools/          # Tool implementations
│   └── artifacts/      # Output formatting
├── examples/           # Example pipelines
└── frontend/           # Web UI
```

For more details, see the [docs portal](https://context-machine-lab.github.io/ContextAgent/).


## 📊 Benchmarks

ContextAgent's context-central design has been validated on multiple research benchmarks:

- **Data Science Tasks**: Efficient context sharing enables streamlined automated ML pipelines
- **Complex Reasoning**: Centralized state tracking improves multi-step reasoning coordination
- **Deep Research**: Search based complex reasoning and report generation

*Detailed benchmark results and comparisons coming soon.*


## 🗺️ Roadmap

- [ ] Persistence Process - Stateful agent workflows
- [ ] Experience Learning - Memory-based reasoning
- [ ] Tool Design - Dynamic tool creation
- [ ] Frontend Support - Enhanced web UI for system interaction and monitoring
- [ ] MCP Support - Full Model Context Protocol integration for extended agent capabilities
- [ ] Claude Code Skill Support - Native integration with Claude Code environment
- [ ] Workflow RAG - Retrieval-augmented generation for complex workflows


## 📚 Documentation

- Hosted docs: [https://context-machine-lab.github.io/ContextAgent/](https://context-machine-lab.github.io/ContextAgent/)
- Deep-dive articles: [DeepWiki](https://deepwiki.com/context-machine-lab/contextagent)
- Local preview:
  ```bash
  uv sync --extra docs
  uv run mkdocs serve
  ```
- Static build:
  ```bash
  uv run mkdocs build
  ```


## 🙏 Acknowledgements

ContextAgent's context-central design is inspired by the multi-agent systems research community and best practices in distributed state management. We are particularly grateful to:

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) - For providing a lightweight, powerful framework for multi-agent workflows and the financial research agent example that demonstrates structured research patterns.
- [Youtu-Agent](https://github.com/TencentCloudADP/youtu-agent) - For its flexible agent framework architecture with open-source model support and tool generation capabilities.
- [agents-deep-research](https://github.com/qx-labs/agents-deep-research) - For its iterative deep research implementation showcasing multi-agent orchestration for complex reasoning tasks.

We thank the developers of these frameworks and the broader LLM community whose work informed this architecture.


## 🤝 Contributing

We welcome contributions! ContextAgent is designed to be a community resource for multi-agent research. Please open an issue or submit a pull request.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📖 Citation

If you use ContextAgent in your research, please cite:

```bibtex
@misc{contextagent2025,
  title={ContextAgent: Lightweight Context-Driven Multi-Agent System Design},
  author={Zhimeng Guo, Hangfan Zhang, Siyuan Xu, Huaisheng Zhu, Teng Xiao, Jingyi Chen, Minhao Cheng},
  year={2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url={https://github.com/context-machine-lab/contextagent}
}
```
