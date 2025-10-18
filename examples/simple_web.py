from pipelines.simple_web import SimpleWebPipeline, WebSearchDebugQuery

# Load the default configuration file and start the pipeline using the one-parameter API.
pipe = SimpleWebPipeline("pipelines/configs/web_searcher.yaml")

query = WebSearchDebugQuery(
    prompt="Find recent breakthroughs in reinforcement learning and cite sources."
)

pipe.run_sync(query)
