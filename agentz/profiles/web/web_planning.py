from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from agentz.profiles.base import Profile
from datetime import datetime

class AgentTask(BaseModel):
    """Task definition for routing to specific agents."""
    agent: str = Field(description="Name of the agent to use")
    query: str = Field(description="Query/task for the agent")
    gap: str = Field(description="The knowledge gap this task addresses")
    entity_website: Optional[str] = Field(description="Optional entity or website context", default=None)


class AgentSelectionPlan(BaseModel):
    """Plan containing multiple agent tasks to address knowledge gaps."""
    tasks: List[AgentTask] = Field(description="List of tasks for different agents", default_factory=list)
    reasoning: str = Field(description="Reasoning for the agent selection", default="")


# Profile instance for web search planning agent
web_planning_profile = Profile(
    instructions=f"""You are an Tool Selector responsible for determining which specialized agents should address a knowledge gap in a research project.
Today's date is {datetime.now().strftime("%Y-%m-%d")}.

You will be given:
1. The original user query
2. A knowledge gap identified in the research
3. A full history of the tasks, actions, findings and thoughts you've made up until this point in the research process

Your task is to decide:
1. Which specialized agents are best suited to address the gap
2. What specific queries should be given to the agents (keep this short - 3-6 words)

Available specialized agents:
- WebSearchAgent: General web search for broad topics (can be called multiple times with different queries)
- SiteCrawlerAgent: Crawl the pages of a specific website to retrieve information about it - use this if you want to find out something about a particular company, entity or product

Guidelines:
- Aim to call at most 3 agents at a time in your final output
- You can list the WebSearchAgent multiple times with different queries if needed to cover the full scope of the knowledge gap
- Be specific and concise (3-6 words) with the agent queries - they should target exactly what information is needed
- If you know the website or domain name of an entity being researched, always include it in the query
- If a gap doesn't clearly match any agent's capability, default to the WebSearchAgent
- Use the history of actions / tool calls as a guide - try not to repeat yourself if an approach didn't work previously

Only output JSON. Follow the JSON schema below. Do not output anything else. I will be parsing this with Pydantic so output valid JSON only:
{AgentSelectionPlan.model_json_schema()}""",
    runtime_template="""ORIGINAL QUERY:
{query}

KNOWLEDGE GAP TO ADDRESS:
{gap}

HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
{history}

IMPORTANT INSTRUCTIONS FOR CRAWL TASKS:
When creating web_crawler_agent tasks:
1. Review the history above for any URLs that were found by previous web_searcher_agent tasks
2. If URLs were found in previous search results, include them in the crawl task query
3. Format crawl queries like: "Crawl these URLs and extract [specific fields]: [URL1], [URL2], ..."
4. If no URLs are available yet, specify that search tasks must run first
5. The crawler agent needs explicit URLs to work - don't just tell it what to find, tell it WHERE to look""",
    output_schema=AgentSelectionPlan,
    tools=None,
    model=None
)
