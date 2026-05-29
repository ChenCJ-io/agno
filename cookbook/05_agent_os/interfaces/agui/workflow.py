"""
Workflow via AG-UI
==================

Exposes an Agno Workflow through the AG-UI interface. The workflow's
lifecycle events (workflow_started, step_started, step_completed,
workflow_completed) are streamed to the AG-UI client as AG-UI events.

A keyword-based Router decides between a quick chat reply (for greetings
and short messages) and the full research-then-summarize pipeline (for
substantive questions). Mirrors the canonical agno selector pattern in
cookbook/04_workflows/05_conditional_branching/router_basic.py.
"""

from typing import List

from agno.agent.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.os import AgentOS
from agno.os.interfaces.agui import AGUI
from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

chat_agent = Agent(
    name="Chat",
    model=OpenAIResponses(id="gpt-5.4-mini"),
    instructions="Reply briefly and warmly to greetings or short small-talk messages.",
    markdown=True,
)

researcher = Agent(
    name="Researcher",
    model=OpenAIResponses(id="gpt-5.4-mini"),
    instructions="Research the topic and return three key facts as bullets.",
    markdown=True,
)

summarizer = Agent(
    name="Summarizer",
    model=OpenAIResponses(id="gpt-5.4-mini"),
    instructions="Summarize the input into a single paragraph.",
    markdown=True,
)

# ---------------------------------------------------------------------------
# Steps + keyword-based selector
# ---------------------------------------------------------------------------

chat_step = Step(name="chat", agent=chat_agent)
research_step = Step(name="research", agent=researcher)
summarize_step = Step(name="summarize", agent=summarizer)


def chat_vs_research_router(step_input: StepInput) -> List[Step]:
    """Route small talk to a quick chat reply; everything else to research+summarize."""
    text = (step_input.input or "").lower().strip()
    chat_signals = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "yo",
        "good morning",
        "good evening",
        "good afternoon",
    }
    if text in chat_signals or len(text.split()) <= 2:
        return [chat_step]
    return [research_step, summarize_step]


# ---------------------------------------------------------------------------
# Workflow + AGUI mount
# ---------------------------------------------------------------------------

research_workflow = Workflow(
    name="Adaptive Workflow",
    description="Greet small talk directly; research and summarize for substantive questions.",
    steps=[
        Router(
            name="chat_or_research_router",
            selector=chat_vs_research_router,
            choices=[chat_step, research_step, summarize_step],
            description="Pick a quick chat reply for small talk, otherwise research+summarize.",
        ),
    ],
)

agent_os = AgentOS(
    workflows=[research_workflow],
    interfaces=[AGUI(workflow=research_workflow)],
)
app = agent_os.get_app()


if __name__ == "__main__":
    agent_os.serve(app="workflow:app", reload=True, port=9001)
