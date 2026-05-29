"""Unit tests for workflow support in the AGUI interface."""

import pytest

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.os.interfaces.agui import AGUI
from agno.workflow import Workflow
from agno.workflow.step import Step


def create_test_workflow() -> Workflow:
    """Create a simple test workflow."""
    agent = Agent(
        name="Test Agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions="You are a helpful assistant",
    )
    step = Step(name="test_step", agent=agent)
    return Workflow(
        name="Test Workflow",
        description="A test workflow for unit testing",
        steps=[step],
    )


def test_agui_with_workflow():
    """AGUI initializes correctly when given a workflow."""
    workflow = create_test_workflow()
    agui = AGUI(workflow=workflow)

    assert agui.workflow is not None
    assert agui.workflow.name == "Test Workflow"
    assert agui.type == "agui"
    assert agui.agent is None
    assert agui.team is None


def test_agui_with_agent():
    """AGUI still initializes correctly when given an agent (regression check)."""
    agent = Agent(name="Test Agent", model=OpenAIChat(id="gpt-4o-mini"))
    agui = AGUI(agent=agent)

    assert agui.agent is not None
    assert agui.workflow is None
    assert agui.team is None


def test_agui_requires_one_entity():
    """AGUI raises ValueError when no entity is provided."""
    with pytest.raises(ValueError, match="requires an agent, team, or workflow"):
        AGUI()


def test_agui_workflow_router_creation():
    """Router with a workflow exposes AGUI routes."""
    workflow = create_test_workflow()
    agui = AGUI(workflow=workflow)
    router = agui.get_router()

    assert router is not None
    assert len(router.routes) > 0
    route_paths = [getattr(route, "path", "") for route in router.routes]
    assert any("/agui" in path or path == "" for path in route_paths)


def test_agui_router_has_status_endpoint():
    """AGUI router with a workflow exposes /status."""
    workflow = create_test_workflow()
    agui = AGUI(workflow=workflow)
    router = agui.get_router()

    route_paths = [getattr(route, "path", "") for route in router.routes]
    assert "/status" in route_paths


def test_agui_prefix_configuration():
    """Custom prefix is preserved on AGUI(workflow=...)."""
    workflow = create_test_workflow()
    agui = AGUI(workflow=workflow, prefix="/custom-agui")

    assert agui.prefix == "/custom-agui"


def test_agui_tags_configuration():
    """Custom tags are preserved on AGUI(workflow=...)."""
    workflow = create_test_workflow()
    agui = AGUI(workflow=workflow, tags=["Custom", "AGUI"])

    assert agui.tags == ["Custom", "AGUI"]


def test_workflow_properties_preserved():
    """Workflow properties round-trip through AGUI without mutation."""
    workflow = Workflow(
        name="Complex Workflow",
        description="A complex test workflow",
        steps=[Step(name="step1", agent=Agent(model=OpenAIChat(id="gpt-4o-mini")))],
    )

    agui = AGUI(workflow=workflow)
    assert agui.workflow is not None
    assert agui.workflow.name == "Complex Workflow"
    assert agui.workflow.description == "A complex test workflow"
