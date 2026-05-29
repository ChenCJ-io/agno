"""Correctness tests for workflow event → AG-UI event translation in agui/utils.py."""

from types import SimpleNamespace

import pytest

pytest.importorskip("ag_ui", reason="ag_ui not installed")

from ag_ui.core import (
    CustomEvent,
    RunErrorEvent,
    RunFinishedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
)

from agno.os.interfaces.agui.utils import (
    EventBuffer,
    _create_events_from_chunk,
    async_stream_agno_response_as_agui_events,
)
from agno.run.agent import RunContentEvent, RunEvent
from agno.run.workflow import (
    WorkflowCancelledEvent,
    WorkflowCompletedEvent,
    WorkflowErrorEvent,
    WorkflowRunEvent,
)


def _emit(chunk):
    events, _, _ = _create_events_from_chunk(chunk, message_id="", message_started=False, event_buffer=EventBuffer())
    return events


def _chunk(event_value, **fields):
    return SimpleNamespace(event=event_value, **fields)


def test_workflow_started_emits_custom_event():
    events = _emit(_chunk(WorkflowRunEvent.workflow_started.value, workflow_name="my_workflow"))
    assert isinstance(events[0], CustomEvent)
    assert events[0].name == "WorkflowStarted"
    assert events[0].value["workflow_name"] == "my_workflow"


def test_workflow_completed_does_not_emit_through_chunk_handler():
    # workflow_completed is routed through _create_completion_events by the
    # stream gate, so _create_events_from_chunk is a no-op for it.
    events = _emit(_chunk(WorkflowRunEvent.workflow_completed.value, workflow_name="my_workflow"))
    assert events == []


def test_workflow_error_does_not_emit_through_chunk_handler():
    # workflow_error is routed through _create_completion_events by the
    # stream gate, so _create_events_from_chunk is a no-op for it.
    events = _emit(_chunk(WorkflowRunEvent.workflow_error.value, error="boom", workflow_name="my_workflow"))
    assert events == []


def test_step_started_emits_step_started_event():
    events = _emit(_chunk(WorkflowRunEvent.step_started.value, step_name="research"))
    assert isinstance(events[0], StepStartedEvent)
    assert events[0].step_name == "research"


def test_step_completed_emits_step_finished_event():
    events = _emit(_chunk(WorkflowRunEvent.step_completed.value, step_name="research"))
    assert isinstance(events[0], StepFinishedEvent)
    assert events[0].step_name == "research"


def test_workflow_agent_started_uses_workflow_name():
    # WorkflowAgentStartedEvent carries workflow_name (NOT agent_name — see
    # libs/agno/agno/run/workflow.py producer sites in workflow/workflow.py).
    events = _emit(_chunk(WorkflowRunEvent.workflow_agent_started.value, workflow_name="summarizer"))
    assert isinstance(events[0], StepStartedEvent)
    assert events[0].step_name == "agent:summarizer"


def test_workflow_agent_started_falls_back_when_workflow_name_missing():
    events = _emit(_chunk(WorkflowRunEvent.workflow_agent_started.value))
    assert isinstance(events[0], StepStartedEvent)
    assert events[0].step_name == "agent:workflow_agent"


def test_condition_started_emits_step_started():
    events = _emit(_chunk(WorkflowRunEvent.condition_execution_started.value, step_name="check_premium"))
    assert isinstance(events[0], StepStartedEvent)
    assert events[0].step_name == "Condition: check_premium"


def test_condition_completed_includes_branch_payload():
    events = _emit(
        _chunk(
            WorkflowRunEvent.condition_execution_completed.value,
            step_name="check_premium",
            condition_result=True,
            branch="premium_path",
        )
    )
    custom = next(e for e in events if isinstance(e, CustomEvent))
    assert custom.value["branch"] == "premium_path"
    assert custom.value["condition_result"] is True
    assert any(isinstance(e, StepFinishedEvent) for e in events)


def test_router_started_includes_selected_steps():
    events = _emit(
        _chunk(
            WorkflowRunEvent.router_execution_started.value,
            step_name="route",
            selected_steps=["step_a", "step_b"],
        )
    )
    custom = next(e for e in events if isinstance(e, CustomEvent))
    assert custom.value["selected_steps"] == ["step_a", "step_b"]


def test_router_completed_includes_executed_steps():
    events = _emit(
        _chunk(
            WorkflowRunEvent.router_execution_completed.value,
            step_name="route",
            executed_steps=["step_a"],
        )
    )
    custom = next(e for e in events if isinstance(e, CustomEvent))
    assert custom.value["executed_steps"] == ["step_a"]


def test_loop_iteration_started_includes_progress():
    events = _emit(_chunk(WorkflowRunEvent.loop_iteration_started.value, iteration=2, max_iterations=3))
    assert isinstance(events[0], StepStartedEvent)
    assert events[0].step_name == "Loop iter 2/3"


def test_parallel_execution_started_emits_step_started():
    events = _emit(_chunk(WorkflowRunEvent.parallel_execution_started.value, step_name="parallel_group"))
    assert isinstance(events[0], StepStartedEvent)
    assert events[0].step_name == "Parallel: parallel_group"


def test_steps_execution_completed_emits_step_finished():
    events = _emit(_chunk(WorkflowRunEvent.steps_execution_completed.value, step_name="sequence"))
    assert isinstance(events[0], StepFinishedEvent)
    assert events[0].step_name == "Steps: sequence"


# === End-to-end async-stream tests for terminal workflow events ===


async def _drive_stream(chunks, is_workflow=False):
    """Drive async_stream_agno_response_as_agui_events with a list of chunks."""

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    events = []
    async for event in async_stream_agno_response_as_agui_events(
        mock_stream(), "thread_1", "run_1", is_workflow=is_workflow
    ):
        events.append(event)
    return events


# --- WorkflowCompletedEvent edge cases ---


async def test_workflow_completed_emits_text_triplet_and_custom_event():
    """Workflow's consolidated final content appears as a clean TextMessage triplet.

    No workflow_name header is prepended — the response is delivered as-is.
    Users see the workflow identity via the reasoning ("Thought for N seconds") card.
    """
    events = await _drive_stream([WorkflowCompletedEvent(workflow_name="wf", content="Final answer")], is_workflow=True)
    deltas = [e.delta for e in events if isinstance(e, TextMessageContentEvent)]
    assert deltas == ["Final answer"]
    assert any(isinstance(e, CustomEvent) and e.name == "WorkflowCompleted" for e in events)
    assert any(isinstance(e, RunFinishedEvent) for e in events)


async def test_workflow_completed_response_has_no_markdown_header():
    """Confirm the final response carries no bold/heading workflow metadata."""
    events = await _drive_stream(
        [WorkflowCompletedEvent(workflow_name="My Workflow", content="Plain answer")], is_workflow=True
    )
    deltas = [e.delta for e in events if isinstance(e, TextMessageContentEvent)]
    assert deltas == ["Plain answer"]
    assert "**" not in deltas[0]
    assert "##" not in deltas[0]


async def test_workflow_completed_with_none_content_emits_only_custom_event():
    """No TextMessage triplet when content is None — just CustomEvent + RunFinished."""
    events = await _drive_stream([WorkflowCompletedEvent(workflow_name="wf", content=None)], is_workflow=True)
    assert not any(isinstance(e, TextMessageContentEvent) for e in events)
    assert any(isinstance(e, CustomEvent) and e.name == "WorkflowCompleted" for e in events)
    assert any(isinstance(e, RunFinishedEvent) for e in events)


async def test_workflow_completed_with_none_metadata_safe():
    """metadata=None must not raise."""
    events = await _drive_stream(
        [WorkflowCompletedEvent(workflow_name="wf", content="hello", metadata=None)], is_workflow=True
    )
    assert any(isinstance(e, TextMessageContentEvent) for e in events)


async def test_workflow_completed_with_agent_direct_response_skips_text_triplet():
    """agent_direct_response=True suppresses the workflow TextMessage triplet (inner already streamed)."""
    events = await _drive_stream(
        [
            WorkflowCompletedEvent(
                workflow_name="wf",
                content="already streamed",
                metadata={"agent_direct_response": True},
            )
        ],
        is_workflow=True,
    )
    assert not any(isinstance(e, TextMessageContentEvent) for e in events)
    assert any(isinstance(e, CustomEvent) and e.name == "WorkflowCompleted" for e in events)
    assert any(isinstance(e, RunFinishedEvent) for e in events)


async def test_workflow_completed_with_pydantic_content_renders_json():
    """Pydantic content is rendered via get_text_from_message (not str())."""
    from pydantic import BaseModel

    class MyOutput(BaseModel):
        name: str
        value: int

    events = await _drive_stream(
        [WorkflowCompletedEvent(workflow_name="wf", content=MyOutput(name="test", value=42))],
        is_workflow=True,
    )
    deltas = [e.delta for e in events if isinstance(e, TextMessageContentEvent)]
    assert deltas, "expected at least one TextMessageContentEvent"
    assert '"name": "test"' in deltas[0]
    assert '"value": 42' in deltas[0]


# --- Suppression behavior ---


async def test_run_content_suppressed_when_is_workflow_true():
    """Inner agent run_content events are dropped when is_workflow=True.

    The final TextMessage delta carries the workflow's consolidated content
    with no header decoration.
    """
    inner = RunContentEvent()
    inner.event = RunEvent.run_content
    inner.content = "inner agent text"
    completion = WorkflowCompletedEvent(workflow_name="wf", content="final")

    events = await _drive_stream([inner, completion], is_workflow=True)
    deltas = [e.delta for e in events if isinstance(e, TextMessageContentEvent)]
    assert "inner agent text" not in deltas
    assert deltas == ["final"]


async def test_workflow_started_opens_reasoning_card_and_step_completions_emit_past_tense():
    """Workflow_started opens a Reasoning card; step_completed appends past-tense entries.

    Only step_completed emits text (step_started intentionally emits nothing) so the
    collapsed "Thought for N seconds" card reads naturally in past tense without an
    awkward "Running step: X" leftover after the run.
    """
    from agno.run.workflow import StepCompletedEvent as RawStepCompletedEvent
    from agno.run.workflow import StepStartedEvent as RawStepStartedEvent
    from agno.run.workflow import WorkflowStartedEvent

    started = WorkflowStartedEvent(workflow_name="wf")
    step1_start = RawStepStartedEvent(step_name="research")
    step1_done = RawStepCompletedEvent(step_name="research")
    step2_start = RawStepStartedEvent(step_name="summarize")
    step2_done = RawStepCompletedEvent(step_name="summarize")
    completion = WorkflowCompletedEvent(workflow_name="wf", content="done")

    events = await _drive_stream(
        [started, step1_start, step1_done, step2_start, step2_done, completion], is_workflow=True
    )

    from ag_ui.core import (
        ReasoningEndEvent,
        ReasoningMessageContentEvent,
        ReasoningMessageEndEvent,
        ReasoningMessageStartEvent,
        ReasoningStartEvent,
    )

    # Reasoning card lifecycle present
    assert any(isinstance(e, ReasoningStartEvent) for e in events)
    assert any(isinstance(e, ReasoningMessageStartEvent) for e in events)
    assert any(isinstance(e, ReasoningMessageEndEvent) for e in events)
    assert any(isinstance(e, ReasoningEndEvent) for e in events)

    # Past-tense only entries (no "Running step:"), one per completed step
    reasoning_deltas = [e.delta for e in events if isinstance(e, ReasoningMessageContentEvent)]
    assert any("Ran step: research" in d for d in reasoning_deltas)
    assert any("Ran step: summarize" in d for d in reasoning_deltas)
    assert not any("Running step" in d for d in reasoning_deltas)
    assert not any("Finished step" in d for d in reasoning_deltas)
    # Each line is double-newline terminated for proper markdown line breaks
    assert all(d.endswith("\n\n") for d in reasoning_deltas)


async def test_workflow_started_emits_workflow_name_as_first_reasoning_delta():
    """The workflow name is the first delta inside the Thinking card.

    Surfaces workflow identity to the user without polluting the final answer —
    renders as the first line of the collapsed "Thought for N seconds" card.
    """
    from agno.run.workflow import WorkflowStartedEvent

    started = WorkflowStartedEvent(workflow_name="Research and Summarize")
    completion = WorkflowCompletedEvent(workflow_name="Research and Summarize", content="done")

    events = await _drive_stream([started, completion], is_workflow=True)

    from ag_ui.core import ReasoningMessageContentEvent

    reasoning_deltas = [e.delta for e in events if isinstance(e, ReasoningMessageContentEvent)]
    assert reasoning_deltas, "expected at least one ReasoningMessageContentEvent"
    assert reasoning_deltas[0] == "Workflow: Research and Summarize\n\n"


async def test_run_content_passed_through_when_is_workflow_false():
    """Inner content flows through when is_workflow=False (existing agent/team path)."""
    inner = RunContentEvent()
    inner.event = RunEvent.run_content
    inner.content = "agent text"

    events = await _drive_stream([inner], is_workflow=False)
    deltas = [e.delta for e in events if isinstance(e, TextMessageContentEvent)]
    assert "agent text" in deltas


# --- Terminal-event ordering ---


async def test_workflow_error_is_last_event_no_run_finished():
    """WorkflowErrorEvent: RunErrorEvent is the terminal; no RunFinishedEvent after."""
    events = await _drive_stream([WorkflowErrorEvent(workflow_name="wf", error="boom")], is_workflow=True)
    assert isinstance(events[-1], RunErrorEvent)
    assert "boom" in events[-1].message
    assert not any(isinstance(e, RunFinishedEvent) for e in events)


async def test_workflow_cancelled_emits_custom_event_then_run_error_no_run_finished():
    """WorkflowCancelledEvent: CustomEvent('WorkflowCancelled') + RunErrorEvent; no RunFinishedEvent."""
    events = await _drive_stream(
        [WorkflowCancelledEvent(workflow_name="wf", reason="user requested")], is_workflow=True
    )
    custom_idx = next(i for i, e in enumerate(events) if isinstance(e, CustomEvent) and e.name == "WorkflowCancelled")
    error_idx = next(i for i, e in enumerate(events) if isinstance(e, RunErrorEvent))
    assert custom_idx < error_idx
    assert isinstance(events[-1], RunErrorEvent)
    assert "user requested" in events[-1].message
    assert not any(isinstance(e, RunFinishedEvent) for e in events)


@pytest.mark.parametrize(
    "chunk_factory",
    [
        lambda: WorkflowErrorEvent(workflow_name="wf", error="boom"),
        lambda: WorkflowCancelledEvent(workflow_name="wf", reason="cancelled"),
    ],
    ids=["error", "cancelled"],
)
async def test_workflow_terminal_no_run_finished(chunk_factory):
    """Both workflow terminal types must NOT emit RunFinishedEvent (AG-UI spec: RunErrorEvent is terminal)."""
    events = await _drive_stream([chunk_factory()], is_workflow=True)
    assert not any(isinstance(e, RunFinishedEvent) for e in events)
    assert isinstance(events[-1], RunErrorEvent)


# --- Bug 2 regression: step_name=None must not crash StepStartedEvent ---


def test_step_started_with_none_step_name_uses_fallback():
    """When chunk has step_name=None, getattr default must kick in via `or` chain.

    Regression for the Bug 2 class: a Step constructed without name= emits
    StepStartedEvent(step_name=None), which would crash AG-UI's Pydantic
    StepStartedEvent (step_name is required str) if we didn't guard with
    `getattr(chunk, "step_name", None) or "workflow_step"`.
    """
    events = _emit(_chunk(WorkflowRunEvent.step_started.value, step_name=None))
    assert isinstance(events[0], StepStartedEvent)
    assert events[0].step_name == "workflow_step"


def test_condition_started_with_none_step_name_uses_fallback():
    events = _emit(_chunk(WorkflowRunEvent.condition_execution_started.value, step_name=None))
    assert isinstance(events[0], StepStartedEvent)
    assert events[0].step_name == "Condition: condition"


# --- Bug 3 regression: step_error must NOT emit RunErrorEvent (spec terminal) ---


def test_step_error_is_not_terminal():
    """step_error is non-fatal — emit CustomEvent + StepFinishedEvent, NOT RunErrorEvent."""
    events = _emit(_chunk(WorkflowRunEvent.step_error.value, error="boom", step_name="research"))
    # Must NOT emit RunErrorEvent (that would be terminal per AG-UI spec)
    assert not any(isinstance(e, RunErrorEvent) for e in events)
    # SHOULD emit CustomEvent("StepError") for client observability
    custom = next(e for e in events if isinstance(e, CustomEvent))
    assert custom.name == "StepError"
    assert custom.value == {"step_name": "research", "error": "boom"}
    # SHOULD emit StepFinishedEvent so the AG-UI client closes the step UI
    assert any(isinstance(e, StepFinishedEvent) and e.step_name == "research" for e in events)


# --- Bug 1 regression: agent_direct_response content must reach the client ---


async def test_agent_direct_response_inner_content_reaches_client():
    """When workflow agent answers directly, inner RunContentEvents must NOT be suppressed.

    Regression for Bug 1: with is_workflow=True, the inner RunContentEvent from the
    workflow agent's direct answer was being suppressed AND the WorkflowCompletedEvent
    text triplet was being skipped (because metadata.agent_direct_response=True).
    Result: blank UI. Fix: track workflow_agent_active state on EventBuffer and
    do NOT suppress inner content during that window.
    """
    from agno.run.agent import RunContentEvent as RawRunContentEvent

    agent_started = SimpleNamespace(event=WorkflowRunEvent.workflow_agent_started.value, workflow_name="wf")
    inner = RawRunContentEvent()
    inner.event = RunEvent.run_content
    inner.content = "Hi, I am answering directly."
    agent_completed = SimpleNamespace(event=WorkflowRunEvent.workflow_agent_completed.value, workflow_name="wf")
    completed = WorkflowCompletedEvent(
        workflow_name="wf",
        content="Hi, I am answering directly.",
        metadata={"agent_direct_response": True},
    )

    events = await _drive_stream([agent_started, inner, agent_completed, completed], is_workflow=True)
    deltas = [e.delta for e in events if isinstance(e, TextMessageContentEvent)]
    # The agent's direct answer MUST reach the client (was getting blanked before).
    assert "Hi, I am answering directly." in deltas
