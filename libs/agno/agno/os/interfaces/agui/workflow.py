"""Workflow-specific event translation for the AG-UI interface.

This file owns ONLY the workflow lifecycle event branches that translate
``WorkflowRunEvent`` events into AG-UI events. It does not handle agent or
team events; those live in ``events.py``.

Two public helpers:
- ``should_suppress_inner_workflow_event``: decides whether an inner agent /
  team content event must be dropped while streaming a workflow (the workflow
  emits its own consolidated content via ``WorkflowCompletedEvent``; inner
  text events would duplicate it).
- ``create_workflow_events_from_chunk``: dispatches a ``WorkflowRunEvent`` to
  the right elif branch and returns the AG-UI events to emit.

Terminal workflow events (workflow_completed / workflow_error /
workflow_cancelled) are NOT handled here — they are routed through the
completion gate in ``streaming.py`` to ``completion.py:create_completion_events``.
"""

from typing import List, Union

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    EventType,
    ReasoningMessageContentEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    StepFinishedEvent,
    StepStartedEvent,
)

from agno.run.agent import RunEvent, RunOutputEvent
from agno.run.team import TeamRunEvent, TeamRunOutputEvent
from agno.run.workflow import WorkflowRunEvent, WorkflowRunOutputEvent
from agno.utils.log import log_debug, log_error

from agno.os.interfaces.agui.state import EventBuffer

# Events to suppress when streaming a workflow. The workflow emits its own
# consolidated content via WorkflowCompletedEvent; inner agent/team text events
# would duplicate that content in the AG-UI client.
_SUPPRESSED_IN_WORKFLOW: frozenset = frozenset(
    {
        RunEvent.run_content.value,
        RunEvent.run_intermediate_content.value,
        TeamRunEvent.run_content.value,
        TeamRunEvent.run_intermediate_content.value,
    }
)


def should_suppress_inner_workflow_event(
    chunk: Union[RunOutputEvent, TeamRunOutputEvent, WorkflowRunOutputEvent],
    event_buffer: EventBuffer,
    is_workflow: bool,
) -> bool:
    """Return True when an inner agent/team content event must be dropped.

    When streaming a workflow, inner agent/team content events would duplicate
    the workflow's consolidated content emitted via ``WorkflowCompletedEvent``.
    EXCEPT during workflow-agent direct-answer (between
    ``workflow_agent_started`` and ``workflow_agent_completed``) — that content
    IS the final answer and ``WorkflowCompletedEvent`` skips emission via
    ``agent_direct_response``. Without this guard the user sees a blank UI on
    direct-answer workflows.
    """
    if not is_workflow:
        return False
    if chunk.event not in _SUPPRESSED_IN_WORKFLOW:
        return False
    if event_buffer.workflow_agent_active:
        return False
    log_debug(f"AGUI: suppressing inner event {chunk.event!r} in workflow stream")
    return True


def create_workflow_events_from_chunk(
    chunk: Union[RunOutputEvent, TeamRunOutputEvent, WorkflowRunOutputEvent],
    event_buffer: EventBuffer,
) -> List[BaseEvent]:
    """Translate a single ``WorkflowRunEvent`` chunk into AG-UI events.

    Caller is responsible for the early-return suppression check
    (``should_suppress_inner_workflow_event``) and for routing terminal
    workflow events (completed / error / cancelled) through the completion
    gate, not through this function.
    """
    events_to_emit: List[BaseEvent] = []

    if chunk.event == WorkflowRunEvent.workflow_started:
        events_to_emit.extend(_create_workflow_started_events(chunk, event_buffer))

    # workflow_completed, workflow_error, and workflow_cancelled are terminal
    # events — routed through the completion gate in
    # stream_agno_response_as_agui_events to _create_completion_events, which
    # emits their content / error / cancel events plus run termination. Do not
    # handle them here.

    elif chunk.event == WorkflowRunEvent.step_error:
        # step_error is a non-terminal event — the workflow may continue or
        # recover. Emit as CustomEvent + StepFinishedEvent (NOT RunErrorEvent
        # which is AG-UI spec terminal: no events may follow RunErrorEvent).
        error_message = getattr(chunk, "error", None) or "Step error occurred"
        step_name = getattr(chunk, "step_name", None) or "unknown_step"
        log_error(f"Step error in {step_name}: {error_message}")
        events_to_emit.append(CustomEvent(name="StepError", value={"step_name": step_name, "error": error_message}))
        events_to_emit.append(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=step_name))

    # Handle workflow step events.
    # We deliberately emit a single past-tense entry per step at completion time
    # (instead of "Running step…" then "Finished step…") so the collapsed
    # "Thought for N seconds" card reads naturally after the run. AG-UI's
    # REASONING_MESSAGE_CONTENT is append-only — we can't rewrite an earlier
    # "Running" line into "Ran" once a step finishes, so the past-tense-only
    # approach avoids the awkward "Running step: X" in a completed transcript.
    elif chunk.event == WorkflowRunEvent.step_started:
        step_name = getattr(chunk, "step_name", None) or "workflow_step"
        events_to_emit.append(StepStartedEvent(type=EventType.STEP_STARTED, step_name=step_name))

    elif chunk.event == WorkflowRunEvent.step_completed:
        step_name = getattr(chunk, "step_name", None) or "workflow_step"
        events_to_emit.append(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=step_name))
        if event_buffer.workflow_reasoning_id is not None:
            events_to_emit.append(
                ReasoningMessageContentEvent(
                    type=EventType.REASONING_MESSAGE_CONTENT,
                    message_id=event_buffer.workflow_reasoning_id,
                    delta=f"Ran step: {step_name}\n\n",
                )
            )

    # Handle workflow agent events.
    # WorkflowAgentStartedEvent / WorkflowAgentCompletedEvent do NOT carry an
    # agent_name field — the producer (workflow/workflow.py) populates only
    # workflow_name / workflow_id / session_id (and content for completed).
    # We label the step with workflow_name; "workflow_agent" is a defensive
    # fallback. WorkflowAgentCompletedEvent.content is intentionally NOT
    # forwarded: the agent's text was already streamed via inner
    # RunContentEvent (which is NOT suppressed during this window — see the
    # workflow_agent_active guard in the suppression check above).
    elif chunk.event == WorkflowRunEvent.workflow_agent_started:
        event_buffer.workflow_agent_active = True
        agent_label = getattr(chunk, "workflow_name", None) or "workflow_agent"
        events_to_emit.append(StepStartedEvent(type=EventType.STEP_STARTED, step_name=f"agent:{agent_label}"))

    elif chunk.event == WorkflowRunEvent.workflow_agent_completed:
        event_buffer.workflow_agent_active = False
        agent_label = getattr(chunk, "workflow_name", None) or "workflow_agent"
        events_to_emit.append(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=f"agent:{agent_label}"))

    # Handle conditional flow events
    elif chunk.event == WorkflowRunEvent.condition_execution_started:
        step_name = getattr(chunk, "step_name", None) or "condition"
        events_to_emit.append(StepStartedEvent(type=EventType.STEP_STARTED, step_name=f"Condition: {step_name}"))

    elif chunk.event == WorkflowRunEvent.condition_execution_completed:
        step_name = getattr(chunk, "step_name", None) or "condition"
        events_to_emit.append(
            CustomEvent(
                name="ConditionExecutionCompleted",
                value={
                    "step_name": step_name,
                    "condition_result": getattr(chunk, "condition_result", None),
                    "branch": getattr(chunk, "branch", None),
                },
            )
        )
        events_to_emit.append(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=f"Condition: {step_name}"))

    elif chunk.event == WorkflowRunEvent.condition_paused:
        step_name = getattr(chunk, "step_name", None) or "condition"
        events_to_emit.append(
            CustomEvent(
                name="ConditionPaused",
                value={"step_name": step_name, "message": f"Condition paused awaiting input: {step_name}"},
            )
        )

    # Handle router events
    elif chunk.event == WorkflowRunEvent.router_execution_started:
        step_name = getattr(chunk, "step_name", None) or "router"
        events_to_emit.append(
            CustomEvent(
                name="RouterExecutionStarted",
                value={"step_name": step_name, "selected_steps": getattr(chunk, "selected_steps", None) or []},
            )
        )
        events_to_emit.append(StepStartedEvent(type=EventType.STEP_STARTED, step_name=f"Router: {step_name}"))

    elif chunk.event == WorkflowRunEvent.router_execution_completed:
        step_name = getattr(chunk, "step_name", None) or "router"
        # executed_steps from producer is Optional[int] (count), not a list.
        events_to_emit.append(
            CustomEvent(
                name="RouterExecutionCompleted",
                value={"step_name": step_name, "executed_steps": getattr(chunk, "executed_steps", None)},
            )
        )
        events_to_emit.append(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=f"Router: {step_name}"))

    elif chunk.event == WorkflowRunEvent.router_paused:
        step_name = getattr(chunk, "step_name", None) or "router"
        events_to_emit.append(
            CustomEvent(
                name="RouterPaused",
                value={
                    "step_name": step_name,
                    "available_choices": getattr(chunk, "available_choices", None) or [],
                    "message": getattr(chunk, "user_input_message", None) or f"Router paused: {step_name}",
                },
            )
        )

    # Handle loop events
    elif chunk.event == WorkflowRunEvent.loop_execution_started:
        step_name = getattr(chunk, "step_name", None) or "loop"
        events_to_emit.append(StepStartedEvent(type=EventType.STEP_STARTED, step_name=f"Loop: {step_name}"))

    elif chunk.event == WorkflowRunEvent.loop_iteration_started:
        iteration = getattr(chunk, "iteration", None) or 0
        max_iterations = getattr(chunk, "max_iterations", None)
        label = f"Loop iter {iteration}/{max_iterations}" if max_iterations else f"Loop iter {iteration}"
        events_to_emit.append(StepStartedEvent(type=EventType.STEP_STARTED, step_name=label))

    elif chunk.event == WorkflowRunEvent.loop_iteration_completed:
        iteration = getattr(chunk, "iteration", None) or 0
        max_iterations = getattr(chunk, "max_iterations", None)
        label = f"Loop iter {iteration}/{max_iterations}" if max_iterations else f"Loop iter {iteration}"
        events_to_emit.append(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=label))

    elif chunk.event == WorkflowRunEvent.loop_execution_completed:
        step_name = getattr(chunk, "step_name", None) or "loop"
        events_to_emit.append(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=f"Loop: {step_name}"))

    # Handle parallel events
    elif chunk.event == WorkflowRunEvent.parallel_execution_started:
        step_name = getattr(chunk, "step_name", None) or "parallel"
        events_to_emit.append(StepStartedEvent(type=EventType.STEP_STARTED, step_name=f"Parallel: {step_name}"))

    elif chunk.event == WorkflowRunEvent.parallel_execution_completed:
        step_name = getattr(chunk, "step_name", None) or "parallel"
        events_to_emit.append(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=f"Parallel: {step_name}"))

    # Handle steps group events
    elif chunk.event == WorkflowRunEvent.steps_execution_started:
        step_name = getattr(chunk, "step_name", None) or "steps"
        events_to_emit.append(StepStartedEvent(type=EventType.STEP_STARTED, step_name=f"Steps: {step_name}"))

    elif chunk.event == WorkflowRunEvent.steps_execution_completed:
        step_name = getattr(chunk, "step_name", None) or "steps"
        events_to_emit.append(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=f"Steps: {step_name}"))

    # Log unmapped workflow events (HITL pause/continue, step_output, etc.) for
    # observability. Falls through silently in other interfaces — debugging
    # "where did my event go?" without this is hard. Does not emit AG-UI events.
    # Guarded explicitly so a misuse that passes a non-WorkflowRunEvent here
    # doesn't get silently swallowed as "unhandled workflow event".
    elif chunk.event in {e.value for e in WorkflowRunEvent}:
        log_debug(f"AGUI: workflow event {chunk.event!r} has no explicit handler (intentional or deferred)")

    return events_to_emit


def _create_workflow_started_events(
    chunk: Union[RunOutputEvent, TeamRunOutputEvent, WorkflowRunOutputEvent],
    event_buffer: EventBuffer,
) -> List[BaseEvent]:
    """Build the events emitted when a workflow run starts.

    Opens a reasoning message that will render as the "Thinking…" card in
    CopilotKit-based clients (Dojo). Step lifecycle deltas accumulate into
    this single card and the card auto-collapses to "Thought for N seconds"
    when we close it on workflow completion.
    """
    events_to_emit: List[BaseEvent] = []
    workflow_name = getattr(chunk, "workflow_name", None) or "workflow"
    events_to_emit.append(
        CustomEvent(
            name="WorkflowStarted",
            value={"workflow_name": workflow_name, "message": f"Starting workflow: {workflow_name}"},
        )
    )
    if event_buffer.workflow_reasoning_id is None:
        wf_reasoning_id = event_buffer.start_workflow_reasoning()
        events_to_emit.append(ReasoningStartEvent(type=EventType.REASONING_START, message_id=wf_reasoning_id))
        events_to_emit.append(
            ReasoningMessageStartEvent(
                type=EventType.REASONING_MESSAGE_START, message_id=wf_reasoning_id, role="reasoning"
            )
        )
        # Emit the workflow name as the first delta inside the Thinking card.
        # Surfaces the workflow identity to the user without polluting the
        # final answer; renders as the first line of "Thought for N seconds".
        events_to_emit.append(
            ReasoningMessageContentEvent(
                type=EventType.REASONING_MESSAGE_CONTENT,
                message_id=wf_reasoning_id,
                delta=f"Workflow: {workflow_name}\n\n",
            )
        )
    return events_to_emit
