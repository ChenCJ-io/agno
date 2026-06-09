"""Run-completion event handling for the AG-UI interface.

Generates the events emitted at the END of an AG-UI run: terminal messages,
state snapshots, error / cancelled markers, external-execution tool prompts,
and the final ``RunFinishedEvent`` (or ``RunErrorEvent`` for terminal
failures). This is called by the streaming gate in ``streaming.py`` whenever
a chunk matches one of the completion-style event types.

Public surface:
- ``create_completion_events``: produce the list of AG-UI events that
  terminate a single run.
"""

import copy
import json
import uuid
from typing import Any, Dict, List, Optional, Union

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    EventType,
    ReasoningEndEvent,
    ReasoningMessageEndEvent,
    RunErrorEvent,
    RunFinishedEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

from agno.run.agent import RunOutputEvent, RunPausedEvent
from agno.run.team import TeamRunOutputEvent
from agno.run.workflow import (
    WorkflowCancelledEvent,
    WorkflowCompletedEvent,
    WorkflowErrorEvent,
    WorkflowRunOutputEvent,
)
from agno.utils.log import log_error, log_warning
from agno.utils.message import get_text_from_message

from agno.os.interfaces.agui.state import EventBuffer


def create_completion_events(
    chunk: Union[RunOutputEvent, TeamRunOutputEvent, WorkflowRunOutputEvent],
    event_buffer: EventBuffer,
    message_started: bool,
    message_id: str,
    thread_id: str,
    run_id: str,
    run_state: Optional[Dict[str, Any]] = None,
) -> List[BaseEvent]:
    """Create events for run completion (or terminal failure / cancellation)."""
    events_to_emit: List[BaseEvent] = []

    # Close orphaned reasoning session if stream ended mid-reasoning
    if event_buffer.reasoning_message_id is not None:
        events_to_emit.append(
            ReasoningMessageEndEvent(type=EventType.REASONING_MESSAGE_END, message_id=event_buffer.reasoning_message_id)
        )
        events_to_emit.append(
            ReasoningEndEvent(type=EventType.REASONING_END, message_id=event_buffer.reasoning_message_id)
        )
        event_buffer.end_reasoning()

    # End remaining active tool calls if needed
    for tool_call_id in list(event_buffer.active_tool_call_ids):
        if tool_call_id not in event_buffer.ended_tool_call_ids:
            events_to_emit.append(
                ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=tool_call_id,
                )
            )

    # End the message and run, denoting the end of the session
    if message_started:
        events_to_emit.append(TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id))

    # Workflow ERROR is terminal in AG-UI spec — emit RunErrorEvent and return.
    # Do NOT also emit RunFinishedEvent (spec: RunErrorEvent is the final event).
    if isinstance(chunk, WorkflowErrorEvent):
        error_msg = getattr(chunk, "error", None) or "Workflow error occurred"
        workflow_name = chunk.workflow_name or "workflow"
        log_error(f"Workflow error in {workflow_name}: {error_msg}")
        events_to_emit.append(RunErrorEvent(type=EventType.RUN_ERROR, message=f"Workflow error: {error_msg}"))
        return events_to_emit

    # Workflow CANCEL is also terminal — emit a CustomEvent marker for client
    # observability (cancel != error semantically) then RunErrorEvent so the
    # AG-UI client treats the run as ended. No RunFinishedEvent follows.
    if isinstance(chunk, WorkflowCancelledEvent):
        reason = getattr(chunk, "reason", None) or "no reason given"
        workflow_name = chunk.workflow_name or "workflow"
        events_to_emit.append(
            CustomEvent(name="WorkflowCancelled", value={"workflow_name": workflow_name, "reason": reason})
        )
        events_to_emit.append(RunErrorEvent(type=EventType.RUN_ERROR, message=f"Workflow cancelled: {reason}"))
        return events_to_emit

    # Workflow COMPLETED — always emit a CustomEvent("WorkflowCompleted") for
    # client observability; additionally emit consolidated content as a fresh
    # TextMessage triplet when content is present (AG-UI requires final text
    # via TextMessage* events; RunFinishedEvent.result is opaque to clients).
    # Skip the triplet when agent_direct_response=True (content was already
    # streamed via inner agent's RunContentEvent — emitting again would
    # duplicate). Falls through to RunFinishedEvent — completion is soft terminal.
    if isinstance(chunk, WorkflowCompletedEvent):
        events_to_emit.extend(_emit_workflow_completed(chunk, event_buffer))

    # Emit external execution tools
    if isinstance(chunk, RunPausedEvent):
        events_to_emit.extend(_emit_external_execution_tools(chunk))

    # Emit final state snapshot before finishing the run (only if frontend opted into state tracking)
    if run_state is not None:
        # Use session_state from RunCompletedEvent (authoritative) if available,
        # otherwise fall back to run_state. Deep-copy so the emitted event
        # doesn't alias the live agent state (consistent with set_state_snapshot).
        authoritative_state = getattr(chunk, "session_state", None)
        final_state = authoritative_state if authoritative_state is not None else run_state
        events_to_emit.append(StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=copy.deepcopy(final_state)))

    events_to_emit.append(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id))

    return events_to_emit


def _emit_workflow_completed(
    chunk: WorkflowCompletedEvent,
    event_buffer: EventBuffer,
) -> List[BaseEvent]:
    """Build the events for a workflow-completed terminal chunk."""
    events_to_emit: List[BaseEvent] = []

    # Close the workflow-progress reasoning card BEFORE the final TextMessage.
    # This makes CopilotKit collapse the "Thinking…" card to "Thought for N
    # seconds" so the final answer renders below as a fresh assistant message.
    if event_buffer.workflow_reasoning_id is not None:
        wf_reasoning_id = event_buffer.workflow_reasoning_id
        events_to_emit.append(
            ReasoningMessageEndEvent(type=EventType.REASONING_MESSAGE_END, message_id=wf_reasoning_id)
        )
        events_to_emit.append(ReasoningEndEvent(type=EventType.REASONING_END, message_id=wf_reasoning_id))
        event_buffer.end_workflow_reasoning()

    if chunk.content is not None:
        # Strict isinstance guard: don't trust falsy non-None metadata
        # (e.g. an int 0 or False would silently coerce to {} via `or`).
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        agent_direct = bool(metadata.get("agent_direct_response"))
        if not agent_direct:
            rendered = get_text_from_message(chunk.content)
            if not rendered:
                log_warning(
                    f"WorkflowCompletedEvent.content was non-None but rendered to empty string; "
                    f"workflow_name={chunk.workflow_name!r}, content_type={type(chunk.content).__name__}"
                )
            if rendered:
                # Emit the workflow's consolidated content as the final
                # assistant message. We intentionally do NOT prepend the
                # workflow name as a header — keeping the response clean
                # without metadata noise. The reasoning card (rendered as
                # "Thought for N seconds") already provides the per-step
                # transcript for users who want to see what ran.
                wf_message_id = str(uuid.uuid4())
                events_to_emit.append(
                    TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=wf_message_id, role="assistant")
                )
                events_to_emit.append(
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT, message_id=wf_message_id, delta=rendered
                    )
                )
                events_to_emit.append(TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=wf_message_id))
    wf_name = chunk.workflow_name or "workflow"
    events_to_emit.append(
        CustomEvent(
            name="WorkflowCompleted",
            value={"workflow_name": wf_name, "message": f"Workflow completed: {wf_name}"},
        )
    )
    return events_to_emit


def _emit_external_execution_tools(chunk: RunPausedEvent) -> List[BaseEvent]:
    """Build the events for external-execution tool requests on a paused run."""
    events_to_emit: List[BaseEvent] = []
    external_tools = chunk.tools_awaiting_external_execution
    if not external_tools:
        return events_to_emit

    # First, emit an assistant message for external tool calls
    assistant_message_id = str(uuid.uuid4())
    events_to_emit.append(
        TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=assistant_message_id,
            role="assistant",
        )
    )

    # Add any text content if present for the assistant message
    if chunk.content:
        events_to_emit.append(
            TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=assistant_message_id,
                delta=str(chunk.content),
            )
        )

    # End the assistant message
    events_to_emit.append(
        TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=assistant_message_id,
        )
    )

    # Emit tool call events for external execution
    for tool in external_tools:
        if tool.tool_call_id is None or tool.tool_name is None:
            continue

        events_to_emit.append(
            ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=tool.tool_call_id,
                tool_call_name=tool.tool_name,
                parent_message_id=assistant_message_id,
            )
        )
        events_to_emit.append(
            ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=tool.tool_call_id,
                # default=str handles non-JSON-serializable types.
                delta=json.dumps(tool.tool_args, default=str),
            )
        )
        events_to_emit.append(
            ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=tool.tool_call_id,
            )
        )

    return events_to_emit
