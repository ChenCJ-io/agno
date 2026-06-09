"""Sync + async stream orchestrators for the AG-UI interface.

These are the two functions the AG-UI router calls. They consume an Agno
response stream (agent / team / workflow), translate each chunk into AG-UI
events via ``events.py`` / ``workflow.py``, and route terminal chunks
through ``completion.py`` so the run always ends cleanly.
"""

from collections.abc import Iterator
from typing import Any, AsyncIterator, Dict, Optional, Union

from ag_ui.core import BaseEvent

from agno.run.agent import RunEvent, RunOutputEvent
from agno.run.team import TeamRunEvent, TeamRunOutputEvent
from agno.run.workflow import WorkflowRunEvent, WorkflowRunOutputEvent

from agno.os.interfaces.agui.completion import create_completion_events
from agno.os.interfaces.agui.events import create_events_from_chunk, emit_event_logic
from agno.os.interfaces.agui.state import EventBuffer


def _is_completion_event(chunk: Union[RunOutputEvent, TeamRunOutputEvent, WorkflowRunOutputEvent]) -> bool:
    """Return True when a chunk should be routed to the completion gate."""
    return (
        chunk.event == RunEvent.run_completed
        or chunk.event == TeamRunEvent.run_completed
        or chunk.event == RunEvent.run_paused
        or chunk.event == TeamRunEvent.run_paused
        or chunk.event == WorkflowRunEvent.workflow_completed
        or chunk.event == WorkflowRunEvent.workflow_error
        or chunk.event == WorkflowRunEvent.workflow_cancelled
    )


def stream_agno_response_as_agui_events(
    response_stream: Iterator[Union[RunOutputEvent, TeamRunOutputEvent, WorkflowRunOutputEvent]],
    thread_id: str,
    run_id: str,
    run_state: Optional[Dict[str, Any]] = None,
    is_workflow: bool = False,
) -> Iterator[BaseEvent]:
    """Map the Agno response stream to AG-UI format, handling event ordering constraints."""
    message_id = ""  # Will be set by EventBuffer when text message starts
    message_started = False
    event_buffer = EventBuffer()
    stream_completed = False

    # Establish baseline state snapshot for delta tracking
    if run_state is not None:
        event_buffer.set_state_snapshot(run_state)

    completion_chunk = None

    for chunk in response_stream:
        if _is_completion_event(chunk):
            # Store completion chunk but don't process it yet
            completion_chunk = chunk
            stream_completed = True
        else:
            events_from_chunk, message_started, message_id = create_events_from_chunk(
                chunk, message_id, message_started, event_buffer, run_state=run_state, is_workflow=is_workflow
            )

            for event in events_from_chunk:
                for emit_event in emit_event_logic(event_buffer=event_buffer, event=event):
                    yield emit_event

    # Process ONLY completion cleanup events, not content from completion chunk
    if completion_chunk:
        completion_events = create_completion_events(
            completion_chunk, event_buffer, message_started, message_id, thread_id, run_id, run_state=run_state
        )
        for event in completion_events:
            for emit_event in emit_event_logic(event_buffer=event_buffer, event=event):
                yield emit_event

    # Ensure completion events are always emitted even when stream ends naturally
    if not stream_completed:
        # Create a synthetic completion event to ensure proper cleanup
        from agno.run.agent import RunCompletedEvent

        synthetic_completion = RunCompletedEvent()
        completion_events = create_completion_events(
            synthetic_completion, event_buffer, message_started, message_id, thread_id, run_id, run_state=run_state
        )
        for event in completion_events:
            for emit_event in emit_event_logic(event_buffer=event_buffer, event=event):
                yield emit_event


async def async_stream_agno_response_as_agui_events(
    response_stream: AsyncIterator[Union[RunOutputEvent, TeamRunOutputEvent, WorkflowRunOutputEvent]],
    thread_id: str,
    run_id: str,
    run_state: Optional[Dict[str, Any]] = None,
    is_workflow: bool = False,
) -> AsyncIterator[BaseEvent]:
    """Async variant of ``stream_agno_response_as_agui_events`` with identical semantics."""
    message_id = ""
    message_started = False
    event_buffer = EventBuffer()
    stream_completed = False

    if run_state is not None:
        event_buffer.set_state_snapshot(run_state)

    completion_chunk = None

    async for chunk in response_stream:
        if _is_completion_event(chunk):
            completion_chunk = chunk
            stream_completed = True
        else:
            events_from_chunk, message_started, message_id = create_events_from_chunk(
                chunk, message_id, message_started, event_buffer, run_state=run_state, is_workflow=is_workflow
            )

            for event in events_from_chunk:
                for emit_event in emit_event_logic(event_buffer=event_buffer, event=event):
                    yield emit_event

    if completion_chunk:
        completion_events = create_completion_events(
            completion_chunk, event_buffer, message_started, message_id, thread_id, run_id, run_state=run_state
        )
        for event in completion_events:
            for emit_event in emit_event_logic(event_buffer=event_buffer, event=event):
                yield emit_event

    if not stream_completed:
        from agno.run.agent import RunCompletedEvent

        synthetic_completion = RunCompletedEvent()
        completion_events = create_completion_events(
            synthetic_completion, event_buffer, message_started, message_id, thread_id, run_id, run_state=run_state
        )
        for event in completion_events:
            for emit_event in emit_event_logic(event_buffer=event_buffer, event=event):
                yield emit_event
