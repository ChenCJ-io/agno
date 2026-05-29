"""Logic used by the AG-UI router."""

import copy
import json
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, Union

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    EventType,
    RawEvent,
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunErrorEvent,
    RunFinishedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.core.types import Message as AGUIMessage
from pydantic import BaseModel

from agno.reasoning.step import ReasoningStep
from agno.run.agent import ReasoningCompletedEvent as AgentReasoningCompletedEvent
from agno.run.agent import ReasoningContentDeltaEvent as AgentReasoningContentDeltaEvent
from agno.run.agent import ReasoningStartedEvent as AgentReasoningStartedEvent
from agno.run.agent import ReasoningStepEvent as AgentReasoningStepEvent
from agno.run.agent import RunContentEvent, RunEvent, RunOutputEvent, RunPausedEvent
from agno.run.team import ReasoningCompletedEvent as TeamReasoningCompletedEvent
from agno.run.team import ReasoningContentDeltaEvent as TeamReasoningContentDeltaEvent
from agno.run.team import ReasoningStartedEvent as TeamReasoningStartedEvent
from agno.run.team import ReasoningStepEvent as TeamReasoningStepEvent
from agno.run.team import RunContentEvent as TeamRunContentEvent
from agno.run.team import TeamRunEvent, TeamRunOutputEvent
from agno.run.workflow import (
    WorkflowCancelledEvent,
    WorkflowCompletedEvent,
    WorkflowErrorEvent,
    WorkflowRunEvent,
    WorkflowRunOutputEvent,
)
from agno.utils.log import log_debug, log_error, log_warning
from agno.utils.message import get_text_from_message

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


def validate_agui_state(state: Any, thread_id: str) -> Optional[Dict[str, Any]]:
    """Validate the given AGUI state is of the expected type (dict)."""
    if state is None:
        return None

    if isinstance(state, dict):
        return state

    if isinstance(state, BaseModel):
        try:
            return state.model_dump()
        except Exception as exc:
            log_warning(f"AGUI state.model_dump() failed (thread {thread_id}): {type(exc).__name__}: {exc}")

    if is_dataclass(state):
        try:
            return asdict(state)  # type: ignore
        except Exception as exc:
            log_warning(f"AGUI asdict(state) failed (thread {thread_id}): {type(exc).__name__}: {exc}")

    if hasattr(state, "to_dict") and callable(getattr(state, "to_dict")):
        try:
            result = state.to_dict()  # type: ignore
            if isinstance(result, dict):
                return result
        except Exception as exc:
            log_warning(f"AGUI state.to_dict() failed (thread {thread_id}): {type(exc).__name__}: {exc}")

    log_warning(f"AGUI state must be a dict, got {type(state).__name__}. State will be ignored. Thread: {thread_id}")
    return None


@dataclass
class EventBuffer:
    """Buffer to manage event ordering constraints, relevant when mapping Agno responses to AG-UI events."""

    active_tool_call_ids: Set[str]  # All currently active tool calls
    ended_tool_call_ids: Set[str]  # All tool calls that have ended
    current_text_message_id: str = ""  # ID of the current text message context (for tool call parenting)
    next_text_message_id: str = ""  # Pre-generated ID for the next text message
    pending_tool_calls_parent_id: str = ""  # Parent message ID for pending tool calls
    reasoning_message_id: Optional[str] = None  # Active reasoning session ID, set by reasoning_started
    reasoning_step_count: int = 0  # Step counter for ReasoningTools (reset each session)
    _last_snapshot: Optional[Dict[str, Any]] = field(
        default=None, repr=False
    )  # Snapshot of last state for delta computation
    workflow_agent_active: bool = False  # True between workflow_agent_started and workflow_agent_completed
    workflow_reasoning_id: Optional[str] = (
        None  # Active workflow-progress reasoning message (rendered as "Thinking…" card)
    )

    def __init__(self):
        self.active_tool_call_ids = set()
        self.ended_tool_call_ids = set()
        self.current_text_message_id = ""
        self.next_text_message_id = str(uuid.uuid4())
        self.pending_tool_calls_parent_id = ""
        self.reasoning_message_id = None
        self.reasoning_step_count = 0
        self._last_snapshot = None
        self.workflow_agent_active = False
        self.workflow_reasoning_id = None

    def start_tool_call(self, tool_call_id: str) -> None:
        """Start a new tool call."""
        self.active_tool_call_ids.add(tool_call_id)

    def end_tool_call(self, tool_call_id: str) -> None:
        """End a tool call."""
        self.active_tool_call_ids.discard(tool_call_id)
        self.ended_tool_call_ids.add(tool_call_id)

    def start_text_message(self) -> str:
        """Start a new text message and return its ID."""
        # Use the pre-generated next ID as current, and generate a new next ID
        self.current_text_message_id = self.next_text_message_id
        self.next_text_message_id = str(uuid.uuid4())
        return self.current_text_message_id

    def get_parent_message_id_for_tool_call(self) -> str:
        """Get the message ID to use as parent for tool calls."""
        # If we have a pending parent ID set (from text message end), use that
        if self.pending_tool_calls_parent_id:
            return self.pending_tool_calls_parent_id
        # Otherwise use current text message ID
        return self.current_text_message_id

    def set_pending_tool_calls_parent_id(self, parent_id: str) -> None:
        """Set the parent message ID for upcoming tool calls."""
        self.pending_tool_calls_parent_id = parent_id

    def clear_pending_tool_calls_parent_id(self) -> None:
        """Clear the pending parent ID when a new text message starts."""
        self.pending_tool_calls_parent_id = ""

    def start_reasoning(self) -> str:
        """Start a new reasoning session and return its message ID."""
        self.reasoning_message_id = str(uuid.uuid4())
        self.reasoning_step_count = 0
        return self.reasoning_message_id

    def set_state_snapshot(self, state: Dict[str, Any]) -> None:
        """Store deep copy of current state for delta computation."""
        self._last_snapshot = copy.deepcopy(state)

    def compute_state_delta(self, current_state: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Compute JSON Patch delta between last snapshot and current state.

        Returns a list of JSON Patch ops (RFC 6902) or None if unchanged/error.
        """
        if self._last_snapshot is None:
            return None
        try:
            import jsonpatch

            patch = jsonpatch.make_patch(self._last_snapshot, current_state)
            ops = patch.patch
            if not ops:
                return None
            return ops
        except Exception as e:
            log_warning(f"Failed to compute state delta: {e}")
            return None

    def next_reasoning_step(self) -> int:
        """Increment and return the current reasoning step number."""
        self.reasoning_step_count += 1
        return self.reasoning_step_count

    def ensure_reasoning_started(self) -> Tuple[str, bool]:
        """Return the active reasoning session ID, starting one if needed.
        Returns (reasoning_id, is_new) where is_new is True if a new session was created.
        """
        if self.reasoning_message_id is not None:
            return self.reasoning_message_id, False
        return self.start_reasoning(), True

    def end_reasoning(self) -> None:
        """End the active reasoning session."""
        self.reasoning_message_id = None
        self.reasoning_step_count = 0

    def start_workflow_reasoning(self) -> str:
        """Start the workflow-progress reasoning message and return its ID."""
        self.workflow_reasoning_id = str(uuid.uuid4())
        return self.workflow_reasoning_id

    def end_workflow_reasoning(self) -> None:
        """End the workflow-progress reasoning message."""
        self.workflow_reasoning_id = None


def extract_agui_user_input(messages: List[AGUIMessage]) -> str:
    """Extract the last user message content from AG-UI messages.

    AG-UI frontends send the full conversation history on every request.
    The agent manages its own history via session DB, so we only need the
    latest user message as input — matching the REST API pattern.
    """
    for msg in reversed(messages):
        if msg.role == "user" and msg.content is not None:
            # UserMessage.content is Union[str, List[InputContent]]
            if isinstance(msg.content, str):
                return msg.content
            # Multimodal: extract text parts
            if isinstance(msg.content, list):
                text_parts = []
                for part in msg.content:
                    if hasattr(part, "type") and part.type == "text" and hasattr(part, "text"):
                        text_parts.append(part.text)
                if text_parts:
                    return "\n".join(text_parts)
    return ""


def extract_team_response_chunk_content(response: TeamRunContentEvent) -> str:
    """Given a response stream chunk, find and extract the content."""

    # Handle Team members' responses
    members_content = []
    if hasattr(response, "member_responses") and response.member_responses:  # type: ignore
        for member_resp in response.member_responses:  # type: ignore
            if isinstance(member_resp, RunContentEvent):
                member_content = extract_response_chunk_content(member_resp)
                if member_content:
                    members_content.append(f"Team member: {member_content}")
            elif isinstance(member_resp, TeamRunContentEvent):
                member_content = extract_team_response_chunk_content(member_resp)
                if member_content:
                    members_content.append(f"Team member: {member_content}")
    members_response = "\n".join(members_content) if members_content else ""

    # Handle structured outputs
    main_content = get_text_from_message(response.content) if response.content is not None else ""

    return main_content + members_response


def extract_response_chunk_content(response: RunContentEvent) -> str:
    """Given a response stream chunk, find and extract the content."""

    if hasattr(response, "messages") and response.messages:  # type: ignore
        for msg in reversed(response.messages):  # type: ignore
            if hasattr(msg, "role") and msg.role == "assistant" and hasattr(msg, "content") and msg.content:
                # Handle structured outputs from messages
                return get_text_from_message(msg.content)

    # Handle structured outputs
    return get_text_from_message(response.content) if response.content is not None else ""


def _create_state_delta_events(
    run_state: Optional[Dict[str, Any]],
    event_buffer: EventBuffer,
) -> List[BaseEvent]:
    """Compute state delta and return StateDeltaEvent if state changed."""
    if run_state is None:
        return []
    ops = event_buffer.compute_state_delta(run_state)
    if ops is None:
        return []
    # Update the snapshot to current state for next delta computation
    event_buffer.set_state_snapshot(run_state)
    return [StateDeltaEvent(type=EventType.STATE_DELTA, delta=ops)]


def _format_reasoning_step_delta(step: Optional[ReasoningStep], step_number: int = 0) -> str:
    """Format a single ReasoningStep as a text delta for REASONING_MESSAGE_CONTENT.

    ReasoningStepEvent.content holds a ReasoningStep object (title, reasoning,
    action, result, confidence). We format just this one step — NOT the
    accumulated reasoning_content field, which duplicates prior steps.
    """
    if step is None:
        return ""
    parts: List[str] = []
    title = step.title or "Thinking"
    if step_number > 0:
        parts.append(f"## Step {step_number}: {title}")
    else:
        parts.append(f"## {title}")
    if step.reasoning:
        parts.append(step.reasoning)
    if step.action:
        parts.append(f"Action: {step.action}")
    if step.result:
        parts.append(f"Result: {step.result}")
    if step.confidence is not None:
        parts.append(f"Confidence: {step.confidence}")
    return "\n".join(parts) + "\n\n" if parts else ""


def _create_events_from_chunk(
    chunk: Union[RunOutputEvent, TeamRunOutputEvent, WorkflowRunOutputEvent],
    message_id: str,
    message_started: bool,
    event_buffer: EventBuffer,
    run_state: Optional[Dict[str, Any]] = None,
    is_workflow: bool = False,
) -> Tuple[List[BaseEvent], bool, str]:
    """
    Process a single chunk and return events to emit + updated message_started state.

    Args:
        chunk: The event chunk to process
        message_id: Current message identifier
        message_started: Whether a message is currently active
        event_buffer: Event buffer for tracking tool call state (includes reasoning session state)
        run_state: Mutable dict reference to the agent's session state (for delta tracking)
        is_workflow: When True, drop inner agent/team content events to avoid
            duplication with WorkflowCompletedEvent.content emitted at completion.

    Returns:
        Tuple of (events_to_emit, new_message_started_state, message_id)
    """
    events_to_emit: List[BaseEvent] = []
    is_content_event = False

    # When streaming a workflow, suppress inner agent/team content events.
    # The workflow's consolidated content is emitted via WorkflowCompletedEvent
    # in _create_completion_events; inner run_content would duplicate it.
    # EXCEPT during workflow-agent direct-answer (between workflow_agent_started
    # and workflow_agent_completed) — that content IS the final answer and
    # WorkflowCompletedEvent skips emission via agent_direct_response. Without
    # this guard the user sees a blank UI on direct-answer workflows.
    if is_workflow and chunk.event in _SUPPRESSED_IN_WORKFLOW and not event_buffer.workflow_agent_active:
        log_debug(f"AGUI: suppressing inner event {chunk.event!r} in workflow stream")
        return events_to_emit, message_started, message_id

    # Extract content if the contextual event is a content event
    if chunk.event == RunEvent.run_content:
        content = extract_response_chunk_content(chunk)  # type: ignore
        is_content_event = True
    elif chunk.event == TeamRunEvent.run_content:
        content = extract_team_response_chunk_content(chunk)  # type: ignore
        is_content_event = True
    else:
        content = None

    # Handle text responses
    if content is not None:
        # Handle the message start event, emitted once per message
        if not message_started:
            message_started = True
            message_id = event_buffer.start_text_message()

            # Clear pending tool calls parent ID when starting new text message
            event_buffer.clear_pending_tool_calls_parent_id()

            start_event = TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=message_id,
                role="assistant",
            )
            events_to_emit.append(start_event)

        # Handle the text content event, emitted once per text chunk
        if content is not None and content != "":
            content_event = TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=message_id,
                delta=content,
            )
            events_to_emit.append(content_event)  # type: ignore

    # Handle starting a new tool
    elif chunk.event == RunEvent.tool_call_started or chunk.event == TeamRunEvent.tool_call_started:
        if chunk.tool is not None:  # type: ignore
            tool_call = chunk.tool  # type: ignore

            # End current text message and handle for tool calls
            current_message_id = message_id
            if message_started:
                # End the current text message
                end_message_event = TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=current_message_id)
                events_to_emit.append(end_message_event)

                # Set this message as the parent for any upcoming tool calls
                # This ensures multiple sequential tool calls all use the same parent
                event_buffer.set_pending_tool_calls_parent_id(current_message_id)

                # Reset message started state and generate new message_id for future messages
                message_started = False
                message_id = str(uuid.uuid4())

            # Get the parent message ID - this will use pending parent if set, ensuring multiple tool calls in sequence have the same parent
            parent_message_id = event_buffer.get_parent_message_id_for_tool_call()

            if not parent_message_id:
                # Create parent message for tool calls without preceding assistant message
                parent_message_id = str(uuid.uuid4())

                # Emit a text message to serve as the parent
                text_start = TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START,
                    message_id=parent_message_id,
                    role="assistant",
                )
                events_to_emit.append(text_start)

                text_end = TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=parent_message_id,
                )
                events_to_emit.append(text_end)

                # Set this as the pending parent for subsequent tool calls in this batch
                event_buffer.set_pending_tool_calls_parent_id(parent_message_id)

            start_event = ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=tool_call.tool_call_id,  # type: ignore
                tool_call_name=tool_call.tool_name,  # type: ignore
                parent_message_id=parent_message_id,
            )
            events_to_emit.append(start_event)

            args_event = ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=tool_call.tool_call_id,  # type: ignore
                # default=str handles non-JSON-serializable types (datetime, UUID, bytes, etc.)
                # that tool implementations commonly produce.
                delta=json.dumps(tool_call.tool_args, default=str),
            )
            events_to_emit.append(args_event)  # type: ignore

    # Handle tool call completion
    elif chunk.event == RunEvent.tool_call_completed or chunk.event == TeamRunEvent.tool_call_completed:
        if chunk.tool is not None:  # type: ignore
            tool_call = chunk.tool  # type: ignore
            if tool_call.tool_call_id not in event_buffer.ended_tool_call_ids:
                end_event = ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=tool_call.tool_call_id,  # type: ignore
                )
                events_to_emit.append(end_event)

                if tool_call.result is not None:
                    result_event = ToolCallResultEvent(
                        type=EventType.TOOL_CALL_RESULT,
                        tool_call_id=tool_call.tool_call_id,  # type: ignore
                        content=str(tool_call.result),
                        role="tool",
                        message_id=str(uuid.uuid4()),
                    )
                    events_to_emit.append(result_event)

                # Emit state delta after tool call completion (state may have been mutated by the tool)
                events_to_emit.extend(_create_state_delta_events(run_state, event_buffer))

    # Handle reasoning events
    elif isinstance(chunk, (AgentReasoningStartedEvent, TeamReasoningStartedEvent)):
        if message_started:
            events_to_emit.append(TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id))
            message_started = False
            message_id = str(uuid.uuid4())
        reasoning_id = event_buffer.start_reasoning()
        events_to_emit.append(ReasoningStartEvent(type=EventType.REASONING_START, message_id=reasoning_id))
        events_to_emit.append(
            ReasoningMessageStartEvent(
                type=EventType.REASONING_MESSAGE_START, message_id=reasoning_id, role="reasoning"
            )
        )

    elif isinstance(chunk, (AgentReasoningContentDeltaEvent, TeamReasoningContentDeltaEvent)):
        if message_started:
            events_to_emit.append(TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id))
            message_started = False
            message_id = str(uuid.uuid4())
        reasoning_id, is_new = event_buffer.ensure_reasoning_started()
        if is_new:
            events_to_emit.append(ReasoningStartEvent(type=EventType.REASONING_START, message_id=reasoning_id))
            events_to_emit.append(
                ReasoningMessageStartEvent(
                    type=EventType.REASONING_MESSAGE_START, message_id=reasoning_id, role="reasoning"
                )
            )
        if chunk.reasoning_content:
            events_to_emit.append(
                ReasoningMessageContentEvent(
                    type=EventType.REASONING_MESSAGE_CONTENT, message_id=reasoning_id, delta=chunk.reasoning_content
                )
            )

    elif isinstance(chunk, (AgentReasoningStepEvent, TeamReasoningStepEvent)):
        if message_started:
            events_to_emit.append(TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id))
            message_started = False
            message_id = str(uuid.uuid4())
        reasoning_id, is_new = event_buffer.ensure_reasoning_started()
        if is_new:
            events_to_emit.append(ReasoningStartEvent(type=EventType.REASONING_START, message_id=reasoning_id))
            events_to_emit.append(
                ReasoningMessageStartEvent(
                    type=EventType.REASONING_MESSAGE_START, message_id=reasoning_id, role="reasoning"
                )
            )
        step_num = event_buffer.next_reasoning_step()
        delta = _format_reasoning_step_delta(chunk.content, step_num)
        if delta:
            events_to_emit.append(
                ReasoningMessageContentEvent(
                    type=EventType.REASONING_MESSAGE_CONTENT, message_id=reasoning_id, delta=delta
                )
            )

    elif isinstance(chunk, (AgentReasoningCompletedEvent, TeamReasoningCompletedEvent)):
        if event_buffer.reasoning_message_id is not None:
            reasoning_id = event_buffer.reasoning_message_id
            events_to_emit.append(
                ReasoningMessageEndEvent(type=EventType.REASONING_MESSAGE_END, message_id=reasoning_id)
            )
            events_to_emit.append(ReasoningEndEvent(type=EventType.REASONING_END, message_id=reasoning_id))
            event_buffer.end_reasoning()

    # Handle workflow-level events
    elif chunk.event == WorkflowRunEvent.workflow_started:
        workflow_name = getattr(chunk, "workflow_name", None) or "workflow"
        events_to_emit.append(
            CustomEvent(
                name="WorkflowStarted",
                value={"workflow_name": workflow_name, "message": f"Starting workflow: {workflow_name}"},
            )
        )
        # Open a reasoning message that will render as the "Thinking…" card in
        # CopilotKit-based clients (Dojo). Step lifecycle deltas accumulate into
        # this single card and the card auto-collapses to "Thought for N seconds"
        # when we close it on workflow completion.
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
    elif chunk.event in {e.value for e in WorkflowRunEvent}:
        log_debug(f"AGUI: workflow event {chunk.event!r} has no explicit handler (intentional or deferred)")

    # Handle custom events
    elif chunk.event in (RunEvent.custom_event, TeamRunEvent.custom_event):
        # Use the name of the event class if available, otherwise default to the CustomEvent
        try:
            custom_event_name = chunk.__class__.__name__
        except Exception:
            custom_event_name = chunk.event

        # Use the complete Agno event as value if parsing it works, else fall back
        # to the content field and surface the to_dict failure so genuine
        # serialization bugs aren't silently masked.
        try:
            custom_event_value = chunk.to_dict()
        except Exception as exc:
            log_warning(f"CustomEvent {custom_event_name}.to_dict() failed; falling back to .content: {exc}")
            custom_event_value = chunk.content  # type: ignore

        custom_event = CustomEvent(name=custom_event_name, value=custom_event_value)
        events_to_emit.append(custom_event)

    # Catch-all: emit unmapped events as RawEvent (skip content events which may have None content)
    elif not is_content_event:
        try:
            raw_dict = chunk.to_dict()
        except Exception:
            raw_dict = {"event": str(chunk.event)}
        events_to_emit.append(RawEvent(type=EventType.RAW, event=raw_dict, source="agno"))

    return events_to_emit, message_started, message_id


def _create_completion_events(
    chunk: Union[RunOutputEvent, TeamRunOutputEvent, WorkflowRunOutputEvent],
    event_buffer: EventBuffer,
    message_started: bool,
    message_id: str,
    thread_id: str,
    run_id: str,
    run_state: Optional[Dict[str, Any]] = None,
) -> List[BaseEvent]:
    """Create events for run completion."""
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
            end_event = ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=tool_call_id,
            )
            events_to_emit.append(end_event)

    # End the message and run, denoting the end of the session
    if message_started:
        end_message_event = TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id)
        events_to_emit.append(end_message_event)

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
                        TextMessageStartEvent(
                            type=EventType.TEXT_MESSAGE_START, message_id=wf_message_id, role="assistant"
                        )
                    )
                    events_to_emit.append(
                        TextMessageContentEvent(
                            type=EventType.TEXT_MESSAGE_CONTENT, message_id=wf_message_id, delta=rendered
                        )
                    )
                    events_to_emit.append(
                        TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=wf_message_id)
                    )
        wf_name = chunk.workflow_name or "workflow"
        events_to_emit.append(
            CustomEvent(
                name="WorkflowCompleted",
                value={"workflow_name": wf_name, "message": f"Workflow completed: {wf_name}"},
            )
        )

    # Emit external execution tools
    if isinstance(chunk, RunPausedEvent):
        external_tools = chunk.tools_awaiting_external_execution
        if external_tools:
            # First, emit an assistant message for external tool calls
            assistant_message_id = str(uuid.uuid4())
            assistant_start_event = TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=assistant_message_id,
                role="assistant",
            )
            events_to_emit.append(assistant_start_event)

            # Add any text content if present for the assistant message
            if chunk.content:
                content_event = TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=assistant_message_id,
                    delta=str(chunk.content),
                )
                events_to_emit.append(content_event)

            # End the assistant message
            assistant_end_event = TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=assistant_message_id,
            )
            events_to_emit.append(assistant_end_event)

            # Emit tool call events for external execution
            for tool in external_tools:
                if tool.tool_call_id is None or tool.tool_name is None:
                    continue

                start_event = ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=tool.tool_call_id,
                    tool_call_name=tool.tool_name,
                    parent_message_id=assistant_message_id,  # Use the assistant message as parent
                )
                events_to_emit.append(start_event)

                args_event = ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=tool.tool_call_id,
                    # default=str handles non-JSON-serializable types.
                    delta=json.dumps(tool.tool_args, default=str),
                )
                events_to_emit.append(args_event)

                end_event = ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=tool.tool_call_id,
                )
                events_to_emit.append(end_event)

    # Emit final state snapshot before finishing the run (only if frontend opted into state tracking)
    if run_state is not None:
        # Use session_state from RunCompletedEvent (authoritative) if available, otherwise fall back to run_state.
        # Deep-copy so the emitted event doesn't alias the live agent state (consistent with set_state_snapshot).
        authoritative_state = getattr(chunk, "session_state", None)
        final_state = authoritative_state if authoritative_state is not None else run_state
        snapshot_event = StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=copy.deepcopy(final_state))
        events_to_emit.append(snapshot_event)

    run_finished_event = RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id)
    events_to_emit.append(run_finished_event)

    return events_to_emit


def _emit_event_logic(event: BaseEvent, event_buffer: EventBuffer) -> List[BaseEvent]:
    """Process an event and return events to actually emit."""
    events_to_emit: List[BaseEvent] = [event]

    # Update the event buffer state for tracking purposes
    if event.type == EventType.TOOL_CALL_START:
        tool_call_id = getattr(event, "tool_call_id", None)
        if tool_call_id:
            event_buffer.start_tool_call(tool_call_id)
    elif event.type == EventType.TOOL_CALL_END:
        tool_call_id = getattr(event, "tool_call_id", None)
        if tool_call_id:
            event_buffer.end_tool_call(tool_call_id)

    return events_to_emit


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
        # Check if this is a completion event
        if (
            chunk.event == RunEvent.run_completed
            or chunk.event == TeamRunEvent.run_completed
            or chunk.event == RunEvent.run_paused
            or chunk.event == TeamRunEvent.run_paused
            or chunk.event == WorkflowRunEvent.workflow_completed
            or chunk.event == WorkflowRunEvent.workflow_error
            or chunk.event == WorkflowRunEvent.workflow_cancelled
        ):
            # Store completion chunk but don't process it yet
            completion_chunk = chunk
            stream_completed = True
        else:
            # Process regular chunk immediately
            events_from_chunk, message_started, message_id = _create_events_from_chunk(
                chunk, message_id, message_started, event_buffer, run_state=run_state, is_workflow=is_workflow
            )

            for event in events_from_chunk:
                events_to_emit = _emit_event_logic(event_buffer=event_buffer, event=event)
                for emit_event in events_to_emit:
                    yield emit_event

    # Process ONLY completion cleanup events, not content from completion chunk
    if completion_chunk:
        completion_events = _create_completion_events(
            completion_chunk, event_buffer, message_started, message_id, thread_id, run_id, run_state=run_state
        )
        for event in completion_events:
            events_to_emit = _emit_event_logic(event_buffer=event_buffer, event=event)
            for emit_event in events_to_emit:
                yield emit_event

    # Ensure completion events are always emitted even when stream ends naturally
    if not stream_completed:
        # Create a synthetic completion event to ensure proper cleanup
        from agno.run.agent import RunCompletedEvent

        synthetic_completion = RunCompletedEvent()
        completion_events = _create_completion_events(
            synthetic_completion, event_buffer, message_started, message_id, thread_id, run_id, run_state=run_state
        )
        for event in completion_events:
            events_to_emit = _emit_event_logic(event_buffer=event_buffer, event=event)
            for emit_event in events_to_emit:
                yield emit_event


# Async version - thin wrapper
async def async_stream_agno_response_as_agui_events(
    response_stream: AsyncIterator[Union[RunOutputEvent, TeamRunOutputEvent, WorkflowRunOutputEvent]],
    thread_id: str,
    run_id: str,
    run_state: Optional[Dict[str, Any]] = None,
    is_workflow: bool = False,
) -> AsyncIterator[BaseEvent]:
    """Map the Agno response stream to AG-UI format, handling event ordering constraints."""
    message_id = ""  # Will be set by EventBuffer when text message starts
    message_started = False
    event_buffer = EventBuffer()
    stream_completed = False

    # Establish baseline state snapshot for delta tracking
    if run_state is not None:
        event_buffer.set_state_snapshot(run_state)

    completion_chunk = None

    async for chunk in response_stream:
        # Check if this is a completion event
        if (
            chunk.event == RunEvent.run_completed
            or chunk.event == TeamRunEvent.run_completed
            or chunk.event == RunEvent.run_paused
            or chunk.event == TeamRunEvent.run_paused
            or chunk.event == WorkflowRunEvent.workflow_completed
            or chunk.event == WorkflowRunEvent.workflow_error
            or chunk.event == WorkflowRunEvent.workflow_cancelled
        ):
            # Store completion chunk but don't process it yet
            completion_chunk = chunk
            stream_completed = True
        else:
            # Process regular chunk immediately
            events_from_chunk, message_started, message_id = _create_events_from_chunk(
                chunk, message_id, message_started, event_buffer, run_state=run_state, is_workflow=is_workflow
            )

            for event in events_from_chunk:
                events_to_emit = _emit_event_logic(event_buffer=event_buffer, event=event)
                for emit_event in events_to_emit:
                    yield emit_event

    # Process ONLY completion cleanup events, not content from completion chunk
    if completion_chunk:
        completion_events = _create_completion_events(
            completion_chunk, event_buffer, message_started, message_id, thread_id, run_id, run_state=run_state
        )
        for event in completion_events:
            events_to_emit = _emit_event_logic(event_buffer=event_buffer, event=event)
            for emit_event in events_to_emit:
                yield emit_event

    # Ensure completion events are always emitted even when stream ends naturally
    if not stream_completed:
        # Create a synthetic completion event to ensure proper cleanup
        from agno.run.agent import RunCompletedEvent

        synthetic_completion = RunCompletedEvent()
        completion_events = _create_completion_events(
            synthetic_completion, event_buffer, message_started, message_id, thread_id, run_id, run_state=run_state
        )
        for event in completion_events:
            events_to_emit = _emit_event_logic(event_buffer=event_buffer, event=event)
            for emit_event in events_to_emit:
                yield emit_event
