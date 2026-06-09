"""Async router exposing an Agno Agent, Team, or Workflow in an AG-UI compatible format."""

import copy
import uuid
from typing import AsyncIterator, Optional, Union

from agno.utils.log import log_error, log_info

try:
    from ag_ui.core import (
        BaseEvent,
        EventType,
        RunAgentInput,
        RunErrorEvent,
        RunStartedEvent,
        StateSnapshotEvent,
    )
    from ag_ui.encoder import EventEncoder
except ImportError as e:
    raise ImportError("`ag_ui` not installed. Please install it with `pip install -U ag-ui-protocol`") from e

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from agno.agent import Agent, RemoteAgent
from agno.os.interfaces.agui.media import extract_agui_media
from agno.os.interfaces.agui.messages import extract_agui_user_input
from agno.os.interfaces.agui.state import validate_agui_state
from agno.os.interfaces.agui.streaming import async_stream_agno_response_as_agui_events
from agno.team.remote import RemoteTeam
from agno.team.team import Team
from agno.workflow import RemoteWorkflow, Workflow


async def run_agent(agent: Union[Agent, RemoteAgent], run_input: RunAgentInput) -> AsyncIterator[BaseEvent]:
    """Run the contextual Agent, mapping AG-UI input messages to Agno format, and streaming the response in AG-UI format."""
    run_id = run_input.run_id or str(uuid.uuid4())

    try:
        # AG-UI frontends send full conversation history every request.
        # Extract only the last user message — agent manages history via session DB.
        user_input = extract_agui_user_input(run_input.messages or [])
        images, audio, videos, files = extract_agui_media(run_input.messages or [])

        yield RunStartedEvent(type=EventType.RUN_STARTED, thread_id=run_input.thread_id, run_id=run_id)

        # Look for user_id in run_input.forwarded_props
        user_id = None
        if run_input.forwarded_props and isinstance(run_input.forwarded_props, dict):
            user_id = run_input.forwarded_props.get("user_id")

        # Validating the session state is of the expected type (dict)
        session_state = validate_agui_state(run_input.state, run_input.thread_id)

        # Emit initial state snapshot if state is provided
        if session_state is not None:
            # Deep-copy so the emitted event doesn't alias the live agent state (consistent with final snapshot).
            yield StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=copy.deepcopy(session_state))

        # Request streaming response from agent
        response_stream = agent.arun(  # type: ignore
            input=user_input,
            session_id=run_input.thread_id,
            stream=True,
            stream_events=True,
            user_id=user_id,
            images=images or None,
            audio=audio or None,
            videos=videos or None,
            files=files or None,
            session_state=session_state,
            run_id=run_id,
        )

        # Stream the response content in AG-UI format
        async for event in async_stream_agno_response_as_agui_events(
            response_stream=response_stream,  # type: ignore
            thread_id=run_input.thread_id,
            run_id=run_id,
            run_state=session_state,
        ):
            yield event

    # Emit a RunErrorEvent if any error occurs
    except Exception as e:
        log_error(f"Error running agent: {str(e)}")
        yield RunErrorEvent(type=EventType.RUN_ERROR, message=str(e))


async def run_team(team: Union[Team, RemoteTeam], input: RunAgentInput) -> AsyncIterator[BaseEvent]:
    """Run the contextual Team, mapping AG-UI input messages to Agno format, and streaming the response in AG-UI format."""
    run_id = input.run_id or str(uuid.uuid4())
    try:
        # AG-UI frontends send full conversation history every request.
        # Extract only the last user message — team manages history via session DB.
        user_input = extract_agui_user_input(input.messages or [])
        images, audio, videos, files = extract_agui_media(input.messages or [])
        yield RunStartedEvent(type=EventType.RUN_STARTED, thread_id=input.thread_id, run_id=run_id)

        # Look for user_id in input.forwarded_props
        user_id = None
        if input.forwarded_props and isinstance(input.forwarded_props, dict):
            user_id = input.forwarded_props.get("user_id")

        # Validating the session state is of the expected type (dict)
        session_state = validate_agui_state(input.state, input.thread_id)

        # Emit initial state snapshot if state is provided
        if session_state is not None:
            # Deep-copy so the emitted event doesn't alias the live agent state (consistent with final snapshot).
            yield StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=copy.deepcopy(session_state))

        # Request streaming response from team
        response_stream = team.arun(  # type: ignore
            input=user_input,
            session_id=input.thread_id,
            stream=True,
            stream_events=True,
            user_id=user_id,
            images=images or None,
            audio=audio or None,
            videos=videos or None,
            files=files or None,
            session_state=session_state,
            run_id=run_id,
        )

        # Stream the response content in AG-UI format
        async for event in async_stream_agno_response_as_agui_events(
            response_stream=response_stream, thread_id=input.thread_id, run_id=run_id, run_state=session_state
        ):
            yield event

    except Exception as e:
        log_error(f"Error running team: {str(e)}")
        yield RunErrorEvent(type=EventType.RUN_ERROR, message=str(e))


async def run_workflow(workflow: Union[Workflow, RemoteWorkflow], input: RunAgentInput) -> AsyncIterator[BaseEvent]:
    """Run the contextual Workflow, mapping AG-UI input messages to Agno format, and streaming the response in AG-UI format."""
    run_id = input.run_id or str(uuid.uuid4())
    try:
        # AG-UI frontends send full conversation history every request.
        # Extract only the last user message — workflow manages history via session DB.
        user_input = extract_agui_user_input(input.messages or [])
        yield RunStartedEvent(type=EventType.RUN_STARTED, thread_id=input.thread_id, run_id=run_id)

        # Look for user_id in input.forwarded_props
        user_id = None
        if input.forwarded_props and isinstance(input.forwarded_props, dict):
            user_id = input.forwarded_props.get("user_id")

        # Validating the session state is of the expected type (dict)
        session_state = validate_agui_state(input.state, input.thread_id)

        # Request streaming response from workflow
        response_stream = workflow.arun(  # type: ignore
            input=user_input,
            session_id=input.thread_id,
            stream=True,
            stream_events=True,
            user_id=user_id,
            session_state=session_state,
            run_id=run_id,
        )

        # Stream the response content in AG-UI format
        async for event in async_stream_agno_response_as_agui_events(
            response_stream=response_stream,
            thread_id=input.thread_id,
            run_id=run_id,
            is_workflow=True,
        ):
            yield event

    except Exception as e:
        # Log full details internally with correlation IDs; surface a sanitized
        # message to the client to avoid leaking internal paths, DSNs, or stack
        # fragments via the AG-UI stream.
        log_error(f"Error running workflow run_id={run_id} thread_id={input.thread_id}: {type(e).__name__}: {e}")
        yield RunErrorEvent(type=EventType.RUN_ERROR, message="Workflow execution failed")


def attach_routes(
    router: APIRouter,
    agent: Optional[Union[Agent, RemoteAgent]] = None,
    team: Optional[Union[Team, RemoteTeam]] = None,
    workflow: Optional[Union[Workflow, RemoteWorkflow]] = None,
) -> APIRouter:
    if agent is None and team is None and workflow is None:
        raise ValueError("Either agent, team, or workflow must be provided.")

    encoder = EventEncoder()

    @router.post(
        "/agui",
        name="run_agent",
    )
    async def run_agent_agui(request: Request, run_input: RunAgentInput):
        # NOTE on identity: user_id is taken from run_input.forwarded_props inside each
        # run_*() helper, which is client-controlled. In production deployments using
        # authentication, the deployer is responsible for binding the authenticated
        # principal to the request before this handler runs (e.g. via middleware that
        # overrides forwarded_props.user_id), or for not trusting user_id for
        # authorization decisions. Sessions are namespaced by thread_id which is also
        # client-controlled.
        async def event_generator():
            if agent:
                source = run_agent(agent, run_input)
            elif team:
                source = run_team(team, run_input)
            elif workflow:
                source = run_workflow(workflow, run_input)
            else:
                return
            async for event in source:
                # Detect client disconnect to stop burning LLM tokens after the
                # browser closes the SSE stream. Without this, workflows with
                # many steps continue running until the upstream model finishes
                # all calls.
                if await request.is_disconnected():
                    log_info(f"AGUI client disconnected; stopping stream for run_id={run_input.run_id}")
                    break
                encoded_event = encoder.encode(event)
                yield encoded_event

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            },
        )

    @router.get("/status")
    async def get_status():
        return {"status": "available"}

    return router
