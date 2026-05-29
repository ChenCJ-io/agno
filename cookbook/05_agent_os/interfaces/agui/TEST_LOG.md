# Test Log: interfaces/agui

> Tests not yet run. Run each file and update this log.

### agent_with_media.py

**Status:** PASS

**Description:** Agent With Media - AG-UI agent (Google Gemini) that accepts multimodal user input.

**Result:** AgentOS boots with the AG-UI interface; /config returns 200. Media sent through POST /agui reaches the Gemini agent, which describes it accurately - verified with an image via CLI and interactively in a browser. Multimodal input path verified end-to-end.

---

### agent_with_silent_tools.py

**Status:** PENDING

**Description:** Silent External Tools - Suppress verbose messages in frontends.

---

### agent_with_tools.py

**Status:** PENDING

**Description:** Agent With Tools.

---

### basic.py

**Status:** PENDING

**Description:** Basic.

---

### multiple_instances.py

**Status:** PENDING

**Description:** Multiple Instances.

---

### reasoning_agent.py

**Status:** PENDING

**Description:** Reasoning Agent.

---

### research_team.py

**Status:** PENDING

**Description:** Research Team.

---

### state_events.py

**Status:** PENDING

**Description:** Outbound state synchronization via STATE_SNAPSHOT + STATE_DELTA events. Emits initial and final STATE_SNAPSHOT events plus STATE_DELTA JSON Patch ops after each state-mutating tool call.

---

### structured_output.py

**Status:** PENDING

**Description:** Structured Output.

---

### workflow.py

**Status:** PASS

**Description:** Workflow via AG-UI - Adaptive Workflow (keyword Router) mounted on AGUI interface.

**Test matrix:**
- Static: format.sh PASS, validate.sh PASS (1 unrelated pre-existing mypy error in redis vectordb)
- Unit: 43 tests across test_agui_workflow_events.py, test_workflow_interfaces.py, test_agui_router.py - all pass
- Integration: 43 Router engine tests pass
- Error paths: workflow_error, workflow_cancel, step_error, None workflow_name/step_name - all covered
- Regression: agent path, team path, mutual exclusivity check - all unaffected
- Manual (Dojo): "hi", "hi there", "thanks", "what is photosynthesis" - all routed correctly, Thinking card shows "Workflow: Adaptive Workflow" plus correct step lifecycle, final answer renders

**Result:** All paths verified. Workflow name visible in Thinking card. Keyword Router routes 1-2 word inputs to chat, 3+ word inputs to research+summarize.

---
