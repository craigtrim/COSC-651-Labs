"""Actual CrewAI orchestration with either the fixture or live course model adapter."""

import json
import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
from crewai import Agent, Task, Crew, Process, BaseLLM
from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel, Field
from .experiments import payload, gather


# Issue #32: CrewAI tasks share the course model and collected evidence.
class CourseLLM(BaseLLM):
    """Adapt a course model to CrewAI's text response interface."""

    def __init__(self, model, task_kind, context):
        """Bind a course model, task kind, and context for subsequent calls."""
        super().__init__(model="course/" + model.model_id, temperature=0)
        self.course_model = model
        self.task_kind = task_kind
        self.context = context

    def call(
        self, messages, tools=None, callbacks=None, available_functions=None, **kwargs
    ):
        """Add messages to the context and return JSON with a final-answer prefix."""
        request = {**self.context, "orchestration_messages": messages}
        response = self.course_model.complete(self.task_kind, request)
        return "Final Answer: " + json.dumps(response)

    def supports_function_calling(self):
        """Report that CrewAI function calling is unsupported."""
        return False

    def get_context_window_size(self):
        """Return the fixed context window of 16,000 tokens advertised to CrewAI."""
        return 16000


def run_crew(model, registry, case):
    """Gather evidence and return parsed JSON from sequential CrewAI tasks."""
    gather(registry)
    context = payload(case, registry)
    agents = []
    tasks = []
    for role, kind, goal in [
        (
            "Evidence researcher",
            "researcher",
            "Return the identifiers of relevant supplied evidence.",
        ),
        (
            "Maintenance analyst",
            "proposal",
            "Produce a maintenance recommendation from the evidence.",
        ),
        (
            "Recommendation reviewer",
            "revision",
            "Check the recommendation against the evidence and return the final report.",
        ),
    ]:
        agent = Agent(
            role=role,
            goal=goal,
            backstory="You work only with synthetic LedgerLine course data. Do not approve or execute changes.",
            llm=CourseLLM(model, kind, context),
            allow_delegation=False,
            max_iter=2,
            verbose=False,
        )
        task = Task(
            description=goal
            + " Use the supplied case and evidence. "
            + json.dumps(context),
            expected_output="One JSON object matching the role output contract.",
            agent=agent,
            context=tasks[-1:] if tasks else [],
        )
        agents.append(agent)
        tasks.append(task)
    result = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        memory=False,
    ).kickoff()
    raw = result.raw
    if raw.startswith("Final Answer:"):
        raw = raw.split(":", 1)[1].strip()
    return json.loads(raw)


class MaintenanceState(BaseModel):
    """Track the case identifier, flow events, and review decision."""

    case_id: str = "D01"
    events: list[str] = Field(default_factory=list)
    decision: str = "pending"


class DemonstrationFlow(Flow[MaintenanceState]):
    """Record demonstration stages and return the final review state."""

    @start()
    def intake(self):
        """Record intake and return the current case identifier."""
        self.state.events.append("intake")
        return self.state.case_id

    @listen(intake)
    def diagnose(self, case_id):
        """Record diagnosis and return needs_review for any supplied case ID."""
        self.state.events.append("diagnose")
        return "needs_review"

    @listen(diagnose)
    def review_gate(self, status):
        """Record the approval wait and return state with the supplied decision."""
        self.state.events.append("await_human_approval")
        self.state.decision = status
        return self.state.model_dump()
