"""Central lifecycle status reference used by help/status reporting."""

from __future__ import annotations

from typing import Any

STATUS_REFERENCE_SCHEMA = "status_reference_v1"

PLAN_STATUS_LIFECYCLE = [
    {
        "status": "pending",
        "meaning": "Plan accepted and waiting to enter planner execution.",
        "typical_transitions": ["running", "failed", "cancelled"],
    },
    {
        "status": "running",
        "meaning": "Planner is actively producing or refining the plan.",
        "typical_transitions": ["awaiting_input", "finalized", "failed", "cancelled"],
    },
    {
        "status": "awaiting_input",
        "meaning": (
            "Enrichment agent is paused waiting for a human response to clarifying questions."
        ),
        "typical_transitions": ["running", "failed", "cancelled"],
    },
    {
        "status": "finalized",
        "meaning": "Plan text finalized; task creation can begin.",
        "typical_transitions": [],
    },
    {
        "status": "failed",
        "meaning": "Plan execution failed and requires manual retry or inspection.",
        "typical_transitions": [],
    },
    {
        "status": "cancelled",
        "meaning": "Plan intentionally stopped and will not execute.",
        "typical_transitions": [],
    },
]

TASK_STATUS_LIFECYCLE = [
    {
        "status": "blocked",
        "meaning": "Task exists but cannot start due to blockers or waiting conditions.",
        "typical_transitions": ["ready", "failed", "cancelled"],
    },
    {
        "status": "ready",
        "meaning": "Task is ready for execution (all blockers cleared).",
        "typical_transitions": ["running", "cancelled"],
    },
    {
        "status": "running",
        "meaning": "Task is claimed and actively executing.",
        "typical_transitions": ["review", "failed", "cancelled"],
    },
    {
        "status": "review",
        "meaning": "Executor completed work and task awaits review result.",
        "typical_transitions": ["approved", "rejected", "failed"],
    },
    {
        "status": "rejected",
        "meaning": "Reviewer rejected changes; auto-transitions to running for re-execution.",
        "typical_transitions": ["running"],
    },
    {
        "status": "approved",
        "meaning": "Code passed review and is ready to merge.",
        "typical_transitions": ["completed", "cancelled"],
    },
    {
        "status": "completed",
        "meaning": "Task merged and done; terminal success state.",
        "typical_transitions": [],
    },
    {
        "status": "failed",
        "meaning": "Task execution/review failed and can be retried manually.",
        "typical_transitions": ["blocked"],
    },
    {
        "status": "cancelled",
        "meaning": "Task intentionally stopped and will not execute.",
        "typical_transitions": [],
    },
]

TASK_CREATION_STATUS_LIFECYCLE = [
    {
        "status": "awaiting_approval",
        "meaning": "Plan finalized but project requires manual approval before tasks are created.",
        "typical_transitions": ["pending"],
    },
    {
        "status": "pending",
        "meaning": "Plan is ready to run task creation for approved plan content.",
        "typical_transitions": ["running", "failed"],
    },
    {
        "status": "running",
        "meaning": "Task creation is currently deriving executable tasks.",
        "typical_transitions": ["completed", "failed"],
    },
    {
        "status": "completed",
        "meaning": "Tasks are fully materialized for execution.",
        "typical_transitions": [],
    },
    {
        "status": "failed",
        "meaning": "Task creation failed and requires manual retry.",
        "typical_transitions": ["pending"],
    },
]

PROMPT_STATUS_LIFECYCLE = [
    {
        "status": "pending",
        "meaning": "Prompt awaiting enrichment.",
        "typical_transitions": ["enriching", "finalized"],
    },
    {
        "status": "enriching",
        "meaning": "Enrichment agent is actively refining the prompt.",
        "typical_transitions": ["awaiting_input", "finalized", "failed"],
    },
    {
        "status": "awaiting_input",
        "meaning": "Enrichment paused waiting for answers to clarifying questions.",
        "typical_transitions": ["enriching", "failed", "cancelled"],
    },
    {
        "status": "finalized",
        "meaning": "Enriched prompt ready for the planner.",
        "typical_transitions": [],
    },
    {
        "status": "failed",
        "meaning": "Enrichment failed.",
        "typical_transitions": [],
    },
    {
        "status": "cancelled",
        "meaning": "Enrichment cancelled.",
        "typical_transitions": [],
    },
]

STATUS_LIFECYCLES = [
    {
        "type": "plan",
        "label": "Plan lifecycle",
        "description": "Statuses used by plan requests.",
        "statuses": PLAN_STATUS_LIFECYCLE,
    },
    {
        "type": "task",
        "label": "Task lifecycle",
        "description": "Statuses used by tasks.",
        "statuses": TASK_STATUS_LIFECYCLE,
    },
    {
        "type": "task_creation",
        "label": "Task-creation lifecycle",
        "description": "Statuses used while materializing tasks from a plan.",
        "statuses": TASK_CREATION_STATUS_LIFECYCLE,
    },
    {
        "type": "prompt",
        "label": "Prompt enrichment lifecycle",
        "description": "Statuses tracking the enrichment phase of a plan's prompt.",
        "statuses": PROMPT_STATUS_LIFECYCLE,
    },
]


def get_status_reference() -> dict[str, Any]:
    """Return a machine-parseable lifecycle reference payload."""
    return {
        "schema": STATUS_REFERENCE_SCHEMA,
        "lifecycles": [
            {
                "type": lifecycle["type"],
                "label": lifecycle["label"],
                "description": lifecycle["description"],
                "statuses": [
                    {
                        "status": status["status"],
                        "meaning": status["meaning"],
                        "typical_transitions": list(status["typical_transitions"]),
                    }
                    for status in lifecycle["statuses"]
                ],
            }
            for lifecycle in STATUS_LIFECYCLES
        ],
    }
