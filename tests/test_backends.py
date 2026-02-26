"""Tests for backend prompt and output schema definitions."""

import pytest

from agm.backends import (
    DEFAULT_QUALITY_GATE,
    ENRICHMENT_OUTPUT_SCHEMA,
    ENRICHMENT_PROMPT_SUFFIX,
    EXECUTOR_PROMPT_SUFFIX,
    IMPLEMENTED_BACKENDS,
    MAX_PARENT_PLAN_CHARS,
    MODEL_CATALOG,
    PLAN_OUTPUT_SCHEMA,
    PLAN_PROMPT_SUFFIX,
    QUALITY_GATE_GENERATE_SCHEMA,
    QUALITY_GATE_PRESETS,
    REFRESH_PROMPT_SUFFIX,
    REVIEW_OUTPUT_SCHEMA,
    TASK_OUTPUT_SCHEMA,
    TASK_PROMPT_SUFFIX,
    THINK_MODEL,
    THREAD_CONFIGS,
    TURN_CONFIGS,
    WORK_MODEL,
    append_suffix_once,
    build_enrichment_continuation_prompt,
    build_enrichment_prompt,
    build_enrichment_resume_prompt,
    build_plan_prompt,
    build_task_creation_prompt,
    get_known_model_ids,
    get_live_models,
    get_runtime_thread_config,
    is_model_available,
    resolve_model_config,
    set_live_models,
)
from agm.jobs_common import _extract_reasoning_summaries


def test_default_quality_gate_is_empty():
    """Default quality gate is empty (agm is project-agnostic)."""
    assert "auto_fix" in DEFAULT_QUALITY_GATE
    assert "checks" in DEFAULT_QUALITY_GATE
    assert DEFAULT_QUALITY_GATE["auto_fix"] == []
    assert DEFAULT_QUALITY_GATE["checks"] == []


def test_plan_output_schema_task_priority_required_enum():
    task_schema = PLAN_OUTPUT_SCHEMA["properties"]["tasks"]["items"]
    priority_schema = task_schema["properties"]["priority"]

    assert priority_schema["type"] == "string"
    assert priority_schema["enum"] == ["high", "medium", "low"]
    assert "priority" in task_schema["required"]
    assert task_schema["additionalProperties"] is False


def test_task_output_schema_task_priority_required_enum():
    task_schema = TASK_OUTPUT_SCHEMA["properties"]["tasks"]["items"]
    priority_schema = task_schema["properties"]["priority"]

    assert priority_schema["type"] == "string"
    assert priority_schema["enum"] == ["high", "medium", "low"]
    assert "priority" in task_schema["required"]
    assert task_schema["additionalProperties"] is False


def test_build_plan_prompt_appends_suffix_once():
    plain = "Plan the API changes"
    assert build_plan_prompt(plain) == plain + PLAN_PROMPT_SUFFIX
    assert build_plan_prompt(plain + PLAN_PROMPT_SUFFIX) == plain + PLAN_PROMPT_SUFFIX
    assert append_suffix_once(plain, PLAN_PROMPT_SUFFIX) == plain + PLAN_PROMPT_SUFFIX
    assert (
        append_suffix_once(plain + PLAN_PROMPT_SUFFIX, PLAN_PROMPT_SUFFIX)
        == plain + PLAN_PROMPT_SUFFIX
    )


def test_build_task_creation_prompt_is_exact_payload():
    plan_text = '{"title":"T","summary":"S","tasks":[]}'
    expected = f"Plan JSON:\n```json\n{plan_text}\n```{TASK_PROMPT_SUFFIX}"
    assert build_task_creation_prompt(plan_text) == expected


# -- THREAD_CONFIGS validation --

EXPECTED_JOB_TYPES = {
    "prompt_enrichment",
    "codebase_exploration",
    "plan_request",
    "task_creation",
    "task_execution",
    "task_review",
    "query",
}


def test_thread_configs_covers_all_job_types():
    assert set(THREAD_CONFIGS.keys()) == EXPECTED_JOB_TYPES


def test_turn_configs_covers_all_job_types():
    assert set(TURN_CONFIGS.keys()) == EXPECTED_JOB_TYPES


@pytest.mark.parametrize("job_type", sorted(EXPECTED_JOB_TYPES))
def test_thread_config_has_required_fields(job_type):
    config = THREAD_CONFIGS[job_type]
    assert "approvalPolicy" in config, f"{job_type} missing approvalPolicy"
    assert config["approvalPolicy"] == "never", f"{job_type} should auto-approve"
    assert "personality" in config, f"{job_type} missing personality"
    assert "model" in config, f"{job_type} missing model"
    assert "sandbox" in config, f"{job_type} missing sandbox"
    assert "developerInstructions" in config, f"{job_type} missing developerInstructions"


# Table of expected sandbox + config per job type
EXPECTED_CONFIGS = {
    "prompt_enrichment": {
        "sandbox": "read-only",
        "developer_instructions_is_none": False,
        "has_output_schema": True,
    },
    "plan_request": {
        "sandbox": "read-only",
        "developer_instructions_is_none": False,
        "has_output_schema": True,
    },
    "task_creation": {
        "sandbox": "read-only",
        "developer_instructions_is_none": False,
        "has_output_schema": True,
    },
    "task_execution": {
        "sandbox": "danger-full-access",
        "developer_instructions_is_none": False,
        "has_output_schema": False,
    },
    "task_review": {
        "sandbox": "read-only",
        "developer_instructions_is_none": False,
        "has_output_schema": True,
    },
    "codebase_exploration": {
        "sandbox": "read-only",
        "developer_instructions_is_none": False,
        "has_output_schema": True,
    },
}


@pytest.mark.parametrize("job_type", sorted(EXPECTED_CONFIGS.keys()))
def test_thread_config_sandbox_mode(job_type):
    """Verify sandbox mode is correct per job type — critical for security."""
    config = THREAD_CONFIGS[job_type]
    expected = EXPECTED_CONFIGS[job_type]
    assert config["sandbox"] == expected["sandbox"], (
        f"{job_type} sandbox should be {expected['sandbox']}"
    )


@pytest.mark.parametrize("job_type", sorted(EXPECTED_CONFIGS.keys()))
def test_thread_config_developer_instructions(job_type):
    """Verify developerInstructions matches expected config per job type."""
    config = THREAD_CONFIGS[job_type]
    expected = EXPECTED_CONFIGS[job_type]
    dev_instr = config["developerInstructions"]
    if expected["developer_instructions_is_none"]:
        assert dev_instr is None, (
            f"{job_type} developerInstructions must be None to preserve plan.md"
        )
    else:
        assert isinstance(dev_instr, str) and len(dev_instr) > 0, (
            f"{job_type} developerInstructions must be a non-empty string"
        )


@pytest.mark.parametrize("job_type", sorted(EXPECTED_CONFIGS.keys()))
def test_turn_config_output_schema(job_type):
    """Verify outputSchema presence/absence per job type."""
    config = TURN_CONFIGS[job_type]
    expected = EXPECTED_CONFIGS[job_type]
    if expected["has_output_schema"]:
        assert "outputSchema" in config, f"{job_type} should have outputSchema"
    else:
        assert "outputSchema" not in config, f"{job_type} should not have outputSchema"


def test_review_output_schema_has_verdict_enum():
    verdict = REVIEW_OUTPUT_SCHEMA["properties"]["verdict"]
    assert verdict["enum"] == ["approve", "reject"]
    assert "verdict" in REVIEW_OUTPUT_SCHEMA["required"]


def test_two_tier_model_assignment():
    """Think tier (enrichment, planner, reviewer) and work tier (task_creation, executor)."""
    think_jobs = {"prompt_enrichment", "codebase_exploration", "plan_request", "task_review"}
    work_jobs = {"task_creation", "task_execution"}
    for job_type in think_jobs:
        assert THREAD_CONFIGS[job_type]["model"] == THINK_MODEL, (
            f"{job_type} should use THINK_MODEL ({THINK_MODEL})"
        )
    for job_type in work_jobs:
        assert THREAD_CONFIGS[job_type]["model"] == WORK_MODEL, (
            f"{job_type} should use WORK_MODEL ({WORK_MODEL})"
        )


# -- Per-agent config overrides --

_PIPELINE_AGENTS = {
    "prompt_enrichment",
    "codebase_exploration",
    "plan_request",
    "task_creation",
    "task_execution",
    "task_review",
}


def test_pipeline_agents_have_config_override():
    """All pipeline agents must have a config dict for thread-level overrides."""
    for job_type in _PIPELINE_AGENTS:
        cfg = THREAD_CONFIGS[job_type].get("config")
        assert isinstance(cfg, dict), f"{job_type} missing config dict"


def test_all_agents_disable_multi_agent():
    """All agent types must hard-disable sub-agent spawning."""
    for job_type in [*_PIPELINE_AGENTS, "query"]:
        features = THREAD_CONFIGS[job_type]["config"].get("features", {})
        assert features.get("multi_agent") is False, (
            f"{job_type} must have features.multi_agent = False"
        )


def test_think_tier_agents_have_live_web_search():
    """Think-tier agents (enrichment, planner) get live web search for research."""
    for job_type in ("prompt_enrichment", "plan_request"):
        assert THREAD_CONFIGS[job_type]["config"]["web_search"] == "live", (
            f"{job_type} should have web_search='live'"
        )


def test_work_tier_agents_have_disabled_web_search():
    """Work-tier and non-research agents don't need web search."""
    for job_type in ("codebase_exploration", "task_creation", "task_execution", "task_review"):
        assert THREAD_CONFIGS[job_type]["config"]["web_search"] == "disabled", (
            f"{job_type} should have web_search='disabled'"
        )


def test_runtime_thread_config_carries_config_dict():
    """get_runtime_thread_config must preserve the config dict from THREAD_CONFIGS."""
    runtime = get_runtime_thread_config("codex", "plan_request")
    assert isinstance(runtime.get("config"), dict)
    assert runtime["config"]["web_search"] == "live"
    assert runtime["config"]["features"]["multi_agent"] is False


def test_runtime_thread_config_deepcopies_config():
    """Config dict must be deep-copied so mutations don't leak back to THREAD_CONFIGS."""
    runtime = get_runtime_thread_config("codex", "task_execution")
    runtime["config"]["web_search"] = "live"  # mutate the copy
    # Original should be unchanged
    assert THREAD_CONFIGS["task_execution"]["config"]["web_search"] == "disabled"


def test_context7_only_in_planner_suffix():
    """Context7 MCP references should only appear in planner prompt suffix."""
    assert "Context7" in PLAN_PROMPT_SUFFIX
    assert "Context7" not in TASK_PROMPT_SUFFIX
    assert "Context7" not in EXECUTOR_PROMPT_SUFFIX
    assert "Context7" not in REFRESH_PROMPT_SUFFIX


def test_all_schemas_have_additional_properties_false():
    """OpenAI structured output requires additionalProperties: false."""
    schemas = [
        ENRICHMENT_OUTPUT_SCHEMA,
        PLAN_OUTPUT_SCHEMA,
        TASK_OUTPUT_SCHEMA,
        REVIEW_OUTPUT_SCHEMA,
    ]
    for schema in schemas:
        assert schema.get("additionalProperties") is False
        # Check nested task items if present
        if "tasks" in schema.get("properties", {}):
            items = schema["properties"]["tasks"].get("items", {})
            if items:
                assert items.get("additionalProperties") is False


def test_implemented_backends_includes_codex():
    assert "codex" in IMPLEMENTED_BACKENDS


def test_model_catalog_has_supported_metadata():
    """Catalog should describe backend, tier, speed, and model metadata."""
    expected_top_keys = {
        "backend",
        "tier",
        "speed",
        "description",
        "effort_defaults",
        "recommendation",
    }
    for _model, model_spec in MODEL_CATALOG.items():
        assert set(expected_top_keys).issubset(model_spec.keys())
        assert model_spec["backend"] in {"codex"}
        assert model_spec["tier"] in {"both", "think", "work"}
        assert isinstance(model_spec["speed"], str)
        assert model_spec["speed"].strip()
        assert isinstance(model_spec["description"], str)
        assert model_spec["description"].strip()
        assert isinstance(model_spec["effort_defaults"], dict)
        assert model_spec["effort_defaults"]["think"] in {"low", "medium", "high"}
        assert model_spec["effort_defaults"]["work"] in {"low", "medium", "high"}
        assert isinstance(model_spec["recommendation"], dict)
        assert "default" in model_spec["recommendation"]


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("codex", {"think_model": "gpt-5.3-codex", "work_model": "gpt-5.3-codex-spark"}),
    ],
)
def test_resolve_model_config_defaults_follow_catalog(monkeypatch, backend, expected):
    for var in {
        "AGM_MODEL_THINK",
        "AGM_MODEL_WORK",
        "AGM_MODEL_THINK_EFFORT",
        "AGM_MODEL_WORK_EFFORT",
    }:
        monkeypatch.delenv(var, raising=False)

    resolved = resolve_model_config(backend, {})
    assert resolved["think_model"] == expected["think_model"]
    assert resolved["work_model"] == expected["work_model"]
    assert resolved["think_effort"] == "high"
    assert resolved["work_effort"] == "high"


def test_resolve_model_config_partial_overrides(monkeypatch):
    for var in {
        "AGM_MODEL_THINK",
        "AGM_MODEL_WORK",
    }:
        monkeypatch.delenv(var, raising=False)

    resolved = resolve_model_config(
        "codex",
        {
            "think_model": "custom-think-model",
            "work_effort": "low",
        },
    )
    assert resolved["think_model"] == "custom-think-model"
    assert resolved["work_model"] == "gpt-5.3-codex-spark"
    assert resolved["think_effort"] == "high"
    assert resolved["work_effort"] == "low"


def test_resolve_model_config_precedence_project_env_catalog(monkeypatch):
    monkeypatch.setenv("AGM_MODEL_THINK", "env-think-model")
    monkeypatch.setenv("AGM_MODEL_WORK", "env-work-model")
    monkeypatch.setenv("AGM_MODEL_THINK_EFFORT", "low")
    monkeypatch.setenv("AGM_MODEL_WORK_EFFORT", "low")
    project_cfg = {"think_model": "project-think-model", "work_effort": "medium"}

    resolved = resolve_model_config("codex", project_cfg)
    assert resolved["think_model"] == "project-think-model"
    assert resolved["think_effort"] == "low"
    assert resolved["work_model"] == "env-work-model"
    assert resolved["work_effort"] == "medium"


def test_resolve_model_config_allows_unknown_models(monkeypatch):
    for var in {
        "AGM_MODEL_THINK_EFFORT",
        "AGM_MODEL_WORK_EFFORT",
    }:
        monkeypatch.delenv(var, raising=False)

    resolved = resolve_model_config(
        "codex",
        {
            "think_model": "my-unknown-codex-think",
            "work_model": "my-unknown-codex-work",
            "think_effort": "invalid",
            "work_effort": "invalid",
        },
    )
    assert resolved["think_model"] == "my-unknown-codex-think"
    assert resolved["work_model"] == "my-unknown-codex-work"
    assert resolved["think_effort"] == "high"
    assert resolved["work_effort"] == "high"


def test_resolve_model_config_normalizes_effort_values(monkeypatch):
    monkeypatch.setenv("AGM_MODEL_WORK_EFFORT", "invalid")
    codex = resolve_model_config(
        "codex",
        {
            "think_effort": "LOW",
            "work_effort": "",
        },
    )

    assert codex["think_effort"] == "low"
    assert codex["work_effort"] == "high"


def test_runtime_thread_config_resolves_model():
    from agm.backends import get_runtime_turn_config

    runtime_plan = get_runtime_thread_config(
        "codex", "plan_request", {"think_model": "custom-model", "think_effort": "low"}
    )
    assert runtime_plan["model"] == "custom-model"
    # collaborationMode no longer exists — effort is on turn config
    assert "collaborationMode" not in runtime_plan

    runtime_turn = get_runtime_turn_config(
        "codex", "plan_request", {"think_model": "custom-model", "think_effort": "low"}
    )
    assert runtime_turn["effort"] == "low"


@pytest.mark.parametrize(
    "job_type,expected_model",
    [
        ("prompt_enrichment", "custom-think"),
        ("codebase_exploration", "custom-think"),
        ("plan_request", "custom-think"),
        ("task_review", "custom-think"),
        ("query", "custom-think"),
        ("task_creation", "custom-work"),
        ("task_execution", "custom-work"),
    ],
)
def test_runtime_thread_config_uses_tier_model(job_type, expected_model):
    runtime = get_runtime_thread_config(
        "codex",
        job_type,
        {
            "think_model": "custom-think",
            "work_model": "custom-work",
            "think_effort": "low",
            "work_effort": "medium",
        },
    )
    assert runtime["model"] == expected_model


@pytest.mark.parametrize(
    "job_type,expected_effort",
    [
        ("prompt_enrichment", "low"),
        ("codebase_exploration", "low"),
        ("plan_request", "low"),
        ("task_review", "low"),
        ("query", "low"),
        ("task_creation", "medium"),
        ("task_execution", "medium"),
    ],
)
def test_runtime_turn_config_uses_tier_effort(job_type, expected_effort):
    from agm.backends import get_runtime_turn_config

    runtime = get_runtime_turn_config(
        "codex",
        job_type,
        {
            "think_model": "custom-think",
            "work_model": "custom-work",
            "think_effort": "low",
            "work_effort": "medium",
        },
    )
    assert runtime["effort"] == expected_effort


# -- Enrichment prompt builders --


def test_build_enrichment_prompt_appends_suffix():
    result = build_enrichment_prompt("Add a login page")
    assert "Add a login page" in result
    assert "enriched_prompt" in result  # From ENRICHMENT_PROMPT_SUFFIX


def test_build_enrichment_resume_prompt_formats_answers():
    questions = [
        {"question": "Which auth method?", "answer": "OAuth2"},
        {"question": "What database?", "answer": "PostgreSQL"},
    ]
    result = build_enrichment_resume_prompt(questions)
    assert "Q: Which auth method?" in result
    assert "A: OAuth2" in result
    assert "Q: What database?" in result
    assert "A: PostgreSQL" in result
    assert "enriched_prompt" in result  # Instruction to update


def test_build_enrichment_resume_prompt_includes_header():
    questions = [
        {"question": "Which auth method?", "answer": "OAuth2", "header": "Auth"},
        {"question": "What database?", "answer": "PostgreSQL", "header": None},
    ]
    result = build_enrichment_resume_prompt(questions)
    assert "Q: [Auth] Which auth method?" in result
    assert "Q: What database?" in result  # No header prefix


def test_enrichment_output_schema_structure():
    """Enrichment schema requires enriched_prompt and questions with structured options."""
    props = ENRICHMENT_OUTPUT_SCHEMA["properties"]
    assert "enriched_prompt" in props
    assert "questions" in props
    assert props["enriched_prompt"]["type"] == "string"
    assert props["questions"]["type"] == "array"
    # Nested question items must have additionalProperties: false
    items = props["questions"]["items"]
    assert items.get("additionalProperties") is False
    assert "question" in items["properties"]
    assert "header" in items["properties"]
    assert "options" in items["properties"]
    assert "multi_select" in items["properties"]
    # Options items are objects with label + description
    option_items = items["properties"]["options"]["items"]
    assert "label" in option_items["properties"]
    assert "description" in option_items["properties"]
    assert set(items["required"]) == {"question", "header", "options", "multi_select"}


# -- Exploration output schema --


def test_exploration_output_schema_structure():
    """Exploration schema requires all six fields."""
    from agm.backends import EXPLORATION_OUTPUT_SCHEMA

    props = EXPLORATION_OUTPUT_SCHEMA["properties"]
    assert "summary" in props
    assert "architecture" in props
    assert "relevant_files" in props
    assert "patterns_to_follow" in props
    assert "reusable_helpers" in props
    assert "test_locations" in props
    assert set(EXPLORATION_OUTPUT_SCHEMA["required"]) == {
        "summary",
        "architecture",
        "relevant_files",
        "patterns_to_follow",
        "reusable_helpers",
        "test_locations",
    }
    # relevant_files items have required fields
    rf_items = props["relevant_files"]["items"]
    assert set(rf_items["required"]) == {"path", "description", "key_symbols"}
    assert rf_items.get("additionalProperties") is False


# -- Exploration prompt --


def test_build_exploration_prompt_appends_suffix():
    from agm.backends import build_exploration_prompt

    result = build_exploration_prompt("Add a login page")
    assert "Add a login page" in result
    assert "Explore the codebase" in result


# -- build_plan_prompt with exploration context --


def test_build_plan_prompt_without_exploration():
    """build_plan_prompt without exploration_context behaves as before."""
    result = build_plan_prompt("Add auth module")
    assert "Add auth module" in result
    assert "<codebase_exploration>" not in result
    assert "Produce a structured implementation plan" in result


def test_build_plan_prompt_with_exploration():
    """build_plan_prompt with exploration_context prepends it in XML tags."""
    result = build_plan_prompt(
        "Add auth module",
        exploration_context='{"summary":"Project uses Flask"}',
    )
    assert "<codebase_exploration>" in result
    assert '{"summary":"Project uses Flask"}' in result
    assert "</codebase_exploration>" in result
    assert "Add auth module" in result
    assert "Produce a structured implementation plan" in result


# -- Enrichment continuation prompt --


def test_build_enrichment_continuation_prompt_includes_parent_context():
    """Continuation prompt includes parent enriched prompt, plan text, and task outcomes."""
    result = build_enrichment_continuation_prompt(
        raw_prompt="Add tests for the auth module",
        parent_enriched_prompt="Modify src/auth.py to add OAuth2 support.",
        parent_plan_text='{"title":"Add OAuth2","tasks":[]}',
        task_outcomes_summary=(
            'Summary: 2 completed, 1 failed\n  - "Add middleware" — status: completed'
        ),
    )
    assert "CONTINUATION" in result
    assert "Previous specification" in result
    assert "Modify src/auth.py" in result
    assert "Previous plan output" in result
    assert '{"title":"Add OAuth2"' in result
    assert "Task outcomes" in result
    assert "2 completed, 1 failed" in result
    assert "Add tests for the auth module" in result
    assert result.endswith(ENRICHMENT_PROMPT_SUFFIX)


def test_build_enrichment_continuation_prompt_handles_missing_parent_data():
    """Continuation prompt gracefully handles None parent data."""
    result = build_enrichment_continuation_prompt(
        raw_prompt="Follow up on the plan",
        parent_enriched_prompt=None,
        parent_plan_text=None,
        task_outcomes_summary=None,
    )
    assert "CONTINUATION" in result
    assert "Follow up on the plan" in result
    assert "Previous specification" not in result
    assert "Previous plan output" not in result
    assert "Task outcomes" not in result
    assert result.endswith(ENRICHMENT_PROMPT_SUFFIX)


def test_build_enrichment_continuation_prompt_truncates_long_plan():
    """Parent plan text is truncated at MAX_PARENT_PLAN_CHARS."""
    long_plan = "x" * (MAX_PARENT_PLAN_CHARS + 1000)
    result = build_enrichment_continuation_prompt(
        raw_prompt="Continue",
        parent_enriched_prompt=None,
        parent_plan_text=long_plan,
        task_outcomes_summary=None,
    )
    assert "... (truncated)" in result
    # The truncated plan should be exactly MAX_PARENT_PLAN_CHARS of 'x' + truncation marker
    assert "x" * MAX_PARENT_PLAN_CHARS in result
    assert "x" * (MAX_PARENT_PLAN_CHARS + 1) not in result


# -- Quality gate presets --


def test_quality_gate_presets_have_required_keys():
    """Each preset must have description and config with auto_fix + checks."""
    assert len(QUALITY_GATE_PRESETS) >= 2
    for name, preset in QUALITY_GATE_PRESETS.items():
        assert "description" in preset, f"{name} missing description"
        assert "config" in preset, f"{name} missing config"
        config = preset["config"]
        assert isinstance(config["auto_fix"], list), f"{name} auto_fix not a list"
        assert isinstance(config["checks"], list), f"{name} checks not a list"
        assert len(config["checks"]) > 0, f"{name} checks should not be empty"


@pytest.mark.parametrize("preset_name", sorted(QUALITY_GATE_PRESETS.keys()))
def test_quality_gate_preset_cmd_arrays(preset_name):
    """Every command in a preset must be a non-empty list of strings."""
    config = QUALITY_GATE_PRESETS[preset_name]["config"]
    for entry in config["auto_fix"] + config["checks"]:
        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["cmd"], list) and len(entry["cmd"]) > 0
        assert all(isinstance(c, str) for c in entry["cmd"])


def test_quality_gate_generate_schema_structure():
    """Generate schema must have auto_fix and checks, additionalProperties false."""
    assert QUALITY_GATE_GENERATE_SCHEMA["type"] == "object"
    assert "auto_fix" in QUALITY_GATE_GENERATE_SCHEMA["properties"]
    assert "checks" in QUALITY_GATE_GENERATE_SCHEMA["properties"]
    assert QUALITY_GATE_GENERATE_SCHEMA.get("additionalProperties") is False
    # Nested items must also have additionalProperties: false
    for key in ("auto_fix", "checks"):
        items = QUALITY_GATE_GENERATE_SCHEMA["properties"][key]["items"]
        assert items.get("additionalProperties") is False


# -- Reasoning summaries --


def test_executor_and_reviewer_turn_configs_have_summary():
    """Executor and reviewer turns must request concise reasoning summaries."""
    for job_type in ("task_execution", "task_review"):
        assert TURN_CONFIGS[job_type].get("summary") == "concise", (
            f"{job_type} should have summary='concise'"
        )


def test_other_turn_configs_have_no_summary():
    """Non-executor/reviewer turns should not request reasoning summaries."""
    for job_type in (
        "prompt_enrichment",
        "codebase_exploration",
        "plan_request",
        "task_creation",
        "query",
    ):
        assert "summary" not in TURN_CONFIGS[job_type], f"{job_type} should not have summary"


def test_extract_reasoning_summaries_basic():
    """Extract summary texts from a turn/completed notification."""
    params = {
        "turn": {
            "items": [
                {
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "Analyzed the codebase"},
                        {"type": "summary_text", "text": "Found the bug in auth.py"},
                    ],
                },
                {"type": "agentMessage", "text": "I fixed the bug."},
            ]
        }
    }
    summaries = _extract_reasoning_summaries(params)
    assert summaries == ["Analyzed the codebase", "Found the bug in auth.py"]


def test_extract_reasoning_summaries_empty():
    """Return empty list when no reasoning items exist."""
    assert _extract_reasoning_summaries({}) == []
    assert _extract_reasoning_summaries({"turn": {}}) == []
    assert _extract_reasoning_summaries({"turn": {"items": []}}) == []
    assert (
        _extract_reasoning_summaries({"turn": {"items": [{"type": "agentMessage", "text": "hi"}]}})
        == []
    )


def test_extract_reasoning_summaries_skips_empty_text():
    """Empty or whitespace-only summary texts are filtered out."""
    params = {
        "turn": {
            "items": [
                {
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": ""},
                        {"type": "summary_text", "text": "  "},
                        {"type": "summary_text", "text": "Actual summary"},
                    ],
                }
            ]
        }
    }
    assert _extract_reasoning_summaries(params) == ["Actual summary"]


# -- Live model catalog --


class TestLiveModelCatalog:
    """Tests for the dynamic model list cache."""

    def setup_method(self):
        """Reset live model cache before each test."""
        import agm.backends

        agm.backends._live_models = None

    def teardown_method(self):
        """Reset live model cache after each test."""
        import agm.backends

        agm.backends._live_models = None

    def test_get_live_models_returns_none_when_not_populated(self):
        assert get_live_models() is None

    def test_set_and_get_live_models(self):
        models = [{"id": "model-a"}, {"id": "model-b"}]
        set_live_models(models)
        assert get_live_models() == models

    def test_get_known_model_ids_static_only(self):
        """When no live data, returns static catalog IDs only."""
        ids = get_known_model_ids()
        assert ids == set(MODEL_CATALOG.keys())

    def test_get_known_model_ids_merged(self):
        """When live data available, merges both sets of IDs."""
        set_live_models([{"id": "live-only-model"}, {"id": "gpt-5.3-codex"}])
        ids = get_known_model_ids()
        assert "live-only-model" in ids
        assert "gpt-5.3-codex" in ids
        assert set(MODEL_CATALOG.keys()).issubset(ids)

    def test_is_model_available_returns_none_without_live_data(self):
        assert is_model_available("gpt-5.3-codex") is None

    def test_is_model_available_true(self):
        set_live_models([{"id": "gpt-5.3-codex"}])
        assert is_model_available("gpt-5.3-codex") is True

    def test_is_model_available_false(self):
        set_live_models([{"id": "gpt-5.3-codex"}])
        assert is_model_available("nonexistent-model") is False
