from datetime import datetime, timezone

import pytest

from syncopate.infra.version_policy import validate_latest_stack_manifest


NOW = datetime(2026, 9, 9, 14, 0, tzinfo=timezone.utc)


def passing_manifest():
    return {
        "mode": "latest_each_component",
        "status": "pass",
        "resolved_at_utc": "2026-09-09T13:30:00Z",
        "max_age_hours": 6,
        "required_components": ["verl", "torch", "vllm"],
        "components": {
            "verl": {
                "installed": "0.9.0",
                "latest_stable": "0.9.0",
                "upstream_main_sha": "a" * 40,
            },
            "torch": {
                "installed": "2.14.0+cu130",
                "latest_stable": "2.14.0",
                "upstream_main_sha": "b" * 40,
            },
            "vllm": {
                "installed": "0.28.0",
                "latest_stable": "0.28.0",
                "upstream_main_sha": "c" * 40,
            },
        },
    }


def test_latest_stack_manifest_accepts_each_independently_current_component():
    result = validate_latest_stack_manifest(passing_manifest(), now_utc=NOW)

    assert result["required_components"] == ["verl", "torch", "vllm"]
    assert result["age_hours"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda x: x.update(status="blocked"), "status must be pass"),
        (
            lambda x: x["components"]["vllm"].update(installed="0.24.0"),
            "vllm is not its latest stable release",
        ),
        (
            lambda x: x["components"]["verl"].update(upstream_main_sha="unknown"),
            "verl must record a 40-character upstream main SHA",
        ),
        (
            lambda x: x.update(resolved_at_utc="2026-09-09T06:00:00Z"),
            "version resolution is older than",
        ),
    ],
)
def test_latest_stack_manifest_rejects_stale_anchored_or_unverifiable_stacks(mutate, message):
    value = passing_manifest()
    mutate(value)

    with pytest.raises(ValueError, match=message):
        validate_latest_stack_manifest(value, now_utc=NOW)
