from datetime import datetime, timezone

import pytest

from syncopate.infra.version_policy import (
    resolve_latest_stack_manifest,
    validate_latest_stack_manifest,
)


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


def version_policy():
    return {
        "mode": "latest_each_component",
        "max_age_hours": 6,
        "required_components": ["verl", "torch"],
        "component_sources": {
            "verl": {
                "distribution": "verl",
                "pypi_project": "verl",
                "github_repo": "verl-project/verl",
            },
            "torch": {
                "distribution": "torch",
                "pypi_project": "torch",
                "github_repo": "pytorch/pytorch",
            },
        },
    }


def fake_fetch_json(url):
    values = {
        "https://pypi.org/pypi/verl/json": {"info": {"version": "0.9.0"}},
        "https://pypi.org/pypi/torch/json": {"info": {"version": "2.14.0"}},
        "https://api.github.com/repos/verl-project/verl": {"default_branch": "main"},
        "https://api.github.com/repos/pytorch/pytorch": {"default_branch": "main"},
        "https://api.github.com/repos/verl-project/verl/commits/main": {"sha": "a" * 40},
        "https://api.github.com/repos/pytorch/pytorch/commits/main": {"sha": "b" * 40},
    }
    return values[url]


def test_resolver_queries_each_official_source_and_builds_a_verifiable_manifest():
    installed = {"verl": "0.9.0", "torch": "2.14.0+cu130"}

    manifest = resolve_latest_stack_manifest(
        version_policy(),
        installed_version=installed.__getitem__,
        fetch_json=fake_fetch_json,
        now_utc=NOW,
    )

    assert manifest["status"] == "pass"
    assert manifest["components"]["torch"] == {
        "distribution": "torch",
        "installed": "2.14.0+cu130",
        "latest_stable": "2.14.0",
        "release_source": "https://pypi.org/pypi/torch/json",
        "upstream_main_sha": "b" * 40,
        "main_source": "https://api.github.com/repos/pytorch/pytorch/commits/main",
    }
    validate_latest_stack_manifest(manifest, now_utc=NOW)


def test_resolver_fails_closed_on_old_install_or_source_error():
    installed = {"verl": "0.8.0", "torch": "2.14.0+cu130"}

    def broken_fetch(url):
        if "pytorch/pytorch/commits" in url:
            raise OSError("network unavailable")
        return fake_fetch_json(url)

    manifest = resolve_latest_stack_manifest(
        version_policy(),
        installed_version=installed.__getitem__,
        fetch_json=broken_fetch,
        now_utc=NOW,
    )

    assert manifest["status"] == "blocked"
    assert any("verl is not its latest stable release" in item for item in manifest["errors"])
    assert any("torch source lookup failed" in item for item in manifest["errors"])
    with pytest.raises(ValueError, match="status must be pass"):
        validate_latest_stack_manifest(manifest, now_utc=NOW)


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
