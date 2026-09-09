"""Fail-closed contract for infra experiments that require current upstream stacks."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any


_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _release_version(value: str) -> str:
    """Ignore only a wheel's local build suffix, such as ``+cu130``."""
    return value.split("+", 1)[0]


def validate_latest_stack_manifest(
    manifest: dict[str, Any], *, now_utc: datetime | None = None
) -> dict[str, Any]:
    """Require every registered top-level component to be independently current.

    Dependency compatibility is deliberately not an exception. If the latest
    releases cannot coexist, preparation must remain blocked and record that
    incompatibility as the experiment result.
    """
    if manifest.get("mode") != "latest_each_component":
        raise ValueError("version policy mode must be latest_each_component")
    if manifest.get("status") != "pass":
        raise ValueError("version policy status must be pass before scheduling")

    required = manifest.get("required_components")
    if not isinstance(required, list) or not required or len(required) != len(set(required)):
        raise ValueError("required_components must be a non-empty unique list")

    try:
        resolved = datetime.fromisoformat(manifest["resolved_at_utc"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as exc:
        raise ValueError("resolved_at_utc must be an ISO-8601 UTC timestamp") from exc
    if resolved.tzinfo is None:
        raise ValueError("resolved_at_utc must include a UTC offset")
    now = now_utc or datetime.now(timezone.utc)
    age_hours = (now.astimezone(timezone.utc) - resolved.astimezone(timezone.utc)).total_seconds() / 3600
    max_age_hours = manifest.get("max_age_hours")
    if not isinstance(max_age_hours, (int, float)) or max_age_hours <= 0:
        raise ValueError("max_age_hours must be positive")
    if age_hours < -5 / 60:
        raise ValueError("version resolution timestamp is in the future")
    if age_hours > max_age_hours:
        raise ValueError(f"version resolution is older than {max_age_hours} hours")

    components = manifest.get("components")
    if not isinstance(components, dict):
        raise ValueError("components must be a mapping")
    for name in required:
        component = components.get(name)
        if not isinstance(component, dict):
            raise ValueError(f"{name} is missing from the version manifest")
        installed = component.get("installed")
        latest = component.get("latest_stable")
        if not isinstance(installed, str) or not isinstance(latest, str):
            raise ValueError(f"{name} must record installed and latest_stable versions")
        if _release_version(installed) != latest:
            raise ValueError(
                f"{name} is not its latest stable release: installed={installed}, latest={latest}"
            )
        sha = component.get("upstream_main_sha")
        if not isinstance(sha, str) or not _GIT_SHA.fullmatch(sha):
            raise ValueError(f"{name} must record a 40-character upstream main SHA")

    return {
        "required_components": required,
        "resolved_at_utc": resolved.astimezone(timezone.utc).isoformat(),
        "age_hours": age_hours,
    }
