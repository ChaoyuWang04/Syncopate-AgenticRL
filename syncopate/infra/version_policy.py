"""Fail-closed contract for infra experiments that require current upstream stacks."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import re
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen


_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _release_version(value: str) -> str:
    """Ignore only a wheel's local build suffix, such as ``+cu130``."""
    return value.split("+", 1)[0]


def _fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "syncopate-version-gate/1"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def resolve_latest_stack_manifest(
    policy: dict[str, Any],
    *,
    installed_version: Callable[[str], str] = importlib.metadata.version,
    fetch_json: Callable[[str], dict[str, Any]] = _fetch_json,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Query official release and repository APIs for every required component."""
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    required = policy.get("required_components")
    sources = policy.get("component_sources")
    manifest: dict[str, Any] = {
        "mode": policy.get("mode"),
        "status": "blocked",
        "resolved_at_utc": now.isoformat().replace("+00:00", "Z"),
        "max_age_hours": policy.get("max_age_hours"),
        "required_components": required,
        "components": {},
        "errors": [],
    }
    if not isinstance(required, list) or not required:
        manifest["errors"].append("required_components must be a non-empty list")
        return manifest
    if not isinstance(sources, dict):
        manifest["errors"].append("component_sources must be a mapping")
        return manifest

    for name in required:
        source = sources.get(name)
        if not isinstance(source, dict):
            manifest["errors"].append(f"{name} has no official source registration")
            continue
        distribution = source.get("distribution")
        pypi_project = source.get("pypi_project")
        github_repo = source.get("github_repo")
        if not all(isinstance(value, str) and value for value in (distribution, pypi_project, github_repo)):
            manifest["errors"].append(f"{name} source registration is incomplete")
            continue
        release_url = f"https://pypi.org/pypi/{quote(pypi_project, safe='')}/json"
        repo_url = f"https://api.github.com/repos/{github_repo}"
        try:
            installed = installed_version(distribution)
            latest = fetch_json(release_url)["info"]["version"]
            default_branch = source.get("branch") or fetch_json(repo_url)["default_branch"]
            main_url = f"{repo_url}/commits/{quote(default_branch, safe='')}"
            main_sha = fetch_json(main_url)["sha"]
        except Exception as exc:
            manifest["errors"].append(f"{name} source lookup failed: {type(exc).__name__}: {exc}")
            continue
        component = {
            "distribution": distribution,
            "installed": installed,
            "latest_stable": latest,
            "release_source": release_url,
            "upstream_main_sha": main_sha,
            "main_source": main_url,
        }
        manifest["components"][name] = component
        if _release_version(installed) != latest:
            manifest["errors"].append(
                f"{name} is not its latest stable release: installed={installed}, latest={latest}"
            )
        if not isinstance(main_sha, str) or not _GIT_SHA.fullmatch(main_sha):
            manifest["errors"].append(f"{name} upstream main SHA is invalid")

    if not manifest["errors"] and len(manifest["components"]) == len(required):
        manifest["status"] = "pass"
    return manifest


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
