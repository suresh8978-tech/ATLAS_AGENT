from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import requests
import urllib3
from requests.auth import HTTPBasicAuth

from .config import Settings


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _create_token(settings: Settings) -> str:
    token_url = f"{settings.atlas_hostname.rstrip('/')}/api/v2/tokens/"
    payload = {
        "description": "Auto-generated token",
        "application": None,
        "scope": "write",
    }

    response = requests.post(
        token_url,
        auth=HTTPBasicAuth(settings.atlas_username, settings.atlas_password),
        json=payload,
        verify=False,  # set True with valid certs
        timeout=30,
    )
    response.raise_for_status()

    token = response.json().get("token")
    if not token:
        raise ValueError("Token not found in response. Check AAP server response.")
    return token


def _extract_event_row(event: dict[str, Any]) -> dict[str, Any]:
    raw_event_data = event.get("event_data") or {}
    if not isinstance(raw_event_data, dict):
        raw_event_data = {}

    stdout = event.get("stdout") or ""
    if isinstance(stdout, str) and len(stdout) > 1200:
        stdout = stdout[:1200] + "\n...<truncated>"

    return {
        "id": event.get("id"),
        "counter": event.get("counter"),
        "created": event.get("created"),
        "event": event.get("event"),
        "event_display": event.get("event_display"),
        "failed": _first_non_none(event.get("failed"), raw_event_data.get("failed")),
        "changed": _first_non_none(event.get("changed"), raw_event_data.get("changed")),
        "host_name": _first_non_none(event.get("host_name"), raw_event_data.get("host")),
        "task": _first_non_none(event.get("task"), raw_event_data.get("task")),
        "role": _first_non_none(event.get("role"), raw_event_data.get("role")),
        "task_action": raw_event_data.get("task_action"),
        "task_path": raw_event_data.get("task_path"),
        "play": _first_non_none(event.get("play"), raw_event_data.get("play")),
        "stdout": stdout,
    }


def fetch_job_events(job_id: int, settings: Settings) -> list[dict[str, Any]]:
    token = _create_token(settings)

    url = f"{settings.atlas_hostname.rstrip('/')}/api/v2/jobs/{job_id}/job_events/"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    normalized: list[dict[str, Any]] = []

    while url:
        response = requests.get(
            url,
            headers=headers,
            verify=False,  # set True with valid certs
            timeout=30,
        )
        response.raise_for_status()

        payload = response.json()
        for event in payload.get("results", []):
            if isinstance(event, dict):
                normalized.append(_extract_event_row(event))

        next_url = payload.get("next")
        if isinstance(next_url, str) and next_url:
            # AAP commonly returns relative pagination links like /api/v2/...; resolve safely.
            url = urljoin(f"{settings.atlas_hostname.rstrip('/')}/", next_url)
        else:
            url = ""

    if not normalized:
        raise ValueError(f"Job {job_id} returned no events")

    return normalized
