from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

from .aap_events import fetch_job_events
from .config import Settings, load_settings


def _build_agent(settings: Settings, model_name: str):
    @tool
    def get_job_events(job_id: int) -> str:
        """Fetch all Ansible Automation Platform job events for a job id."""
        events = fetch_job_events(job_id=job_id, settings=settings)
        payload: dict[str, Any] = {
            "job_id": job_id,
            "event_count": len(events),
            "events": events,
        }
        return json.dumps(payload, ensure_ascii=True)

    llm = ChatAnthropic(
        model=model_name,
        temperature=0,
        max_tokens=1800,
    )

    system_prompt = (
        "You are an expert Ansible execution diagnostics analyst. "
        "Always call get_job_events(job_id) first before producing analysis. "
        "Then produce a concise, practical report with these sections:\n"
        "1. Executive Summary\n"
        "2. Timeline Highlights\n"
        "3. Failure Signals\n"
        "4. Probable Root Cause\n"
        "5. Recommended Next Checks"
    )

    return create_agent(model=llm, tools=[get_job_events], system_prompt=system_prompt)


def _extract_final_text(message: BaseMessage | Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def analyze_job(job_id: int | str, settings: Settings | None = None) -> str:
    runtime_settings = settings or load_settings()

    normalized_job_id = int(job_id)
    user_prompt = (
        "Analyze Ansible Automation Platform execution events for job id "
        f"{normalized_job_id}. Focus on failures, error patterns, and likely "
        "causes based on the event stream."
    )

    model_candidates = [runtime_settings.llm_name]
    if "/" in runtime_settings.llm_name:
        model_candidates.append(runtime_settings.llm_name.split("/", maxsplit=1)[1])

    last_error: Exception | None = None
    result = None
    for model_name in dict.fromkeys(model_candidates):
        try:
            agent = _build_agent(runtime_settings, model_name=model_name)
            result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
            break
        except Exception as exc:  # noqa: BLE001 - surface provider errors to caller.
            last_error = exc
            if "Invalid model name" in str(exc) and model_name != model_candidates[-1]:
                continue
            raise

    if result is None:
        if last_error:
            raise last_error
        raise RuntimeError("Agent invocation failed without an explicit error")

    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("Agent produced no response")

    return _extract_final_text(messages[-1]).strip()
