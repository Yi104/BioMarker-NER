from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import request


@dataclass
class LLMOptions:
    provider: str = "none"  # none | ollama | openai | anthropic | gemini
    model: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512


def _build_prompt(question: str, evidence_bundle: Dict[str, Any]) -> str:
    # Keep prompt deterministic and citation-grounded.
    evidence_json = json.dumps(evidence_bundle, ensure_ascii=False)
    return (
        "You are a biomedical evidence summarizer.\n"
        "Rules:\n"
        "1) Use only the provided evidence.\n"
        "2) If evidence is insufficient, say 'insufficient evidence'.\n"
        "3) Attach PMID references for every claim.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence JSON:\n{evidence_json}\n"
    )


def _run_none(question: str, evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    # No-model mode: return structured evidence only.
    return {
        "provider": "none",
        "mode": "evidence_only",
        "question": question,
        "summary": "",
        "evidence": evidence_bundle,
    }


def _run_ollama(prompt: str, options: LLMOptions) -> Dict[str, Any]:
    base_url = options.base_url or "http://localhost:11434"
    model = options.model or "llama3.1:8b"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": options.temperature},
    }
    req = request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return {
        "provider": "ollama",
        "model": model,
        "summary": body.get("response", ""),
    }


def summarize_with_provider(
    question: str,
    evidence_bundle: Dict[str, Any],
    options: LLMOptions,
) -> Dict[str, Any]:
    provider = (options.provider or "none").strip().lower()
    prompt = _build_prompt(question, evidence_bundle)

    if provider == "none":
        return _run_none(question, evidence_bundle)
    if provider == "ollama":
        return _run_ollama(prompt, options)
    if provider in {"openai", "anthropic", "gemini"}:
        # BYO-key providers are intentionally not hardwired yet.
        # This keeps costs and dependency choices under user control.
        return {
            "provider": provider,
            "error": "provider_not_wired",
            "message": "Use BYO API client integration for this provider.",
        }
    return {
        "provider": provider,
        "error": "unknown_provider",
        "message": "Supported providers: none, ollama, openai, anthropic, gemini",
    }


def list_supported_providers() -> List[str]:
    return ["none", "ollama", "openai", "anthropic", "gemini"]
