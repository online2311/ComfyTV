"""OpenAI-compatible chat client for the script writer.

The request matches the Local LLM provider: POST ``{base}/chat/completions``
with a Bearer token when one is configured. Keys are read here and are not
returned in the public config.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .settings import get_setting

_PACKAGE_PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "script_writer"
_DEFAULT_MODEL = "deepseek-v4-pro"
_DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com/v1"
_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)


class ScriptError(ValueError):
    def __init__(self, message: str, *, status: int = 400):
        super().__init__(message)
        self.http_status = status


def load_prompt(name: str, root=None) -> str:
    if root is not None:
        override = Path(root) / "script" / "prompts" / f"{name}.md"
        if override.is_file():
            return override.read_text(encoding="utf-8")
    path = _PACKAGE_PROMPTS / f"{name}.md"
    if not path.is_file():
        raise ScriptError(f"prompt file missing: {name}")
    return path.read_text(encoding="utf-8")


def load_sections(root=None) -> list[str]:
    if root is not None:
        override = Path(root) / "script" / "prompts" / "outline_sections.txt"
        if override.is_file():
            text = override.read_text(encoding="utf-8")
            return [line.strip() for line in text.splitlines() if line.strip()]
    path = _PACKAGE_PROMPTS / "outline_sections.txt"
    if not path.is_file():
        raise ScriptError("outline section file missing")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_key_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^[A-Za-z0-9_]+=", stripped):
            return stripped.split("=", 1)[1].strip().strip("'\"")
        return stripped
    return ""


def _key_from_env(provider: str) -> str:
    if provider == "local":
        named = os.environ.get("COMFYTV_LOCAL_LLM_API_KEY", "")
        if named.strip():
            return named.strip()
        return _read_key_file(Path.home() / ".config" / "llm-keys" / "local")
    for name in ("COMFYTV_SCRIPT_LLM_API_KEY", "DEEPSEEK_API_KEY"):
        value = os.environ.get(name, "")
        if value.strip():
            return value.strip()
    file_env = os.environ.get("COMFYTV_SCRIPT_LLM_KEY_FILE", "")
    if file_env.strip():
        found = _read_key_file(Path(file_env.strip()))
        if found:
            return found
    for relative in ("deepseek", "deepseek.key"):
        found = _read_key_file(Path.home() / ".config" / "llm-keys" / relative)
        if found:
            return found
    return ""


def _clean_base(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urllib.parse.urlsplit(raw)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    if parts.username or parts.password:
        parts = parts._replace(netloc=host)
    cleaned = urllib.parse.urlunsplit(parts._replace(query="", fragment=""))
    return cleaned.rstrip("/")


def _config_llm(root) -> dict:
    path = Path(root) / "script" / "config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    llm = data.get("llm") if isinstance(data, dict) else None
    return llm if isinstance(llm, dict) else {}


def _local_base_from_settings() -> str:
    try:
        value = get_setting("bot-local-llm-url")
    except Exception:
        return ""
    return str(value or "").strip()


def resolve_llm(root, request: dict | None = None) -> dict:
    """Resolve provider, model, and base URL. ``public`` has no API key."""
    request = request if isinstance(request, dict) else {}
    stored = _config_llm(root)
    provider = str(request.get("provider") or stored.get("provider") or "deepseek").strip().lower()
    if provider not in ("deepseek", "local"):
        raise ScriptError("llm provider must be deepseek or local")
    if provider == "local":
        model = str(
            request.get("model") or stored.get("local_model") or stored.get("model")
            or os.environ.get("COMFYTV_LOCAL_LLM_MODEL") or "local"
        ).strip()
        base = (
            request.get("base_url") or stored.get("local_base_url")
            or os.environ.get("COMFYTV_LOCAL_LLM_BASE_URL")
            or os.environ.get("COMFYTV_SCRIPT_LLM_BASE_URL")
            or _local_base_from_settings()
        )
    else:
        model = str(
            request.get("model") or stored.get("model")
            or os.environ.get("COMFYTV_SCRIPT_LLM_MODEL") or _DEFAULT_MODEL
        ).strip()
        base = (
            request.get("base_url") or stored.get("base_url")
            or os.environ.get("COMFYTV_SCRIPT_LLM_BASE_URL")
            or os.environ.get("DEEPSEEK_BASE_URL")
            or _DEFAULT_DEEPSEEK_BASE
        )
    base_url = _clean_base(str(base or ""))
    api_key = _key_from_env(provider)
    public = {"provider": provider, "model": model, "base_url": base_url}
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "public": public,
    }


def redact(text: str, secret: str = "") -> str:
    cleaned = text or ""
    if secret:
        cleaned = cleaned.replace(secret, "[redacted]")
    return _BEARER_RE.sub("Bearer [redacted]", cleaned)


class OpenAICompatChat:
    """Small synchronous client. Same chat-completions body as bot/local_llm.py."""

    def __init__(self, *, base_url: str, model: str, api_key: str = "",
                 timeout: float = 120, require_key: bool = False):
        self.base_url = _clean_base(base_url)
        self.model = model
        self._api_key = api_key or ""
        self.timeout = timeout
        self.require_key = require_key
        self.last_usage = None

    def complete(self, *, system: str, user: str, model: str | None = None) -> str:
        if self.require_key and not self._api_key:
            raise ScriptError("API key is not configured", status=400)
        if not self.base_url:
            raise ScriptError("LLM base URL is not configured", status=400)
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = redact(exc.read().decode("utf-8", errors="replace"), self._api_key)
            detail = detail[:300]
            raise ScriptError(
                f"LLM request failed with HTTP {exc.code}: {detail}",
                status=502,
            ) from None
        except urllib.error.URLError as exc:
            reason = redact(str(exc.reason), self._api_key)
            raise ScriptError(f"LLM request failed: {reason}", status=502) from None
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ScriptError("LLM response was not JSON", status=502) from exc
        usage = data.get("usage") if isinstance(data, dict) else None
        self.last_usage = usage if isinstance(usage, dict) else None
        return _message_text(data)


def _message_text(data: dict) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    message = (choices or [{}])[0].get("message") if isinstance(choices, list) else None
    if not isinstance(message, dict):
        raise ScriptError("LLM response has no message", status=502)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    raise ScriptError("LLM response content was empty", status=502)


def build_client(root, request: dict | None = None) -> OpenAICompatChat:
    resolved = resolve_llm(root, request)
    return OpenAICompatChat(
        base_url=resolved["base_url"],
        model=resolved["model"],
        api_key=resolved["api_key"],
        require_key=resolved["provider"] != "local",
    )
