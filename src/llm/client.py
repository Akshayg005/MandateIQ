"""The only module that talks to Gemini. Two things live here:

  GeminiLike  -- the Protocol normalizer.py, intent.py, narrator.py depend on.
                 Tests run against an in-memory double, never the network.
  GeminiClient -- the real google-genai-SDK-backed implementation, used only
                  at the actual integration edge.

Must never be imported from src/model/ or src/policy/ -- the LLM edge
discipline mirrors src/execute/razorpay_client.py.
"""
from __future__ import annotations

import time
from typing import Callable, Protocol, runtime_checkable, TypeVar

_T = TypeVar("_T")

# The free-tier key is rate-limited to ~15 requests/minute PER MODEL,
# measured against the live API (429 on the 17th call in a burst, recovered
# within a minute -- DECISIONS.md, 2026-08-30). A golden-set run over 50
# rows WILL hit this in ordinary operation, not as an edge case -- confirmed
# by golden_check.py's first live run. 6 retries at the API's own suggested
# delay (observed 26-28s) covers a sustained run through 2-3 quota-window
# rollovers with headroom.
_MAX_RETRIES = 6
_DEFAULT_RETRY_DELAY_S = 15.0


class GeminiClientError(RuntimeError):
    """Network error, quota exhaustion (429), or a malformed response that
    isn't a clean forced function call."""


@runtime_checkable
class GeminiLike(Protocol):
    """What normalizer.py, intent.py, narrator.py depend on. A test double
    implementing this Protocol structurally satisfies isinstance(double, GeminiLike)
    without importing the real google-genai package at all."""

    def forced_call(
        self,
        *,
        model: str,
        system: str,
        user: str,
        tool_name: str,
        tool_schema: dict,
        temperature: float = 0.0,
    ) -> dict:
        """One forced-function-call round trip. Returns the function-call
        ARGS dict (not the raw API response). Must raise GeminiClientError
        on anything that isn't a clean forced call -- there is no code path
        in this Protocol's contract that returns free text; a caller never
        has to check for it."""
        ...

    def generate_text(self, *, model: str, system: str, user: str, temperature: float = 0.3) -> str:
        """Plain prose, no tool. The narrator's only need."""
        ...


class GeminiClient:
    """The real google-genai-SDK-backed implementation. Constructed lazily --
    importing this module must require neither network access nor even the
    google-genai package to be installed (mirrors RazorpayClient's own
    documented lazy-construction discipline).
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Lazily import the real SDK only on construction, not at module import."""
        import os

        api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy initialization of the actual google.genai client."""
        if self._client is None:
            import google.genai

            self._client = google.genai.Client(api_key=self._api_key)
        return self._client

    @staticmethod
    def _retry_delay_seconds(exc) -> float:
        """The API's own suggested wait, parsed from the 429's RetryInfo
        detail. Falls back to a fixed default rather than raising if the
        shape isn't what's expected -- a third-party error body's exact
        structure is not something this client controls or can assume."""
        try:
            for d in exc.details.get("error", {}).get("details", []):
                if str(d.get("@type", "")).endswith("RetryInfo"):
                    delay = str(d.get("retryDelay", ""))
                    if delay.endswith("s"):
                        return float(delay[:-1])
        except Exception:
            pass
        return _DEFAULT_RETRY_DELAY_S

    def _call_with_backoff(self, fn: Callable[[], _T], *, label: str) -> _T:
        """Retry ONLY on 429 RESOURCE_EXHAUSTED, honouring the API's own
        retry-after hint. Anything else -- a bad key, a malformed request, a
        genuine 5xx -- is not a rate limit and is not retried; silently
        retrying a non-transient error would just mask it several times
        slower instead of surfacing it."""
        from google.genai import errors

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return fn()
            except errors.ClientError as e:
                if getattr(e, "code", None) != 429:
                    raise GeminiClientError(f"{label} failed: {e}") from e
                last_exc = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(self._retry_delay_seconds(e))
            except Exception as e:
                raise GeminiClientError(f"{label} failed: {e}") from e
        raise GeminiClientError(
            f"{label} failed after {_MAX_RETRIES} attempts, still rate-limited: {last_exc}"
        ) from last_exc

    def forced_call(
        self,
        *,
        model: str,
        system: str,
        user: str,
        tool_name: str,
        tool_schema: dict,
        temperature: float = 0.0,
    ) -> dict:
        """tool_config.mode="ANY" + allowed_function_names=[tool_name] is what
        makes a free-text response structurally impossible -- this is the
        mechanism CLAUDE.md's "malformed JSON is structurally impossible"
        claim rests on, verified against the live API before adoption
        (DECISIONS.md, 2026-08-30): both adopted models returned a clean
        functionCall with empty stray text on every probed call.

        automatic_function_calling is explicitly disabled: the SDK's default
        behaviour is to EXECUTE a declared function itself if it can resolve
        one, which this client must never do -- forced_call returns args for
        the CALLER to act on, it does not invoke anything on Google's side.
        """
        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system,
            tools=[types.Tool(function_declarations=[tool_schema])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[tool_name],
                )
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self._call_with_backoff(
            lambda: client.models.generate_content(model=model, contents=user, config=config),
            label=f"forced_call to {model!r}",
        )

        parts = []
        if response.candidates:
            parts = response.candidates[0].content.parts or []
        fc = next((p.function_call for p in parts if getattr(p, "function_call", None)), None)
        if fc is None:
            # Should be unreachable under mode=ANY -- probed and confirmed
            # against the live API. Treated as a provider anomaly, not
            # silently coerced into an empty dict: a caller trusting this
            # return value to always contain the tool's required fields
            # must never receive one that doesn't.
            stray = "".join(getattr(p, "text", "") or "" for p in parts)
            raise GeminiClientError(
                f"model {model!r} returned no function call under forced tool-use "
                f"(mode=ANY, allowed={tool_name!r}) -- stray text: {stray[:200]!r}"
            )
        return dict(fc.args or {})

    def generate_text(self, *, model: str, system: str, user: str, temperature: float = 0.3) -> str:
        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(temperature=temperature, system_instruction=system)
        response = self._call_with_backoff(
            lambda: client.models.generate_content(model=model, contents=user, config=config),
            label=f"generate_text to {model!r}",
        )
        return response.text or ""
