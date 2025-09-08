import os
import sys
import types
import asyncio
import pytest
from fastapi.testclient import TestClient

from main import (
    app,
    metrics,
    metrics_lock,
    rate_limiter,
    ChatMessage,
    ChatRequest,
    build_prompt,
    get_client,
    generate_with_retries,
    llm_router_chat,
    _split_system_and_messages,
)


client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Reset global state and env between tests for isolation."""
    # Reset metrics
    with metrics_lock:
        metrics["requests_total"] = 0
        metrics["requests_by_status"].clear()
        metrics["response_time_seconds"].clear()
        metrics["rate_limited_total"] = 0
        metrics["generation_errors_total"] = 0
    # Reset rate limiter
    rate_limiter.clear()
    # Default sane envs
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10")
    monkeypatch.setenv("RATE_LIMIT_WINDOW", "60")
    monkeypatch.setenv("REQUEST_SIZE_LIMIT_BYTES", "1048576")
    monkeypatch.setenv("INPUT_CHAR_LIMIT", "8000")
    monkeypatch.setenv("HF_MAX_RETRIES", "1")
    monkeypatch.setenv("HF_RETRY_DELAY", "0.0")
    yield


def _valid_chat_body():
    return {
        "messages": [
            {"role": "system", "content": "Behave"},
            {"role": "user", "content": "Hello"},
        ],
        "max_new_tokens": 10,
        "temperature": 0.1,
        "top_p": 0.9,
    }


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"


def test_health_without_token_and_with_features(monkeypatch):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "degraded"
    assert data["features"]["rate_limiting"] is True
    assert data["features"]["authentication"] is False

    # With token and API_KEY
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "t")
    monkeypatch.setenv("API_KEY", "s")
    r2 = client.get("/health")
    data2 = r2.json()
    assert data2["status"] == "healthy"
    assert data2["features"]["authentication"] is True


def test_readiness_token_missing_and_present(monkeypatch):
    r = client.get("/readiness")
    assert r.status_code == 503

    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "t")
    r2 = client.get("/readiness")
    assert r2.status_code == 200
    assert r2.json()["status"] == "ready"


def test_request_size_limit_middleware_413(monkeypatch):
    # Force very small limit and set large Content-Length header
    monkeypatch.setenv("REQUEST_SIZE_LIMIT_BYTES", "1")
    headers = {"Content-Length": "9999"}
    r = client.post("/v1/chat", headers=headers, json=_valid_chat_body())
    assert r.status_code == 413


def test_metrics_empty_then_nonempty():
    # Initially empty
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["http_request_duration_seconds"]["avg"] == 0
    assert data["http_request_duration_seconds"]["p50"] == 0
    assert data["http_request_duration_seconds"]["p95"] == 0
    assert data["http_request_duration_seconds"]["p99"] == 0

    # Produce some traffic
    for _ in range(3):
        client.get("/health")
    r2 = client.get("/metrics")
    data2 = r2.json()
    assert data2["http_requests_total"] >= 4  # includes metrics calls too
    for k in ["avg", "p50", "p95", "p99"]:
        assert data2["http_request_duration_seconds"][k] >= 0


def test_default_system_prompt_used(monkeypatch):
    # Ensure when no system message is provided, router still works and adds default
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "t")

    class DummyHF:
        def text_generation(self, prompt, **kwargs):  # noqa: ARG002
            assert "<|system|>" in prompt
            return "OK"

    monkeypatch.setattr("main.get_client", lambda model_id=None: DummyHF())
    out = llm_router_chat([ChatMessage(role="user", content="hi")], None, 0.1, 0.9, 10)
    assert out == "OK"


def test_chat_logging_prompt_try_except(monkeypatch):
    # Mock router and also force build_prompt to raise once to hit except branch
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setattr("main.llm_router_chat", lambda *a, **k: "X")

    calls = {"n": 0}

    def flaky_build(messages):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return "<|system|>\nS\n\n<|assistant|>\n"

    monkeypatch.setattr("main.build_prompt", flaky_build)
    # First call will hit except path, second will log normally
    r1 = client.post("/v1/chat", json=_valid_chat_body(), headers={"Authorization": "Bearer secret"})
    r2 = client.post("/v1/chat", json=_valid_chat_body(), headers={"Authorization": "Bearer secret"})
    assert r1.status_code == 200 and r2.status_code == 200


def test_build_prompt_variants(monkeypatch):
    msgs = [
        ChatMessage(role="system", content="You are test."),
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi"),
    ]
    p = build_prompt(msgs)
    assert p.startswith("<|system|>") and p.endswith("<|assistant|>\n")

    # Invalid INPUT_CHAR_LIMIT -> default used
    monkeypatch.setenv("INPUT_CHAR_LIMIT", "not-an-int")
    p2 = build_prompt([ChatMessage(role="user", content="A")])
    assert "<|user|>" in p2

    # Trimming behavior with very small limit keeps the suffix (up to the limit)
    monkeypatch.setenv("INPUT_CHAR_LIMIT", "5")
    p3 = build_prompt([ChatMessage(role="user", content="X" * 200)])
    # When limit < len("<|assistant|>\n"), we keep the last `limit` chars of the suffix
    assert p3.endswith("<|assistant|>\n"[-5:])
    assert len(p3) == 5

    # Zero limit -> no trimming
    monkeypatch.setenv("INPUT_CHAR_LIMIT", "0")
    p4 = build_prompt([ChatMessage(role="user", content="short")])
    assert p4.endswith("<|assistant|>\n")


def test_models_validation_errors():
    with pytest.raises(ValueError):
        ChatMessage(role="user", content="   ")

    with pytest.raises(ValueError):
        ChatRequest(messages=[ChatMessage(role="assistant", content="hi")])


def test_get_client_token_missing_and_present(monkeypatch):
    with pytest.raises(RuntimeError):
        get_client()

    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "t")
    c = get_client()
    from huggingface_hub import InferenceClient

    assert isinstance(c, InferenceClient)


def test_generate_with_retries_success_and_failure(monkeypatch):
    async def body():
        # Speed up retries
        async def fast_sleep(_):
            fut = asyncio.get_running_loop().create_future()
            fut.set_result(None)
            await fut

        monkeypatch.setattr(asyncio, "sleep", fast_sleep)

        class Dummy:
            def __init__(self):
                self.calls = 0

            def text_generation(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls < 2:
                    raise RuntimeError("fail once")
                return "OK"

        # Success on retry
        monkeypatch.setenv("HF_MAX_RETRIES", "2")
        d = Dummy()
        out = await generate_with_retries(d, "prompt")
        assert out == "OK"

        # Always failing -> increments generation_errors_total
        class DummyFail:
            def text_generation(self, *_args, **_kwargs):
                raise RuntimeError("always fail")

        before = metrics["generation_errors_total"]
        with pytest.raises(RuntimeError):
            await generate_with_retries(DummyFail(), "p")
        assert metrics["generation_errors_total"] == before + 1

    asyncio.run(body())


def test_auth_dependency_via_chat(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    # Missing token -> 401
    r = client.post("/v1/chat", json=_valid_chat_body())
    assert r.status_code == 401

    # Wrong token -> 401
    r2 = client.post("/v1/chat", json=_valid_chat_body(), headers={"Authorization": "Bearer nope"})
    assert r2.status_code == 401


def test_chat_success_with_auth_and_no_rate_limit(monkeypatch):
    # Auth ok and router mocked
    monkeypatch.setenv("API_KEY", "secret")

    def fake_router(messages, model, temperature, top_p, max_new_tokens):  # noqa: ARG001
        return "Hello there"

    monkeypatch.setattr("main.llm_router_chat", fake_router)

    r = client.post(
        "/v1/chat",
        json=_valid_chat_body(),
        headers={"Authorization": "Bearer secret"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"] == "Hello there"


def test_rate_limit_enabled_and_disabled(monkeypatch):
    # Disabled
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "0")
    monkeypatch.setattr("main.llm_router_chat", lambda *a, **k: "ok")
    for _ in range(3):
        r = client.post("/v1/chat", json=_valid_chat_body())
        assert r.status_code in (200, 401)

    # Enabled and enforce
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setattr("main.llm_router_chat", lambda *a, **k: "ok")

    # First request passes
    r1 = client.post("/v1/chat", json=_valid_chat_body())
    assert r1.status_code in (200, 401, 503, 429)
    # Second should be rate-limited
    r2 = client.post("/v1/chat", json=_valid_chat_body())
    assert r2.status_code == 429


def test_metrics_by_status_captures_various_codes(monkeypatch):
    # 200
    client.get("/")
    # 503 from readiness without token
    client.get("/readiness")
    # 413 from request size
    monkeypatch.setenv("REQUEST_SIZE_LIMIT_BYTES", "1")
    client.post("/v1/chat", headers={"Content-Length": "9999"}, json=_valid_chat_body())

    data = client.get("/metrics").json()
    codes = set(map(int, data["http_requests_by_status"].keys()))
    assert 200 in codes
    assert 503 in codes
    assert 413 in codes


def test_emit_request_log_error_path_and_502(monkeypatch):
    # Force auth required and present, but router fails to trigger 502 branch
    monkeypatch.setenv("API_KEY", "secret")

    def boom_router(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr("main.llm_router_chat", boom_router)

    r = client.post(
        "/v1/chat",
        json=_valid_chat_body(),
        headers={"Authorization": "Bearer secret"},
    )
    # Should hit generation failure path -> 502
    assert r.status_code == 502


def test_auth_success_and_chat_runs(monkeypatch):
    # Ensure auth path returns credentials and chat executes successfully
    monkeypatch.setenv("API_KEY", "secret")

    monkeypatch.setattr("main.llm_router_chat", lambda *a, **k: "hi")

    r = client.post(
        "/v1/chat",
        json=_valid_chat_body(),
        headers={"Authorization": "Bearer secret"},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"


def test_chat_without_api_key_dependency_skips(monkeypatch):
    """Covers verify_api_key early return when API_KEY is not set (CI missing line)."""
    # Ensure no API key is configured
    monkeypatch.delenv("API_KEY", raising=False)
    # Make sure rate limit doesn't block and router returns quickly
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10")
    monkeypatch.setattr("main.llm_router_chat", lambda *a, **k: "ok-no-auth")

    r = client.post("/v1/chat", json=_valid_chat_body())
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"] == "ok-no-auth"


def test_llm_router_hf_fallback_and_ollama_openai_paths(monkeypatch):
    # Force provider to an unsupported one to use fallback logic
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    # Ensure HF token is present for fallback client
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "t")

    # Patch get_client to a dummy with text_generation to cover HF path
    class DummyHF:
        def text_generation(self, prompt, **kwargs):  # noqa: ARG002
            # Ensure prompt was constructed from messages and not empty
            assert isinstance(prompt, str) and len(prompt) > 0
            return "HF-OK"

    monkeypatch.setattr("main.get_client", lambda model_id=None: DummyHF())

    out = llm_router_chat(
        [ChatMessage(role="user", content="hi")],
        model_override=None,
        temperature=0.1,
        top_p=0.9,
        max_new_tokens=10,
    )
    assert out == "HF-OK"

    # Now cover the ollama OpenAI-style path by faking openai client
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    class DummyOpenAIChatChoices:
        def __init__(self):
            self.message = types.SimpleNamespace(content="OLLAMA-OK")

    class DummyOpenAIChat:
        def __init__(self):
            self.choices = [DummyOpenAIChatChoices()]

    class DummyOpenAIClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature, top_p, max_tokens):  # noqa: ARG002
                    # Ensure we get a system + user message
                    assert isinstance(messages, list) and len(messages) >= 2
                    return DummyOpenAIChat()

        def __init__(self, api_key=None, base_url=None):  # noqa: ARG002
            pass

    # Inject dummy openai.OpenAI into module namespace
    openai_mod = types.SimpleNamespace(OpenAI=DummyOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", openai_mod)

    out2 = llm_router_chat(
        [ChatMessage(role="user", content="hello")],
        model_override="qwen2.5:7b-instruct",
        temperature=0.1,
        top_p=0.9,
        max_new_tokens=10,
    )
    assert out2 == "OLLAMA-OK"


def test_llm_router_anthropic_branch_failure_then_hf(monkeypatch):
    # Set provider to anthropic, but raise inside anthropic client, then fall back to HF
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "t")

    class BoomAnthropic:
        class Anthropic:
            def __init__(self, api_key=None):  # noqa: ARG002
                pass

            class messages:
                @staticmethod
                def create(**kwargs):  # noqa: ARG002
                    raise RuntimeError("fail anthropic")

    monkeypatch.setitem(sys.modules, "anthropic", BoomAnthropic)

    class DummyHF:
        def text_generation(self, prompt, **kwargs):  # noqa: ARG002
            return "HF-OK2"

    monkeypatch.setattr("main.get_client", lambda model_id=None: DummyHF())

    out = llm_router_chat(
        [ChatMessage(role="user", content="hi")],
        model_override=None,
        temperature=0.1,
        top_p=0.9,
        max_new_tokens=10,
    )
    assert out == "HF-OK2"


def test_llm_router_explicit_providers_openai_openrouter_groq(monkeypatch):
    # Dummy OpenAI-compatible client that always returns a fixed message
    class DummyOpenAIChatChoices:
        def __init__(self, text):
            self.message = types.SimpleNamespace(content=text)

    class DummyOpenAIChat:
        def __init__(self, text):
            self.choices = [DummyOpenAIChatChoices(text)]

    class DummyOpenAIClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature, top_p, max_tokens):  # noqa: ARG002
                    assert isinstance(messages, list) and len(messages) >= 2
                    return DummyOpenAIChat("OPENAI-OK")

        def __init__(self, api_key=None, base_url=None):  # noqa: ARG002
            pass

    # Inject dummy module
    openai_mod = types.SimpleNamespace(OpenAI=DummyOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", openai_mod)

    # openai branch
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    out_openai = llm_router_chat([ChatMessage(role="user", content="hi")], None, 0.1, 0.9, 10)
    assert out_openai == "OPENAI-OK"

    # openrouter branch
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    out_or = llm_router_chat([ChatMessage(role="user", content="hi")], None, 0.1, 0.9, 10)
    assert out_or == "OPENAI-OK"

    # groq branch
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    out_groq = llm_router_chat([ChatMessage(role="user", content="hi")], None, 0.1, 0.9, 10)
    assert out_groq == "OPENAI-OK"


def test_llm_router_all_providers_fail_raises(monkeypatch):
    # Force all providers to fail and HF to raise
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    # Ensure HF not available
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)

    # OpenAI dummy that raises on create (both explicit and opportunistic)
    class DummyOpenAIClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):  # noqa: ARG002
                    raise RuntimeError("nope")

        def __init__(self, api_key=None, base_url=None):  # noqa: ARG002
            pass

    openai_mod = types.SimpleNamespace(OpenAI=DummyOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", openai_mod)

    # Anthropic client that raises
    class BoomAnthropic:
        class Anthropic:
            class messages:
                @staticmethod
                def create(**kwargs):  # noqa: ARG002
                    raise RuntimeError("nope")  # pragma: no cover

            def __init__(self, api_key=None):  # noqa: ARG002
                pass  # pragma: no cover

    monkeypatch.setitem(sys.modules, "anthropic", BoomAnthropic)

    # HF fallback raises inside get_client call path
    def raise_get_client(model_id=None):  # noqa: ARG002
        raise RuntimeError("no token")

    monkeypatch.setattr("main.get_client", raise_get_client)

    with pytest.raises(RuntimeError):
        llm_router_chat([ChatMessage(role="user", content="x")], None, 0.1, 0.9, 10)


def test_split_system_and_messages_extracts_system():
    system, msgs = _split_system_and_messages(
        [
            ChatMessage(role="system", content="SYS"),
            ChatMessage(role="user", content="U1"),
            ChatMessage(role="assistant", content="A1"),
        ]
    )
    assert system == "SYS"
    assert msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"


def test_llm_router_anthropic_success(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    class DummyContent:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class DummyAnthropic:
        class messages:
            @staticmethod
            def create(**kwargs):  # noqa: ARG002
                # Mix object and dict forms to hit both branches
                return types.SimpleNamespace(
                    content=[
                        DummyContent("ANTH-OK"),
                        {"type": "text", "text": "MORE"},
                    ]
                )

        def __init__(self, api_key=None):  # noqa: ARG002
            pass

    anth = types.SimpleNamespace(Anthropic=DummyAnthropic)
    monkeypatch.setitem(sys.modules, "anthropic", anth)

    # Include a non-user/assistant role using dict to bypass model validation
    # Monkeypatch split to inject a bad role into anthropic conversion path
    from main import _split_system_and_messages as real_split

    def fake_split(messages):  # noqa: ARG001
        sys_prompt, msgs = real_split([ChatMessage(role="user", content="hi")])
        # Prepend a message with role 'tool' to trigger remap
        return sys_prompt, [{"role": "tool", "content": "ignored"}] + msgs

    monkeypatch.setattr("main._split_system_and_messages", fake_split)
    out = llm_router_chat([ChatMessage(role="user", content="x")], None, 0.1, 0.9, 10)
    assert out.startswith("ANTH-OK")


def test_llm_router_opportunistic_ollama_success(monkeypatch):
    # Provider unknown triggers opportunistic loop; first is ollama base_url
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)

    class DummyOpenAIChatChoices:
        def __init__(self):
            self.message = types.SimpleNamespace(content="OP-OK")

    class DummyOpenAIChat:
        def __init__(self):
            self.choices = [DummyOpenAIChatChoices()]

    class DummyOpenAIClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):  # noqa: ARG002
                    return DummyOpenAIChat()

        def __init__(self, api_key=None, base_url=None):  # noqa: ARG002
            pass

    openai_mod = types.SimpleNamespace(OpenAI=DummyOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", openai_mod)

    out = llm_router_chat([ChatMessage(role="user", content="go")], None, 0.1, 0.9, 10)
    assert out == "OP-OK"


def test_request_size_limit_invalid_env_parsing(monkeypatch):
    # Invalid env should fall back to default and not crash
    monkeypatch.setenv("REQUEST_SIZE_LIMIT_BYTES", "not-an-int")
    r = client.get("/")
    assert r.status_code == 200


def test_metrics_trim_response_times_to_last_1000():
    # Prefill metrics with more than 1000 entries to hit trimming branch
    from main import metrics, metrics_lock

    with metrics_lock:
        metrics["response_time_seconds"] = [0.001] * 1005

    # Trigger any request to pass through logging_middleware and update metrics
    r = client.get("/")
    assert r.status_code == 200

    with metrics_lock:
        assert len(metrics["response_time_seconds"]) == 1000


def test_build_prompt_extremely_small_char_limit():
    """Test edge case where char_limit is smaller than assistant_suffix."""
    from main import build_prompt, ChatMessage

    # Set an extremely small character limit
    os.environ["INPUT_CHAR_LIMIT"] = "5"  # Much smaller than "<|assistant|>\n" (15 chars)

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="This is a very long message that will exceed the char limit"),
        ChatMessage(role="assistant", content="Previous response"),
        ChatMessage(role="user", content="Another long message"),
    ]

    prompt = build_prompt(messages)

    # When keep_chars is 0 or negative, it should use assistant_suffix[-char_limit:]
    # This covers the line: prompt = prefix[-keep_chars:] + assistant_suffix if keep_chars else assistant_suffix[-char_limit:]
    assert len(prompt) <= 5
    # The assistant_suffix is "<|assistant|>\n" (15 chars), so [-5:] would be "nt|>\n"
    assert prompt == "nt|>\n"

    # Clean up
    os.environ["INPUT_CHAR_LIMIT"] = "8000"
