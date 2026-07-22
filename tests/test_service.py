import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest

from app import service


def workflow_data() -> dict[str, object]:
    return {
        "10": {
            "inputs": {"text": "原始指令", "clip": ["1", 0]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "API Instruction"},
        },
        "20": {
            "inputs": {"images": ["2", 0]},
            "_meta": {"title": "API Output"},
        },
    }


def write_settings(
    fs, root: Path, token: str = "secret", url: str = "http://127.0.0.1:8188"
) -> None:
    config = root / "config"
    fs.create_file(config / "api_token.txt", contents=token)
    fs.create_file(config / "comfy_url.txt", contents=url)
    fs.create_file(
        config / "llm_url.txt", contents="http://llm.local/v1/chat/completions"
    )
    fs.create_file(config / "llm_api_key.txt", contents="llm-secret")
    fs.create_file(config / "llm_model.txt", contents="prompt-model")


@pytest.mark.parametrize(
    "missing",
    [
        "api_token.txt",
        "comfy_url.txt",
        "llm_url.txt",
        "llm_api_key.txt",
        "llm_model.txt",
    ],
)
def test_missing_configuration_stops_startup(fs, missing: str) -> None:
    root = Path("/api")
    write_settings(fs, root)
    fs.remove_object(str(root / "config" / missing))

    with pytest.raises(RuntimeError, match=missing):
        service.load_settings(root)


@pytest.mark.parametrize(
    ("token", "url"),
    [
        ("   ", "http://127.0.0.1:8188"),
        ("secret", "   "),
        ("line1\nline2", "http://127.0.0.1:8188"),
        ("secret", "http://127.0.0.1:8188\n/path"),
        ("secret", "ftp://127.0.0.1"),
        ("secret", "http://user:pass@127.0.0.1"),
        ("secret", "http://127.0.0.1/path"),
        ("secret", "http://127.0.0.1?query=1"),
        ("secret", "http://127.0.0.1#fragment"),
    ],
)
def test_invalid_configuration_stops_startup(fs, token: str, url: str) -> None:
    root = Path("/api")
    write_settings(fs, root, token, url)

    with pytest.raises(RuntimeError):
        service.load_settings(root)


def test_configuration_paths_ignore_current_directory(fs, monkeypatch) -> None:
    root = Path("/project")
    write_settings(fs, root, "\ufeff secret ", "\ufeffhttp://127.0.0.1:8188/")
    fs.create_dir("/elsewhere")
    monkeypatch.setattr(service, "PROJECT_ROOT", root)
    monkeypatch.chdir("/elsewhere")

    settings = service.load_settings()

    assert settings.token == "secret"
    assert settings.comfy_url == "http://127.0.0.1:8188"
    assert settings.llm_url == "http://llm.local/v1/chat/completions"
    assert settings.llm_api_key == "llm-secret"
    assert settings.llm_model == "prompt-model"


def test_system_prompt_is_loaded_from_project_root(fs, monkeypatch) -> None:
    root = Path("/project")
    fs.create_file(root / "prompt/system.md", contents="\ufeff 系统指令\n第二行 ")
    monkeypatch.setattr(service, "PROJECT_ROOT", root)

    assert service.load_system_prompt() == "系统指令\n第二行"


def test_llm_preprocessing_uses_openai_chat_completions_format() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  a rainy neon street  "}}]},
        )

    async def run() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await service.preprocess_instruction(
                client,
                service.Settings(
                    "api-secret",
                    "http://comfy.local",
                    "http://llm.local/v1/chat/completions",
                    "llm-secret",
                    "prompt-model",
                ),
                "系统指令",
                "生成雨夜街道",
            )

    result = asyncio.run(run())

    assert (
        result
        == """a rainy neon street

safe
(mature:-1), (aged down:1)
(simple background:-1.25)
(shiny skin:-1)
(flat color, anime coloring:2)
rim light, light particles, cinematic lighting
depth of field, strong perspective, blurry background"""
    )
    assert captured == {
        "authorization": "Bearer llm-secret",
        "payload": {
            "model": "prompt-model",
            "messages": [
                {"role": "system", "content": "系统指令"},
                {"role": "user", "content": "生成雨夜街道"},
            ],
        },
    }


def llm_settings() -> service.Settings:
    return service.Settings(
        "api-secret",
        "http://comfy.local",
        "http://llm.local/v1/chat/completions",
        "llm-secret",
        "prompt-model",
    )


@pytest.mark.parametrize("failure", ["network", 408, 429, 500, 502, 503, 504])
def test_llm_transient_failure_retries_once_and_returns_success(
    failure: str | int, monkeypatch
) -> None:
    attempts = 0
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if failure == "network":
                raise httpx.ConnectError("连接失败", request=request)
            return httpx.Response(failure, json={"error": "暂时失败"})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "retry success"}}]}
        )

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    async def run() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await service.preprocess_instruction(
                client, llm_settings(), "系统指令", "生成雨夜"
            )

    assert asyncio.run(run()).startswith("retry success\n")
    assert attempts == 2
    assert sleeps == [3.0]


@pytest.mark.parametrize("failure", ["network", 503])
def test_llm_two_transient_failures_raise_upstream_error(
    failure: str | int, monkeypatch
) -> None:
    attempts = 0
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if failure == "network":
            raise httpx.ReadTimeout("超时", request=request)
        return httpx.Response(failure, json={"error": "暂时失败"})

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(service.UpstreamError):
                await service.preprocess_instruction(
                    client, llm_settings(), "系统指令", "生成雨夜"
                )

    asyncio.run(run())
    assert attempts == 2
    assert sleeps == [3.0]


@pytest.mark.parametrize("failure", [400, "invalid_json", "empty_content"])
def test_llm_permanent_or_damaged_response_is_not_retried(failure: int | str) -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if failure == "invalid_json":
            return httpx.Response(200, text="not json")
        if failure == "empty_content":
            return httpx.Response(
                200, json={"choices": [{"message": {"content": " "}}]}
            )
        return httpx.Response(failure, json={"error": "永久失败"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(service.UpstreamError):
                await service.preprocess_instruction(
                    client, llm_settings(), "系统指令", "生成雨夜"
                )

    asyncio.run(run())
    assert attempts == 1


def test_unique_workflow_markers_are_resolved() -> None:
    template = service.resolve_workflow(workflow_data())

    assert template.instruction_node_id == "10"
    assert template.output_node_id == "20"


def test_committed_workflow_can_build_prompt() -> None:
    template = service.load_workflow()
    prompt = service.build_prompt(template, "生成雨夜街道")

    assert prompt[template.instruction_node_id]["inputs"]["text"] == "生成雨夜街道"


@pytest.mark.parametrize("title", ["API Instruction", "API Output"])
def test_missing_workflow_marker_stops_startup(title: str) -> None:
    data = workflow_data()
    for node in data.values():
        if node["_meta"]["title"] == title:
            node["_meta"]["title"] = "缺失标记"

    with pytest.raises(RuntimeError, match=title):
        service.resolve_workflow(data)


@pytest.mark.parametrize("title", ["API Instruction", "API Output"])
def test_duplicate_workflow_marker_stops_startup(title: str) -> None:
    data = workflow_data()
    source = next(node for node in data.values() if node["_meta"]["title"] == title)
    data["30"] = copy.deepcopy(source)

    with pytest.raises(RuntimeError, match=title):
        service.resolve_workflow(data)


def test_instruction_marker_requires_text_input() -> None:
    data = workflow_data()
    del data["10"]["inputs"]["text"]

    with pytest.raises(RuntimeError, match="inputs.text"):
        service.resolve_workflow(data)


def test_build_prompt_only_changes_instruction() -> None:
    template = service.resolve_workflow(workflow_data())
    original = copy.deepcopy(template.data)

    prompt = service.build_prompt(template, "生成雨夜街道")

    expected = copy.deepcopy(original)
    expected["10"]["inputs"]["text"] = "生成雨夜街道"
    assert prompt == expected
    assert template.data == original
    assert prompt["20"]["inputs"] == original["20"]["inputs"]


def successful_history(*images: dict[str, object]) -> dict[str, object]:
    return {
        "job-id": {
            "status": {"status_str": "success", "completed": True},
            "outputs": {"20": {"images": list(images)}},
            "prompt": ["敏感工作流"],
        }
    }


@pytest.mark.parametrize("image_count", [1, 2])
def test_completed_history_parses_one_or_many_images(image_count: int) -> None:
    one = {"filename": "one.png", "subfolder": "api\\job", "type": "output"}
    two = {"filename": "two.png", "subfolder": "api/job", "type": "output"}

    status, files = service.parse_history(
        successful_history(*(one, two)[:image_count]), "job-id", "20"
    )

    assert status == "completed"
    assert (
        files
        == [
            {"filename": "one.png", "subfolder": "api/job", "type": "output"},
            two,
        ][:image_count]
    )


def test_completed_history_only_returns_selected_output_files() -> None:
    history = successful_history(
        {"filename": "final.png", "subfolder": "api", "type": "output"},
        {"filename": "temp.png", "subfolder": "temp", "type": "temp"},
    )
    history["job-id"]["outputs"]["99"] = {
        "images": [{"filename": "other.png", "subfolder": "manual", "type": "output"}]
    }

    status, files = service.parse_history(history, "job-id", "20")

    assert status == "completed"
    assert files == [{"filename": "final.png", "subfolder": "api", "type": "output"}]


def test_successful_history_without_output_file_is_rejected() -> None:
    with pytest.raises(service.UpstreamError):
        service.parse_history(successful_history(), "job-id", "20")


def test_failed_history_returns_stable_status() -> None:
    history = {
        "job-id": {"status": {"status_str": "error", "completed": False}, "outputs": {}}
    }

    assert service.parse_history(history, "job-id", "20") == ("failed", None)


@pytest.mark.parametrize("section", ["queue_running", "queue_pending"])
def test_queue_running_or_pending_recognizes_job(section: str) -> None:
    queue = {"queue_running": [], "queue_pending": []}
    queue[section] = [[1, "job-id", {}, {}, []]]

    assert service.is_queued(queue, "job-id") is True
    assert service.is_queued(queue, "other-id") is False


def generation_service(
    handler,
) -> tuple[httpx.AsyncClient, service.GenerationService]:
    client = httpx.AsyncClient(
        base_url="http://comfy.local", transport=httpx.MockTransport(handler)
    )
    return client, service.GenerationService(
        client,
        client,
        llm_settings(),
        "系统指令",
        service.resolve_workflow(workflow_data()),
    )


def test_history_result_missing_returns_none_without_querying_queue() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={})

    async def run() -> service.GenerationResult | None:
        client, generation = generation_service(handler)
        try:
            return await generation.history_result("job-id")
        finally:
            await client.aclose()

    assert asyncio.run(run()) is None
    assert paths == ["/history/job-id"]


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_history_result_reuses_history_parsing(status: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/view":
            return httpx.Response(
                200, content=b"WEBP", headers={"content-type": "image/webp"}
            )
        if status == "failed":
            return httpx.Response(
                200,
                json={"job-id": {"status": {"status_str": "error"}, "outputs": {}}},
            )
        return httpx.Response(
            200,
            json=successful_history(
                {"filename": "result.webp", "subfolder": "api", "type": "output"}
            ),
        )

    async def run() -> service.GenerationResult | None:
        client, generation = generation_service(handler)
        try:
            return await generation.history_result("job-id")
        finally:
            await client.aclose()

    result = asyncio.run(run())
    assert result is not None
    assert result.status == status
    assert result.image == (b"WEBP" if status == "completed" else None)


@pytest.mark.parametrize("failure", ["network", 408, 429, 500, 502, 503, 504])
def test_comfy_get_transient_failure_has_distinct_error(failure: str | int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            raise httpx.ConnectError("连接失败", request=request)
        return httpx.Response(failure, json={"error": "暂时失败"})

    async def run() -> None:
        client, generation = generation_service(handler)
        try:
            with pytest.raises(service.TransientUpstreamError):
                await generation.history_result("job-id")
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("failure", [400, "damaged"])
def test_comfy_get_permanent_or_damaged_response_stays_permanent(
    failure: int | str,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        if failure == "damaged":
            return httpx.Response(200, json=[])
        return httpx.Response(failure, json={"error": "永久失败"})

    async def run() -> None:
        client, generation = generation_service(handler)
        try:
            with pytest.raises(service.UpstreamError) as caught:
                await generation.history_result("job-id")
            assert not isinstance(caught.value, service.TransientUpstreamError)
        finally:
            await client.aclose()

    asyncio.run(run())
