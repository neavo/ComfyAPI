import copy
import json
from pathlib import Path

import httpx
import pytest

from app import service


def workflow_data() -> dict[str, object]:
    return {
        "10": {
            "inputs": {"text": "原始指令", "seed": 1},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "API Instruction"},
        },
        "20": {
            "inputs": {"images": ["2", 0], "noise_seed": 2},
            "_meta": {"title": "API Output"},
        },
    }


def settings() -> service.Settings:
    return service.Settings(
        "api-secret",
        "http://comfy.local",
        "http://llm.local/v1/chat/completions",
        "llm-secret",
        "prompt-model",
    )


def generation(client: httpx.AsyncClient) -> service.GenerationService:
    return service.GenerationService(
        client,
        client,
        settings(),
        "系统指令",
        service.resolve_workflow(workflow_data()),
    )


def write_settings(fs, root: Path) -> None:
    config = root / "config"
    for name, value in {
        "api_token.txt": "\ufeff secret ",
        "comfy_url.txt": "http://127.0.0.1:8188/",
        "llm_url.txt": "http://llm.local/v1/chat/completions",
        "llm_api_key.txt": "llm-secret",
        "llm_model.txt": "prompt-model",
    }.items():
        fs.create_file(config / name, contents=value)


def test_settings_load_from_project_root(fs, monkeypatch) -> None:
    root = Path("/project")
    write_settings(fs, root)
    fs.create_dir("/elsewhere")
    monkeypatch.setattr(service, "PROJECT_ROOT", root)
    monkeypatch.chdir("/elsewhere")

    loaded = service.load_settings()

    assert loaded == service.Settings(
        "secret",
        "http://127.0.0.1:8188",
        "http://llm.local/v1/chat/completions",
        "llm-secret",
        "prompt-model",
    )


def test_missing_configuration_stops_startup(fs) -> None:
    root = Path("/project")
    write_settings(fs, root)
    fs.remove_object(str(root / "config/llm_model.txt"))

    with pytest.raises(RuntimeError, match="llm_model.txt"):
        service.load_settings(root)


@pytest.mark.parametrize(
    "url",
    ["ftp://127.0.0.1", "http://user:pass@127.0.0.1", "http://127.0.0.1/path"],
)
def test_invalid_comfy_url_stops_startup(fs, url: str) -> None:
    root = Path("/project")
    write_settings(fs, root)
    (root / "config/comfy_url.txt").write_text(url, encoding="utf-8")

    with pytest.raises(RuntimeError, match="comfy_url.txt"):
        service.load_settings(root)


def test_system_prompt_loads_multiline_utf8(fs, monkeypatch) -> None:
    root = Path("/project")
    fs.create_file(root / "prompt/system.md", contents="\ufeff 系统指令\n第二行 ")
    monkeypatch.setattr(service, "PROJECT_ROOT", root)

    assert service.load_system_prompt() == "系统指令\n第二行"


@pytest.mark.anyio
async def test_llm_preprocessing_uses_chat_completions_contract() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  a rainy neon street  "}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await service.preprocess_instruction(
            client,
            settings(),
            "系统指令",
            "生成雨夜街道",
        )

    assert result == f"a rainy neon street\n{service.PROMPT_SUFFIX}"
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


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["network", 503])
async def test_llm_retries_one_transient_failure(
    failure: str | int,
    monkeypatch,
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
            return httpx.Response(failure)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "retry success"}}]},
        )

    monkeypatch.setattr(service.asyncio, "sleep", record_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await service.preprocess_instruction(
            client,
            settings(),
            "系统指令",
            "生成雨夜",
        )

    assert result.startswith("retry success\n")
    assert attempts == 2
    assert sleeps == [service.RETRY_DELAY]


@pytest.mark.anyio
async def test_llm_permanent_failure_is_not_retried() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(service.UpstreamError):
            await service.preprocess_instruction(
                client,
                settings(),
                "系统指令",
                "生成雨夜",
            )

    assert attempts == 1


def test_workflow_markers_are_resolved() -> None:
    template = service.resolve_workflow(workflow_data())

    assert (template.instruction_node_id, template.output_node_id) == ("10", "20")


def test_committed_workflow_builds_prompt() -> None:
    template = service.load_workflow()

    prompt = service.build_prompt(template, "生成雨夜街道")

    assert prompt[template.instruction_node_id]["inputs"]["text"] == "生成雨夜街道"


@pytest.mark.parametrize("title", ["API Instruction", "API Output"])
def test_missing_or_duplicate_workflow_marker_stops_startup(title: str) -> None:
    missing = workflow_data()
    source = next(node for node in missing.values() if node["_meta"]["title"] == title)
    source["_meta"]["title"] = "缺失标记"

    with pytest.raises(RuntimeError, match=title):
        service.resolve_workflow(missing)

    duplicate = workflow_data()
    source = next(
        node for node in duplicate.values() if node["_meta"]["title"] == title
    )
    duplicate["30"] = copy.deepcopy(source)

    with pytest.raises(RuntimeError, match=title):
        service.resolve_workflow(duplicate)


def test_instruction_marker_requires_text_input() -> None:
    data = workflow_data()
    del data["10"]["inputs"]["text"]

    with pytest.raises(RuntimeError, match="inputs.text"):
        service.resolve_workflow(data)


def test_build_prompt_copies_template_and_randomizes_seeds(monkeypatch) -> None:
    template = service.resolve_workflow(workflow_data())
    original = copy.deepcopy(template.data)
    seeds = iter([101, 202])
    monkeypatch.setattr(service.random, "randint", lambda *_: next(seeds))

    prompt = service.build_prompt(template, "生成雨夜街道")

    assert prompt["10"]["inputs"] == {"text": "生成雨夜街道", "seed": 101}
    assert prompt["20"]["inputs"]["noise_seed"] == 202
    assert template.data == original


def successful_history(*images: dict[str, object]) -> dict[str, object]:
    return {
        "job-id": {
            "status": {"status_str": "success"},
            "outputs": {"20": {"images": list(images)}},
        }
    }


def test_history_selects_first_output_image() -> None:
    history = successful_history(
        {"filename": "preview.png", "subfolder": "temp", "type": "temp"},
        {"filename": "final.png", "subfolder": "api\\job", "type": "output"},
        {"filename": "second.png", "subfolder": "api", "type": "output"},
    )
    history["job-id"]["outputs"]["99"] = {
        "images": [{"filename": "other.png", "subfolder": "", "type": "output"}]
    }

    assert service.parse_history(history, "job-id", "20") == {
        "filename": "final.png",
        "subfolder": "api/job",
        "type": "output",
    }


def test_history_reports_missing_failed_and_damaged_results() -> None:
    assert service.parse_history({}, "job-id", "20") is None
    assert (
        service.parse_history(
            {"job-id": {"status": {"status_str": "error"}}},
            "job-id",
            "20",
        )
        == "failed"
    )
    with pytest.raises(service.UpstreamError):
        service.parse_history(successful_history(), "job-id", "20")


@pytest.mark.parametrize("section", ["queue_running", "queue_pending"])
def test_queue_recognizes_running_or_pending_job(section: str) -> None:
    queue = {"queue_running": [], "queue_pending": []}
    queue[section] = [[1, "job-id"]]

    assert service.is_queued(queue, "job-id") is True
    assert service.is_queued(queue, "other-id") is False


@pytest.mark.anyio
async def test_passthrough_submission_skips_llm_and_removes_markers() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/prompt"
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"prompt_id": captured["prompt_id"]})

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        job_id = await generation(client).submit(
            "保留 启用透传模式中间启用透传模式 空格"
        )

    assert captured["prompt_id"] == job_id
    assert captured["prompt"]["10"]["inputs"]["text"] == (
        f"保留 中间 空格\n{service.PROMPT_SUFFIX}"
    )


@pytest.mark.anyio
async def test_passthrough_without_instruction_is_rejected_before_submission() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("不应访问上游")

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(service.InstructionError):
            await generation(client).submit("启用透传模式启用透传模式")


@pytest.mark.anyio
async def test_llm_failure_is_distinguished_from_comfy_failure() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"choices": []})

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(service.LlmUpstreamError):
            await generation(client).submit("生成雨夜")

    assert paths == ["/v1/chat/completions"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("queue", "expected"),
    [
        ({"queue_running": [], "queue_pending": [[1, "job-id"]]}, "processing"),
        ({"queue_running": [], "queue_pending": []}, "missing"),
    ],
)
async def test_missing_history_falls_back_to_queue(
    queue: dict[str, object],
    expected: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={} if request.url.path.startswith("/history") else queue
        )

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await generation(client).result("job-id")

    assert result.status == expected


@pytest.mark.anyio
async def test_completed_result_downloads_selected_image() -> None:
    requested_view: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/view":
            requested_view.update(request.url.params)
            return httpx.Response(
                200,
                content=b"WEBP",
                headers={"content-type": "image/webp"},
            )
        return httpx.Response(
            200,
            json=successful_history(
                {"filename": "final.webp", "subfolder": "api", "type": "output"}
            ),
        )

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await generation(client).result("job-id")

    assert result == service.GenerationResult("completed", b"WEBP", "image/webp")
    assert requested_view == {
        "filename": "final.webp",
        "subfolder": "api",
        "type": "output",
    }


@pytest.mark.anyio
async def test_upstream_failure_raises_stable_service_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(service.UpstreamError):
            await generation(client).result("job-id")
