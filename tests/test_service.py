import copy
import json
from pathlib import Path

import httpx
import pytest

from app import service
from app.comfy import ComfyError, ComfyJob, DownloadedImage


def workflow_data(input_name: str = "text") -> dict[str, object]:
    return {
        "10": {
            "inputs": {input_name: "原始输入", "seed": 1},
            "class_type": "Input",
            "_meta": {"title": "api_input"},
        },
        "20": {
            "inputs": {"source": ["10", 0], "noise_seed": 2},
            "class_type": "Output",
            "_meta": {"title": "api_output"},
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


class FakeComfy:
    def __init__(self, job: ComfyJob = ComfyJob("missing")) -> None:
        self.job_value = job
        self.submissions: list[tuple[str, dict[str, object]]] = []
        self.uploads: list[tuple[str, bytes, str]] = []

    async def upload_image(
        self,
        filename: str,
        image: bytes,
        media_type: str,
    ) -> str:
        self.uploads.append((filename, image, media_type))
        return f"api/image_to_text/{filename}"

    async def submit(self, job_id: str, prompt: dict[str, object]) -> None:
        self.submissions.append((job_id, prompt))

    async def job(self, _: str) -> ComfyJob:
        return self.job_value

    async def download_image(self, _: object) -> DownloadedImage:
        return DownloadedImage(b"WEBP", "image/webp")


def text_to_image(
    client: httpx.AsyncClient,
    comfy: FakeComfy | None = None,
) -> service.TextToImageService:
    return service.TextToImageService(
        comfy or FakeComfy(),
        client,
        settings(),
        "系统指令",
        service.resolve_workflow(workflow_data(), "text"),
    )


def write_config(root: Path, **overrides: str | None) -> None:
    values: dict[str, str | None] = {
        "api_token": " secret ",
        "comfy_url": "http://127.0.0.1:8188/",
        "llm_url": "http://llm.local/v1/chat/completions",
        "llm_api_key": "llm-secret",
        "llm_model": "prompt-model",
        **overrides,
    }
    config = root / "config"
    config.mkdir(parents=True)
    (config / "config.toml").write_text(
        "".join(
            f"{name} = {json.dumps(value, ensure_ascii=False)}\n"
            for name, value in values.items()
            if value is not None
        ),
        encoding="utf-8",
    )


def test_settings_load_from_project_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "project"
    write_config(root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(service, "PROJECT_ROOT", root)
    monkeypatch.chdir(elsewhere)

    loaded = service.load_settings()

    assert loaded == service.Settings(
        "secret",
        "http://127.0.0.1:8188",
        "http://llm.local/v1/chat/completions",
        "llm-secret",
        "prompt-model",
    )


def test_missing_configuration_stops_startup(tmp_path: Path) -> None:
    root = tmp_path / "project"
    write_config(root, llm_model=None)

    with pytest.raises(RuntimeError, match="llm_model"):
        service.load_settings(root)


@pytest.mark.parametrize(
    "url",
    ["ftp://127.0.0.1", "http://user:pass@127.0.0.1", "http://127.0.0.1/path"],
)
def test_invalid_comfy_url_stops_startup(tmp_path: Path, url: str) -> None:
    root = tmp_path / "project"
    write_config(root, comfy_url=url)

    with pytest.raises(RuntimeError, match="comfy_url"):
        service.load_settings(root)


def test_system_prompt_loads_multiline_utf8(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    path = root / "prompt" / "system.md"
    path.parent.mkdir(parents=True)
    path.write_text("\ufeff 系统指令\n第二行 ", encoding="utf-8")
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
        with pytest.raises(service.LlmUpstreamError):
            await service.preprocess_instruction(
                client,
                settings(),
                "系统指令",
                "生成雨夜",
            )

    assert attempts == 1


def test_committed_workflows_build_prompts() -> None:
    text = service.load_workflow(service.TEXT_TO_IMAGE_WORKFLOW_PATH, "text")
    image = service.load_workflow(service.IMAGE_TO_TEXT_WORKFLOW_PATH, "image")

    assert (
        text.data[text.output_node_id]["inputs"]["filename_prefix"]
        == "api/%date:yyyyMMdd%/%date:yyyyMMdd%"
    )
    assert (
        service.build_workflow(text, "生成雨夜")[text.input_node_id]["inputs"]["text"]
        == "生成雨夜"
    )
    assert (
        service.build_workflow(image, "api/input.webp")[image.input_node_id]["inputs"][
            "image"
        ]
        == "api/input.webp"
    )


@pytest.mark.parametrize("title", ["api_input", "api_output"])
def test_missing_or_duplicate_workflow_marker_stops_startup(title: str) -> None:
    missing = workflow_data()
    source = next(node for node in missing.values() if node["_meta"]["title"] == title)
    source["_meta"]["title"] = "缺失标记"

    with pytest.raises(RuntimeError, match=title):
        service.resolve_workflow(missing, "text")

    duplicate = workflow_data()
    source = next(
        node for node in duplicate.values() if node["_meta"]["title"] == title
    )
    duplicate["30"] = copy.deepcopy(source)

    with pytest.raises(RuntimeError, match=title):
        service.resolve_workflow(duplicate, "text")


def test_input_marker_requires_configured_field() -> None:
    data = workflow_data()
    del data["10"]["inputs"]["text"]

    with pytest.raises(RuntimeError, match="inputs.text"):
        service.resolve_workflow(data, "text")


def test_build_workflow_only_randomizes_seeds_when_requested(monkeypatch) -> None:
    template = service.resolve_workflow(workflow_data(), "text")
    original = copy.deepcopy(template.data)
    seeds = iter([101, 202])
    monkeypatch.setattr(service.random, "randint", lambda *_: next(seeds))

    stable = service.build_workflow(template, "稳定")
    randomized = service.build_workflow(
        template,
        "随机",
        randomize_seeds=True,
    )

    assert stable["10"]["inputs"] == {"text": "稳定", "seed": 1}
    assert randomized["10"]["inputs"] == {"text": "随机", "seed": 101}
    assert randomized["20"]["inputs"]["noise_seed"] == 202
    assert template.data == original


@pytest.mark.anyio
async def test_text_to_image_passthrough_skips_llm() -> None:
    comfy = FakeComfy()

    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("透传模式不应访问 LLM")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as client:
        job_id = await text_to_image(client, comfy).submit(
            "保留 启用透传模式中间启用透传模式 空格"
        )

    assert comfy.submissions[0][0] == job_id
    assert comfy.submissions[0][1]["10"]["inputs"]["text"] == (
        f"保留 中间 空格\n{service.PROMPT_SUFFIX}"
    )


@pytest.mark.anyio
async def test_text_to_image_returns_downloaded_output() -> None:
    comfy = FakeComfy(
        ComfyJob(
            "completed",
            {
                "20": {
                    "images": [
                        {"filename": "preview.png", "type": "temp"},
                        {
                            "filename": "final.webp",
                            "subfolder": "api",
                            "type": "output",
                        },
                    ]
                }
            },
        )
    )
    async with httpx.AsyncClient() as client:
        result = await text_to_image(client, comfy).result("job")

    assert result == service.ImageResult("completed", b"WEBP", "image/webp")


@pytest.mark.anyio
async def test_image_to_text_uploads_and_returns_raw_text() -> None:
    comfy = FakeComfy(
        ComfyJob("completed", {"20": {"text": ["  description\n\ntags  "]}})
    )
    image_to_text = service.ImageToTextService(
        comfy,
        service.resolve_workflow(workflow_data("image"), "image"),
    )

    job_id = await image_to_text.submit(b"PNG", "image/png")
    result = await image_to_text.result(job_id)

    assert comfy.uploads == [(f"{job_id}.png", b"PNG", "image/png")]
    assert comfy.submissions[0][1]["10"]["inputs"]["image"] == (
        f"api/image_to_text/{job_id}.png"
    )
    assert result == service.TextResult("completed", "description\n\ntags")


@pytest.mark.anyio
@pytest.mark.parametrize("output", [{}, {"text": [123]}])
async def test_completed_job_without_expected_output_is_upstream_error(
    output: dict[str, object],
) -> None:
    comfy = FakeComfy(ComfyJob("completed", {"20": output}))
    image_to_text = service.ImageToTextService(
        comfy,
        service.resolve_workflow(workflow_data("image"), "image"),
    )

    with pytest.raises(ComfyError):
        await image_to_text.result("job")
