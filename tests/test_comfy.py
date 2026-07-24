import json

import httpx
import pytest

from app.comfy import ComfyClient, ComfyError, GenerationResult


JOB_ID = "550e8400-e29b-41d4-a716-446655440000"
PROMPT = {"10": {"inputs": {"text": "生成雨夜"}}}


def successful_history(*images: dict[str, object]) -> dict[str, object]:
    return {
        JOB_ID: {
            "status": {"status_str": "success"},
            "outputs": {"20": {"images": list(images)}},
        }
    }


@pytest.mark.anyio
async def test_submit_sends_only_native_prompt_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"prompt_id": JOB_ID})

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        await ComfyClient(client, "20").submit(JOB_ID, PROMPT)

    assert captured == {"prompt_id": JOB_ID, "prompt": PROMPT}


@pytest.mark.anyio
@pytest.mark.parametrize("section", ["queue_running", "queue_pending"])
async def test_result_reports_queued_jobs_as_processing(section: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/history"):
            return httpx.Response(200, json={})
        queue = {"queue_running": [], "queue_pending": []}
        queue[section] = [[1, JOB_ID]]
        return httpx.Response(200, json=queue)

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await ComfyClient(client, "20").result(JOB_ID)

    assert result == GenerationResult("processing")


@pytest.mark.anyio
async def test_result_downloads_first_output_image_from_configured_node() -> None:
    requested_view: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/view":
            requested_view.update(request.url.params)
            return httpx.Response(
                200,
                content=b"WEBP",
                headers={"content-type": "image/webp"},
            )
        history = successful_history(
            {"filename": "preview.png", "subfolder": "temp", "type": "temp"},
            {"filename": "final.webp", "subfolder": "api\\job", "type": "output"},
            {"filename": "second.webp", "subfolder": "api", "type": "output"},
        )
        history[JOB_ID]["outputs"]["99"] = {
            "images": [{"filename": "other.webp", "subfolder": "", "type": "output"}]
        }
        return httpx.Response(200, json=history)

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await ComfyClient(client, "20").result(JOB_ID)

    assert result == GenerationResult("completed", b"WEBP", "image/webp")
    assert requested_view == {
        "filename": "final.webp",
        "subfolder": "api/job",
        "type": "output",
    }


@pytest.mark.anyio
async def test_result_reports_failed_history_as_failed() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={JOB_ID: {"status": {"status_str": "error"}}},
        )

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await ComfyClient(client, "20").result(JOB_ID)

    assert result == GenerationResult("failed")


@pytest.mark.anyio
async def test_result_rechecks_history_after_job_leaves_queue() -> None:
    history_queries = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal history_queries
        if request.url.path.startswith("/history"):
            history_queries += 1
            history = (
                {}
                if history_queries == 1
                else successful_history(
                    {"filename": "final.webp", "subfolder": "", "type": "output"}
                )
            )
            return httpx.Response(200, json=history)
        if request.url.path == "/queue":
            return httpx.Response(
                200,
                json={"queue_running": [], "queue_pending": []},
            )
        return httpx.Response(200, content=b"WEBP")

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await ComfyClient(client, "20").result(JOB_ID)

    assert result.status == "completed"
    assert history_queries == 2


@pytest.mark.anyio
async def test_result_reports_missing_only_after_history_recheck() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/queue":
            return httpx.Response(
                200,
                json={"queue_running": [], "queue_pending": []},
            )
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await ComfyClient(client, "20").result(JOB_ID)

    assert result == GenerationResult("missing")
    assert paths == [f"/history/{JOB_ID}", "/queue", f"/history/{JOB_ID}"]


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["network", "http", "json", "structure"])
async def test_protocol_failures_raise_comfy_error(failure: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            raise httpx.ConnectError("连接失败", request=request)
        if failure == "http":
            return httpx.Response(503)
        if failure == "json":
            return httpx.Response(200, content=b"not-json")
        return httpx.Response(200, json={JOB_ID: {"status": "broken"}})

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ComfyError):
            await ComfyClient(client, "20").result(JOB_ID)
