import json

import httpx
import pytest

from app.comfy import ComfyClient, ComfyError, ComfyJob, DownloadedImage


JOB_ID = "550e8400-e29b-41d4-a716-446655440000"
PROMPT = {"10": {"inputs": {"text": "生成雨夜"}}}


def successful_history(outputs: object) -> dict[str, object]:
    return {
        JOB_ID: {
            "status": {"status_str": "success"},
            "outputs": outputs,
        }
    }


@pytest.mark.anyio
async def test_upload_image_returns_workflow_path() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "name": "job.webp",
                "subfolder": "api\\image_to_text",
                "type": "input",
            },
        )

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        path = await ComfyClient(client).upload_image(
            "job.webp",
            b"WEBP",
            "image/webp",
        )

    assert path == "api/image_to_text/job.webp"
    assert str(captured["content_type"]).startswith("multipart/form-data;")
    body = bytes(captured["body"])
    assert all(
        value in body
        for value in (
            b'name="type"',
            b"input",
            b'name="subfolder"',
            b"api/image_to_text",
            b'filename="job.webp"',
            b"WEBP",
        )
    )


@pytest.mark.anyio
async def test_submit_sends_native_prompt_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"prompt_id": JOB_ID})

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        await ComfyClient(client).submit(JOB_ID, PROMPT)

    assert captured == {"prompt_id": JOB_ID, "prompt": PROMPT}


@pytest.mark.anyio
@pytest.mark.parametrize("section", ["queue_running", "queue_pending"])
async def test_job_reports_queued_tasks_as_processing(section: str) -> None:
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
        result = await ComfyClient(client).job(JOB_ID)

    assert result == ComfyJob("processing")


@pytest.mark.anyio
async def test_job_preserves_completed_workflow_outputs() -> None:
    outputs = {
        "20": {"images": [{"filename": "final.webp"}]},
        "30": {"text": ["a rainy street"]},
    }

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=successful_history(outputs))
        ),
    ) as client:
        result = await ComfyClient(client).job(JOB_ID)

    assert result == ComfyJob("completed", outputs)


@pytest.mark.anyio
async def test_download_image_uses_output_metadata() -> None:
    requested: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requested.update(request.url.params)
        return httpx.Response(
            200,
            content=b"WEBP",
            headers={"content-type": "image/webp"},
        )

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await ComfyClient(client).download_image(
            {
                "filename": "final.webp",
                "subfolder": "api\\job",
                "type": "output",
            }
        )

    assert result == DownloadedImage(b"WEBP", "image/webp")
    assert requested == {
        "filename": "final.webp",
        "subfolder": "api/job",
        "type": "output",
    }


@pytest.mark.anyio
async def test_job_reports_failed_history() -> None:
    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={JOB_ID: {"status": {"status_str": "error"}}},
            )
        ),
    ) as client:
        result = await ComfyClient(client).job(JOB_ID)

    assert result == ComfyJob("failed")


@pytest.mark.anyio
async def test_job_rechecks_history_after_queue_race() -> None:
    history_queries = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal history_queries
        if request.url.path.startswith("/history"):
            history_queries += 1
            payload = (
                {}
                if history_queries == 1
                else successful_history({"20": {"images": []}})
            )
            return httpx.Response(200, json=payload)
        return httpx.Response(
            200,
            json={"queue_running": [], "queue_pending": []},
        )

    async with httpx.AsyncClient(
        base_url="http://comfy.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await ComfyClient(client).job(JOB_ID)

    assert result.status == "completed"
    assert history_queries == 2


@pytest.mark.anyio
async def test_job_reports_missing_only_after_history_recheck() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
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
        result = await ComfyClient(client).job(JOB_ID)

    assert result == ComfyJob("missing")


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
            await ComfyClient(client).job(JOB_ID)
