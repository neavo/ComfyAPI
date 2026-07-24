import json

import httpx
import pytest

from app.telegram_api import (
    BackendApi,
    ImageApiResult,
    TelegramApi,
    TextApiResult,
    TransientBackendApiError,
)

JOB_ID = "550e8400-e29b-41d4-a716-446655440000"


@pytest.mark.anyio
async def test_backend_api_submits_both_task_types() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, json={"id": JOB_ID})

    async with httpx.AsyncClient(
        base_url="http://api.local",
        transport=httpx.MockTransport(handler),
    ) as client:
        backend = BackendApi(client)
        text_id = await backend.submit_text_to_image("画一只猫", False)
        image_id = await backend.submit_image_to_text(b"PNG", "image/png")

    assert (text_id, image_id) == (JOB_ID, JOB_ID)
    assert requests[0].url.path == "/text_to_image"
    assert json.loads(requests[0].content) == {
        "instruction": "画一只猫",
        "safe_mode": False,
    }
    assert requests[1].url.path == "/image_to_text"
    assert requests[1].headers["content-type"] == "image/png"
    assert requests[1].content == b"PNG"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("path", "response", "expected"),
    [
        (
            f"/text_to_image/{JOB_ID}",
            httpx.Response(
                200,
                content=b"WEBP",
                headers={"content-type": "image/webp"},
            ),
            ImageApiResult("completed", b"WEBP", "image/webp"),
        ),
        (
            f"/image_to_text/{JOB_ID}",
            httpx.Response(200, json={"text": "prompt"}),
            TextApiResult("completed", "prompt"),
        ),
        (f"/text_to_image/{JOB_ID}", httpx.Response(404), ImageApiResult("missing")),
        (f"/image_to_text/{JOB_ID}", httpx.Response(500), TextApiResult("failed")),
    ],
)
async def test_backend_api_maps_result_contracts(
    path: str,
    response: httpx.Response,
    expected: ImageApiResult | TextApiResult,
) -> None:
    async with httpx.AsyncClient(
        base_url="http://api.local",
        transport=httpx.MockTransport(lambda _: response),
    ) as client:
        backend = BackendApi(client)
        result = (
            await backend.text_to_image_result(JOB_ID)
            if path.startswith("/text_to_image")
            else await backend.image_to_text_result(JOB_ID)
        )

    assert result == expected


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [202, 502])
async def test_backend_api_treats_pending_and_transient_results_differently(
    status_code: int,
) -> None:
    async with httpx.AsyncClient(
        base_url="http://api.local",
        transport=httpx.MockTransport(lambda _: httpx.Response(status_code)),
    ) as client:
        backend = BackendApi(client)
        if status_code == 202:
            assert await backend.image_to_text_result(JOB_ID) is None
        else:
            with pytest.raises(TransientBackendApiError):
                await backend.image_to_text_result(JOB_ID)


@pytest.mark.anyio
async def test_telegram_api_downloads_file() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/getFile"):
            return httpx.Response(
                200,
                json={"ok": True, "result": {"file_path": "photos/input.jpg"}},
            )
        return httpx.Response(200, content=b"JPEG")

    async with httpx.AsyncClient(
        base_url="https://api.telegram.org",
        transport=httpx.MockTransport(handler),
    ) as client:
        image = await TelegramApi("TOKEN", client).download_file("largest")

    assert image == b"JPEG"
    assert paths == ["/botTOKEN/getFile", "/file/botTOKEN/photos/input.jpg"]


@pytest.mark.anyio
async def test_telegram_429_uses_retry_after(monkeypatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                json={
                    "ok": False,
                    "description": "Too Many Requests",
                    "parameters": {"retry_after": 7},
                },
            )
        return httpx.Response(200, json={"ok": True, "result": {}})

    monkeypatch.setattr("app.telegram_api.asyncio.sleep", record_sleep)
    async with httpx.AsyncClient(
        base_url="https://api.telegram.org",
        transport=httpx.MockTransport(handler),
    ) as client:
        delivered = await TelegramApi("TOKEN", client).send_message({}, "状态")

    assert delivered is True
    assert attempts == 2
    assert sleeps == [7]
