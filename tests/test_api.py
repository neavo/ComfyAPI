from types import SimpleNamespace

import httpx
import pytest

from app.api import MAX_IMAGE_BYTES
from app.comfy import ComfyError
from app.main import app
from app.service import (
    ImageResult,
    InstructionError,
    LlmUpstreamError,
    TextResult,
)

JOB_ID = "550e8400-e29b-41d4-a716-446655440000"
PNG = b"\x89PNG\r\n\x1a\nimage"


class FakeTextToImage:
    def __init__(
        self,
        *,
        submit: str | Exception = JOB_ID,
        result: ImageResult | Exception = ImageResult("missing"),
    ) -> None:
        self.submit_value = submit
        self.result_value = result
        self.instructions: list[str] = []
        self.job_ids: list[str] = []

    async def submit(self, instruction: str) -> str:
        self.instructions.append(instruction)
        if isinstance(self.submit_value, Exception):
            raise self.submit_value
        return self.submit_value

    async def result(self, job_id: str) -> ImageResult:
        self.job_ids.append(job_id)
        if isinstance(self.result_value, Exception):
            raise self.result_value
        return self.result_value


class FakeImageToText:
    def __init__(
        self,
        *,
        submit: str | Exception = JOB_ID,
        result: TextResult | Exception = TextResult("missing"),
    ) -> None:
        self.submit_value = submit
        self.result_value = result
        self.images: list[tuple[bytes, str]] = []
        self.job_ids: list[str] = []

    async def submit(self, image: bytes, media_type: str) -> str:
        self.images.append((image, media_type))
        if isinstance(self.submit_value, Exception):
            raise self.submit_value
        return self.submit_value

    async def result(self, job_id: str) -> TextResult:
        self.job_ids.append(job_id)
        if isinstance(self.result_value, Exception):
            raise self.result_value
        return self.result_value


async def request(
    method: str,
    path: str,
    *,
    text_to_image: FakeTextToImage | None = None,
    image_to_text: FakeImageToText | None = None,
    token: str | None = "secret",
    json: object | None = None,
    content: bytes | None = None,
    content_type: str | None = None,
) -> httpx.Response:
    app.state.settings = SimpleNamespace(token="secret")
    app.state.text_to_image = text_to_image or FakeTextToImage()
    app.state.image_to_text = image_to_text or FakeImageToText()
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    if content_type is not None:
        headers["Content-Type"] = content_type
    async with httpx.AsyncClient(
        base_url="http://api.local",
        transport=httpx.ASGITransport(app=app),
    ) as client:
        return await client.request(
            method,
            path,
            headers=headers,
            json=json,
            content=content,
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("POST", "/text_to_image", {"json": {"instruction": "雨夜"}}),
        ("GET", f"/text_to_image/{JOB_ID}", {}),
        ("POST", "/image_to_text", {"content": PNG, "content_type": "image/png"}),
        ("GET", f"/image_to_text/{JOB_ID}", {}),
    ],
)
@pytest.mark.parametrize("token", [None, "wrong"])
async def test_all_routes_require_bearer_token(
    method: str,
    path: str,
    kwargs: dict[str, object],
    token: str | None,
) -> None:
    response = await request(method, path, token=token, **kwargs)

    assert response.status_code == 401


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["/text_to_image", "/new"])
async def test_text_to_image_routes_share_submission_contract(path: str) -> None:
    service = FakeTextToImage()

    response = await request(
        "POST",
        path,
        text_to_image=service,
        json={"instruction": "  生成雨夜街道  "},
    )

    assert response.status_code == 202
    assert response.json() == {"id": JOB_ID}
    assert service.instructions == ["生成雨夜街道"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {"instruction": " "},
        {"instruction": "x" * 4097},
        {"instruction": "雨夜", "seed": 1},
    ],
)
async def test_text_to_image_rejects_invalid_json(body: object) -> None:
    service = FakeTextToImage()

    response = await request(
        "POST",
        "/text_to_image",
        text_to_image=service,
        json=body,
    )

    assert response.status_code == 422
    assert service.instructions == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (InstructionError("无有效指令"), 422, "无有效指令"),
        (LlmUpstreamError("敏感 LLM 错误"), 502, "LLM upstream error"),
        (ComfyError("敏感节点错误"), 502, "ComfyUI upstream error"),
    ],
)
async def test_text_to_image_sanitizes_submission_errors(
    error: Exception,
    status_code: int,
    detail: str,
) -> None:
    response = await request(
        "POST",
        "/text_to_image",
        text_to_image=FakeTextToImage(submit=error),
        json={"instruction": "内部指令"},
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert "敏感" not in response.text


@pytest.mark.anyio
async def test_image_to_text_accepts_raw_image() -> None:
    service = FakeImageToText()

    response = await request(
        "POST",
        "/image_to_text",
        image_to_text=service,
        content=PNG,
        content_type="image/png",
    )

    assert response.status_code == 202
    assert response.json() == {"id": JOB_ID}
    assert service.images == [(PNG, "image/png")]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("content", "content_type", "status_code"),
    [
        (b"", "image/png", 422),
        (b"not png", "image/png", 415),
        (PNG, "image/gif", 415),
    ],
)
async def test_image_to_text_rejects_invalid_images(
    content: bytes,
    content_type: str,
    status_code: int,
) -> None:
    service = FakeImageToText()

    response = await request(
        "POST",
        "/image_to_text",
        image_to_text=service,
        content=content,
        content_type=content_type,
    )

    assert response.status_code == status_code
    assert service.images == []


@pytest.mark.anyio
async def test_image_to_text_rejects_oversized_body() -> None:
    service = FakeImageToText()

    response = await request(
        "POST",
        "/image_to_text",
        image_to_text=service,
        content=b"x" * (MAX_IMAGE_BYTES + 1),
        content_type="image/jpeg",
    )

    assert response.status_code == 413
    assert service.images == []


@pytest.mark.anyio
async def test_image_to_text_sanitizes_submission_error() -> None:
    response = await request(
        "POST",
        "/image_to_text",
        image_to_text=FakeImageToText(submit=ComfyError("敏感上传错误")),
        content=PNG,
        content_type="image/png",
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "ComfyUI upstream error"}
    assert "敏感" not in response.text


@pytest.mark.anyio
@pytest.mark.parametrize("path", [f"/text_to_image/{JOB_ID}", f"/result/{JOB_ID}"])
async def test_text_to_image_result_routes_return_image(path: str) -> None:
    service = FakeTextToImage(result=ImageResult("completed", b"WEBP", "image/webp"))

    response = await request("GET", path, text_to_image=service)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.content == b"WEBP"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("result", "status_code", "detail"),
    [
        (ImageResult("processing"), 202, None),
        (ImageResult("failed"), 500, "generation failed"),
        (ImageResult("missing"), 404, "Task not found"),
    ],
)
async def test_text_to_image_maps_task_status(
    result: ImageResult,
    status_code: int,
    detail: str | None,
) -> None:
    response = await request(
        "GET",
        f"/text_to_image/{JOB_ID}",
        text_to_image=FakeTextToImage(result=result),
    )

    assert response.status_code == status_code
    if detail is None:
        assert response.content == b""
    else:
        assert response.json() == {"detail": detail}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("result", "status_code", "body"),
    [
        (
            TextResult("completed", " description\n\ntags "),
            200,
            {"text": " description\n\ntags "},
        ),
        (TextResult("processing"), 202, None),
        (TextResult("failed"), 500, {"detail": "generation failed"}),
        (TextResult("missing"), 404, {"detail": "Task not found"}),
    ],
)
async def test_image_to_text_maps_task_status(
    result: TextResult,
    status_code: int,
    body: dict[str, str] | None,
) -> None:
    response = await request(
        "GET",
        f"/image_to_text/{JOB_ID}",
        image_to_text=FakeImageToText(result=result),
    )

    assert response.status_code == status_code
    if body is None:
        assert response.content == b""
    else:
        assert response.json() == body


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("path", "service_name"),
    [
        (f"/text_to_image/{JOB_ID}", "text"),
        (f"/image_to_text/{JOB_ID}", "image"),
    ],
)
async def test_result_routes_sanitize_upstream_errors(
    path: str,
    service_name: str,
) -> None:
    kwargs = (
        {"text_to_image": FakeTextToImage(result=ComfyError("内部错误"))}
        if service_name == "text"
        else {"image_to_text": FakeImageToText(result=ComfyError("内部错误"))}
    )

    response = await request("GET", path, **kwargs)

    assert response.status_code == 502
    assert response.json() == {"detail": "ComfyUI upstream error"}
    assert "内部错误" not in response.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    "path",
    ["/text_to_image/not-a-uuid", "/image_to_text/not-a-uuid"],
)
async def test_result_routes_reject_invalid_uuid(path: str) -> None:
    response = await request("GET", path)

    assert response.status_code == 422
