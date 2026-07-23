from types import SimpleNamespace

import httpx
import pytest

from app.main import app
from app.service import (
    GenerationResult,
    InstructionError,
    LlmUpstreamError,
    UpstreamError,
)

JOB_ID = "550e8400-e29b-41d4-a716-446655440000"


class FakeGeneration:
    def __init__(
        self,
        *,
        submit: str | Exception = JOB_ID,
        result: GenerationResult | Exception = GenerationResult("missing"),
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

    async def result(self, job_id: str) -> GenerationResult:
        self.job_ids.append(job_id)
        if isinstance(self.result_value, Exception):
            raise self.result_value
        return self.result_value


async def request(
    method: str,
    path: str,
    generation: FakeGeneration,
    *,
    token: str | None = "secret",
    body: object | None = None,
) -> httpx.Response:
    app.state.settings = SimpleNamespace(token="secret")
    app.state.generation = generation
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    async with httpx.AsyncClient(
        base_url="http://api.local",
        transport=httpx.ASGITransport(app=app),
    ) as client:
        return await client.request(method, path, headers=headers, json=body)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/new", {"instruction": "雨夜"}),
        ("GET", f"/result/{JOB_ID}", None),
    ],
)
@pytest.mark.parametrize("token", [None, "wrong"])
async def test_missing_or_wrong_token_returns_401(
    method: str,
    path: str,
    body: object | None,
    token: str | None,
) -> None:
    response = await request(
        method,
        path,
        FakeGeneration(),
        token=token,
        body=body,
    )

    assert response.status_code == 401


@pytest.mark.anyio
async def test_new_normalizes_instruction_and_returns_only_job_id() -> None:
    generation = FakeGeneration()

    response = await request(
        "POST",
        "/new",
        generation,
        body={"instruction": "  生成雨夜街道  "},
    )

    assert response.status_code == 202
    assert response.json() == {"id": JOB_ID}
    assert generation.instructions == ["生成雨夜街道"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {"instruction": " "},
        {"instruction": "x" * 4097},
        {"instruction": "雨夜", "seed": 1},
    ],
)
async def test_new_rejects_invalid_request_body(body: object) -> None:
    generation = FakeGeneration()

    response = await request("POST", "/new", generation, body=body)

    assert response.status_code == 422
    assert generation.instructions == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (InstructionError("无有效指令"), 422, "无有效指令"),
        (LlmUpstreamError("敏感 LLM 错误"), 502, "LLM upstream error"),
        (UpstreamError("敏感节点错误"), 502, "ComfyUI upstream error"),
    ],
)
async def test_new_maps_service_errors_to_public_responses(
    error: Exception,
    status_code: int,
    detail: str,
) -> None:
    response = await request(
        "POST",
        "/new",
        FakeGeneration(submit=error),
        body={"instruction": "内部指令"},
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert "敏感" not in response.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("result", "status_code", "detail"),
    [
        (GenerationResult("processing"), 400, "Task is still processing"),
        (GenerationResult("failed"), 500, "generation failed"),
        (GenerationResult("missing"), 404, "Task not found"),
    ],
)
async def test_result_maps_service_status(
    result: GenerationResult,
    status_code: int,
    detail: str,
) -> None:
    generation = FakeGeneration(result=result)

    response = await request("GET", f"/result/{JOB_ID}", generation)

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert generation.job_ids == [JOB_ID]


@pytest.mark.anyio
async def test_result_returns_completed_image() -> None:
    response = await request(
        "GET",
        f"/result/{JOB_ID}",
        FakeGeneration(result=GenerationResult("completed", b"WEBP", "image/webp")),
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.content == b"WEBP"


@pytest.mark.anyio
async def test_result_sanitizes_upstream_error() -> None:
    response = await request(
        "GET",
        f"/result/{JOB_ID}",
        FakeGeneration(result=UpstreamError("内部错误")),
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "ComfyUI upstream error"}
    assert "内部错误" not in response.text


@pytest.mark.anyio
async def test_result_rejects_invalid_uuid() -> None:
    generation = FakeGeneration()

    response = await request("GET", "/result/not-a-uuid", generation)

    assert response.status_code == 422
    assert generation.job_ids == []
