import asyncio
import json
from collections.abc import Callable
from uuid import UUID

import httpx
import pytest

from app.main import app
from app.service import GenerationService, Settings, resolve_workflow


def workflow_data() -> dict[str, object]:
    return {
        "10": {
            "inputs": {"text": "原始指令"},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "API Instruction"},
        },
        "20": {
            "inputs": {},
            "_meta": {"title": "API Output"},
        },
    }


def response_json(status: int, data: object) -> httpx.Response:
    return httpx.Response(status, json=data)


def request(
    method: str,
    path: str,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    token: str | None = "secret",
    body: object | None = None,
    llm_handler: Callable[[httpx.Request], httpx.Response] | None = None,
) -> httpx.Response:
    async def run() -> httpx.Response:
        def route(req: httpx.Request) -> httpx.Response:
            if req.url.host == "llm.local":
                if llm_handler is not None:
                    return llm_handler(req)
                instruction = json.loads(req.content)["messages"][1]["content"]
                return response_json(
                    200, {"choices": [{"message": {"content": instruction}}]}
                )
            return handler(req)

        upstream = httpx.AsyncClient(
            base_url="http://comfy.local",
            transport=httpx.MockTransport(route),
        )
        settings = Settings(
            "secret",
            "http://comfy.local",
            "http://llm.local/v1/chat/completions",
            "llm-secret",
            "prompt-model",
        )
        app.state.settings = settings
        app.state.generation = GenerationService(
            upstream,
            upstream,
            settings,
            "系统指令",
            resolve_workflow(workflow_data()),
        )
        headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
        async with httpx.AsyncClient(
            base_url="http://api.local",
            transport=httpx.ASGITransport(app=app),
        ) as client:
            result = await client.request(method, path, headers=headers, json=body)
        await upstream.aclose()
        return result

    return asyncio.run(run())


def unused_upstream(_: httpx.Request) -> httpx.Response:
    raise AssertionError("本场景不应访问上游")


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/new", {"instruction": "雨夜"}),
        ("GET", "/result/550e8400-e29b-41d4-a716-446655440000", None),
    ],
)
@pytest.mark.parametrize("token", [None, "wrong"])
def test_missing_or_wrong_token_returns_401(
    method: str, path: str, body: object | None, token: str | None
) -> None:
    response = request(method, path, unused_upstream, token=token, body=body)

    assert response.status_code == 401


def test_new_accepts_instruction_and_returns_only_uuid() -> None:
    captured: dict[str, object] = {}

    def upstream(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return response_json(
            200, {"prompt_id": captured["prompt_id"], "number": 1, "node_errors": {}}
        )

    def llm(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        assert payload["messages"] == [
            {"role": "system", "content": "系统指令"},
            {"role": "user", "content": "生成雨夜街道"},
        ]
        return response_json(
            200, {"choices": [{"message": {"content": "a rainy neon street"}}]}
        )

    response = request(
        "POST",
        "/new",
        upstream,
        body={"instruction": "  生成雨夜街道  "},
        llm_handler=llm,
    )

    assert response.status_code == 202
    assert set(response.json()) == {"id"}
    UUID(response.json()["id"])
    assert (
        captured["prompt"]["10"]["inputs"]["text"]
        == """a rainy neon street

safe
(mature:-1), (aged down:1)
(simple background:-1.25)
(shiny skin:-1)
(flat color, anime coloring:2)
rim light, light particles, cinematic lighting
depth of field, strong perspective, blurry background"""
    )
    assert captured["extra_data"] == {"extra_pnginfo": {"workflow": captured["prompt"]}}
    assert "生成雨夜街道" not in response.text


def test_new_does_not_submit_to_comfyui_when_llm_fails() -> None:
    def llm(_: httpx.Request) -> httpx.Response:
        return response_json(200, {"choices": []})

    response = request(
        "POST",
        "/new",
        unused_upstream,
        body={"instruction": "内部指令"},
        llm_handler=llm,
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM upstream error"}
    assert "内部指令" not in response.text


def test_new_retries_one_transient_llm_failure_and_returns_202(monkeypatch) -> None:
    attempts = 0

    async def no_wait(_: float) -> None:
        return None

    def llm(req: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return response_json(503, {"error": "暂时失败"})
        return response_json(
            200, {"choices": [{"message": {"content": "a rainy street"}}]}
        )

    def upstream(req: httpx.Request) -> httpx.Response:
        prompt_id = json.loads(req.content)["prompt_id"]
        return response_json(200, {"prompt_id": prompt_id})

    monkeypatch.setattr(asyncio, "sleep", no_wait)

    response = request(
        "POST",
        "/new",
        upstream,
        body={"instruction": "雨夜街道"},
        llm_handler=llm,
    )

    assert response.status_code == 202
    assert attempts == 2


@pytest.mark.parametrize(
    "body",
    [
        {"instruction": ""},
        {"instruction": " "},
        {"instruction": "x" * 4097},
        {"instruction": "雨夜", "seed": 1},
    ],
)
def test_new_rejects_invalid_request_body(body: object) -> None:
    response = request("POST", "/new", unused_upstream, body=body)

    assert response.status_code == 422


@pytest.mark.parametrize("failure", ["status", "disconnect", "timeout", "different_id"])
def test_new_maps_upstream_failures_to_sanitized_502(failure: str) -> None:
    def upstream(req: httpx.Request) -> httpx.Response:
        prompt_id = json.loads(req.content)["prompt_id"]
        if failure == "status":
            return response_json(
                400, {"node_errors": {"10": {"errors": ["敏感节点错误"]}}}
            )
        if failure == "disconnect":
            raise httpx.ConnectError("连接失败", request=req)
        if failure == "timeout":
            raise httpx.ReadTimeout("超时", request=req)
        return response_json(200, {"prompt_id": prompt_id + "-different"})

    response = request("POST", "/new", upstream, body={"instruction": "内部指令"})

    assert response.status_code == 502
    assert response.json() == {"detail": "ComfyUI upstream error"}
    assert "内部指令" not in response.text
    assert "敏感节点错误" not in response.text


JOB_ID = "550e8400-e29b-41d4-a716-446655440000"


def test_result_returns_processing_from_shared_queue() -> None:
    def upstream(req: httpx.Request) -> httpx.Response:
        if req.url.path.startswith("/history/"):
            return response_json(200, {})
        return response_json(
            200, {"queue_running": [], "queue_pending": [[1, JOB_ID, {}, {}, []]]}
        )

    response = request("GET", f"/result/{JOB_ID}", upstream)

    assert response.status_code == 400
    assert response.json() == {"detail": "Task is still processing"}


def test_result_returns_first_completed_image_without_history_leak() -> None:
    requested_view: dict[str, str] = {}

    def upstream(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/view":
            requested_view.update(req.url.params)
            return httpx.Response(
                200, content=b"WEBP", headers={"content-type": "image/webp"}
            )
        return response_json(
            200,
            {
                JOB_ID: {
                    "status": {"status_str": "success", "completed": True},
                    "outputs": {
                        "20": {
                            "images": [
                                {
                                    "filename": "20121212_00001_.webp",
                                    "subfolder": "api",
                                    "type": "output",
                                },
                                {
                                    "filename": "20121212_00002_.webp",
                                    "subfolder": "api",
                                    "type": "output",
                                },
                            ]
                        }
                    },
                    "prompt": ["绝不泄露的完整工作流"],
                }
            },
        )

    response = request("GET", f"/result/{JOB_ID}", upstream)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.content == b"WEBP"
    assert requested_view == {
        "filename": "20121212_00001_.webp",
        "subfolder": "api",
        "type": "output",
    }


def test_result_returns_failed_without_upstream_error_details() -> None:
    def upstream(_: httpx.Request) -> httpx.Response:
        return response_json(
            200,
            {
                JOB_ID: {
                    "status": {"status_str": "error"},
                    "outputs": {},
                    "error": "节点堆栈",
                }
            },
        )

    response = request("GET", f"/result/{JOB_ID}", upstream)

    assert response.status_code == 500
    assert response.json() == {"detail": "generation failed"}


def test_result_maps_image_read_failure_to_sanitized_502() -> None:
    def upstream(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/view":
            return response_json(503, {"error": "内部图片错误"})
        return response_json(
            200,
            {
                JOB_ID: {
                    "status": {"status_str": "success"},
                    "outputs": {
                        "20": {
                            "images": [
                                {
                                    "filename": "result.png",
                                    "subfolder": "api",
                                    "type": "output",
                                }
                            ]
                        }
                    },
                }
            },
        )

    response = request("GET", f"/result/{JOB_ID}", upstream)

    assert response.status_code == 502
    assert response.json() == {"detail": "ComfyUI upstream error"}
    assert "内部图片错误" not in response.text


def test_result_returns_404_when_history_and_queue_do_not_contain_job() -> None:
    def upstream(req: httpx.Request) -> httpx.Response:
        if req.url.path.startswith("/history/"):
            return response_json(200, {})
        return response_json(200, {"queue_running": [], "queue_pending": []})

    response = request("GET", f"/result/{JOB_ID}", upstream)

    assert response.status_code == 404


@pytest.mark.parametrize(
    "job_id",
    [JOB_ID.upper(), JOB_ID.replace("-", ""), "{" + JOB_ID + "}", "not-a-uuid"],
)
def test_result_rejects_noncanonical_uuid(job_id: str) -> None:
    response = request("GET", f"/result/{job_id}", unused_upstream)

    assert response.status_code == 422


def test_result_maps_damaged_upstream_response_to_502() -> None:
    response = request(
        "GET",
        f"/result/{JOB_ID}",
        lambda _: response_json(
            200, {JOB_ID: {"status": {"status_str": "success"}, "outputs": {}}}
        ),
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "ComfyUI upstream error"}


@pytest.mark.parametrize("failure", ["status", "disconnect", "timeout", "invalid_json"])
def test_result_maps_query_failures_to_sanitized_502(failure: str) -> None:
    def upstream(req: httpx.Request) -> httpx.Response:
        if failure == "status":
            return response_json(500, {"error": "内部错误"})
        if failure == "disconnect":
            raise httpx.ConnectError("连接失败", request=req)
        if failure == "timeout":
            raise httpx.ReadTimeout("超时", request=req)
        return httpx.Response(200, text="not json")

    response = request("GET", f"/result/{JOB_ID}", upstream)

    assert response.status_code == 502
    assert response.json() == {"detail": "ComfyUI upstream error"}
    assert "内部错误" not in response.text
