import asyncio
import contextlib
import json
from unittest.mock import patch

import httpx
import pytest

from app.telegram import (
    JOB_TIMEOUT,
    QUEUE_CAPACITY,
    GenerationApi,
    GenerationApiError,
    GenerationResult,
    TelegramApi,
    TelegramBot,
    TelegramError,
    TelegramJob,
    TransientGenerationApiError,
    extract_instruction,
)

JOB_ID = "550e8400-e29b-41d4-a716-446655440000"


@pytest.mark.anyio
async def test_generation_api_submits_authenticated_instruction() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(202, json={"id": JOB_ID})

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:48188",
        headers={"Authorization": "Bearer TOKEN"},
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await GenerationApi(client).submit("画一只猫")

    assert result == JOB_ID
    assert captured == {
        "method": "POST",
        "path": "/new",
        "authorization": "Bearer TOKEN",
        "body": {"instruction": "画一只猫"},
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (200, b"WEBP", "completed"),
        (202, b"", None),
        (404, b"{}", "missing"),
        (500, b"{}", "failed"),
    ],
)
async def test_generation_api_maps_result_status(
    status: int,
    body: bytes,
    expected: str | None,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            content=body,
            headers={"content-type": "image/webp"},
        )

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:48188",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await GenerationApi(client).result(JOB_ID)

    assert (result.status if result else None) == expected
    if result and result.status == "completed":
        assert (result.image, result.media_type) == (b"WEBP", "image/webp")


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["network", 502])
async def test_generation_api_marks_query_outage_transient(
    failure: str | int,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            raise httpx.ConnectError("连接失败", request=request)
        return httpx.Response(502)

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:48188",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(TransientGenerationApiError):
            await GenerationApi(client).result(JOB_ID)


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["submit", "result"])
async def test_generation_api_rejects_broken_contract(operation: str) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return (
            httpx.Response(202, json={})
            if operation == "submit"
            else httpx.Response(418)
        )

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:48188",
        transport=httpx.MockTransport(handler),
    ) as client:
        api = GenerationApi(client)
        with pytest.raises(GenerationApiError):
            await getattr(api, operation)("画猫" if operation == "submit" else JOB_ID)


def update(
    text: object = "@PainterBot 画一只猫",
    *,
    update_id: int = 1,
    message_id: int = 7,
    chat_type: str = "supergroup",
    is_bot: bool = False,
    thread_id: int | None = 11,
) -> dict[str, object]:
    message: dict[str, object] = {
        "message_id": message_id,
        "from": {"is_bot": is_bot},
        "chat": {"id": -1001, "type": chat_type},
        "text": text,
    }
    if thread_id is not None:
        message["message_thread_id"] = thread_id
    return {"update_id": update_id, "message": message}


class FakeApi:
    def __init__(self) -> None:
        self.messages: list[tuple[dict[str, str], str]] = []
        self.photos: list[tuple[dict[str, str], bytes, str]] = []
        self.message_results: list[bool] = []

    async def get_username(self) -> str:
        return "PainterBot"

    async def get_updates(self, _: int | None) -> list[dict[str, object]]:
        await asyncio.Future()

    async def send_message(self, reply: dict[str, str], text: str) -> bool:
        self.messages.append((reply, text))
        return self.message_results.pop(0) if self.message_results else True

    async def send_photo(
        self,
        reply: dict[str, str],
        image: bytes,
        media_type: str,
    ) -> bool:
        self.photos.append((reply, image, media_type))
        return True


class FakeGeneration:
    def __init__(self, results: list[object] | None = None) -> None:
        self.instructions: list[str] = []
        self.results = iter(
            results
            if results is not None
            else [GenerationResult("completed", b"WEBP", "image/webp")]
        )
        self.result_calls = 0

    async def submit(self, instruction: str) -> str:
        self.instructions.append(instruction)
        return f"job-{len(self.instructions)}"

    async def result(self, _: str) -> GenerationResult | None:
        self.result_calls += 1
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


async def drain_queue(bot: TelegramBot, poll_delay: float = 0) -> None:
    with patch("app.telegram.RESULT_POLL_DELAY", poll_delay):
        worker = asyncio.create_task(bot.worker(1))
        await asyncio.wait_for(bot.queue.join(), 1)
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("@PainterBot 画一只猫", "画一只猫"),
        ("@painterbot\n雨夜街道", "雨夜街道"),
        ("@PainterBot", ""),
        ("请让 @PainterBot 画一只猫", None),
        ("@PainterBotExtra 画一只猫", None),
    ],
)
def test_only_leading_exact_mention_becomes_instruction(
    text: str,
    expected: str | None,
) -> None:
    assert extract_instruction(text, "PainterBot") == expected


@pytest.mark.anyio
@pytest.mark.parametrize(
    "candidate",
    [
        update("普通聊天"),
        update(is_bot=True),
        update(text=None),
        {"update_id": 1, "edited_message": {}},
    ],
)
async def test_irrelevant_update_has_no_effect(candidate: dict[str, object]) -> None:
    api = FakeApi()
    generation = FakeGeneration()
    bot = TelegramBot(api, generation)

    await bot.accept_update(candidate, "PainterBot")

    assert bot.queue.empty()
    assert (api.messages, api.photos, generation.instructions) == ([], [], [])


@pytest.mark.anyio
async def test_private_text_generates_and_replies_with_photo() -> None:
    api = FakeApi()
    generation = FakeGeneration()
    bot = TelegramBot(api, generation)

    await bot.accept_update(
        update("  画一只猫  ", chat_type="private", thread_id=None),
        "PainterBot",
    )
    await drain_queue(bot)

    assert generation.instructions == ["画一只猫"]
    assert [text for _, text in api.messages] == ["正在生成…"]
    assert api.photos == [
        (
            {
                "chat_id": "-1001",
                "reply_parameters": '{"message_id": 7, "allow_sending_without_reply": true}',
            },
            b"WEBP",
            "image/webp",
        )
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("text", ["/start", "/START@painterbot payload"])
async def test_private_start_explains_usage(text: str) -> None:
    api = FakeApi()
    bot = TelegramBot(api, FakeGeneration())

    await bot.accept_update(
        update(text, chat_type="private", thread_id=None),
        "PainterBot",
    )

    assert bot.queue.empty()
    assert [text for _, text in api.messages] == ["请直接发送生图描述"]


@pytest.mark.anyio
async def test_private_unknown_command_is_ignored() -> None:
    api = FakeApi()
    bot = TelegramBot(api, FakeGeneration())

    await bot.accept_update(
        update("/help", chat_type="private", thread_id=None),
        "PainterBot",
    )

    assert bot.queue.empty()
    assert api.messages == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("text", "chat_type", "reply"),
    [
        ("@PainterBot", "supergroup", "请在 @PainterBot 后输入生图描述"),
        (
            "@PainterBot " + "猫" * 4097,
            "supergroup",
            "生图描述长度必须为 1 至 4096 个字符",
        ),
        ("   ", "private", "请输入生图描述"),
        ("猫" * 4097, "private", "生图描述长度必须为 1 至 4096 个字符"),
    ],
)
async def test_invalid_instruction_replies_without_queueing(
    text: str,
    chat_type: str,
    reply: str,
) -> None:
    api = FakeApi()
    bot = TelegramBot(api, FakeGeneration())

    await bot.accept_update(
        update(text, chat_type=chat_type, thread_id=None),
        "PainterBot",
    )

    assert bot.queue.empty()
    assert [text for _, text in api.messages] == [reply]


@pytest.mark.anyio
async def test_full_queue_replies_busy() -> None:
    api = FakeApi()
    bot = TelegramBot(api, FakeGeneration())

    for number in range(QUEUE_CAPACITY + 1):
        await bot.accept_update(
            update(update_id=number, message_id=number),
            "PainterBot",
        )

    assert bot.queue.qsize() == QUEUE_CAPACITY
    assert [text for _, text in api.messages] == ["当前生成任务较多，请稍后再试"]


@pytest.mark.anyio
async def test_long_poll_retries_transient_failure(monkeypatch) -> None:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    class PollApi(FakeApi):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def get_updates(self, _: int | None) -> list[dict[str, object]]:
            self.calls += 1
            if self.calls == 1:
                raise TelegramError("瞬时失败", retryable=True)
            raise TelegramError("永久失败")

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    with pytest.raises(TelegramError, match="永久失败"):
        await TelegramBot(PollApi(), FakeGeneration())._receive("PainterBot")

    assert sleeps == [3.0]


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

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    async with httpx.AsyncClient(
        base_url="https://api.telegram.org",
        transport=httpx.MockTransport(handler),
    ) as client:
        delivered = await TelegramApi("TOKEN", client).send_message({}, "状态")

    assert delivered is True
    assert attempts == 2
    assert sleeps == [7]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failure", "expected_attempts"),
    [("network", 3), (400, 1)],
)
async def test_outbound_retries_only_transient_failure(
    failure: str | int,
    expected_attempts: int,
    monkeypatch,
) -> None:
    attempts = 0

    async def no_wait(_: float) -> None:
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if failure == "network":
            raise httpx.ConnectError("连接失败", request=request)
        return httpx.Response(failure, json={"ok": False, "description": "失败"})

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    async with httpx.AsyncClient(
        base_url="https://api.telegram.org",
        transport=httpx.MockTransport(handler),
    ) as client:
        delivered = await TelegramApi("TOKEN", client).send_message({}, "状态")

    assert delivered is False
    assert attempts == expected_attempts


@pytest.mark.anyio
async def test_failed_processing_notice_does_not_stop_generation() -> None:
    api = FakeApi()
    api.message_results = [False]
    generation = FakeGeneration()
    bot = TelegramBot(api, generation)

    await bot.accept_update(update("@PainterBot 画猫"), "PainterBot")
    await drain_queue(bot)

    assert generation.instructions == ["画猫"]
    assert len(api.photos) == 1


@pytest.mark.anyio
async def test_polling_survives_pending_and_transient_results() -> None:
    api = FakeApi()
    generation = FakeGeneration(
        [
            None,
            TransientGenerationApiError("短暂失败"),
            GenerationResult("completed", b"WEBP", "image/webp"),
        ]
    )
    bot = TelegramBot(api, generation)

    await bot.accept_update(update("@PainterBot 画猫"), "PainterBot")
    await drain_queue(bot)

    assert generation.result_calls == 3
    assert len(api.photos) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (GenerationResult("failed"), "生成失败，请重试"),
        (GenerationResult("missing"), "生成任务已丢失，请重试"),
        (GenerationApiError("协议异常"), "生图服务暂时异常，请稍后重试"),
    ],
)
async def test_generation_failure_returns_stable_message(
    result: GenerationResult | Exception,
    expected: str,
) -> None:
    api = FakeApi()
    bot = TelegramBot(api, FakeGeneration([result]))

    await bot.accept_update(update("@PainterBot 画猫"), "PainterBot")
    await drain_queue(bot)

    assert [text for _, text in api.messages] == ["正在生成…", expected]
    assert api.photos == []


@pytest.mark.anyio
async def test_expired_job_does_not_block_next_job() -> None:
    class TimeoutGeneration(FakeGeneration):
        async def submit(self, instruction: str) -> str:
            self.instructions.append(instruction)
            if instruction == "超时任务":
                await asyncio.Future()
            return "job-ok"

    api = FakeApi()
    generation = TimeoutGeneration()
    bot = TelegramBot(api, generation)
    loop = asyncio.get_running_loop()
    bot.queue.put_nowait(TelegramJob(-1001, 1, None, "超时任务", loop.time()))
    bot.queue.put_nowait(
        TelegramJob(-1001, 2, None, "后续任务", loop.time() + JOB_TIMEOUT)
    )

    await drain_queue(bot)

    assert "生成超时，请稍后重试" in [text for _, text in api.messages]
    assert generation.instructions == ["超时任务", "后续任务"]
    assert len(api.photos) == 1


@pytest.mark.anyio
async def test_completed_result_replies_in_topic_with_webp() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {}})

    async with httpx.AsyncClient(
        base_url="https://api.telegram.org",
        transport=httpx.MockTransport(handler),
    ) as client:
        bot = TelegramBot(TelegramApi("TOKEN", client), FakeGeneration())
        await bot.accept_update(update("@PainterBot 画猫"), "PainterBot")
        await drain_queue(bot)

    assert [request.url.path.rsplit("/", 1)[-1] for request in requests] == [
        "sendMessage",
        "sendPhoto",
    ]
    photo = requests[1].content
    assert b"WEBP" in photo
    assert b'filename="result.webp"' in photo
    assert b"image/webp" in photo
    assert b'"message_id": 7' in photo
    assert b"11" in photo
