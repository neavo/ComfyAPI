import asyncio
import contextlib
from unittest.mock import patch
from uuid import UUID

import httpx
import pytest

from app.telegram import (
    GenerationApi,
    GenerationApiError,
    GenerationResult,
    TransientGenerationApiError,
    JOB_TIMEOUT,
    QUEUE_CAPACITY,
    TelegramApi,
    TelegramBot,
    TelegramError,
    TelegramJob,
    extract_instruction,
)

JOB_ID = "550e8400-e29b-41d4-a716-446655440000"


def test_generation_api_submit_sends_exact_authenticated_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/new"
        assert request.headers["authorization"] == "Bearer TOKEN"
        assert request.read() == b'{"instruction":"\xe7\x94\xbb\xe4\xb8\x80\xe5\x8f\xaa\xe7\x8c\xab"}'
        return httpx.Response(202, json={"id": JOB_ID})

    async def run() -> str:
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:48188",
            headers={"Authorization": "Bearer TOKEN"},
            transport=httpx.MockTransport(handler),
        ) as client:
            return await GenerationApi(client).submit("画一只猫")

    assert asyncio.run(run()) == JOB_ID
    assert str(UUID(JOB_ID)) == JOB_ID


@pytest.mark.parametrize(
    ("status", "body", "headers", "expected"),
    [
        (200, b"WEBP", {"content-type": "image/webp"}, "completed"),
        (400, b'{"detail":"Task is still processing"}', {}, None),
        (404, b'{"detail":"Task not found"}', {}, None),
        (500, b'{"detail":"generation failed"}', {}, "failed"),
    ],
)
def test_generation_api_result_maps_public_protocol(
    status: int,
    body: bytes,
    headers: dict[str, str],
    expected: str | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == f"/result/{JOB_ID}"
        assert request.headers["authorization"] == "Bearer TOKEN"
        return httpx.Response(status, content=body, headers=headers)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:48188",
            headers={"Authorization": "Bearer TOKEN"},
            transport=httpx.MockTransport(handler),
        ) as client:
            return await GenerationApi(client).result(JOB_ID)

    result = asyncio.run(run())
    assert (result.status if result is not None else None) == expected
    if expected == "completed":
        assert result.image == b"WEBP"
        assert result.media_type == "image/webp"


@pytest.mark.parametrize("failure", [502, "network"])
def test_generation_api_result_treats_upstream_failures_as_transient(
    failure: int | str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            raise httpx.ConnectError("连接失败", request=request)
        return httpx.Response(502, json={"detail": "ComfyUI upstream error"})

    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:48188",
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(TransientGenerationApiError):
                await GenerationApi(client).result(JOB_ID)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("operation", "status", "body"),
    [
        ("submit", 202, b'{"id":"not-a-uuid"}'),
        ("submit", 202, b"{}"),
        ("submit", 202, b"not-json"),
        ("submit", 401, b'{"detail":"Unauthorized"}'),
        ("submit", 502, b'{"detail":"upstream"}'),
        ("result", 200, b""),
        ("result", 401, b'{"detail":"Unauthorized"}'),
        ("result", 400, b"not-json"),
        ("result", 400, b'{"detail":"unknown"}'),
        ("result", 404, b'{"detail":"unknown"}'),
        ("result", 418, b'{"detail":"unknown"}'),
        ("result", 503, b'{"detail":"unknown"}'),
    ],
)
def test_generation_api_rejects_damaged_or_unknown_protocol(
    operation: str, status: int, body: bytes
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body)

    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:48188",
            transport=httpx.MockTransport(handler),
        ) as client:
            api = GenerationApi(client)
            with pytest.raises(GenerationApiError):
                await getattr(api, operation)("画猫" if operation == "submit" else JOB_ID)

    asyncio.run(run())


def test_generation_api_submit_network_failure_is_not_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("连接失败", request=request)

    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:48188",
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(GenerationApiError):
                await GenerationApi(client).submit("画猫")

    asyncio.run(run())
    assert attempts == 1


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

    async def get_me(self) -> dict[str, str]:
        return {"username": "PainterBot"}

    async def get_updates(self, _: int | None) -> list[dict[str, object]]:
        await asyncio.Future()

    async def send_message(self, reply: dict[str, str], text: str) -> bool:
        self.messages.append((reply, text))
        return self.message_results.pop(0) if self.message_results else True

    async def send_photo(
        self, reply: dict[str, str], image: bytes, media_type: str
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

    async def submit(self, instruction: str) -> str:
        self.instructions.append(instruction)
        return f"job-{len(self.instructions)}"

    async def result(self, _: str) -> GenerationResult | None:
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
def test_only_leading_exact_bot_mention_becomes_instruction(
    text: str, expected: str | None
) -> None:
    assert extract_instruction(text, "PainterBot") == expected


@pytest.mark.parametrize(
    "candidate",
    [
        update("普通聊天"),
        update(is_bot=True),
        update(chat_type="private"),
        update(text=None),
        {"update_id": 1, "edited_message": {}},
    ],
)
def test_irrelevant_updates_do_not_queue_send_or_generate(
    candidate: dict[str, object],
) -> None:
    async def run() -> tuple[FakeApi, FakeGeneration, int]:
        api = FakeApi()
        generation = FakeGeneration()
        bot = TelegramBot(api, generation)
        await bot.accept_update(candidate, "PainterBot")
        return api, generation, bot.queue.qsize()

    api, generation, queued = asyncio.run(run())
    assert queued == 0
    assert api.messages == []
    assert api.photos == []
    assert generation.instructions == []


def test_valid_message_enters_queue_with_deadline_starting_at_enqueue() -> None:
    async def run() -> tuple[TelegramJob, float, float]:
        bot = TelegramBot(FakeApi(), FakeGeneration())
        loop = asyncio.get_running_loop()
        before = loop.time()
        await bot.accept_update(update(), "PainterBot")
        after = loop.time()
        return bot.queue.get_nowait(), before, after

    job, before, after = asyncio.run(run())
    assert job.instruction == "画一只猫"
    assert before + JOB_TIMEOUT <= job.deadline <= after + JOB_TIMEOUT


@pytest.mark.parametrize(
    ("text", "reply"),
    [
        ("@PainterBot", "请在 @PainterBot 后输入生图描述"),
        (
            "@PainterBot " + "猫" * 4097,
            "生图描述长度必须为 1 至 4096 个字符",
        ),
    ],
)
def test_empty_or_oversized_instruction_replies_without_queueing(
    text: str, reply: str
) -> None:
    async def run() -> tuple[FakeApi, int]:
        api = FakeApi()
        bot = TelegramBot(api, FakeGeneration())
        await bot.accept_update(update(text), "PainterBot")
        return api, bot.queue.qsize()

    api, queued = asyncio.run(run())
    assert queued == 0
    assert [text for _, text in api.messages] == [reply]


def test_twenty_waiting_jobs_fill_queue_and_next_message_gets_busy_reply() -> None:
    async def run() -> tuple[FakeApi, TelegramBot]:
        api = FakeApi()
        bot = TelegramBot(api, FakeGeneration())
        for number in range(QUEUE_CAPACITY + 1):
            await bot.accept_update(
                update(update_id=number, message_id=number), "PainterBot"
            )
        return api, bot

    api, bot = asyncio.run(run())
    assert bot.queue.qsize() == QUEUE_CAPACITY == 20
    assert [text for _, text in api.messages] == ["当前生成任务较多，请稍后再试"]


def test_two_workers_bound_concurrency_and_leave_third_job_waiting() -> None:
    class BatchApi(FakeApi):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def get_updates(self, _: int | None) -> list[dict[str, object]]:
            self.calls += 1
            if self.calls == 1:
                return [update(update_id=i, message_id=i) for i in range(3)]
            await asyncio.Future()

    class BlockingGeneration(FakeGeneration):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.maximum = 0
            self.two_started = asyncio.Event()
            self.release = asyncio.Event()

        async def submit(self, instruction: str) -> str:
            self.instructions.append(instruction)
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            if self.active == 2:
                self.two_started.set()
            await self.release.wait()
            self.active -= 1
            return f"job-{len(self.instructions)}"

        async def result(self, _: str) -> GenerationResult:
            return GenerationResult("completed", b"WEBP", "image/webp")

    async def run() -> tuple[int, int]:
        api = BatchApi()
        generation = BlockingGeneration()
        bot = TelegramBot(api, generation)
        task = asyncio.create_task(bot.run())
        await asyncio.wait_for(generation.two_started.wait(), 1)
        state = generation.maximum, bot.queue.qsize()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return state

    maximum, queued = asyncio.run(run())
    assert maximum == 2
    assert queued == 1


def test_long_poll_transient_failure_waits_three_seconds_but_permanent_failure_stops(
    monkeypatch,
) -> None:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    class PollApi(FakeApi):
        def __init__(self, transient_first: bool) -> None:
            super().__init__()
            self.transient_first = transient_first
            self.calls = 0

        async def get_updates(self, _: int | None) -> list[dict[str, object]]:
            self.calls += 1
            if self.transient_first and self.calls == 1:
                raise TelegramError("瞬时轮询失败", retryable=True)
            raise TelegramError("永久轮询失败")

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    async def run() -> None:
        transient = TelegramBot(PollApi(True), FakeGeneration())
        with pytest.raises(ExceptionGroup):
            await transient.run()
        permanent = TelegramBot(PollApi(False), FakeGeneration())
        with pytest.raises(ExceptionGroup):
            await permanent.run()

    asyncio.run(run())
    assert sleeps == [3.0]


def test_telegram_429_uses_retry_after(monkeypatch) -> None:
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

    async def run() -> bool:
        async with httpx.AsyncClient(
            base_url="https://api.telegram.org",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await TelegramApi("TOKEN", client).send_message({}, "状态")

    assert asyncio.run(run()) is True
    assert attempts == 2
    assert sleeps == [7.0]


@pytest.mark.parametrize(
    ("failure", "expected_attempts", "expected_sleeps"),
    [("network", 3, [3.0, 3.0]), (400, 1, [])],
)
def test_outbound_retries_only_transient_failures(
    failure: str | int,
    expected_attempts: int,
    expected_sleeps: list[float],
    monkeypatch,
) -> None:
    attempts = 0
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if failure == "network":
            raise httpx.ConnectError("连接失败", request=request)
        return httpx.Response(failure, json={"ok": False, "description": "失败"})

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    async def run() -> bool:
        async with httpx.AsyncClient(
            base_url="https://api.telegram.org",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await TelegramApi("TOKEN", client).send_message({}, "状态")

    assert asyncio.run(run()) is False
    assert attempts == expected_attempts
    assert sleeps == expected_sleeps


def test_failed_processing_notice_does_not_prevent_generation() -> None:
    async def run() -> tuple[FakeApi, FakeGeneration]:
        api = FakeApi()
        api.message_results = [False]
        generation = FakeGeneration()
        bot = TelegramBot(api, generation)
        await bot.accept_update(update("@PainterBot 画猫"), "PainterBot")
        await drain_queue(bot)
        return api, generation

    api, generation = asyncio.run(run())
    assert generation.instructions == ["画猫"]
    assert len(api.photos) == 1


def test_result_polling_waits_three_seconds_before_every_query(
    monkeypatch,
) -> None:
    events: list[tuple[str, object]] = []

    async def record_sleep(delay: float) -> None:
        events.append(("等待", delay))

    class ObservedGeneration(FakeGeneration):
        async def result(self, job_id: str) -> GenerationResult | None:
            events.append(("查询", job_id))
            return await super().result(job_id)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    generation = ObservedGeneration(
        [
            None,
            TransientGenerationApiError("短暂失败"),
            GenerationResult("completed", b"WEBP", "image/webp"),
        ]
    )

    async def run() -> None:
        bot = TelegramBot(FakeApi(), generation)
        await bot.accept_update(update("@PainterBot 画猫"), "PainterBot")
        await drain_queue(bot, poll_delay=3.0)

    asyncio.run(run())
    assert events == [
        ("等待", 3.0),
        ("查询", "job-1"),
        ("等待", 3.0),
        ("查询", "job-1"),
        ("等待", 3.0),
        ("查询", "job-1"),
    ]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (GenerationResult("failed"), "生成失败，请重试"),
        (GenerationApiError("协议异常"), "生图服务暂时异常，请稍后重试"),
    ],
)
def test_generation_failure_returns_stable_message(
    result: GenerationResult | Exception, expected: str
) -> None:
    generation = FakeGeneration([result])

    async def run() -> FakeApi:
        api = FakeApi()
        bot = TelegramBot(api, generation)
        await bot.accept_update(update("@PainterBot 画猫"), "PainterBot")
        await drain_queue(bot)
        return api

    api = asyncio.run(run())
    assert [text for _, text in api.messages] == ["正在生成…", expected]
    assert api.photos == []


def test_submit_permanent_failure_returns_stable_message() -> None:
    class FailedSubmit(FakeGeneration):
        async def submit(self, instruction: str) -> str:
            self.instructions.append(instruction)
            raise GenerationApiError("提交失败")

    async def run() -> FakeApi:
        api = FakeApi()
        bot = TelegramBot(api, FailedSubmit())
        await bot.accept_update(update("@PainterBot 画猫"), "PainterBot")
        await drain_queue(bot)
        return api

    api = asyncio.run(run())
    assert [text for _, text in api.messages] == [
        "正在生成…",
        "生图服务暂时异常，请稍后重试",
    ]


def test_expired_job_times_out_and_worker_processes_next_job() -> None:
    class TimeoutGeneration(FakeGeneration):
        async def submit(self, instruction: str) -> str:
            self.instructions.append(instruction)
            if instruction == "超时任务":
                await asyncio.Future()
            return "job-ok"

        async def result(self, _: str) -> GenerationResult:
            return GenerationResult("completed", b"WEBP", "image/webp")

    async def run() -> tuple[FakeApi, TimeoutGeneration]:
        api = FakeApi()
        generation = TimeoutGeneration()
        bot = TelegramBot(api, generation)
        loop = asyncio.get_running_loop()
        bot.queue.put_nowait(TelegramJob(-1001, 1, None, "超时任务", loop.time()))
        bot.queue.put_nowait(TelegramJob(-1001, 2, None, "后续任务", loop.time() + 60))
        await drain_queue(bot)
        return api, generation

    api, generation = asyncio.run(run())
    assert "生成超时，请稍后重试" in [text for _, text in api.messages]
    assert generation.instructions == ["超时任务", "后续任务"]
    assert len(api.photos) == 1


def test_unexpected_worker_error_does_not_block_next_job() -> None:
    class FlakyGeneration(FakeGeneration):
        async def submit(self, instruction: str) -> str:
            self.instructions.append(instruction)
            if instruction == "崩溃任务":
                raise RuntimeError("意外错误")
            return "job-ok"

        async def result(self, _: str) -> GenerationResult:
            return GenerationResult("completed", b"WEBP", "image/webp")

    async def run() -> tuple[FakeApi, FlakyGeneration]:
        api = FakeApi()
        generation = FlakyGeneration()
        bot = TelegramBot(api, generation)
        loop = asyncio.get_running_loop()
        for message_id, instruction in enumerate(("崩溃任务", "后续任务"), 1):
            bot.queue.put_nowait(
                TelegramJob(-1001, message_id, None, instruction, loop.time() + 60)
            )
        await drain_queue(bot)
        return api, generation

    api, generation = asyncio.run(run())
    assert generation.instructions == ["崩溃任务", "后续任务"]
    assert len(api.photos) == 1


def test_completed_result_replies_in_topic_with_webp_upload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {}})

    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="https://api.telegram.org",
            transport=httpx.MockTransport(handler),
        ) as client:
            bot = TelegramBot(TelegramApi("TOKEN", client), FakeGeneration())
            await bot.accept_update(update("@PainterBot 画猫"), "PainterBot")
            await drain_queue(bot)

    asyncio.run(run())
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
