import asyncio
import contextlib

import httpx
import pytest

from app.service import GenerationResult, TransientUpstreamError
from app.telegram import (
    JOB_TIMEOUT,
    QUEUE_CAPACITY,
    TelegramApi,
    TelegramBot,
    TelegramError,
    TelegramJob,
    extract_instruction,
)


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

    async def history_result(self, _: str) -> GenerationResult | None:
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


async def drain_queue(bot: TelegramBot) -> None:
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

        async def history_result(self, _: str) -> GenerationResult:
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


def test_missing_or_transient_history_uses_two_three_five_five_delays(
    monkeypatch,
) -> None:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    generation = FakeGeneration(
        [
            None,
            TransientUpstreamError("短暂失败"),
            None,
            None,
            GenerationResult("completed", b"WEBP", "image/webp"),
        ]
    )

    async def run() -> None:
        bot = TelegramBot(FakeApi(), generation)
        await bot.accept_update(update("@PainterBot 画猫"), "PainterBot")
        await drain_queue(bot)

    asyncio.run(run())
    assert sleeps == [2.0, 3.0, 5.0, 5.0]


def test_expired_job_times_out_and_worker_processes_next_job() -> None:
    class TimeoutGeneration(FakeGeneration):
        async def submit(self, instruction: str) -> str:
            self.instructions.append(instruction)
            if instruction == "超时任务":
                await asyncio.Future()
            return "job-ok"

        async def history_result(self, _: str) -> GenerationResult:
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

        async def history_result(self, _: str) -> GenerationResult:
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
