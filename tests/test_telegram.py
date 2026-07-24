import asyncio
import contextlib
from unittest.mock import patch

import pytest

from app.telegram import (
    JOB_TIMEOUT,
    QUEUE_CAPACITY,
    ImageToTextJob,
    TelegramBot,
    TextToImageJob,
    extract_image,
    extract_instruction,
    split_message,
)
from app.telegram_api import (
    MAX_IMAGE_BYTES,
    BackendApiError,
    ImageApiResult,
    TextApiResult,
    TransientBackendApiError,
)


def update(
    text: object = "@PainterBot 画一只猫",
    *,
    caption: object | None = None,
    photo: object | None = None,
    document: object | None = None,
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
    }
    if text is not None:
        message["text"] = text
    if caption is not None:
        message["caption"] = caption
    if photo is not None:
        message["photo"] = photo
    if document is not None:
        message["document"] = document
    if thread_id is not None:
        message["message_thread_id"] = thread_id
    return {"update_id": update_id, "message": message}


class FakeTelegram:
    def __init__(self) -> None:
        self.messages: list[tuple[dict[str, str], str]] = []
        self.photos: list[tuple[dict[str, str], bytes, str]] = []
        self.downloads: list[str] = []
        self.image = b"IMAGE"
        self.message_results: list[bool] = []

    async def get_username(self) -> str:
        return "PainterBot"

    async def get_updates(self, _: int | None) -> list[dict[str, object]]:
        await asyncio.Future()

    async def download_file(self, file_id: str) -> bytes:
        self.downloads.append(file_id)
        return self.image

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


class FakeBackend:
    def __init__(
        self,
        image_results: list[object] | None = None,
        text_results: list[object] | None = None,
    ) -> None:
        self.instructions: list[str] = []
        self.images: list[tuple[bytes, str]] = []
        self.image_results = iter(
            image_results
            if image_results is not None
            else [ImageApiResult("completed", b"WEBP", "image/webp")]
        )
        self.text_results = iter(
            text_results
            if text_results is not None
            else [TextApiResult("completed", "reverse prompt")]
        )
        self.result_calls = 0

    async def submit_text_to_image(self, instruction: str) -> str:
        self.instructions.append(instruction)
        return f"image-{len(self.instructions)}"

    async def submit_image_to_text(self, image: bytes, media_type: str) -> str:
        self.images.append((image, media_type))
        return f"text-{len(self.images)}"

    async def text_to_image_result(self, _: str) -> ImageApiResult | None:
        self.result_calls += 1
        result = next(self.image_results)
        if isinstance(result, Exception):
            raise result
        return result

    async def image_to_text_result(self, _: str) -> TextApiResult | None:
        self.result_calls += 1
        result = next(self.text_results)
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


def test_image_extraction_selects_largest_photo_and_image_documents() -> None:
    photo = extract_image(
        {
            "photo": [
                {"file_id": "small", "file_size": 100, "width": 100, "height": 100},
                {"file_id": "large", "file_size": 500, "width": 500, "height": 500},
            ]
        }
    )
    document = extract_image(
        {
            "document": {
                "file_id": "original",
                "file_size": 600,
                "mime_type": "image/png",
            }
        }
    )

    assert photo is not None and (photo.file_id, photo.media_type) == (
        "large",
        "image/jpeg",
    )
    assert document is not None and (document.file_id, document.media_type) == (
        "original",
        "image/png",
    )
    assert (
        extract_image({"document": {"file_id": "text", "mime_type": "text/plain"}})
        is None
    )


def test_long_text_is_split_without_data_loss() -> None:
    text = ("word " * 2000).strip()

    chunks = split_message(text, 100)

    assert all(0 < len(chunk) <= 100 for chunk in chunks)
    assert " ".join(chunks) == text


@pytest.mark.anyio
async def test_private_text_generates_and_replies_with_photo() -> None:
    telegram = FakeTelegram()
    backend = FakeBackend()
    bot = TelegramBot(telegram, backend)

    await bot.accept_update(
        update("  画一只猫  ", chat_type="private", thread_id=None),
        "PainterBot",
    )
    await drain_queue(bot)

    assert backend.instructions == ["画一只猫"]
    assert [text for _, text in telegram.messages] == ["正在生成…"]
    assert telegram.photos[0][1:] == (b"WEBP", "image/webp")


@pytest.mark.anyio
async def test_private_photo_downloads_and_returns_prompt() -> None:
    telegram = FakeTelegram()
    backend = FakeBackend()
    bot = TelegramBot(telegram, backend)

    await bot.accept_update(
        update(
            None,
            photo=[
                {"file_id": "small", "file_size": 100},
                {"file_id": "large", "file_size": 200},
            ],
            chat_type="private",
            thread_id=None,
        ),
        "PainterBot",
    )
    await drain_queue(bot)

    assert telegram.downloads == ["large"]
    assert backend.images == [(b"IMAGE", "image/jpeg")]
    assert [text for _, text in telegram.messages] == [
        "正在反推提示词…",
        "reverse prompt",
    ]


@pytest.mark.anyio
async def test_group_image_requires_leading_mention_in_caption() -> None:
    telegram = FakeTelegram()
    backend = FakeBackend()
    bot = TelegramBot(telegram, backend)
    photo = [{"file_id": "photo", "file_size": 100}]

    await bot.accept_update(
        update(None, photo=photo, caption="普通图片"),
        "PainterBot",
    )
    await bot.accept_update(
        update(None, photo=photo, caption="@PainterBot 请反推"),
        "PainterBot",
    )
    await drain_queue(bot)

    assert telegram.downloads == ["photo"]
    assert backend.images == [(b"IMAGE", "image/jpeg")]


@pytest.mark.anyio
async def test_private_image_document_is_supported() -> None:
    telegram = FakeTelegram()
    backend = FakeBackend()
    bot = TelegramBot(telegram, backend)

    await bot.accept_update(
        update(
            None,
            document={
                "file_id": "original",
                "file_size": 100,
                "mime_type": "image/webp",
            },
            chat_type="private",
        ),
        "PainterBot",
    )
    await drain_queue(bot)

    assert backend.images == [(b"IMAGE", "image/webp")]


@pytest.mark.anyio
async def test_oversized_image_is_rejected_before_queueing() -> None:
    telegram = FakeTelegram()
    bot = TelegramBot(telegram, FakeBackend())

    await bot.accept_update(
        update(
            None,
            photo=[{"file_id": "huge", "file_size": MAX_IMAGE_BYTES + 1}],
            chat_type="private",
        ),
        "PainterBot",
    )

    assert bot.queue.empty()
    assert [text for _, text in telegram.messages] == ["图片不能超过 10 MiB"]
    assert telegram.downloads == []


@pytest.mark.anyio
@pytest.mark.parametrize("text", ["/start", "/START@painterbot payload"])
async def test_private_start_explains_both_features(text: str) -> None:
    telegram = FakeTelegram()
    bot = TelegramBot(telegram, FakeBackend())

    await bot.accept_update(
        update(text, chat_type="private", thread_id=None),
        "PainterBot",
    )

    assert bot.queue.empty()
    assert [text for _, text in telegram.messages] == [
        "发送文字生成图片；直接发送图片可反推提示词"
    ]


@pytest.mark.anyio
async def test_shared_queue_rejects_tasks_when_full() -> None:
    telegram = FakeTelegram()
    bot = TelegramBot(telegram, FakeBackend())

    for number in range(QUEUE_CAPACITY + 1):
        await bot.accept_update(
            update(update_id=number, message_id=number),
            "PainterBot",
        )

    assert bot.queue.qsize() == QUEUE_CAPACITY
    assert [text for _, text in telegram.messages] == ["当前任务较多，请稍后再试"]


@pytest.mark.anyio
async def test_polling_survives_pending_and_transient_results() -> None:
    telegram = FakeTelegram()
    backend = FakeBackend(
        image_results=[
            None,
            TransientBackendApiError("短暂失败"),
            ImageApiResult("completed", b"WEBP", "image/webp"),
        ]
    )
    bot = TelegramBot(telegram, backend)

    await bot.accept_update(update("@PainterBot 画猫"), "PainterBot")
    await drain_queue(bot)

    assert backend.result_calls == 3
    assert len(telegram.photos) == 1


@pytest.mark.anyio
async def test_long_reverse_prompt_is_delivered_in_multiple_messages() -> None:
    telegram = FakeTelegram()
    prompt = ("word " * 2000).strip()
    bot = TelegramBot(
        telegram,
        FakeBackend(text_results=[TextApiResult("completed", prompt)]),
    )

    await bot.accept_update(
        update(None, photo=[{"file_id": "photo"}], chat_type="private"),
        "PainterBot",
    )
    await drain_queue(bot)

    delivered = [text for _, text in telegram.messages][1:]
    assert len(delivered) > 1
    assert " ".join(delivered) == prompt


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("job", "results", "expected"),
    [
        (
            TextToImageJob(-1001, 7, None, "画猫", float("inf")),
            {"image_results": [ImageApiResult("failed")]},
            "生成失败，请重试",
        ),
        (
            ImageToTextJob(-1001, 7, None, "photo", "image/jpeg", float("inf")),
            {"text_results": [TextApiResult("missing")]},
            "反推任务已丢失，请重试",
        ),
        (
            TextToImageJob(-1001, 7, None, "画猫", float("inf")),
            {"image_results": [BackendApiError("协议异常")]},
            "生图服务暂时异常，请稍后重试",
        ),
    ],
)
async def test_task_failures_return_stable_messages(
    job: TextToImageJob | ImageToTextJob,
    results: dict[str, list[object]],
    expected: str,
) -> None:
    telegram = FakeTelegram()
    bot = TelegramBot(telegram, FakeBackend(**results))
    bot.queue.put_nowait(job)

    await drain_queue(bot)

    assert [text for _, text in telegram.messages][-1] == expected
    assert telegram.photos == []


@pytest.mark.anyio
async def test_expired_job_does_not_block_next_job() -> None:
    class TimeoutBackend(FakeBackend):
        async def submit_text_to_image(self, instruction: str) -> str:
            self.instructions.append(instruction)
            if instruction == "超时任务":
                await asyncio.Future()
            return "job-ok"

    telegram = FakeTelegram()
    backend = TimeoutBackend()
    bot = TelegramBot(telegram, backend)
    loop = asyncio.get_running_loop()
    bot.queue.put_nowait(TextToImageJob(-1001, 1, None, "超时任务", loop.time()))
    bot.queue.put_nowait(
        TextToImageJob(-1001, 2, None, "后续任务", loop.time() + JOB_TIMEOUT)
    )

    await drain_queue(bot)

    assert "任务超时，请稍后重试" in [text for _, text in telegram.messages]
    assert backend.instructions == ["超时任务", "后续任务"]
    assert len(telegram.photos) == 1


@pytest.mark.anyio
async def test_completed_image_replies_in_original_topic() -> None:
    telegram = FakeTelegram()
    bot = TelegramBot(telegram, FakeBackend())

    await bot.accept_update(update("@PainterBot 画猫"), "PainterBot")
    await drain_queue(bot)

    reply = telegram.photos[0][0]
    assert reply["chat_id"] == "-1001"
    assert '"message_id": 7' in reply["reply_parameters"]
    assert reply["message_thread_id"] == "11"
