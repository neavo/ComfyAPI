import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx

from .service import (
    IMAGE_EXTENSIONS,
    MAX_IMAGE_BYTES,
    load_config,
    required_setting,
)
from .telegram_api import (
    RECONNECT_DELAY,
    BackendApi,
    BackendApiError,
    TelegramApi,
    TelegramError,
    TransientBackendApiError,
)

WORKER_COUNT = 2
QUEUE_CAPACITY = 20
JOB_TIMEOUT = 180.0
RESULT_POLL_DELAY = 3.0
TELEGRAM_MESSAGE_LIMIT = 4096
API_URL = "http://127.0.0.1:48188"
HELP_TEXT = """🤖 使用帮助

🎨 生图
发送文字描述；群聊格式：@机器人 生图描述

🔍 反推提示词
发送照片或 JPEG、PNG、WebP 图片文件；群聊请在图片说明中 @机器人

⚡ 透传提示词
格式：启用透传模式 <提示词>
作用：跳过自动扩写"""
LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TextToImageJob:
    reply: dict[str, str]
    instruction: str
    deadline: float
    safe_mode: bool = True


@dataclass(frozen=True, slots=True)
class ImageToTextJob:
    reply: dict[str, str]
    file_id: str
    media_type: str
    deadline: float


@dataclass(frozen=True, slots=True)
class ImageAttachment:
    file_id: str
    media_type: str
    size: int | None


def extract_instruction(text: str, username: str) -> str | None:
    mention = f"@{username}"
    if not text.casefold().startswith(mention.casefold()):
        return None
    if len(text) > len(mention) and not text[len(mention)].isspace():
        return None
    return text[len(mention) :].strip()


def extract_image(message: dict[str, Any]) -> ImageAttachment | None:
    photos = message.get("photo")
    if isinstance(photos, list):
        candidates = [
            photo
            for photo in photos
            if isinstance(photo, dict) and isinstance(photo.get("file_id"), str)
        ]
        if candidates:
            photo = max(
                candidates,
                key=lambda item: (
                    item.get("file_size")
                    if isinstance(item.get("file_size"), int)
                    else 0,
                    (
                        item.get("width") * item.get("height")
                        if isinstance(item.get("width"), int)
                        and isinstance(item.get("height"), int)
                        else 0
                    ),
                ),
            )
            size = photo.get("file_size")
            return ImageAttachment(
                photo["file_id"],
                "image/jpeg",
                size if isinstance(size, int) else None,
            )

    document = message.get("document")
    if not isinstance(document, dict):
        return None
    file_id = document.get("file_id")
    media_type = document.get("mime_type")
    size = document.get("file_size")
    if not isinstance(file_id, str) or media_type not in IMAGE_EXTENSIONS:
        return None
    return ImageAttachment(
        file_id,
        media_type,
        size if isinstance(size, int) else None,
    )


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        newline = remaining.rfind("\n", 0, limit + 1)
        space = remaining.rfind(" ", 0, limit + 1)
        cut = max(newline, space)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def load_safe_mode_exempt_chat_ids(
    config: dict[str, object],
) -> frozenset[int]:
    name = "tg_safe_mode_exempt_chat_ids"
    value = config.get(name, [])
    if not isinstance(value, list) or any(
        not isinstance(chat_id, int) or isinstance(chat_id, bool) for chat_id in value
    ):
        raise RuntimeError(f"配置项 {name} 必须是整数数组")
    return frozenset(value)


class TelegramBot:
    def __init__(
        self,
        api: TelegramApi,
        backend: BackendApi,
        safe_mode_exempt_chat_ids: frozenset[int] = frozenset(),
    ) -> None:
        self.api = api
        self.backend = backend
        self.safe_mode_exempt_chat_ids = safe_mode_exempt_chat_ids
        self.queue: asyncio.Queue[TextToImageJob | ImageToTextJob] = asyncio.Queue(
            QUEUE_CAPACITY
        )

    async def run(self) -> None:
        username = await self._username()
        LOGGER.info("Telegram 机器人 @%s 已启动", username)
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(self._receive(username))
            for worker_id in range(1, WORKER_COUNT + 1):
                tasks.create_task(self.worker(worker_id))

    async def _username(self) -> str:
        while True:
            try:
                return await self.api.get_username()
            except TelegramError as error:
                if not error.retryable:
                    raise
                delay = (
                    error.retry_after
                    if error.retry_after is not None
                    else RECONNECT_DELAY
                )
                LOGGER.warning("Telegram getMe 失败（%s），%.1f 秒后重试", error, delay)
                await asyncio.sleep(delay)

    async def _receive(self, username: str) -> None:
        offset: int | None = None
        while True:
            try:
                updates = await self.api.get_updates(offset)
            except TelegramError as error:
                if not error.retryable:
                    raise
                delay = (
                    error.retry_after
                    if error.retry_after is not None
                    else RECONNECT_DELAY
                )
                LOGGER.warning("Telegram 长轮询失败（%s），%.1f 秒后重连", error, delay)
                await asyncio.sleep(delay)
                continue
            for update in updates:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = max(offset or update_id + 1, update_id + 1)
                await self.accept_update(update, username)

    async def accept_update(self, update: dict[str, Any], username: str) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return
        sender = message.get("from")
        chat = message.get("chat")
        message_id = message.get("message_id")
        chat_type = chat.get("type") if isinstance(chat, dict) else None
        if (
            not isinstance(sender, dict)
            or sender.get("is_bot") is True
            or not isinstance(chat, dict)
            or chat_type not in {"private", "group", "supergroup"}
            or not isinstance(chat.get("id"), int)
            or not isinstance(message_id, int)
        ):
            return

        thread_id = message.get("message_thread_id")
        reply = self.reply_data(chat["id"], message_id, thread_id)
        attachment = extract_image(message)
        if attachment is not None:
            if chat_type != "private":
                caption = message.get("caption")
                if (
                    not isinstance(caption, str)
                    or extract_instruction(caption, username) is None
                ):
                    return
            if attachment.size is not None and attachment.size > MAX_IMAGE_BYTES:
                await self.api.send_message(reply, "图片不能超过 10 MiB")
                return
            await self._enqueue(
                ImageToTextJob(
                    reply,
                    attachment.file_id,
                    attachment.media_type,
                    asyncio.get_running_loop().time() + JOB_TIMEOUT,
                ),
                reply,
            )
            return

        text = message.get("text")
        if not isinstance(text, str):
            return
        if chat_type == "private":
            instruction = text.strip()
            if instruction:
                command = instruction.split(maxsplit=1)[0].casefold()
                if command in {"/help", f"/help@{username}".casefold()}:
                    await self.api.send_message(reply, HELP_TEXT)
                    return
                if instruction.startswith("/"):
                    return
        else:
            instruction = extract_instruction(text, username)
            if instruction is None:
                return

        if not instruction:
            prompt = "请输入生图描述" if chat_type == "private" else HELP_TEXT
            await self.api.send_message(reply, prompt)
            return
        if len(instruction) > 4096:
            await self.api.send_message(reply, "生图描述长度必须为 1 至 4096 个字符")
            return
        await self._enqueue(
            TextToImageJob(
                reply,
                instruction,
                asyncio.get_running_loop().time() + JOB_TIMEOUT,
                chat_type == "private"
                or chat["id"] not in self.safe_mode_exempt_chat_ids,
            ),
            reply,
        )

    async def _enqueue(
        self,
        job: TextToImageJob | ImageToTextJob,
        reply: dict[str, str],
    ) -> None:
        if self.queue.full():
            await self.api.send_message(reply, "当前任务较多，请稍后再试")
            return
        self.queue.put_nowait(job)

    async def worker(self, worker_id: int) -> None:
        while True:
            job = await self.queue.get()
            try:
                try:
                    async with asyncio.timeout_at(job.deadline):
                        await self._process(job)
                except TimeoutError:
                    await self.api.send_message(job.reply, "任务超时，请稍后重试")
                    LOGGER.error(
                        "Telegram 任务超时：worker=%s reply=%s",
                        worker_id,
                        job.reply,
                    )
                except Exception:
                    LOGGER.exception(
                        "Telegram Worker 未预期失败：worker=%s reply=%s",
                        worker_id,
                        job.reply,
                    )
            finally:
                self.queue.task_done()

    async def _process(self, job: TextToImageJob | ImageToTextJob) -> None:
        if isinstance(job, TextToImageJob):
            await self._process_text_to_image(job)
        else:
            await self._process_image_to_text(job)

    async def _process_text_to_image(self, job: TextToImageJob) -> None:
        reply = job.reply
        await self.api.send_message(reply, "正在生成图片，请稍后 …")
        job_id: str | None = None
        try:
            job_id = await self.backend.submit_text_to_image(
                job.instruction,
                job.safe_mode,
            )
            result = await self._poll(
                job_id,
                self.backend.text_to_image_result,
                job,
                "文生图",
            )
        except BackendApiError as error:
            LOGGER.error(
                "Telegram 文生图 API 任务失败：reply=%s job_id=%s error=%s",
                job.reply,
                job_id,
                error,
            )
            await self.api.send_message(reply, "生图服务暂时异常，请稍后重试")
            return

        if result.status == "failed":
            await self.api.send_message(reply, "生成失败，请重试")
            return
        if result.status == "missing":
            await self.api.send_message(reply, "生成任务已丢失，请重试")
            return
        if result.image is None:
            raise RuntimeError("文生图完成结果缺少图片")
        delivered = await self.api.send_photo(
            reply,
            result.image,
            result.media_type or "image/webp",
        )
        log = LOGGER.info if delivered else LOGGER.error
        log(
            "Telegram 文生图任务%s：reply=%s job_id=%s",
            "完成" if delivered else "图片交付失败",
            job.reply,
            job_id,
        )

    async def _process_image_to_text(self, job: ImageToTextJob) -> None:
        reply = job.reply
        await self.api.send_message(reply, "正在反推提示词，请稍后 …")
        try:
            image = await self.api.download_file(job.file_id)
        except TelegramError as error:
            LOGGER.error(
                "Telegram 图片下载失败：reply=%s error=%s",
                job.reply,
                error,
            )
            await self.api.send_message(reply, "图片下载失败，请重试")
            return

        job_id: str | None = None
        try:
            job_id = await self.backend.submit_image_to_text(image, job.media_type)
            result = await self._poll(
                job_id,
                self.backend.image_to_text_result,
                job,
                "图生文",
            )
        except BackendApiError as error:
            LOGGER.error(
                "Telegram 图生文 API 任务失败：reply=%s job_id=%s error=%s",
                job.reply,
                job_id,
                error,
            )
            await self.api.send_message(reply, "反推服务暂时异常，请稍后重试")
            return

        if result.status == "failed":
            await self.api.send_message(reply, "反推失败，请重试")
            return
        if result.status == "missing":
            await self.api.send_message(reply, "反推任务已丢失，请重试")
            return
        if result.text is None:
            raise RuntimeError("图生文完成结果缺少文本")
        delivered = True
        for chunk in split_message(result.text):
            delivered = await self.api.send_message(reply, chunk) and delivered
        log = LOGGER.info if delivered else LOGGER.error
        log(
            "Telegram 图生文任务%s：reply=%s job_id=%s",
            "完成" if delivered else "文本交付失败",
            job.reply,
            job_id,
        )

    async def _poll(
        self,
        job_id: str,
        fetch: Callable[[str], Awaitable[T | None]],
        job: TextToImageJob | ImageToTextJob,
        operation: str,
    ) -> T:
        while True:
            await asyncio.sleep(RESULT_POLL_DELAY)
            try:
                result = await fetch(job_id)
            except TransientBackendApiError as error:
                result = None
                LOGGER.warning(
                    "%s API 查询瞬时失败：reply=%s job_id=%s error=%s",
                    operation,
                    job.reply,
                    job_id,
                    error,
                )
            if result is not None:
                return result
            LOGGER.info(
                "%s API 尚无结果：reply=%s job_id=%s，%.1f 秒后重试",
                operation,
                job.reply,
                job_id,
                RESULT_POLL_DELAY,
            )

    @staticmethod
    def reply_data(
        chat_id: int, message_id: int, message_thread_id: object
    ) -> dict[str, str]:
        data = {
            "chat_id": str(chat_id),
            "reply_parameters": json.dumps(
                {"message_id": message_id, "allow_sending_without_reply": True}
            ),
        }
        if isinstance(message_thread_id, int):
            data["message_thread_id"] = str(message_thread_id)
        return data


async def main() -> None:
    config = load_config()
    telegram_token = required_setting(config, "tg_bot_token")
    api_token = required_setting(config, "api_token")
    safe_mode_exempt_chat_ids = load_safe_mode_exempt_chat_ids(config)
    async with (
        httpx.AsyncClient(
            base_url=API_URL,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=httpx.Timeout(JOB_TIMEOUT, connect=5.0),
        ) as backend_client,
        httpx.AsyncClient(
            base_url="https://api.telegram.org",
            timeout=httpx.Timeout(40.0, connect=10.0),
        ) as telegram_client,
    ):
        await TelegramBot(
            TelegramApi(telegram_token, telegram_client),
            BackendApi(backend_client),
            safe_mode_exempt_chat_ids,
        ).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main())
