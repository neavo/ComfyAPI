import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .service import (
    LLM_ATTEMPTS as SERVICE_LLM_ATTEMPTS,
    PROJECT_ROOT,
    GenerationService,
    TransientUpstreamError,
    UpstreamError,
    load_settings,
    load_system_prompt,
    load_workflow,
)

WORKER_COUNT = 2
QUEUE_CAPACITY = 20
RECONNECT_DELAY = 3.0
JOB_TIMEOUT = 180.0
OUTBOUND_ATTEMPTS = 3
LLM_ATTEMPTS = SERVICE_LLM_ATTEMPTS
RESULT_POLL_DELAYS = (2.0, 3.0, 5.0)
TRANSIENT_STATUS_CODES = {500, 502, 503, 504}
LOGGER = logging.getLogger(__name__)


class TelegramError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class TelegramJob:
    chat_id: int
    message_id: int
    message_thread_id: int | None
    instruction: str
    deadline: float


def extract_instruction(text: str, username: str) -> str | None:
    mention = f"@{username}"
    if not text.casefold().startswith(mention.casefold()):
        return None
    if len(text) > len(mention) and not text[len(mention)].isspace():
        return None
    return text[len(mention) :].strip()


def load_bot_token(path: Path | None = None) -> str:
    path = PROJECT_ROOT / "config" / "tg_bot_token.txt" if path is None else path
    try:
        token = path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"无法读取配置文件 {path.name}: {error}") from error
    if not token or "\n" in token or "\r" in token:
        raise RuntimeError(f"配置文件 {path.name} 必须包含单行非空内容")
    return token


class TelegramApi:
    def __init__(self, token: str, client: httpx.AsyncClient) -> None:
        self.token = token
        self.client = client

    async def get_me(self) -> dict[str, Any]:
        result = await self._call("getMe")
        if not isinstance(result, dict):
            raise TelegramError("getMe 响应缺少机器人信息", retryable=True)
        return result

    async def get_updates(self, offset: int | None) -> list[dict[str, Any]]:
        data = {"timeout": "30", "allowed_updates": json.dumps(["message"])}
        if offset is not None:
            data["offset"] = str(offset)
        result = await self._call("getUpdates", data)
        if not isinstance(result, list) or not all(
            isinstance(update, dict) for update in result
        ):
            raise TelegramError("getUpdates 响应不是更新列表", retryable=True)
        return result

    async def send_message(self, reply: dict[str, str], text: str) -> bool:
        return await self._send("sendMessage", {**reply, "text": text})

    async def send_photo(
        self, reply: dict[str, str], image: bytes, media_type: str
    ) -> bool:
        return await self._send(
            "sendPhoto",
            reply,
            {"photo": ("result.webp", image, media_type)},
        )

    async def _send(
        self,
        method: str,
        data: dict[str, str],
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> bool:
        for attempt in range(1, OUTBOUND_ATTEMPTS + 1):
            try:
                await self._call(method, data, files)
                return True
            except TelegramError as error:
                if not error.retryable or attempt == OUTBOUND_ATTEMPTS:
                    LOGGER.error(
                        "Telegram %s 第 %s/%s 次失败后停止：%s",
                        method,
                        attempt,
                        OUTBOUND_ATTEMPTS,
                        error,
                    )
                    return False
                delay = (
                    error.retry_after
                    if error.retry_after is not None
                    else RECONNECT_DELAY
                )
                LOGGER.warning(
                    "Telegram %s 第 %s/%s 次失败（%s），%.1f 秒后重试",
                    method,
                    attempt,
                    OUTBOUND_ATTEMPTS,
                    error,
                    delay,
                )
                await asyncio.sleep(delay)
        raise AssertionError("Telegram 出站重试循环未返回")

    async def _call(
        self,
        method: str,
        data: dict[str, str] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> object:
        try:
            response = await self.client.post(
                f"/bot{self.token}/{method}", data=data, files=files
            )
        except httpx.RequestError as error:
            raise TelegramError(
                f"请求失败：{type(error).__name__}", retryable=True
            ) from None

        try:
            payload = response.json()
        except ValueError as error:
            retryable = response.is_success or response.status_code in (
                {429} | TRANSIENT_STATUS_CODES
            )
            raise TelegramError("响应不是合法 JSON", retryable=retryable) from error

        parameters = payload.get("parameters") if isinstance(payload, dict) else None
        raw_retry_after = (
            parameters.get("retry_after") if isinstance(parameters, dict) else None
        )
        retry_after = (
            float(raw_retry_after)
            if isinstance(raw_retry_after, (int, float))
            and not isinstance(raw_retry_after, bool)
            and raw_retry_after >= 0
            else None
        )
        description = payload.get("description") if isinstance(payload, dict) else None
        message = (
            description
            if isinstance(description, str)
            else f"HTTP {response.status_code}"
        )

        if response.status_code == 429 or retry_after is not None:
            raise TelegramError(message, retryable=True, retry_after=retry_after)
        if response.status_code in TRANSIENT_STATUS_CODES:
            raise TelegramError(message, retryable=True)
        if response.is_error:
            raise TelegramError(message)
        if (
            not isinstance(payload, dict)
            or "ok" not in payload
            or "result" not in payload
        ):
            raise TelegramError("响应缺少 Bot API 状态", retryable=True)
        if payload.get("ok") is not True:
            raise TelegramError(message)
        return payload.get("result")


class TelegramBot:
    def __init__(self, api: TelegramApi, generation: GenerationService) -> None:
        self.api = api
        self.generation = generation
        self.queue: asyncio.Queue[TelegramJob] = asyncio.Queue(QUEUE_CAPACITY)

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
                identity = await self.api.get_me()
                username = identity.get("username")
                if not isinstance(username, str) or not username:
                    raise TelegramError("getMe 响应缺少机器人用户名", retryable=True)
                return username
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
        text = message.get("text")
        message_id = message.get("message_id")
        if (
            not isinstance(sender, dict)
            or sender.get("is_bot") is True
            or not isinstance(chat, dict)
            or chat.get("type") not in {"group", "supergroup"}
            or not isinstance(chat.get("id"), int)
            or not isinstance(message_id, int)
            or not isinstance(text, str)
        ):
            return

        instruction = extract_instruction(text, username)
        if instruction is None:
            return
        thread_id = message.get("message_thread_id")
        reply = self.reply_data(chat["id"], message_id, thread_id)
        if not instruction:
            await self.api.send_message(reply, f"请在 @{username} 后输入生图描述")
            return
        if len(instruction) > 4096:
            await self.api.send_message(reply, "生图描述长度必须为 1 至 4096 个字符")
            return
        if self.queue.full():
            await self.api.send_message(reply, "当前生成任务较多，请稍后再试")
            return

        self.queue.put_nowait(
            TelegramJob(
                chat["id"],
                message_id,
                thread_id if isinstance(thread_id, int) else None,
                instruction,
                asyncio.get_running_loop().time() + JOB_TIMEOUT,
            )
        )

    async def worker(self, worker_id: int) -> None:
        while True:
            job = await self.queue.get()
            try:
                try:
                    async with asyncio.timeout_at(job.deadline):
                        await self._process(job)
                except TimeoutError:
                    reply = self.reply_data(
                        job.chat_id, job.message_id, job.message_thread_id
                    )
                    await self.api.send_message(reply, "生成超时，请稍后重试")
                    LOGGER.error(
                        "Telegram 任务超时：worker=%s chat_id=%s message_id=%s",
                        worker_id,
                        job.chat_id,
                        job.message_id,
                    )
                except Exception:
                    LOGGER.exception(
                        "Telegram Worker 未预期失败：worker=%s chat_id=%s message_id=%s",
                        worker_id,
                        job.chat_id,
                        job.message_id,
                    )
            finally:
                self.queue.task_done()

    async def _process(self, job: TelegramJob) -> None:
        reply = self.reply_data(job.chat_id, job.message_id, job.message_thread_id)
        if not await self.api.send_message(reply, "正在生成…"):
            LOGGER.error(
                "Telegram 状态消息交付失败，继续生成：chat_id=%s message_id=%s",
                job.chat_id,
                job.message_id,
            )

        job_id: str | None = None
        try:
            job_id = await self.generation.submit(job.instruction)
            delay_index = 0
            while True:
                try:
                    result = await self.generation.history_result(job_id)
                except TransientUpstreamError as error:
                    result = None
                    LOGGER.warning(
                        "ComfyUI 历史查询瞬时失败：chat_id=%s message_id=%s "
                        "job_id=%s error=%s",
                        job.chat_id,
                        job.message_id,
                        job_id,
                        error,
                    )
                if result is not None:
                    break
                delay = RESULT_POLL_DELAYS[
                    min(delay_index, len(RESULT_POLL_DELAYS) - 1)
                ]
                delay_index += 1
                LOGGER.info(
                    "ComfyUI 历史尚无结果：chat_id=%s message_id=%s job_id=%s，"
                    "%.1f 秒后重试",
                    job.chat_id,
                    job.message_id,
                    job_id,
                    delay,
                )
                await asyncio.sleep(delay)
        except UpstreamError as error:
            LOGGER.error(
                "Telegram 上游任务失败：chat_id=%s message_id=%s job_id=%s error=%s",
                job.chat_id,
                job.message_id,
                job_id,
                error,
            )
            await self.api.send_message(reply, "生图服务暂时异常，请稍后重试")
            return

        if result.status == "failed":
            await self.api.send_message(reply, "生成失败，请重试")
            LOGGER.error(
                "Telegram 生成失败：chat_id=%s message_id=%s job_id=%s",
                job.chat_id,
                job.message_id,
                job_id,
            )
            return
        if result.status != "completed" or result.image is None:
            raise RuntimeError(f"history_result 返回意外状态：{result.status}")
        delivered = await self.api.send_photo(
            reply, result.image, result.media_type or "image/webp"
        )
        log = LOGGER.info if delivered else LOGGER.error
        log(
            "Telegram 任务%s：chat_id=%s message_id=%s job_id=%s",
            "完成" if delivered else "图片交付失败",
            job.chat_id,
            job.message_id,
            job_id,
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
    token = load_bot_token()
    settings = load_settings()
    system_prompt = load_system_prompt()
    workflow = load_workflow()
    async with (
        httpx.AsyncClient(base_url=settings.comfy_url, timeout=5.0) as comfy_client,
        httpx.AsyncClient(timeout=60.0) as llm_client,
        httpx.AsyncClient(
            base_url="https://api.telegram.org",
            timeout=httpx.Timeout(40.0, connect=10.0),
        ) as telegram_client,
    ):
        generation = GenerationService(
            comfy_client,
            llm_client,
            settings,
            system_prompt,
            workflow,
        )
        await TelegramBot(TelegramApi(token, telegram_client), generation).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
