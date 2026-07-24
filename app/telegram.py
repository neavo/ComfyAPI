import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from .service import PROJECT_ROOT, read_required

WORKER_COUNT = 2
QUEUE_CAPACITY = 20
RECONNECT_DELAY = 3.0
JOB_TIMEOUT = 180.0
OUTBOUND_ATTEMPTS = 3
RESULT_POLL_DELAY = 3.0
TRANSIENT_STATUS_CODES = {500, 502, 503, 504}
GENERATION_API_URL = "http://127.0.0.1:48188"
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


@dataclass(frozen=True, slots=True)
class GenerationResult:
    status: Literal["completed", "failed", "missing"]
    image: bytes | None = None
    media_type: str | None = None


class GenerationApiError(RuntimeError):
    pass


class TransientGenerationApiError(GenerationApiError):
    pass


class GenerationApi:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def submit(self, instruction: str) -> str:
        try:
            response = await self.client.post("/new", json={"instruction": instruction})
        except httpx.RequestError as error:
            raise GenerationApiError(f"提交请求失败：{type(error).__name__}") from None
        if response.status_code != 202:
            raise GenerationApiError(f"提交响应异常：HTTP {response.status_code}")
        try:
            job_id = response.json()["id"]
        except (ValueError, KeyError, TypeError):
            raise GenerationApiError("提交响应缺少任务 ID") from None
        if not isinstance(job_id, str):
            raise GenerationApiError("提交响应缺少任务 ID")
        return job_id

    async def result(self, job_id: str) -> GenerationResult | None:
        try:
            response = await self.client.get(f"/result/{job_id}")
        except httpx.RequestError as error:
            raise TransientGenerationApiError(
                f"查询请求失败：{type(error).__name__}"
            ) from None
        if response.status_code == 200:
            if not response.content:
                raise GenerationApiError("完成响应图片为空")
            return GenerationResult(
                "completed", response.content, response.headers.get("content-type")
            )
        if response.status_code == 502:
            raise TransientGenerationApiError("查询响应异常：HTTP 502")
        if response.status_code == 202:
            return None
        if response.status_code == 404:
            return GenerationResult("missing")
        if response.status_code == 500:
            return GenerationResult("failed")
        raise GenerationApiError(f"查询响应异常：HTTP {response.status_code}")


class TelegramApi:
    def __init__(self, token: str, client: httpx.AsyncClient) -> None:
        self.token = token
        self.client = client

    async def get_username(self) -> str:
        result = await self._call("getMe")
        username = result.get("username") if isinstance(result, dict) else None
        if not isinstance(username, str) or not username:
            raise TelegramError("getMe 响应缺少机器人用户名", retryable=True)
        return username

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

        description = payload.get("description") if isinstance(payload, dict) else None
        message = (
            description
            if isinstance(description, str)
            else f"HTTP {response.status_code}"
        )

        if response.status_code == 429:
            parameters = (
                payload.get("parameters") if isinstance(payload, dict) else None
            )
            retry_after = (
                parameters.get("retry_after") if isinstance(parameters, dict) else None
            )
            if not isinstance(retry_after, (int, float)) or retry_after < 0:
                retry_after = None
            raise TelegramError(message, retryable=True, retry_after=retry_after)
        if response.status_code in TRANSIENT_STATUS_CODES:
            raise TelegramError(message, retryable=True)
        if (
            response.is_error
            or not isinstance(payload, dict)
            or payload.get("ok") is not True
        ):
            raise TelegramError(message)
        return payload.get("result")


class TelegramBot:
    def __init__(self, api: TelegramApi, generation: GenerationApi) -> None:
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
        text = message.get("text")
        message_id = message.get("message_id")
        chat_type = chat.get("type") if isinstance(chat, dict) else None
        if (
            not isinstance(sender, dict)
            or sender.get("is_bot") is True
            or not isinstance(chat, dict)
            or chat_type not in {"private", "group", "supergroup"}
            or not isinstance(chat.get("id"), int)
            or not isinstance(message_id, int)
            or not isinstance(text, str)
        ):
            return

        thread_id = message.get("message_thread_id")
        reply = self.reply_data(chat["id"], message_id, thread_id)
        if chat_type == "private":
            instruction = text.strip()
            if instruction:
                command = instruction.split(maxsplit=1)[0].casefold()
                if command in {"/start", f"/start@{username}".casefold()}:
                    await self.api.send_message(reply, "请直接发送生图描述")
                    return
                if instruction.startswith("/"):
                    return
        else:
            instruction = extract_instruction(text, username)
            if instruction is None:
                return

        if not instruction:
            prompt = (
                "请输入生图描述"
                if chat_type == "private"
                else f"请在 @{username} 后输入生图描述"
            )
            await self.api.send_message(reply, prompt)
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
            while True:
                await asyncio.sleep(RESULT_POLL_DELAY)
                try:
                    result = await self.generation.result(job_id)
                except TransientGenerationApiError as error:
                    result = None
                    LOGGER.warning(
                        "生成 API 查询瞬时失败：chat_id=%s message_id=%s "
                        "job_id=%s error=%s",
                        job.chat_id,
                        job.message_id,
                        job_id,
                        error,
                    )
                if result is not None:
                    break
                LOGGER.info(
                    "生成 API 尚无结果：chat_id=%s message_id=%s job_id=%s，"
                    "%.1f 秒后重试",
                    job.chat_id,
                    job.message_id,
                    job_id,
                    RESULT_POLL_DELAY,
                )
        except GenerationApiError as error:
            LOGGER.error(
                "Telegram 生成 API 任务失败：chat_id=%s message_id=%s "
                "job_id=%s error=%s",
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
        if result.status == "missing":
            await self.api.send_message(reply, "生成任务已丢失，请重试")
            LOGGER.error(
                "Telegram 生成任务丢失：chat_id=%s message_id=%s job_id=%s",
                job.chat_id,
                job.message_id,
                job_id,
            )
            return
        if result.status != "completed" or result.image is None:
            raise RuntimeError(f"result 返回意外状态：{result.status}")
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
    config = PROJECT_ROOT / "config"
    telegram_token = read_required(config / "tg_bot_token.txt")
    api_token = read_required(config / "api_token.txt")
    async with (
        httpx.AsyncClient(
            base_url=GENERATION_API_URL,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=httpx.Timeout(JOB_TIMEOUT, connect=5.0),
        ) as generation_client,
        httpx.AsyncClient(
            base_url="https://api.telegram.org",
            timeout=httpx.Timeout(40.0, connect=10.0),
        ) as telegram_client,
    ):
        await TelegramBot(
            TelegramApi(telegram_token, telegram_client),
            GenerationApi(generation_client),
        ).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
