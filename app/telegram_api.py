import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from .service import MAX_IMAGE_BYTES

OUTBOUND_ATTEMPTS = 3
RECONNECT_DELAY = 3.0
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


class BackendApiError(RuntimeError):
    pass


class TransientBackendApiError(BackendApiError):
    pass


@dataclass(frozen=True, slots=True)
class ImageApiResult:
    status: Literal["completed", "failed", "missing"]
    image: bytes | None = None
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class TextApiResult:
    status: Literal["completed", "failed", "missing"]
    text: str | None = None


class BackendApi:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def submit_text_to_image(
        self,
        instruction: str,
        safe_mode: bool,
    ) -> str:
        return await self._submit(
            "/text_to_image",
            json={"instruction": instruction, "safe_mode": safe_mode},
        )

    async def submit_image_to_text(self, image: bytes, media_type: str) -> str:
        return await self._submit(
            "/image_to_text",
            content=image,
            headers={"Content-Type": media_type},
        )

    async def text_to_image_result(self, job_id: str) -> ImageApiResult | None:
        response = await self._get(f"/text_to_image/{job_id}")
        if response.status_code == 200:
            if not response.content:
                raise BackendApiError("完成响应图片为空")
            return ImageApiResult(
                "completed",
                response.content,
                response.headers.get("content-type"),
            )
        status = self._result_status(response)
        return None if status is None else ImageApiResult(status)

    async def image_to_text_result(self, job_id: str) -> TextApiResult | None:
        response = await self._get(f"/image_to_text/{job_id}")
        if response.status_code == 200:
            try:
                text = response.json()["text"]
            except (ValueError, KeyError, TypeError):
                raise BackendApiError("完成响应缺少文本") from None
            if not isinstance(text, str) or not text:
                raise BackendApiError("完成响应缺少文本")
            return TextApiResult("completed", text)
        status = self._result_status(response)
        return None if status is None else TextApiResult(status)

    async def _submit(self, path: str, **kwargs: Any) -> str:
        try:
            response = await self.client.post(path, **kwargs)
        except httpx.RequestError as error:
            raise BackendApiError(f"提交请求失败：{type(error).__name__}") from None
        if response.status_code != 202:
            raise BackendApiError(f"提交响应异常：HTTP {response.status_code}")
        try:
            job_id = response.json()["id"]
        except (ValueError, KeyError, TypeError):
            raise BackendApiError("提交响应缺少任务 ID") from None
        if not isinstance(job_id, str):
            raise BackendApiError("提交响应缺少任务 ID")
        return job_id

    async def _get(self, path: str) -> httpx.Response:
        try:
            return await self.client.get(path)
        except httpx.RequestError as error:
            raise TransientBackendApiError(
                f"查询请求失败：{type(error).__name__}"
            ) from None

    @staticmethod
    def _result_status(
        response: httpx.Response,
    ) -> Literal["failed", "missing"] | None:
        if response.status_code == 502:
            raise TransientBackendApiError("查询响应异常：HTTP 502")
        if response.status_code == 202:
            return None
        if response.status_code == 404:
            return "missing"
        if response.status_code == 500:
            return "failed"
        raise BackendApiError(f"查询响应异常：HTTP {response.status_code}")


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

    async def download_file(self, file_id: str) -> bytes:
        result = await self._call("getFile", {"file_id": file_id})
        file_path = result.get("file_path") if isinstance(result, dict) else None
        if not isinstance(file_path, str) or not file_path:
            raise TelegramError("getFile 响应缺少 file_path")
        try:
            async with self.client.stream(
                "GET", f"/file/bot{self.token}/{file_path}"
            ) as response:
                if response.status_code in TRANSIENT_STATUS_CODES:
                    raise TelegramError(
                        f"文件下载返回 HTTP {response.status_code}",
                        retryable=True,
                    )
                if response.is_error:
                    raise TelegramError(f"文件下载返回 HTTP {response.status_code}")
                image = bytearray()
                async for chunk in response.aiter_bytes():
                    image.extend(chunk)
                    if len(image) > MAX_IMAGE_BYTES:
                        raise TelegramError("图片超过 10 MiB")
        except httpx.RequestError as error:
            raise TelegramError(
                f"文件下载失败：{type(error).__name__}",
                retryable=True,
            ) from None
        if not image:
            raise TelegramError("下载图片为空")
        return bytes(image)

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
        return False

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
