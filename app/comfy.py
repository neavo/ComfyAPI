import logging
from dataclasses import dataclass
from typing import Any, Literal

import httpx


LOGGER = logging.getLogger(__name__)
JobStatus = Literal["processing", "completed", "failed", "missing"]


class ComfyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ComfyJob:
    status: JobStatus
    outputs: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DownloadedImage:
    content: bytes
    media_type: str


class ComfyClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def upload_image(
        self,
        filename: str,
        image: bytes,
        media_type: str,
    ) -> str:
        try:
            response = await self.client.post(
                "/upload/image",
                data={
                    "type": "input",
                    "subfolder": "api/image_to_text",
                    "overwrite": "false",
                },
                files={"image": (filename, image, media_type)},
                timeout=30.0,
            )
        except httpx.RequestError as error:
            raise ComfyError(
                f"POST /upload/image 请求失败: {type(error).__name__}"
            ) from error
        payload = self._json(response, "POST /upload/image")
        if response.is_error:
            raise ComfyError(f"POST /upload/image 返回 HTTP {response.status_code}")
        if not isinstance(payload, dict):
            raise ComfyError("POST /upload/image 响应不是对象")
        name = payload.get("name")
        subfolder = payload.get("subfolder")
        image_type = payload.get("type")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(subfolder, str)
            or image_type != "input"
        ):
            raise ComfyError("POST /upload/image 响应字段损坏")
        return "/".join(part for part in (subfolder.replace("\\", "/"), name) if part)

    async def submit(self, job_id: str, prompt: dict[str, Any]) -> None:
        try:
            response = await self.client.post(
                "/prompt",
                json={"prompt_id": job_id, "prompt": prompt},
            )
        except httpx.RequestError as error:
            raise ComfyError(
                f"POST /prompt 请求失败: {type(error).__name__}"
            ) from error
        payload = self._json(response, "POST /prompt")
        if response.is_error:
            node_errors = (
                payload.get("node_errors") if isinstance(payload, dict) else None
            )
            LOGGER.error(
                "ComfyUI 拒绝任务 %s，状态码 %s，node_errors=%r",
                job_id,
                response.status_code,
                node_errors,
            )
            raise ComfyError(f"POST /prompt 返回 HTTP {response.status_code}")
        if not isinstance(payload, dict) or payload.get("prompt_id") != job_id:
            raise ComfyError("POST /prompt 返回的 prompt_id 不匹配")

    async def job(self, job_id: str) -> ComfyJob:
        job = await self._history(job_id)
        if job is not None:
            return job
        if await self._queued(job_id):
            return ComfyJob("processing")
        return await self._history(job_id) or ComfyJob("missing")

    async def download_image(self, image: object) -> DownloadedImage:
        if not isinstance(image, dict):
            raise ComfyError("图片元数据结构损坏")
        if image.get("type") != "output":
            raise ComfyError("图片不是 output 文件")
        if not all(
            isinstance(image.get(field), str) for field in ("filename", "subfolder")
        ):
            raise ComfyError("图片元数据字段损坏")
        response = await self._get(
            "/view",
            params={
                "filename": image["filename"],
                "subfolder": image["subfolder"].replace("\\", "/"),
                "type": "output",
            },
        )
        return DownloadedImage(
            response.content,
            response.headers.get("content-type", "image/webp"),
        )

    async def _history(self, job_id: str) -> ComfyJob | None:
        response = await self._get(f"/history/{job_id}")
        history = self._json(response, "GET /history")
        if not isinstance(history, dict):
            raise ComfyError("history 响应不是对象")
        if job_id not in history:
            return None
        record = history[job_id]
        if not isinstance(record, dict) or not isinstance(record.get("status"), dict):
            raise ComfyError("history 任务结构损坏")
        status = record["status"].get("status_str")
        if status == "error":
            return ComfyJob("failed")
        if status != "success":
            raise ComfyError("history 任务状态未知")
        outputs = record.get("outputs")
        if not isinstance(outputs, dict):
            raise ComfyError("成功任务缺少 outputs")
        return ComfyJob("completed", outputs)

    async def _queued(self, job_id: str) -> bool:
        response = await self._get("/queue")
        queue = self._json(response, "GET /queue")
        if not isinstance(queue, dict):
            raise ComfyError("queue 响应不是对象")
        for name in ("queue_running", "queue_pending"):
            items = queue.get(name)
            if not isinstance(items, list):
                raise ComfyError(f"queue 缺少 {name}")
            for item in items:
                if not isinstance(item, list) or len(item) < 2:
                    raise ComfyError("queue 项结构损坏")
                if item[1] == job_id:
                    return True
        return False

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        try:
            response = await self.client.get(path, params=params)
        except httpx.RequestError as error:
            raise ComfyError(f"GET {path} 请求失败: {type(error).__name__}") from error
        if response.is_error:
            raise ComfyError(f"GET {path} 返回 HTTP {response.status_code}")
        return response

    @staticmethod
    def _json(response: httpx.Response, endpoint: str) -> object:
        try:
            return response.json()
        except ValueError as error:
            raise ComfyError(f"{endpoint} 返回了无效 JSON") from error
