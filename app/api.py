import logging
import secrets
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, field_validator

from .comfy import ComfyError
from .service import (
    ImageResult,
    ImageToTextService,
    InstructionError,
    LlmUpstreamError,
    MAX_IMAGE_BYTES,
    TextResult,
    TextToImageService,
    normalize_instruction,
)

LOGGER = logging.getLogger(__name__)
UPSTREAM_DETAIL = "ComfyUI upstream error"
LLM_UPSTREAM_DETAIL = "LLM upstream error"
IMAGE_SIGNATURES = {
    "image/jpeg": lambda data: data.startswith(b"\xff\xd8\xff"),
    "image/png": lambda data: data.startswith(b"\x89PNG\r\n\x1a\n"),
    "image/webp": lambda data: (
        len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    ),
}


class TextToImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str

    @field_validator("instruction")
    @classmethod
    def validate_instruction(cls, value: str) -> str:
        return normalize_instruction(value)


class IdResponse(BaseModel):
    id: str


class TextResponse(BaseModel):
    text: str


router = APIRouter()
bearer = HTTPBearer(auto_error=False)


def authenticate(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> None:
    if credentials is None or not secrets.compare_digest(
        credentials.credentials, request.app.state.settings.token
    ):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post(
    "/new",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IdResponse,
    include_in_schema=False,
)
@router.post(
    "/text_to_image",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IdResponse,
)
async def text_to_image(
    body: TextToImageRequest,
    request: Request,
    _: None = Depends(authenticate),
) -> IdResponse:
    service: TextToImageService = request.app.state.text_to_image
    try:
        job_id = await service.submit(body.instruction)
    except InstructionError as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
        ) from error
    except LlmUpstreamError as error:
        LOGGER.error("指令预处理失败：%s", error)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, LLM_UPSTREAM_DETAIL) from error
    except ComfyError as error:
        LOGGER.error("文生图任务提交失败：%s", error)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, UPSTREAM_DETAIL) from error
    LOGGER.info("文生图任务 %s 提交成功", job_id)
    return IdResponse(id=job_id)


@router.get("/result/{id}", include_in_schema=False)
@router.get("/text_to_image/{id}")
async def text_to_image_result(
    id: UUID,
    request: Request,
    _: None = Depends(authenticate),
):
    job_id = str(id)
    service: TextToImageService = request.app.state.text_to_image
    try:
        output = await service.result(job_id)
    except ComfyError as error:
        LOGGER.error("文生图任务 %s 查询失败：%s", job_id, error)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, UPSTREAM_DETAIL) from error
    LOGGER.info("文生图任务 %s 状态：%s", job_id, output.status)
    return _image_response(output)


@router.post(
    "/image_to_text",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IdResponse,
)
async def image_to_text(
    request: Request,
    _: None = Depends(authenticate),
) -> IdResponse:
    image, media_type = await _read_image(request)
    service: ImageToTextService = request.app.state.image_to_text
    try:
        job_id = await service.submit(image, media_type)
    except ComfyError as error:
        LOGGER.error("图生文任务提交失败：%s", error)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, UPSTREAM_DETAIL) from error
    LOGGER.info("图生文任务 %s 提交成功", job_id)
    return IdResponse(id=job_id)


@router.get("/image_to_text/{id}", response_model=TextResponse)
async def image_to_text_result(
    id: UUID,
    request: Request,
    _: None = Depends(authenticate),
):
    job_id = str(id)
    service: ImageToTextService = request.app.state.image_to_text
    try:
        output = await service.result(job_id)
    except ComfyError as error:
        LOGGER.error("图生文任务 %s 查询失败：%s", job_id, error)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, UPSTREAM_DETAIL) from error
    LOGGER.info("图生文任务 %s 状态：%s", job_id, output.status)
    return _text_response(output)


async def _read_image(request: Request) -> tuple[bytes, str]:
    media_type = request.headers.get("content-type", "").partition(";")[0].lower()
    if media_type not in IMAGE_SIGNATURES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Supported image types: JPEG, PNG, WebP",
        )
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_IMAGE_BYTES:
                raise HTTPException(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    "Image exceeds 10 MiB",
                )
        except ValueError:
            pass

    image = bytearray()
    async for chunk in request.stream():
        image.extend(chunk)
        if len(image) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                "Image exceeds 10 MiB",
            )
    if not image:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Image is empty",
        )
    data = bytes(image)
    if not IMAGE_SIGNATURES[media_type](data):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Image content does not match Content-Type",
        )
    return data, media_type


def _image_response(output: ImageResult) -> Response:
    if output.status == "processing":
        return Response(status_code=status.HTTP_202_ACCEPTED)
    if output.status == "completed" and output.image is not None:
        return Response(content=output.image, media_type=output.media_type)
    if output.status == "failed":
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "generation failed")
    raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")


def _text_response(output: TextResult) -> TextResponse | Response:
    if output.status == "processing":
        return Response(status_code=status.HTTP_202_ACCEPTED)
    if output.status == "completed" and output.text is not None:
        return TextResponse(text=output.text)
    if output.status == "failed":
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "generation failed")
    raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
