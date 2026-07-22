import logging
import secrets
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, field_validator

from .service import (
    GenerationService,
    LlmUpstreamError,
    UpstreamError,
    normalize_instruction,
)

LOGGER = logging.getLogger(__name__)
UPSTREAM_DETAIL = "ComfyUI upstream error"
LLM_UPSTREAM_DETAIL = "LLM upstream error"


class NewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str

    @field_validator("instruction")
    @classmethod
    def validate_instruction(cls, value: str) -> str:
        return normalize_instruction(value)


class IdResponse(BaseModel):
    id: str


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


@router.post("/new", status_code=status.HTTP_202_ACCEPTED, response_model=IdResponse)
async def new_job(
    body: NewRequest, request: Request, _: None = Depends(authenticate)
) -> IdResponse:
    generation: GenerationService = request.app.state.generation
    try:
        job_id = await generation.submit(body.instruction)
    except LlmUpstreamError as error:
        LOGGER.error("指令预处理失败：%s", error)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, LLM_UPSTREAM_DETAIL) from error
    except UpstreamError as error:
        LOGGER.error("任务提交失败：%s", error)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, UPSTREAM_DETAIL) from error
    LOGGER.info("任务 %s 提交成功", job_id)
    return IdResponse(id=job_id)


@router.get("/result/{id}")
async def result(id: str, request: Request, _: None = Depends(authenticate)):
    try:
        parsed = UUID(id)
    except ValueError as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid canonical UUID"
        ) from error
    if str(parsed) != id:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid canonical UUID"
        )

    generation: GenerationService = request.app.state.generation
    try:
        output = await generation.result(id)
    except UpstreamError as error:
        LOGGER.error("任务 %s 查询失败：%s", id, error)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, UPSTREAM_DETAIL) from error
    LOGGER.info("任务 %s 状态：%s", id, output.status)
    if output.status == "processing":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Task is still processing")
    if output.status == "completed":
        return Response(content=output.image, media_type=output.media_type)
    if output.status == "failed":
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "generation failed")
    raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
