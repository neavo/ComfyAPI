import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from .api import router
from .comfy import ComfyClient
from .service import (
    IMAGE_TO_TEXT_WORKFLOW_PATH,
    TEXT_TO_IMAGE_WORKFLOW_PATH,
    ImageToTextService,
    TextToImageService,
    load_settings,
    load_system_prompt,
    load_workflow,
)

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = load_settings()
    system_prompt = load_system_prompt()
    text_to_image_workflow = load_workflow(TEXT_TO_IMAGE_WORKFLOW_PATH, "text")
    image_to_text_workflow = load_workflow(IMAGE_TO_TEXT_WORKFLOW_PATH, "image")
    async with (
        httpx.AsyncClient(base_url=settings.comfy_url, timeout=5.0) as comfy_http,
        httpx.AsyncClient(timeout=60.0) as llm_client,
    ):
        comfy = ComfyClient(comfy_http)
        application.state.settings = settings
        application.state.text_to_image = TextToImageService(
            comfy,
            llm_client,
            settings,
            system_prompt,
            text_to_image_workflow,
        )
        application.state.image_to_text = ImageToTextService(
            comfy,
            image_to_text_workflow,
        )
        LOGGER.info(
            "服务启动，工作流校验通过：text_to_image=%s/%s image_to_text=%s/%s",
            text_to_image_workflow.input_node_id,
            text_to_image_workflow.output_node_id,
            image_to_text_workflow.input_node_id,
            image_to_text_workflow.output_node_id,
        )
        yield


app = FastAPI(
    lifespan=lifespan,
    redirect_slashes=False,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.include_router(router)
