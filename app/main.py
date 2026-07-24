import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from .api import router
from .comfy import ComfyClient
from .service import (
    GenerationService,
    load_settings,
    load_system_prompt,
    load_workflow,
)

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = load_settings()
    system_prompt = load_system_prompt()
    workflow = load_workflow()
    async with (
        httpx.AsyncClient(base_url=settings.comfy_url, timeout=5.0) as comfy_client,
        httpx.AsyncClient(timeout=60.0) as llm_client,
    ):
        application.state.settings = settings
        application.state.generation = GenerationService(
            ComfyClient(comfy_client, workflow.output_node_id),
            llm_client,
            settings,
            system_prompt,
            workflow,
        )
        LOGGER.info(
            "服务启动，工作流校验通过：instruction=%s output=%s",
            workflow.instruction_node_id,
            workflow.output_node_id,
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
