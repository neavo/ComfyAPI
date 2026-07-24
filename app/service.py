import asyncio
import copy
import json
import logging
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from .comfy import ComfyClient, ComfyError, JobStatus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_TO_IMAGE_WORKFLOW_PATH = PROJECT_ROOT / "workflows" / "generation.json"
IMAGE_TO_TEXT_WORKFLOW_PATH = PROJECT_ROOT / "workflows" / "image_to_text.json"
INPUT_MARKER = "api_input"
OUTPUT_MARKER = "api_output"
PASSTHROUGH_MARKER = "启用透传模式"
PROMPT_SUFFIX = """
safe
(mature:-1), aged down
(simple background:-1.25)
(shiny skin:-1), flat color, anime coloring
masterpiece, best quality, score_7
bloom, light particles, cinematic lighting
depth of field, strong perspective, blurry background"""
LOGGER = logging.getLogger(__name__)
LLM_ATTEMPTS = 2
RETRY_DELAY = 3.0
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
IMAGE_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


class LlmUpstreamError(RuntimeError):
    pass


class InstructionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Settings:
    token: str
    comfy_url: str
    llm_url: str
    llm_api_key: str
    llm_model: str


@dataclass(frozen=True, slots=True)
class WorkflowTemplate:
    data: dict[str, Any]
    input_node_id: str
    output_node_id: str
    input_name: str


@dataclass(frozen=True, slots=True)
class ImageResult:
    status: JobStatus
    image: bytes | None = None
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class TextResult:
    status: JobStatus
    text: str | None = None


class TextToImageService:
    def __init__(
        self,
        comfy: ComfyClient,
        llm_client: httpx.AsyncClient,
        settings: Settings,
        system_prompt: str,
        workflow: WorkflowTemplate,
    ) -> None:
        self.comfy = comfy
        self.llm_client = llm_client
        self.settings = settings
        self.system_prompt = system_prompt
        self.workflow = workflow

    async def submit(self, instruction: str) -> str:
        job_id = str(uuid4())
        if PASSTHROUGH_MARKER in instruction:
            instruction = normalize_instruction(
                instruction.replace(PASSTHROUGH_MARKER, "")
            )
            prompt = f"{instruction}\n{PROMPT_SUFFIX}"
        else:
            prompt = await preprocess_instruction(
                self.llm_client,
                self.settings,
                self.system_prompt,
                instruction,
            )
        await self.comfy.submit(
            job_id,
            build_workflow(self.workflow, prompt, randomize_seeds=True),
        )
        return job_id

    async def result(self, job_id: str) -> ImageResult:
        job = await self.comfy.job(job_id)
        if job.status != "completed":
            return ImageResult(job.status)
        output = job.outputs.get(self.workflow.output_node_id) if job.outputs else None
        images = output.get("images") if isinstance(output, dict) else None
        if not isinstance(images, list):
            raise ComfyError("成功任务缺少 api_output images")
        image = next(
            (
                item
                for item in images
                if isinstance(item, dict) and item.get("type") == "output"
            ),
            None,
        )
        if image is None:
            raise ComfyError("成功任务没有 output 图片")
        downloaded = await self.comfy.download_image(image)
        return ImageResult("completed", downloaded.content, downloaded.media_type)


class ImageToTextService:
    def __init__(self, comfy: ComfyClient, workflow: WorkflowTemplate) -> None:
        self.comfy = comfy
        self.workflow = workflow

    async def submit(self, image: bytes, media_type: str) -> str:
        job_id = str(uuid4())
        filename = f"{job_id}.{IMAGE_EXTENSIONS[media_type]}"
        uploaded = await self.comfy.upload_image(filename, image, media_type)
        await self.comfy.submit(
            job_id,
            build_workflow(self.workflow, uploaded),
        )
        return job_id

    async def result(self, job_id: str) -> TextResult:
        job = await self.comfy.job(job_id)
        if job.status != "completed":
            return TextResult(job.status)
        output = job.outputs.get(self.workflow.output_node_id) if job.outputs else None
        texts = output.get("text") if isinstance(output, dict) else None
        text = (
            texts[0].strip()
            if isinstance(texts, list) and texts and isinstance(texts[0], str)
            else None
        )
        if not text:
            raise ComfyError("成功任务缺少 api_output text")
        return TextResult("completed", text)


def normalize_instruction(value: str) -> str:
    value = value.strip()
    if not 1 <= len(value) <= 4096:
        raise InstructionError("instruction 长度必须为 1 至 4096 个字符")
    return value


def load_config(root: Path | None = None) -> dict[str, object]:
    root = PROJECT_ROOT if root is None else root
    path = root / "config" / "config.toml"
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"无法加载配置文件 {path}: {error}") from error


def required_setting(config: dict[str, object], name: str) -> str:
    value = config.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"配置项 {name} 必须是非空字符串")
    return value.strip()


def load_settings(root: Path | None = None) -> Settings:
    config = load_config(root)
    token = required_setting(config, "api_token")
    comfy_url = _http_url("comfy_url", required_setting(config, "comfy_url"), True)
    llm_url = _http_url("llm_url", required_setting(config, "llm_url"))
    return Settings(
        token,
        comfy_url,
        llm_url,
        required_setting(config, "llm_api_key"),
        required_setting(config, "llm_model"),
    )


def _http_url(name: str, value: str, root: bool = False) -> str:
    try:
        url = httpx.URL(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} 不是合法 URL") from error
    if (
        url.scheme not in {"http", "https"}
        or not url.host
        or url.userinfo
        or (root and url.path != "/")
    ):
        kind = "根 URL" if root else "URL"
        raise RuntimeError(f"{name} 必须是无用户信息的 HTTP/HTTPS {kind}")
    return str(url).rstrip("/") if root else str(url)


def load_system_prompt(path: Path | None = None) -> str:
    path = PROJECT_ROOT / "prompt" / "system.md" if path is None else path
    try:
        prompt = path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"无法读取系统指令 {path}: {error}") from error
    if not prompt:
        raise RuntimeError(f"系统指令 {path} 不能为空")
    return prompt


def load_workflow(path: Path, input_name: str) -> WorkflowTemplate:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法加载工作流 {path}: {error}") from error
    return resolve_workflow(data, input_name)


def resolve_workflow(data: object, input_name: str) -> WorkflowTemplate:
    if not isinstance(data, dict):
        raise RuntimeError("工作流必须是节点对象")

    def unique_node(title: str) -> tuple[str, dict[str, Any]]:
        matches = [
            (node_id, node)
            for node_id, node in data.items()
            if isinstance(node_id, str)
            and isinstance(node, dict)
            and isinstance(node.get("_meta"), dict)
            and node["_meta"].get("title") == title
        ]
        if len(matches) != 1:
            raise RuntimeError(f"工作流必须恰好包含一个 {title} 节点")
        return matches[0]

    input_id, input_node = unique_node(INPUT_MARKER)
    if (
        not isinstance(input_node.get("inputs"), dict)
        or input_name not in input_node["inputs"]
    ):
        raise RuntimeError(f"{INPUT_MARKER} 节点必须包含 inputs.{input_name}")

    output_id, _ = unique_node(OUTPUT_MARKER)
    return WorkflowTemplate(data, input_id, output_id, input_name)


def build_workflow(
    template: WorkflowTemplate,
    value: str,
    *,
    randomize_seeds: bool = False,
) -> dict[str, Any]:
    prompt = copy.deepcopy(template.data)
    prompt[template.input_node_id]["inputs"][template.input_name] = value
    if randomize_seeds:
        _randomize_seeds(prompt)
    return prompt


def _randomize_seeds(prompt: dict[str, Any]) -> None:
    for node in prompt.values():
        inputs = node.get("inputs", {})
        if "noise_seed" in inputs:
            inputs["noise_seed"] = random.randint(0, 2**63 - 1)
        if "seed" in inputs:
            inputs["seed"] = random.randint(0, 2**63 - 1)


async def preprocess_instruction(
    client: httpx.AsyncClient,
    settings: Settings,
    system_prompt: str,
    instruction: str,
) -> str:
    for attempt in range(1, LLM_ATTEMPTS + 1):
        failure: str | None = None
        try:
            response = await client.post(
                settings.llm_url,
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                json={
                    "model": settings.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": instruction},
                    ],
                },
            )
        except httpx.RequestError as error:
            failure = type(error).__name__
        else:
            if response.status_code in TRANSIENT_STATUS_CODES:
                failure = f"HTTP {response.status_code}"
            else:
                if response.is_error:
                    raise LlmUpstreamError(f"LLM 返回 HTTP {response.status_code}")
                try:
                    payload = response.json()
                except ValueError as error:
                    raise LlmUpstreamError("LLM 返回了无效 JSON") from error
                choices = payload.get("choices") if isinstance(payload, dict) else None
                message = (
                    choices[0].get("message")
                    if isinstance(choices, list) and choices
                    else None
                )
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str) or not content.strip():
                    raise LlmUpstreamError(
                        "LLM 响应缺少非空 choices[0].message.content"
                    )
                return f"{content.strip()}\n{PROMPT_SUFFIX}"
        if attempt == LLM_ATTEMPTS:
            raise LlmUpstreamError(f"LLM 请求失败: {failure}")
        LOGGER.warning(
            "LLM 请求第 %s/%s 次失败（%s），%.1f 秒后重试",
            attempt,
            LLM_ATTEMPTS,
            failure,
            RETRY_DELAY,
        )
        await asyncio.sleep(RETRY_DELAY)
