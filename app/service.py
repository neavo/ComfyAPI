import asyncio
import copy
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = PROJECT_ROOT / "workflows" / "generation.json"
PASSTHROUGH_MARKER = "启用透传模式"
PROMPT_SUFFIX = """
safe
(mature:-1), (aged down:1)
(simple background:-1.25)
(shiny skin:-1), flat color, anime coloring
masterpiece, best quality, score_7
bloom, light particles, cinematic lighting
depth of field, strong perspective, blurry background"""
LOGGER = logging.getLogger(__name__)
LLM_ATTEMPTS = 2
RETRY_DELAY = 3.0
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class UpstreamError(RuntimeError):
    pass


class LlmUpstreamError(UpstreamError):
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
    instruction_node_id: str
    output_node_id: str


@dataclass(frozen=True, slots=True)
class GenerationResult:
    status: Literal["processing", "completed", "failed", "missing"]
    image: bytes | None = None
    media_type: str | None = None


class GenerationService:
    def __init__(
        self,
        comfy_client: httpx.AsyncClient,
        llm_client: httpx.AsyncClient,
        settings: Settings,
        system_prompt: str,
        workflow: WorkflowTemplate,
    ) -> None:
        self.comfy_client = comfy_client
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
            try:
                prompt = await preprocess_instruction(
                    self.llm_client,
                    self.settings,
                    self.system_prompt,
                    instruction,
                )
            except UpstreamError as error:
                raise LlmUpstreamError(str(error)) from error
        await submit_prompt(self.comfy_client, self.workflow, job_id, prompt)
        return job_id

    async def result(self, job_id: str) -> GenerationResult:
        history = await query_history(self.comfy_client, self.workflow, job_id)
        if history == "failed":
            return GenerationResult("failed")
        if history is None:
            return GenerationResult(await query_queue(self.comfy_client, job_id))
        return await read_result_image(self.comfy_client, history)


async def read_result_image(
    client: httpx.AsyncClient, file: dict[str, str]
) -> GenerationResult:
    image = await get_upstream(client, "/view", params=file)
    return GenerationResult(
        "completed", image.content, image.headers.get("content-type", "image/webp")
    )


def normalize_instruction(value: str) -> str:
    value = value.strip()
    if not 1 <= len(value) <= 4096:
        raise InstructionError("instruction 长度必须为 1 至 4096 个字符")
    return value


def read_required(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"无法读取配置文件 {path.name}: {error}") from error
    if not value or "\n" in value or "\r" in value:
        raise RuntimeError(f"配置文件 {path.name} 必须包含单行非空内容")
    return value


def load_settings(root: Path | None = None) -> Settings:
    root = PROJECT_ROOT if root is None else root
    config = root / "config"
    token = read_required(config / "api_token.txt")
    comfy_url = _http_url(
        "comfy_url.txt", read_required(config / "comfy_url.txt"), True
    )
    llm_url = _http_url("llm_url.txt", read_required(config / "llm_url.txt"))
    return Settings(
        token,
        comfy_url,
        llm_url,
        read_required(config / "llm_api_key.txt"),
        read_required(config / "llm_model.txt"),
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


def load_workflow(path: Path = WORKFLOW_PATH) -> WorkflowTemplate:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法加载工作流 {path}: {error}") from error
    return resolve_workflow(data)


def resolve_workflow(data: object) -> WorkflowTemplate:
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

    instruction_id, instruction = unique_node("API Instruction")
    if (
        not isinstance(instruction.get("inputs"), dict)
        or "text" not in instruction["inputs"]
    ):
        raise RuntimeError("API Instruction 节点必须包含 inputs.text")

    output_id, _ = unique_node("API Output")
    return WorkflowTemplate(data, instruction_id, output_id)


def build_prompt(template: WorkflowTemplate, instruction: str) -> dict[str, Any]:
    prompt = copy.deepcopy(template.data)
    prompt[template.instruction_node_id]["inputs"]["text"] = instruction
    _randomize_seeds(prompt)
    return prompt


def _randomize_seeds(prompt: dict[str, Any]) -> None:
    for node in prompt.values():
        inputs = node.get("inputs", {})
        if "noise_seed" in inputs:
            inputs["noise_seed"] = random.randint(0, 2**63 - 1)
        if "seed" in inputs:
            inputs["seed"] = random.randint(0, 2**63 - 1)


def parse_history(
    history: object,
    job_id: str,
    output_node_id: str,
) -> dict[str, str] | Literal["failed"] | None:
    if not isinstance(history, dict):
        raise UpstreamError("history 响应不是对象")
    if job_id not in history:
        return None
    record = history[job_id]
    if not isinstance(record, dict) or not isinstance(record.get("status"), dict):
        raise UpstreamError("history 任务结构损坏")
    status = record["status"].get("status_str")
    if status == "error":
        return "failed"
    if status != "success":
        raise UpstreamError("history 任务状态未知")

    outputs = record.get("outputs")
    output = outputs.get(output_node_id) if isinstance(outputs, dict) else None
    images = output.get("images") if isinstance(output, dict) else None
    if not isinstance(images, list):
        raise UpstreamError("成功任务缺少 API Output images")
    for image in images:
        if not isinstance(image, dict):
            raise UpstreamError("图片元数据结构损坏")
        if image.get("type") != "output":
            continue
        if not all(
            isinstance(image.get(field), str) for field in ("filename", "subfolder")
        ):
            raise UpstreamError("图片元数据字段损坏")
        return {
            "filename": image["filename"],
            "subfolder": image["subfolder"].replace("\\", "/"),
            "type": "output",
        }
    raise UpstreamError("成功任务没有 output 文件")


def is_queued(queue: object, job_id: str) -> bool:
    if not isinstance(queue, dict):
        raise UpstreamError("queue 响应不是对象")
    sections: list[list[object]] = []
    for name in ("queue_running", "queue_pending"):
        items = queue.get(name)
        if not isinstance(items, list):
            raise UpstreamError(f"queue 缺少 {name}")
        sections.append(items)
    for items in sections:
        for item in items:
            if not isinstance(item, list) or len(item) < 2:
                raise UpstreamError("queue 项结构损坏")
            if item[1] == job_id:
                return True
    return False


def _json(response: httpx.Response, endpoint: str) -> object:
    try:
        return response.json()
    except ValueError as error:
        raise UpstreamError(f"{endpoint} 返回了无效 JSON") from error


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
                    # "thinking": {"type": "disabled"},
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
                    raise UpstreamError(f"LLM 返回 HTTP {response.status_code}")
                payload = _json(response, "LLM")
                choices = payload.get("choices") if isinstance(payload, dict) else None
                message = (
                    choices[0].get("message")
                    if isinstance(choices, list) and choices
                    else None
                )
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str) or not content.strip():
                    raise UpstreamError("LLM 响应缺少非空 choices[0].message.content")
                return f"{content.strip()}\n{PROMPT_SUFFIX}"
        if attempt == LLM_ATTEMPTS:
            raise UpstreamError(f"LLM 请求失败: {failure}")
        LOGGER.warning(
            "LLM 请求第 %s/%s 次失败（%s），%.1f 秒后重试",
            attempt,
            LLM_ATTEMPTS,
            failure,
            RETRY_DELAY,
        )
        await asyncio.sleep(RETRY_DELAY)


async def submit_prompt(
    client: httpx.AsyncClient,
    template: WorkflowTemplate,
    job_id: str,
    instruction: str,
) -> None:
    prompt = build_prompt(template, instruction)
    try:
        response = await client.post(
            "/prompt",
            json={
                "prompt_id": job_id,
                "prompt": prompt,
                "extra_data": {"extra_pnginfo": {"workflow": prompt}},
            },
        )
    except httpx.RequestError as error:
        raise UpstreamError(f"POST /prompt 请求失败: {type(error).__name__}") from error
    payload = _json(response, "POST /prompt")
    if response.is_error:
        node_errors = payload.get("node_errors") if isinstance(payload, dict) else None
        LOGGER.error(
            "ComfyUI 拒绝任务 %s，状态码 %s，node_errors=%r",
            job_id,
            response.status_code,
            node_errors,
        )
        raise UpstreamError(f"POST /prompt 返回 HTTP {response.status_code}")
    if not isinstance(payload, dict) or payload.get("prompt_id") != job_id:
        raise UpstreamError("POST /prompt 返回的 prompt_id 不匹配")


async def get_upstream(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: dict[str, str] | None = None,
) -> httpx.Response:
    try:
        response = await client.get(path, params=params)
    except httpx.RequestError as error:
        raise UpstreamError(f"GET {path} 请求失败: {type(error).__name__}") from error
    if response.status_code in TRANSIENT_STATUS_CODES:
        raise UpstreamError(f"GET {path} 返回 HTTP {response.status_code}")
    if response.is_error:
        raise UpstreamError(f"GET {path} 返回 HTTP {response.status_code}")
    return response


async def query_history(
    client: httpx.AsyncClient,
    template: WorkflowTemplate,
    job_id: str,
) -> dict[str, str] | Literal["failed"] | None:
    response = await get_upstream(client, f"/history/{job_id}")
    return parse_history(
        _json(response, "GET /history"), job_id, template.output_node_id
    )


async def query_queue(
    client: httpx.AsyncClient, job_id: str
) -> Literal["processing", "missing"]:
    response = await get_upstream(client, "/queue")
    queue = _json(response, "GET /queue")
    return "processing" if is_queued(queue, job_id) else "missing"
