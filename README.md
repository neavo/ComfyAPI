# ComfyUI 指令生图服务

这个服务先通过 OpenAI 兼容的 Chat Completions 接口和 `prompt/system.md` 扩写自然语言指令，再提交到既有 ComfyUI 实例，并直接使用 ComfyUI 的共享队列、历史和 GPU。FastAPI 只公开 `POST /new` 与 `GET /result/{id}`；可选的 Telegram 机器人作为独立进程复用同一套生图代码。两个入口都不接收工作流、模型、尺寸或保存路径。

## 准备工作流

先在目标 ComfyUI 中实际运行工作流，再以 **API Format** 导出为 `workflows/generation.json`。工作流必须满足：

- 接收指令的唯一文本节点标题为 `API Instruction`，且存在 `inputs.text`。
- 唯一最终 `Save Image (LoraManager)` 节点标题为 `API Output`，目标 ComfyUI 必须已安装 LoRA Manager 节点包。
- `API Output.inputs.filename_prefix` 必须为 `api/%date:yyyyMMdd%`；使用有损 WebP、质量 95，并启用工作流嵌入、元数据和文件名计数器，关闭配方保存。
- 图片保存为 `ComfyUI/output/api/YYYYMMDD_00001_.webp`；日期和递增编号由 ComfyUI 生成，服务不会改写。

部署前用预先生成的规范 UUID 调用 ComfyUI `POST /prompt`，确认响应返回相同 `prompt_id`，并确认 `/history/{id}` 与 `/queue` 可查询该 ID。目标实例不满足时先升级 ComfyUI，不在本服务中增加旧版分支。

## 安装与配置

要求 Python 3.11 或更高版本及 `uv`。在项目根目录执行：

```powershell
uv sync
New-Item -ItemType Directory -Force .\config | Out-Null
Set-Content -LiteralPath .\config\api_token.txt -NoNewline -Encoding UTF8 "替换为随机长TOKEN"
Set-Content -LiteralPath .\config\comfy_url.txt -NoNewline -Encoding UTF8 "http://127.0.0.1:8188"
Set-Content -LiteralPath .\config\llm_url.txt -NoNewline -Encoding UTF8 "https://HOST/v1/chat/completions"
Set-Content -LiteralPath .\config\llm_api_key.txt -NoNewline -Encoding UTF8 "替换为LLM密钥"
Set-Content -LiteralPath .\config\llm_model.txt -NoNewline -Encoding UTF8 "替换为模型名"
Set-Content -LiteralPath .\config\tg_bot_token.txt -NoNewline -Encoding UTF8 "替换为Telegram机器人TOKEN"
```

`config` 下的配置文件只允许单行非空 UTF-8 内容，均已被 `.gitignore` 排除。`llm_url.txt` 填写完整的 Chat Completions 地址。服务不读取环境变量，也没有默认值；`prompt/system.md` 在启动时加载，缺失或为空会阻止对应进程启动。`tg_bot_token.txt` 仅供机器人进程读取，不影响 FastAPI 启动。

## 启动

Windows 可直接运行项目根目录下的启动脚本：

```powershell
.\app_api.bat
.\app_tgbot.bat
```

也可分别手动启动。FastAPI：

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 48188
```

机器人作为独立进程启动：

```powershell
uv run python -m app.telegram
```

将机器人加入目标群后，发送 `@机器人用户名 生图描述` 即可触发。机器人通过长轮询接收群组和超级群组中的明确提及，生成完成后引用原消息发送图片。FastAPI 与机器人故障互不影响，但二者提交的任务会进入同一个 ComfyUI 队列。

在 BotFather 中关闭机器人的 Group Privacy Mode，使其能接收全群消息；普通消息会在本地直接忽略。提及必须位于消息开头，用户名匹配忽略大小写，用户名后必须是空白或消息结束。空指令、超过 4096 个字符的指令、私聊、机器人消息和非文本消息都不会提交生图任务。

机器人同时运行最多 2 个生图任务，另有 20 个等待位置；队列已满时立即回复稍后重试，不会阻塞长轮询。每个任务从进入等待队列起最多保留 180 秒，超时会释放处理位置。Telegram 发送和 LLM 瞬时失败会进行有限重试，ComfyUI 结果查询按 2、3、5 秒节奏退避且只读取历史结果。群组话题 ID、原消息引用关系和 WebP 图片格式保持不变。

等待队列和任务状态只存在于当前机器人进程内。重启后不会恢复已接收任务，也不保证跨重启的精确一次生成或发送。

生产环境使用现有 Windows 进程管理方式；没有既有方案时，用任务计划程序分别配置两个进程“系统启动时运行”和失败重启。防火墙只对外开放 443，不开放 48188 或 8188。Caddy 示例：

```caddyfile
api.example.com {
    reverse_proxy 127.0.0.1:48188
}
```

## API 接入指南

所有请求均需携带 `Authorization: Bearer TOKEN`。

### 1. 创建任务

```http
POST /new
Content-Type: application/json
Authorization: Bearer TOKEN

{"instruction":"一只戴耳机的橘猫"}
```

`instruction` 长度为 1～4096 个字符。成功返回 HTTP 202：

```json
{"id":"550e8400-e29b-41d4-a716-446655440000"}
```

### 2. 查询结果

```http
GET /result/{id}
Authorization: Bearer TOKEN
```

任务完成时直接返回第一张最终输出图片。任务仍在队列中返回 400，调用方可每 2～5 秒重试；生成失败返回 500。

### 3. PowerShell 示例

```powershell
$headers = @{ Authorization = "Bearer TOKEN" }
$job = Invoke-RestMethod -Method Post -Uri "https://api.example.com/new" `
  -Headers $headers -ContentType "application/json" `
  -Body (@{ instruction = "一只戴耳机的橘猫" } | ConvertTo-Json)

Invoke-RestMethod -Method Get -Uri "https://api.example.com/result/$($job.id)" `
  -Headers $headers -OutFile ".\result.webp"
```

| 状态码 | 含义 |
| --- | --- |
| 202 | 任务已提交 |
| 400 | 任务仍在处理中 |
| 401 | Token 错误或缺失 |
| 404 | 任务不存在 |
| 500 | 生成失败 |
| 422 | 参数或任务 ID 格式错误 |
| 502 | LLM、ComfyUI 或图片读取上游异常 |

## 验证与排错

```powershell
uv run pytest
```

- 启动时报配置错误：检查 `config` 下六个文本文件是否存在、是否只有一行；FastAPI 使用其中五个，机器人还使用 `tg_bot_token.txt`。`comfy_url.txt` 必须是无路径、用户信息、查询和片段的 HTTP/HTTPS 根 URL。
- 启动时报工作流错误：重新以 API Format 导出，并检查两个节点标题、节点类型和静态前缀。
- 返回 `LLM upstream error`：检查 LLM 地址、密钥、模型及兼容响应中的 `choices[0].message.content`。
- 返回 `ComfyUI upstream error`：确认 ComfyUI 可从本机访问，随后查看本机服务日志中的状态码或 `node_errors` 摘要。
- 返回 404：确认 ID 正确，且 ComfyUI 历史尚未被外部清理。
- 机器人看不到群消息：确认已加入目标群并关闭 Group Privacy Mode；无需改用 Webhook 或命令模式。
- 机器人回复“当前生成任务较多”：已有 2 个任务运行且 20 个任务等待，等待现有任务完成或超时后重试。
- 机器人回复“生成超时”：任务从进入等待队列起已超过 180 秒；检查 LLM 与 ComfyUI 延迟，后续任务会继续使用已释放的处理位置。
- Telegram 或 ComfyUI 短暂断连：查看日志中的操作、尝试次数、任务 ID 和等待秒数；进程会在有限重试或下一次历史查询中继续工作。
