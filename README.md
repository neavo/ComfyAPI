# ComfyUI 图像双向 API

复用现有 ComfyUI 队列提供文生图、Florence-2 图生文和 Telegram 机器人。

## 准备

需要 Python 3.11+、`uv`、可访问的 ComfyUI，以及 ComfyUI-Florence2 与
`PreviewAny` 节点。

安装依赖：

```powershell
uv sync
```

在 `config` 目录创建以下单行 UTF-8 文本文件：

| 文件 | 内容 |
| --- | --- |
| `api_token.txt` | API Bearer Token |
| `comfy_url.txt` | ComfyUI 根地址，如 `http://127.0.0.1:8188` |
| `llm_url.txt` | OpenAI 兼容的 Chat Completions 完整地址 |
| `llm_api_key.txt` | 文生图指令扩写使用的 LLM API Key |
| `llm_model.txt` | 文生图指令扩写使用的模型名 |
| `tg_bot_token.txt` | Telegram Bot Token，仅机器人需要 |

同时确保 `prompt/system.md` 非空。

## 工作流

服务启动时加载：

| 文件 | 用途 | `api_input` 输入 |
| --- | --- | --- |
| `workflows/generation.json` | 文生图 | `inputs.text` |
| `workflows/image_to_text.json` | Florence-2 图生文 | `inputs.image` |

两个工作流都必须以 **API Format** 导出，并恰好包含标题为 `api_input` 和
`api_output` 的节点。图生文工作流的 `api_output` 必须把 Florence caption
写入 ComfyUI history 的 `text` 字段。

## 启动

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 48188
uv run python -m app.telegram
```

Windows 也可分别运行 `app_api.bat` 和 `app_tgbot.bat`。Telegram 使用
`api_token.txt` 调用本机 `http://127.0.0.1:48188`，不直连 ComfyUI 或 LLM。

## HTTP API

所有请求均需携带：

```http
Authorization: Bearer TOKEN
```

### 文生图

```http
POST /text_to_image
Content-Type: application/json

{"instruction":"一只戴耳机的橘猫"}
```

`instruction` 长度为 1～4096 个字符。成功返回 HTTP 202：

```json
{"id":"550e8400-e29b-41d4-a716-446655440000"}
```

查询结果：

```http
GET /text_to_image/{id}
```

完成时返回图片；处理中返回空响应体 HTTP 202。

### 图生文

请求体直接发送 JPEG、PNG 或 WebP 图片，最大 10 MiB：

```http
POST /image_to_text
Content-Type: image/webp

<图片二进制>
```

成功返回 HTTP 202 和任务 ID。查询结果：

```http
GET /image_to_text/{id}
```

完成时返回 Florence-2 的原始文本：

```json
{"text":"image description\n\ntag1, tag2, tag3"}
```

处理中返回空响应体 HTTP 202。HTTP 404 表示任务不存在，500 表示工作流失败，
502 表示 ComfyUI 或 LLM 上游异常。

### PowerShell 示例

```powershell
$headers = @{ Authorization = "Bearer TOKEN" }

$generation = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:48188/text_to_image" `
  -Headers $headers -ContentType "application/json" `
  -Body (@{ instruction = "一只戴耳机的橘猫" } | ConvertTo-Json)

$caption = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:48188/image_to_text" `
  -Headers $headers -ContentType "image/webp" `
  -InFile ".\input.webp"
```

`/new` 和 `/result/{id}` 是 `0.2.x` 的文生图兼容别名，将在 `0.3.0` 删除。

## Telegram

- 私聊发送文字：生成图片。
- 私聊直接发送照片或 JPEG、PNG、WebP 图片文件：反推提示词。
- 群聊中，文字或图片说明必须以 `@机器人用户名` 开头。
- `/start` 显示使用提示。

在 BotFather 中关闭 **Group Privacy Mode** 后再将机器人加入群组。机器人使用
同一个有界队列处理两类任务；重启后不会恢复未完成任务。

## 运维与排错

- 图生文上传文件保存在 ComfyUI `input/api/image_to_text`，部署侧应按保留期清理。
- 工作流错误：确认以 API Format 导出，并检查 `api_input`、`api_output`。
- 图生文缺少文本：确认 `api_output` 在 `/history/{prompt_id}` 中产生非空 `text`。
- `LLM upstream error`：检查文生图扩写模型地址、密钥和模型名。
- `ComfyUI upstream error`：检查 ComfyUI 地址、节点安装情况和服务日志。
- 机器人看不到群消息：确认已加入群组并关闭 Group Privacy Mode。

测试：

```powershell
uv run pytest
```
